// One-launch decode MoE block for Qwen3.5/3.6-MoE on gfx90a (plan 13 stage 1, plan 23).
//
// For M BF16 rows of the post-attention-norm hidden state x [M, 2048] (the model sends launches
// of M <= 2; M <= 8 is supported) it computes, in one persistent launch,
//   logits = x @ w_router^T                     (BF16, as the router linear stores them)
//   ids, w = softmax top-8 of logits, renormalized (fp32; lowest expert id wins ties)
//   a_k    = silu(x @ w13[e_k, :256]^T) * (x @ w13[e_k, 256:]^T)     (BF16 at the head's points)
//   routed = sum_k bf16(w_k * (a_k @ w2[e_k]^T))                       (fixed slot order)
//   shared = bf16(silu(x @ w_sgu[:256]^T) * (x @ w_sgu[256:]^T)) @ w_sdown^T
//   out    = bf16(bf16(routed) + sigmoid(x . w_sgate) * bf16(shared))
// which is the HIP decode path of Qwen2MoeSparseMoeBlock.forward (router linear, Triton
// router, shared expert MLP, fused_experts, fused_gate_sigmoid_mul_add) without the
// all-reduce, which stays outside.
//
// Schedule: phase A runs the router rows first and signals an arrival counter, then the
// shared-expert and gate rows; every workgroup waits for the router counter, routes all rows
// redundantly and runs the w13 tiles of the distinct experts; one grid barrier; phase B runs
// w2 + shared down per (row pair, token). Counters are self-resetting: the barrier count
// returns to 0, its generation only increases, and workgroup 0 clears the router counter
// after the barrier, when every workgroup has passed its wait. Nothing needs a host reset, so
// the launch is safe to capture in a graph and replay. A spin that exceeds SPIN_MAX sets the
// error word S_ERR instead of hanging; the host reads and clears it
// (hip_decode_moe_block.check_error), since the counters are then not trustworthy.
//
// The grid must be co-resident (hip_decode_moe_block_max_grid) or the barrier cannot finish.
// The workspace holds the counters, so launches that share one workspace must be ordered
// (one stream per device, as the model uses it); two concurrent launches on different streams
// need different workspaces.
//
// Variant bits (template V, chosen by the host; plan 23):
//   F_TOPK  top-8 by DPP/readlane wave max on a 32-bit key (bf16 logit order, lowest id on
//           ties) instead of the shuffle butterfly on a 64-bit key; ids and weights are
//           bitwise identical.
//   F_PIPE  the w13 and w2 loops load the next task's weights before reducing the current one
//           (two tasks of HBM reads in flight per wave); same math.
//   F_DSUM  wave/half-wave sums by DPP within rows and readlane across rows (fixed tree,
//           per row, independent of M and grid; differs in rounding from the butterfly).
//   F_STAMP per-wave s_memrealtime stamps at phase boundaries into the workspace (debug).
//
// Coherence (gfx90a): agent-scope release fence in every wave before an arrival
// (s_waitcnt vmcnt(0)), glc atomics on the counters, agent-scope acquire fence
// (buffer_wbinvl1_vol) before reading data written by other workgroups.
//
// Numerics: fp32 accumulation with explicit fmaf in a fixed per-lane order, fixed-tree wave
// reductions, no float atomics; each row's arithmetic depends only on that row, so results
// are bitwise repeatable and batch invariant for any M and grid size.

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <torch/all.h>

#include <utility>

// Same rounding as the stage-0 bench (built with -ffp-contract=off); fma only where written.
#pragma clang fp contract(off)

namespace hip_decode_moe {

#define DEVI __device__ __forceinline__
#define AGENT __HIP_MEMORY_SCOPE_AGENT

constexpr int WPB = 4;
constexpr int NT = WPB * 64;
constexpr int HID = 2048;
constexpr int NE = 256, INTER = 256, TOPK = 8, MAX_M = 8;
constexpr int NROUTER = NE, NSGU = 2 * INTER;
constexpr int R_STRIDE = NROUTER + NSGU;  // r row = [router logits | shared gate_up]
constexpr unsigned SPIN_MAX = 1u << 24;

constexpr int F_TOPK = 1, F_PIPE = 2, F_DSUM = 4, F_STAMP = 8;
constexpr int NVARIANT = 16;  // variants instantiated for M <= 2; M > 2 runs variant 0
constexpr int NSTAMP = 8, STAMP_WG = 256;

// Workspace (bytes); zero-initialised once by the caller and kept for the process.
constexpr int WS_SYNC = 0;                                    // unsigned[256]
constexpr int WS_R = 1024;                                    // bf16 [8][768]
constexpr int WS_GATE = WS_R + MAX_M * R_STRIDE * 2;          // f32 [8]
constexpr int WS_ACT = 16384;                                 // bf16 [8][8][256]
constexpr int WS_IDS = WS_ACT + MAX_M * TOPK * INTER * 2;     // i32 [8][8]
constexpr int WS_WTS = WS_IDS + MAX_M * TOPK * 4;             // f32 [8][8]
constexpr int WS_STAMP = 65536;                               // u64 [256 wg][4 wave][8]
constexpr int WS_BYTES = WS_STAMP + STAMP_WG * WPB * NSTAMP * 8;
static_assert(WS_GATE + MAX_M * 4 <= WS_ACT, "workspace layout");
static_assert(WS_WTS + MAX_M * TOPK * 4 <= WS_STAMP, "workspace layout");
// sync words, one per 256-byte line
constexpr int S_COUNT = 0, S_GEN = 64, S_ERR = 128, S_ROUTER = 192;

DEVI float lo(uint32_t v) { return __uint_as_float(v << 16); }
DEVI float hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
DEVI uint32_t f2bf(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;
  u += 0x7fffu + ((u >> 16) & 1u);
  return u >> 16;
}
DEVI float bfr(float f) { return __uint_as_float(f2bf(f) << 16); }
DEVI void dot2(uint32_t w, uint32_t x, float& a) {
  a = fmaf(lo(w), lo(x), a);
  a = fmaf(hi(w), hi(x), a);
}
DEVI void dot8(const uint4& w, const uint4& x, float& a) {
  dot2(w.x, x.x, a);
  dot2(w.y, x.y, a);
  dot2(w.z, x.z, a);
  dot2(w.w, x.w, a);
}
DEVI float wave_sum(float v) {
#pragma unroll
  for (int o = 32; o > 0; o >>= 1) v += __shfl_xor(v, o);
  return v;
}
DEVI float half_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o);
  return v;
}

// DPP within a row of 16 lanes: xor 1, xor 2 (quad_perm), half-row mirror, row mirror. After
// the four steps every lane of a row holds the same value (each step combines a lane with its
// partner, and + / max are commutative).
template <int CTRL>
DEVI uint32_t dpp(uint32_t v) {
  return (uint32_t)__builtin_amdgcn_update_dpp(0, (int)v, CTRL, 0xf, 0xf, false);
}
constexpr int DPP_XOR1 = 0xB1, DPP_XOR2 = 0x4E, DPP_HMIRROR = 0x141, DPP_MIRROR = 0x140;
DEVI float fadd_dpp_row(float v) {
  v = v + __uint_as_float(dpp<DPP_XOR1>(__float_as_uint(v)));
  v = v + __uint_as_float(dpp<DPP_XOR2>(__float_as_uint(v)));
  v = v + __uint_as_float(dpp<DPP_HMIRROR>(__float_as_uint(v)));
  v = v + __uint_as_float(dpp<DPP_MIRROR>(__float_as_uint(v)));
  return v;
}
DEVI float rl(float v, int l) { return __uint_as_float((uint32_t)__builtin_amdgcn_readlane((int)__float_as_uint(v), l)); }
DEVI float wave_sum_dpp(float v) {
  v = fadd_dpp_row(v);
  return (rl(v, 0) + rl(v, 16)) + (rl(v, 32) + rl(v, 48));
}
DEVI float half_sum_dpp(float v, int lane) {
  v = fadd_dpp_row(v);
  const float s0 = rl(v, 0) + rl(v, 16), s1 = rl(v, 32) + rl(v, 48);
  return lane < 32 ? s0 : s1;
}
DEVI uint32_t umax(uint32_t a, uint32_t b) { return a > b ? a : b; }
DEVI uint32_t wave_max_dpp(uint32_t v) {
  v = umax(v, dpp<DPP_XOR1>(v));
  v = umax(v, dpp<DPP_XOR2>(v));
  v = umax(v, dpp<DPP_HMIRROR>(v));
  v = umax(v, dpp<DPP_MIRROR>(v));
  const uint32_t a = (uint32_t)__builtin_amdgcn_readlane((int)v, 0), b = (uint32_t)__builtin_amdgcn_readlane((int)v, 16);
  const uint32_t c = (uint32_t)__builtin_amdgcn_readlane((int)v, 32), d = (uint32_t)__builtin_amdgcn_readlane((int)v, 48);
  return umax(umax(a, b), umax(c, d));
}
template <int V>
DEVI float wsum(float v) {
  if constexpr ((V & F_DSUM) != 0) return wave_sum_dpp(v);
  else return wave_sum(v);
}
template <int V>
DEVI float hsum(float v, int lane) {
  if constexpr ((V & F_DSUM) != 0) return half_sum_dpp(v, lane);
  else return half_sum(v);
}

DEVI float silu(float x) { return x / (1.0f + expf(-x)); }
DEVI uint4 ld16(const uint16_t* p) { return *reinterpret_cast<const uint4*>(p); }
DEVI uint32_t ford(float f) {
  uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
DEVI float iford(uint32_t k) { return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k); }
// Inverse of ford for a key whose low 16 bits were dropped (the value is a bf16, so ford's low
// 16 bits are 0 for positive and 0xffff for negative values and carry no order).
DEVI float iford_hi16(uint32_t key) {
  const uint32_t h = key & 0xffff0000u;
  return __uint_as_float((h & 0x80000000u) ? (h & 0x7fffffffu) : (~h & 0xffff0000u));
}

DEVI void spin_until_geq(unsigned* p, unsigned target, unsigned* err) {
  unsigned n = 0;
  while (__hip_atomic_load(p, __ATOMIC_RELAXED, AGENT) < target) {
    __builtin_amdgcn_s_sleep(1);
    if (++n > SPIN_MAX) {
      __hip_atomic_store(err, 1u, __ATOMIC_RELAXED, AGENT);
      break;
    }
  }
}

DEVI void grid_barrier(unsigned* s) {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned G = gridDim.x;
    const unsigned g = __hip_atomic_load(s + S_GEN, __ATOMIC_RELAXED, AGENT);
    const unsigned a = __hip_atomic_fetch_add(s + S_COUNT, 1u, __ATOMIC_ACQ_REL, AGENT);
    if (a == G - 1) {
      __hip_atomic_store(s + S_COUNT, 0u, __ATOMIC_RELAXED, AGENT);
      __hip_atomic_store(s + S_GEN, g + 1u, __ATOMIC_RELEASE, AGENT);
    } else {
      unsigned n = 0;
      while (__hip_atomic_load(s + S_GEN, __ATOMIC_RELAXED, AGENT) == g) {
        __builtin_amdgcn_s_sleep(1);
        if (++n > SPIN_MAX) {
          __hip_atomic_store(s + S_ERR, 1u, __ATOMIC_RELAXED, AGENT);
          break;
        }
      }
    }
  }
  __syncthreads();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}

struct Args {
  const uint16_t* x;         // [M, 2048]
  const uint16_t* w_router;  // [256, 2048]
  const uint16_t* w_sgu;     // [512, 2048]: gate rows 0..255, up rows 256..511
  const uint16_t* w_sgate;   // [2048]
  const uint16_t* w_sdown;   // [2048, 256]
  const uint16_t* w13;       // [256, 512, 2048]: gate rows 0..255, up rows 256..511
  const uint16_t* w2;        // [256, 2048, 256]
  uint16_t* out;             // [M, 2048]
  uint8_t* ws;
};

template <int M>
DEVI void load_x(const uint16_t* x, int lane, uint4 (&xr)[M][4]) {
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int j = 0; j < 4; ++j) xr[m][j] = ld16(x + (size_t)m * HID + j * 512 + lane * 8);
}

DEVI void load_pair(const uint16_t* r0, const uint16_t* r1, int lane, uint4 (&w)[2][4]) {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    w[0][j] = ld16(r0 + j * 512 + lane * 8);
    w[1][j] = r1 ? ld16(r1 + j * 512 + lane * 8) : make_uint4(0u, 0u, 0u, 0u);
  }
}

template <int M, int V>
DEVI void dots_pair(const uint4 (&w)[2][4], const uint4 (&xr)[M][4], float (&o)[2][M]) {
#pragma unroll
  for (int q = 0; q < 2; ++q)
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float a = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) dot8(w[q][j], xr[m][j], a);
      o[q][m] = wsum<V>(a);
    }
}

template <int M, int V>
DEVI void gemv_pair(const uint16_t* r0, const uint16_t* r1, const uint4 (&xr)[M][4], int lane,
                    float (&o)[2][M]) {
  uint4 w[2][4];
  load_pair(r0, r1, lane, w);
  dots_pair<M, V>(w, xr, o);
}

// Phase B task inputs: 8 routed w2 half-rows, the shared down half-row, the 8 act vectors and
// the shared gate/up values for this lane's 8 columns.
struct BT {
  uint4 wv[TOPK], av[TOPK], sw, g4, u4;
};

template <int M, int V>
__global__ void __launch_bounds__(NT) moe_block_kernel(Args a) {
  const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6, tid = threadIdx.x;
  const int gw = blockIdx.x * WPB + wave, NW = gridDim.x * WPB;
  unsigned* sync = reinterpret_cast<unsigned*>(a.ws + WS_SYNC);
  uint16_t* r = reinterpret_cast<uint16_t*>(a.ws + WS_R);
  float* gate = reinterpret_cast<float*>(a.ws + WS_GATE);
  uint16_t* act = reinterpret_cast<uint16_t*>(a.ws + WS_ACT);
  unsigned long long* stamps = reinterpret_cast<unsigned long long*>(a.ws + WS_STAMP);
  auto stamp = [&](int p) {
    if constexpr ((V & F_STAMP) != 0) {
      if (lane == 0 && blockIdx.x < STAMP_WG)
        stamps[(blockIdx.x * WPB + wave) * NSTAMP + p] = __builtin_amdgcn_s_memrealtime();
    }
  };
  stamp(0);

  __shared__ int s_ids[MAX_M][TOPK];
  __shared__ float s_w[MAX_M][TOPK];
  __shared__ int s_ulist[MAX_M * TOPK];
  __shared__ int s_uidx[NE];
  __shared__ signed char s_slot[MAX_M * TOPK][MAX_M];
  __shared__ int s_wcnt[WPB];
  __shared__ int s_nu;

  uint4 xr[M][4];
  load_x<M>(a.x, lane, xr);

  // ---- phase A: router rows (pair tasks 0..127), arrive; shared gate_up (128..383), gate (384)
  constexpr int NRP = NROUTER / 2, NSP = NSGU / 2, NTASK_A = NRP + NSP + 1;
  int t = gw;
  for (; t < NRP; t += NW) {
    const int n = 2 * t;
    const uint16_t* W = a.w_router + (size_t)n * HID;
    float o[2][M];
    gemv_pair<M, V>(W, W + HID, xr, lane, o);
    if (lane == 0)
#pragma unroll
      for (int m = 0; m < M; ++m)
        *reinterpret_cast<uint32_t*>(r + m * R_STRIDE + n) = f2bf(o[0][m]) | (f2bf(o[1][m]) << 16);
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  __syncthreads();
  if (tid == 0) __hip_atomic_fetch_add(sync + S_ROUTER, 1u, __ATOMIC_RELEASE, AGENT);
  for (; t < NTASK_A; t += NW) {
    float o[2][M];
    if (t < NRP + NSP) {
      const int n = 2 * (t - NRP);
      const uint16_t* W = a.w_sgu + (size_t)n * HID;
      gemv_pair<M, V>(W, W + HID, xr, lane, o);
      if (lane == 0)
#pragma unroll
        for (int m = 0; m < M; ++m)
          *reinterpret_cast<uint32_t*>(r + m * R_STRIDE + NROUTER + n) = f2bf(o[0][m]) | (f2bf(o[1][m]) << 16);
    } else {
      gemv_pair<M, V>(a.w_sgate, nullptr, xr, lane, o);
      if (lane == 0)
#pragma unroll
        for (int m = 0; m < M; ++m) gate[m] = o[0][m];
    }
  }
  stamp(1);
  if (tid == 0) spin_until_geq(sync + S_ROUTER, gridDim.x, sync + S_ERR);
  __syncthreads();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  stamp(2);

  // ---- routing, redundantly in every workgroup: softmax top-8 per row (lowest id wins ties),
  // the ascending list of distinct experts and the (expert, row) -> slot table.
  for (int m = wave; m < M; m += WPB) {
    const uint2 lv = *reinterpret_cast<const uint2*>(r + m * R_STRIDE + lane * 4);
    float v[4] = {lo(lv.x), hi(lv.x), lo(lv.y), hi(lv.y)};
    float top[TOPK];
    int tix[TOPK];
    if constexpr ((V & F_TOPK) != 0) {
      // 32-bit key: ford(bf16 logit) high half | (0xffff - id). The logits are bf16, so this
      // orders exactly as the 64-bit key below; unique per expert, so one max per round.
      constexpr uint32_t KO = 0x007f0000u;  // ford(-inf) high half: a taken expert
      uint32_t key[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) key[q] = (ford(v[q]) & 0xffff0000u) | (0xffffu - (uint32_t)(lane * 4 + q));
#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        const uint32_t best = wave_max_dpp(umax(umax(key[0], key[1]), umax(key[2], key[3])));
        const int idx = 0xffff - (int)(best & 0xffffu);
        top[k] = iford_hi16(best);
        tix[k] = idx;
#pragma unroll
        for (int q = 0; q < 4; ++q)
          if (idx == lane * 4 + q) key[q] = KO | (0xffffu - (uint32_t)idx);
      }
    } else {
#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        uint64_t best = 0;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const uint64_t key = ((uint64_t)ford(v[q]) << 32) | (uint32_t)(0xffff - (lane * 4 + q));
          best = key > best ? key : best;
        }
#pragma unroll
        for (int o = 32; o > 0; o >>= 1) {
          const uint32_t oh = (uint32_t)__shfl_xor((int)(uint32_t)(best >> 32), o);
          const uint32_t ol = (uint32_t)__shfl_xor((int)(uint32_t)best, o);
          const uint64_t other = ((uint64_t)oh << 32) | ol;
          best = other > best ? other : best;
        }
        const int idx = 0xffff - (int)(best & 0xffffu);
        top[k] = iford((uint32_t)(best >> 32));
        tix[k] = idx;
#pragma unroll
        for (int q = 0; q < 4; ++q)
          if (idx == lane * 4 + q) v[q] = -INFINITY;
      }
    }
    float e[TOPK], sum = 0.f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      e[k] = expf(top[k] - top[0]);
      sum += e[k];
    }
    if (lane == 0)
#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        s_ids[m][k] = tix[k];
        s_w[m][k] = e[k] / sum;
      }
  }
  __syncthreads();
  {
    bool used = false;
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int k = 0; k < TOPK; ++k) used |= (s_ids[m][k] == tid);
    const uint64_t bal = __ballot(used);
    const int pre = __popcll(bal & ((1ull << lane) - 1ull));
    if (lane == 0) s_wcnt[wave] = __popcll(bal);
    for (int i = tid; i < MAX_M * TOPK * MAX_M; i += NT) (&s_slot[0][0])[i] = -1;
    __syncthreads();
    int off = 0, tot = 0;
#pragma unroll
    for (int w = 0; w < WPB; ++w) {
      off += (w < wave) ? s_wcnt[w] : 0;
      tot += s_wcnt[w];
    }
    if (used) {
      s_ulist[off + pre] = tid;
      s_uidx[tid] = off + pre;
    }
    if (tid == 0) s_nu = tot;
    __syncthreads();
    if (tid < M * TOPK) {
      s_slot[s_uidx[s_ids[tid / TOPK][tid % TOPK]]][tid / TOPK] = (signed char)(tid % TOPK);
      if (blockIdx.x == 0) {
        reinterpret_cast<int*>(a.ws + WS_IDS)[tid] = s_ids[tid / TOPK][tid % TOPK];
        reinterpret_cast<float*>(a.ws + WS_WTS)[tid] = s_w[tid / TOPK][tid % TOPK];
      }
    }
    __syncthreads();
  }
  stamp(3);

  // ---- w13 tiles of the distinct experts: task = (expert, intermediate column i)
  {
    const int ntask = s_nu * INTER;
    auto w13_out = [&](int t2, const float (&o)[2][M]) {
      const int u = t2 / INTER, i = t2 % INTER;
      if (lane == 0)
#pragma unroll
        for (int m = 0; m < M; ++m) {
          const int k = s_slot[u][m];
          if (k >= 0) act[(m * TOPK + k) * INTER + i] = (uint16_t)f2bf(silu(bfr(o[0][m])) * bfr(o[1][m]));
        }
    };
    if constexpr ((V & F_PIPE) != 0) {
      // Next task's 8 KB is issued before this task's reduction, so two tasks are in flight.
      auto load13 = [&](int t2, uint4 (&w)[2][4]) {
        const uint16_t* W = a.w13 + (size_t)s_ulist[t2 / INTER] * NSGU * HID + (size_t)(t2 % INTER) * HID;
        load_pair(W, W + (size_t)INTER * HID, lane, w);
      };
      uint4 wc[2][4], wn[2][4];
      if (gw < ntask) load13(gw, wc);
      for (int t2 = gw; t2 < ntask; t2 += NW) {
        const bool more = t2 + NW < ntask;
        if (more) load13(t2 + NW, wn);
        float o[2][M];
        dots_pair<M, V>(wc, xr, o);
        w13_out(t2, o);
        if (more) {
#pragma unroll
          for (int q = 0; q < 2; ++q)
#pragma unroll
            for (int j = 0; j < 4; ++j) wc[q][j] = wn[q][j];
        }
      }
    } else {
      for (int t2 = gw; t2 < ntask; t2 += NW) {
        const uint16_t* W = a.w13 + (size_t)s_ulist[t2 / INTER] * NSGU * HID + (size_t)(t2 % INTER) * HID;
        float o[2][M];
        gemv_pair<M, V>(W, W + (size_t)INTER * HID, xr, lane, o);
        w13_out(t2, o);
      }
    }
  }
  stamp(4);
  grid_barrier(sync);
  if (blockIdx.x == 0 && tid == 0) __hip_atomic_store(sync + S_ROUTER, 0u, __ATOMIC_RELAXED, AGENT);
  stamp(5);

  // ---- phase B: task = (row pair p, token m); half-wave per output row
  {
    const int half = lane >> 5, kk = (lane & 31) * 8;
    const int ntask = M * (HID / 2);
    auto loadB = [&](int t3, BT& b) {
      const int m = t3 / (HID / 2), row = 2 * (t3 % (HID / 2)) + half;
#pragma unroll
      for (int k = 0; k < TOPK; ++k) b.wv[k] = ld16(a.w2 + ((size_t)s_ids[m][k] * HID + row) * INTER + kk);
      b.sw = ld16(a.w_sdown + (size_t)row * INTER + kk);
#pragma unroll
      for (int k = 0; k < TOPK; ++k) b.av[k] = ld16(act + (m * TOPK + k) * INTER + kk);
      b.g4 = ld16(r + m * R_STRIDE + NROUTER + kk);
      b.u4 = ld16(r + m * R_STRIDE + NROUTER + INTER + kk);
    };
    auto computeB = [&](int t3, const BT& b) {
      const int m = t3 / (HID / 2), row = 2 * (t3 % (HID / 2)) + half;
      float routed = 0.f;
#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        float d = 0.f;
        dot8(b.wv[k], b.av[k], d);
        d = hsum<V>(d, lane);
        routed += bfr(d * s_w[m][k]);
      }
      uint4 sa;
      sa.x = f2bf(silu(lo(b.g4.x)) * lo(b.u4.x)) | (f2bf(silu(hi(b.g4.x)) * hi(b.u4.x)) << 16);
      sa.y = f2bf(silu(lo(b.g4.y)) * lo(b.u4.y)) | (f2bf(silu(hi(b.g4.y)) * hi(b.u4.y)) << 16);
      sa.z = f2bf(silu(lo(b.g4.z)) * lo(b.u4.z)) | (f2bf(silu(hi(b.g4.z)) * hi(b.u4.z)) << 16);
      sa.w = f2bf(silu(lo(b.g4.w)) * lo(b.u4.w)) | (f2bf(silu(hi(b.g4.w)) * hi(b.u4.w)) << 16);
      float sd = 0.f;
      dot8(b.sw, sa, sd);
      sd = hsum<V>(sd, lane);
      const float sig = 1.0f / (1.0f + expf(-gate[m]));
      if ((lane & 31) == 0) a.out[(size_t)m * HID + row] = (uint16_t)f2bf(bfr(routed) + sig * bfr(sd));
    };
    if constexpr ((V & F_PIPE) != 0) {
      BT bc, bn;
      if (gw < ntask) loadB(gw, bc);
      for (int t3 = gw; t3 < ntask; t3 += NW) {
        const bool more = t3 + NW < ntask;
        if (more) loadB(t3 + NW, bn);
        computeB(t3, bc);
        if (more) bc = bn;
      }
    } else {
      for (int t3 = gw; t3 < ntask; t3 += NW) {
        BT b;
        loadB(t3, b);
        computeB(t3, b);
      }
    }
  }
  stamp(6);
}

template <int M, int V>
int max_grid_for() {
  int per_cu = 0, dev = 0;
  hipDeviceProp_t p;
  if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&p, dev) != hipSuccess) return 0;
  if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, moe_block_kernel<M, V>, NT, 0) != hipSuccess) return 0;
  return per_cu * p.multiProcessorCount;
}

// Variant dispatch for M <= 2 (all 16 variants); M > 2 has variant 0 only.
template <int M>
struct VTable {
  template <int V>
  static void launch(dim3 g, dim3 b, hipStream_t s, const Args& a) {
    hipLaunchKernelGGL((moe_block_kernel<M, V>), g, b, 0, s, a);
  }
  template <int... Vs>
  static void launch_v(int v, dim3 g, dim3 b, hipStream_t s, const Args& a, std::integer_sequence<int, Vs...>) {
    ((v == Vs ? (launch<Vs>(g, b, s, a), 0) : 0), ...);
  }
  template <int... Vs>
  static int grid_v(int v, std::integer_sequence<int, Vs...>) {
    int r = 0;
    ((v == Vs ? (r = max_grid_for<M, Vs>(), 0) : 0), ...);
    return r;
  }
};

template <int M>
void launch_m(int v, dim3 g, dim3 b, hipStream_t s, const Args& a) {
  if constexpr (M <= 2) VTable<M>::launch_v(v, g, b, s, a, std::make_integer_sequence<int, NVARIANT>{});
  else VTable<M>::template launch<0>(g, b, s, a);
}
template <int M>
int grid_m(int v) {
  if constexpr (M <= 2) return VTable<M>::grid_v(v, std::make_integer_sequence<int, NVARIANT>{});
  else return v == 0 ? max_grid_for<M, 0>() : 0;
}

}  // namespace hip_decode_moe

// Largest co-resident grid (workgroups) for M rows and a variant; 0 if it cannot be determined
// or the variant is not built for this M.
int64_t hip_decode_moe_block_max_grid(int64_t m, int64_t variant) {
  using namespace hip_decode_moe;
  if (variant < 0 || variant >= NVARIANT) return 0;
  const int v = (int)variant;
  switch (m) {
    case 1: return grid_m<1>(v);
    case 2: return grid_m<2>(v);
    case 3: return grid_m<3>(v);
    case 4: return grid_m<4>(v);
    case 5: return grid_m<5>(v);
    case 6: return grid_m<6>(v);
    case 7: return grid_m<7>(v);
    case 8: return grid_m<8>(v);
    default: return 0;
  }
}

void hip_decode_moe_block(
    torch::Tensor& x,
    torch::Tensor& w_router,
    torch::Tensor& w_sgu,
    torch::Tensor& w_sgate,
    torch::Tensor& w_sdown,
    torch::Tensor& w13,
    torch::Tensor& w2,
    torch::Tensor& out,
    torch::Tensor& ws,
    int64_t grid,
    int64_t variant) {
  using namespace hip_decode_moe;
  const int64_t M = x.size(0);
  TORCH_CHECK(M >= 1 && M <= MAX_M, "hip_decode_moe_block: M must be in 1..8");
  TORCH_CHECK(variant >= 0 && variant < NVARIANT && (M <= 2 || variant == 0),
              "hip_decode_moe_block: variant ", variant, " is not built for M=", M);
  TORCH_CHECK(x.dim() == 2 && x.size(1) == HID && x.is_contiguous(), "x must be [M, 2048] contiguous");
  TORCH_CHECK(out.sizes() == x.sizes() && out.is_contiguous(), "out must match x");
  TORCH_CHECK(w_router.size(0) == NE && w_router.size(1) == HID && w_router.is_contiguous(), "w_router");
  TORCH_CHECK(w_sgu.size(0) == NSGU && w_sgu.size(1) == HID && w_sgu.is_contiguous(), "w_sgu");
  TORCH_CHECK(w_sgate.numel() == HID && w_sgate.is_contiguous(), "w_sgate");
  TORCH_CHECK(w_sdown.size(0) == HID && w_sdown.size(1) == INTER && w_sdown.is_contiguous(), "w_sdown");
  TORCH_CHECK(w13.dim() == 3 && w13.size(0) == NE && w13.size(1) == NSGU && w13.size(2) == HID && w13.is_contiguous(), "w13");
  TORCH_CHECK(w2.dim() == 3 && w2.size(0) == NE && w2.size(1) == HID && w2.size(2) == INTER && w2.is_contiguous(), "w2");
  TORCH_CHECK(ws.numel() * ws.element_size() >= WS_BYTES && ws.is_contiguous(), "workspace too small");
  TORCH_CHECK(ws.device() == x.device(), "workspace must be on x's device");
  for (const torch::Tensor* t : {&x, &w_router, &w_sgu, &w_sgate, &w_sdown, &w13, &w2, &out})
    TORCH_CHECK(t->scalar_type() == at::kBFloat16, "hip_decode_moe_block: BF16 tensors only");
  TORCH_CHECK(grid >= 1, "grid");

  const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(x.device());
  auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
  Args a;
  a.x = reinterpret_cast<const uint16_t*>(x.data_ptr());
  a.w_router = reinterpret_cast<const uint16_t*>(w_router.data_ptr());
  a.w_sgu = reinterpret_cast<const uint16_t*>(w_sgu.data_ptr());
  a.w_sgate = reinterpret_cast<const uint16_t*>(w_sgate.data_ptr());
  a.w_sdown = reinterpret_cast<const uint16_t*>(w_sdown.data_ptr());
  a.w13 = reinterpret_cast<const uint16_t*>(w13.data_ptr());
  a.w2 = reinterpret_cast<const uint16_t*>(w2.data_ptr());
  a.out = reinterpret_cast<uint16_t*>(out.data_ptr());
  a.ws = reinterpret_cast<uint8_t*>(ws.data_ptr());
  const dim3 g((unsigned)grid), b(NT);
  const int v = (int)variant;
  switch (M) {
    case 1: launch_m<1>(v, g, b, stream, a); break;
    case 2: launch_m<2>(v, g, b, stream, a); break;
    case 3: launch_m<3>(v, g, b, stream, a); break;
    case 4: launch_m<4>(v, g, b, stream, a); break;
    case 5: launch_m<5>(v, g, b, stream, a); break;
    case 6: launch_m<6>(v, g, b, stream, a); break;
    case 7: launch_m<7>(v, g, b, stream, a); break;
    default: launch_m<8>(v, g, b, stream, a); break;
  }
  const hipError_t err = hipGetLastError();
  TORCH_CHECK(err == hipSuccess, "hip_decode_moe_block launch failed: ", hipGetErrorString(err));
}
