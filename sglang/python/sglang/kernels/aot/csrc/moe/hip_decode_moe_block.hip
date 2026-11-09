// One-launch decode MoE block for Qwen3.5/3.6-MoE on gfx90a (plans 13, 23, 28).
//
// For M BF16 rows of the post-attention-norm hidden state x [M, 2048] (M <= 8 per launch; the
// host splits larger batches) it computes, in one persistent launch,
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
// w2 + shared down. Counters are self-resetting: the barrier count returns to 0, its
// generation only increases, and workgroup 0 clears the router counter after the barrier,
// when every workgroup has passed its wait. Nothing needs a host reset, so the launch is safe
// to capture in a graph and replay. A spin that exceeds SPIN_MAX sets the error word S_ERR
// instead of hanging; the host reads and clears it (hip_decode_moe_block.check_error /
// poll_error), since the counters are then not trustworthy.
//
// Phase B, two forms:
//   per token (variants 0, 21): task = (row pair, token); a half-wave gathers the 8 routed w2
//     half-rows and the shared-down half-row of one output row.
//   expert-stationary (F_ES, plan 28): workgroup g owns a block of output row pairs and, for
//     every distinct expert and the shared down, reads that expert's contiguous slab of those
//     rows (waves 1..3; one item = one row pair of one expert, 1 KB), applies it to every token
//     routed to the expert, and keeps bf16(w_k * d) per (token, slot, row) in LDS; then the
//     workgroup sums the 8 slots in slot order and adds the shared term. Each (row, expert,
//     token) dot is the same lane mapping and tree as the per-token form, so the output is
//     bitwise equal to variant 21 for any M and grid. With F_PRE the B waves issue their first
//     two chunks of w2 loads before the grid barrier (after a release fence, so the barrier does
//     not wait for them), and wave 0 alone runs the barrier and stages the act vectors into LDS.
//
// The grid must be co-resident (hip_decode_moe_block_max_grid) or the barrier cannot finish.
// The workspace holds the counters, so launches that share one workspace must be ordered
// (one stream per device, as the model uses it); two concurrent launches on different streams
// need different workspaces.
//
// Variant bits (template V, chosen by the host):
//   F_TOPK  top-8 by DPP/readlane wave max on a 32-bit key; ids and weights are bitwise those of
//           the 64-bit shuffle butterfly.
//   F_DSUM  wave/half-wave sums by DPP within rows and readlane across rows (fixed tree, per
//           row, independent of M and grid; differs in rounding from the butterfly).
//   F_STAMP per-wave s_memrealtime stamps at phase boundaries into the workspace (debug).
//   F_LDSB  per-token phase B reads act, the shared act and the gate from LDS.
//   F_ES    expert-stationary phase B (above).
//   F_PRE   with F_ES: first w2 chunks issued before the grid barrier.
//   F_CB4   with F_ES: 4-item w2 windows instead of 8 (smaller code, fewer loads in flight).
//   F_XG    x is read from L2 per dot instead of held in registers (always on for M > XREG_MAX);
//           w13 then computes dots only for the tokens routed to each expert. Same math.
//
// Coherence (gfx90a): agent-scope release fence before an arrival (s_waitcnt vmcnt(0)), glc
// atomics on the counters, agent-scope acquire (buffer_wbinvl1_vol) before reading data written
// by other workgroups. Under F_PRE only wave 0 acquires: the L1 is per CU, and the B waves
// read other workgroups' data only after a workgroup barrier that follows wave 0's acquire.
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
constexpr int XREG_MAX = 3;  // x in registers up to this M (M = 4 needs > 256 VGPRs)

constexpr int F_TOPK = 1, F_DSUM = 4, F_STAMP = 8, F_LDSB = 16, F_ES = 128, F_PRE = 256, F_XG = 512, F_CB4 = 1024;
constexpr int V_REF = F_TOPK | F_DSUM | F_LDSB;  // plan 23 default (21): per-token phase B
constexpr int V_DEF = F_TOPK | F_DSUM | F_ES | F_PRE;  // 389
// Built (M, variant) pairs: 0 for M <= 8 (reference), 21 and its stamped build for M <= 2 (the
// plan 23 default, the bitwise reference for the expert-stationary form), the default and its
// stamped build for M <= 8; ES without pre-issue and with 4-item windows for the bench, M <= 2.
constexpr bool built_mv(int m, int v) {
  return (v == 0 && m <= 8) || ((v == V_REF || v == (V_REF | F_STAMP)) && m <= 2) ||
         ((v == V_DEF || v == (V_DEF | F_STAMP)) && m <= 8) ||
         ((v == (F_TOPK | F_DSUM | F_ES) || v == (V_DEF | F_CB4)) && m <= 2);
}
using AllV = std::integer_sequence<int, 0, V_REF, V_REF | F_STAMP, V_DEF, V_DEF | F_STAMP, F_TOPK | F_DSUM | F_ES,
                                   V_DEF | F_CB4>;
inline bool built(int64_t m, int64_t v) { return m >= 1 && m <= MAX_M && v >= 0 && v < 4096 && built_mv((int)m, (int)v); }
constexpr int NSTAMP = 16, STAMP_WG = 256;

// Expert-stationary phase B: row pairs per workgroup block, w2 items per chunk per B wave.
constexpr int PMAX = 12, RMAX = 2 * PMAX, NSLOT = TOPK + 1, CB = 8, NBW = WPB - 1;
static_assert(CB <= 64 && (CB & (CB - 1)) == 0, "one lane per chunk item");
constexpr int NBLK_MIN = (HID / 2 + PMAX - 1) / PMAX;

// Workspace (bytes); zero-initialised once by the caller and kept for the process.
constexpr int WS_SYNC = 0;                                    // unsigned[256]
constexpr int WS_R = 1024;                                    // bf16 [8][768]
constexpr int WS_GATE = WS_R + MAX_M * R_STRIDE * 2;          // f32 [8]
constexpr int WS_ACT = 16384;                                 // bf16 [8][8][256]
constexpr int WS_IDS = WS_ACT + MAX_M * TOPK * INTER * 2;     // i32 [8][8]
constexpr int WS_WTS = WS_IDS + MAX_M * TOPK * 4;             // f32 [8][8]
constexpr int WS_STAMP = 65536;                               // u64 [256 wg][4 wave][16]
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
// Load through an explicit global address-space pointer: a pointer rebuilt from integers is
// otherwise generic, and flat loads also count in lgkmcnt, so LDS waits would wait for them.
DEVI uint4 ld16g(uint64_t addr) {
#if defined(__HIP_DEVICE_COMPILE__)
  typedef const uint4 __attribute__((address_space(1)))* gptr16;
  return *reinterpret_cast<gptr16>(addr);
#else
  return *reinterpret_cast<const uint4*>(addr);
#endif
}
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
DEVI int ufirst(int v) { return __builtin_amdgcn_readfirstlane(v); }
DEVI uint4 silu_mul4(const uint4& g4, const uint4& u4) {
  uint4 sa;
  sa.x = f2bf(silu(lo(g4.x)) * lo(u4.x)) | (f2bf(silu(hi(g4.x)) * hi(u4.x)) << 16);
  sa.y = f2bf(silu(lo(g4.y)) * lo(u4.y)) | (f2bf(silu(hi(g4.y)) * hi(u4.y)) << 16);
  sa.z = f2bf(silu(lo(g4.z)) * lo(u4.z)) | (f2bf(silu(hi(g4.z)) * hi(u4.z)) << 16);
  sa.w = f2bf(silu(lo(g4.w)) * lo(u4.w)) | (f2bf(silu(hi(g4.w)) * hi(u4.w)) << 16);
  return sa;
}

// Workgroup barrier over LDS only: unlike __syncthreads it does not wait for outstanding
// global loads, so w2 loads issued before it stay in flight.
DEVI void lds_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
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

// Arrival and wait of one workgroup at the grid barrier, by thread 0.
DEVI void grid_arrive_wait(unsigned* s) {
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

DEVI void grid_barrier(unsigned* s) {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  __syncthreads();
  if (threadIdx.x == 0) grid_arrive_wait(s);
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

DEVI void load_x1(const uint16_t* x, int m, int lane, uint4 (&xm)[4]) {
#pragma unroll
  for (int j = 0; j < 4; ++j) xm[j] = ld16(x + (size_t)m * HID + j * 512 + lane * 8);
}

template <int M>
DEVI void load_x(const uint16_t* x, int lane, uint4 (&xr)[M][4]) {
#pragma unroll
  for (int m = 0; m < M; ++m) load_x1(x, m, lane, xr[m]);
}

DEVI void load_pair(const uint16_t* r0, const uint16_t* r1, int lane, uint4 (&w)[2][4]) {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    w[0][j] = ld16(r0 + j * 512 + lane * 8);
    w[1][j] = r1 ? ld16(r1 + j * 512 + lane * 8) : make_uint4(0u, 0u, 0u, 0u);
  }
}

// Row-pair dots for one token; the per-lane order and tree of every (row, token) dot are the
// same whether x comes from registers or from L2.
template <int V>
DEVI void dots_pair1(const uint4 (&w)[2][4], const uint4 (&xm)[4], float& o0, float& o1) {
  float a0 = 0.f, a1 = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) dot8(w[0][j], xm[j], a0);
#pragma unroll
  for (int j = 0; j < 4; ++j) dot8(w[1][j], xm[j], a1);
  o0 = wsum<V>(a0);
  o1 = wsum<V>(a1);
}

// Phase A row pair for every token; emit(m, o0, o1). XG: x from L2 in groups of 4 tokens.
template <int M, int V, bool XG, class F>
DEVI void gemv_pair_all(const uint16_t* r0, const uint16_t* r1, const uint16_t* x, const uint4 (&xr)[XG ? 1 : M][4],
                        int lane, F emit) {
  uint4 w[2][4];
  load_pair(r0, r1, lane, w);
  if constexpr (!XG) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float o0, o1;
      dots_pair1<V>(w, xr[m], o0, o1);
      emit(m, o0, o1);
    }
  } else {
#pragma unroll 1
    for (int m0 = 0; m0 < M; m0 += 4) {
      uint4 xg[4][4];
#pragma unroll
      for (int t = 0; t < 4; ++t) load_x1(x, m0 + t < M ? m0 + t : M - 1, lane, xg[t]);
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        if (m0 + t < M) {
          float o0, o1;
          dots_pair1<V>(w, xg[t], o0, o1);
          emit(m0 + t, o0, o1);
        }
      }
    }
  }
}

template <int M, int V>
__global__ void __launch_bounds__(NT) moe_block_kernel(Args a) {
  constexpr bool XG = M > XREG_MAX || (V & F_XG) != 0;
  constexpr bool ES = (V & F_ES) != 0;
  constexpr bool PRE = ES && (V & F_PRE) != 0;
  constexpr bool ACT_LDS = ES || (V & F_LDSB) != 0;
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
  __shared__ int s_ulist[MAX_M * TOPK + 1];
  __shared__ int s_uidx[NE];
  __shared__ signed char s_slot[MAX_M * TOPK + 1][MAX_M];
  __shared__ unsigned s_umask[MAX_M * TOPK + 1];
  __shared__ int s_wcnt[WPB];
  __shared__ int s_nu;
  __shared__ uint4 s_actv[ACT_LDS ? M * TOPK * INTER / 8 : 1];
  __shared__ uint4 s_sa[ACT_LDS ? M * INTER / 8 : 1];
  __shared__ float s_sig[M];
  __shared__ float s_part[ES ? M * NSLOT * RMAX : 1];

  uint4 xr[XG ? 1 : M][4];
  if constexpr (!XG) load_x<M>(a.x, lane, xr);

  // ---- phase A: router rows (pair tasks 0..127), arrive; shared gate_up (128..383), gate (384)
  constexpr int NRP = NROUTER / 2, NSP = NSGU / 2, NTASK_A = NRP + NSP + 1;
  int t = gw;
  for (; t < NRP; t += NW) {
    const int n = 2 * t;
    const uint16_t* W = a.w_router + (size_t)n * HID;
    gemv_pair_all<M, V, XG>(W, W + HID, a.x, xr, lane, [&](int m, float o0, float o1) {
      if (lane == 0) *reinterpret_cast<uint32_t*>(r + m * R_STRIDE + n) = f2bf(o0) | (f2bf(o1) << 16);
    });
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  __syncthreads();
  if (tid == 0) __hip_atomic_fetch_add(sync + S_ROUTER, 1u, __ATOMIC_RELEASE, AGENT);
  for (; t < NTASK_A; t += NW) {
    if (t < NRP + NSP) {
      const int n = 2 * (t - NRP);
      const uint16_t* W = a.w_sgu + (size_t)n * HID;
      gemv_pair_all<M, V, XG>(W, W + HID, a.x, xr, lane, [&](int m, float o0, float o1) {
        if (lane == 0) *reinterpret_cast<uint32_t*>(r + m * R_STRIDE + NROUTER + n) = f2bf(o0) | (f2bf(o1) << 16);
      });
    } else {
      gemv_pair_all<M, V, XG>(a.w_sgate, nullptr, a.x, xr, lane, [&](int m, float o0, float) {
        if (lane == 0) gate[m] = o0;
      });
    }
  }
  stamp(1);
  if (tid == 0) spin_until_geq(sync + S_ROUTER, gridDim.x, sync + S_ERR);
  __syncthreads();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  stamp(2);

  // ---- routing, redundantly in every workgroup: softmax top-8 per row (lowest id wins ties),
  // the ascending list of distinct experts, the (expert, row) -> slot table and per-expert
  // token masks.
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
    for (int i = tid; i < (MAX_M * TOPK + 1) * MAX_M; i += NT) (&s_slot[0][0])[i] = -1;
    if (tid <= MAX_M * TOPK) s_umask[tid] = 0u;
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
    if (tid == 0) {
      s_nu = tot;
      s_ulist[tot] = 0;  // the shared-down unit of phase B; never used as an expert id
    }
    __syncthreads();
    if (tid < M * TOPK) {
      const int u = s_uidx[s_ids[tid / TOPK][tid % TOPK]];
      s_slot[u][tid / TOPK] = (signed char)(tid % TOPK);
      atomicOr(&s_umask[u], 1u << (tid / TOPK));
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
    for (int t2 = gw; t2 < ntask; t2 += NW) {
      const int u = t2 / INTER, i = t2 % INTER;
      const uint16_t* W = a.w13 + (size_t)s_ulist[u] * NSGU * HID + (size_t)i * HID;
      auto put = [&](int m, int k, float o0, float o1) {
        if (lane == 0) act[(m * TOPK + k) * INTER + i] = (uint16_t)f2bf(silu(bfr(o0)) * bfr(o1));
      };
      if constexpr (!XG) {
        gemv_pair_all<M, V, false>(W, W + (size_t)INTER * HID, a.x, xr, lane, [&](int m, float o0, float o1) {
          const int k = s_slot[u][m];
          if (k >= 0) put(m, k, o0, o1);
        });
      } else {
        // Only the tokens routed to this expert.
        uint4 w[2][4];
        load_pair(W, W + (size_t)INTER * HID, lane, w);
        unsigned mask = (unsigned)ufirst((int)s_umask[u]);
        while (mask) {
          const int m = __builtin_ctz(mask);
          mask &= mask - 1u;
          uint4 xm[4];
          load_x1(a.x, m, lane, xm);
          float o0, o1;
          dots_pair1<V>(w, xm, o0, o1);
          put(m, ufirst(s_slot[u][m]), o0, o1);
        }
      }
    }
  }
  stamp(4);

  if constexpr (!ES) {
    stamp(5);
    grid_barrier(sync);
    if (blockIdx.x == 0 && tid == 0) __hip_atomic_store(sync + S_ROUTER, 0u, __ATOMIC_RELAXED, AGENT);
    // ---- phase B per token: task = (row pair p, token m); half-wave per output row
    const int half = lane >> 5, kk = (lane & 31) * 8;
    const int ntask = M * (HID / 2);
    if constexpr (ACT_LDS) {
      // Stage the act vectors, the shared expert's act and the sigmoid gate once per
      // workgroup, so a task's only global loads are its w2 and shared-down half-rows.
      for (int i = tid; i < M * TOPK * INTER / 8; i += NT) s_actv[i] = ld16(act + i * 8);
      for (int i = tid; i < M * INTER / 8; i += NT) {
        const int m = i / (INTER / 8), c = (i % (INTER / 8)) * 8;
        s_sa[i] = silu_mul4(ld16(r + m * R_STRIDE + NROUTER + c), ld16(r + m * R_STRIDE + NROUTER + INTER + c));
      }
      if (tid < M) s_sig[tid] = 1.0f / (1.0f + expf(-gate[tid]));
      __syncthreads();
    }
    stamp(6);
    for (int t3 = gw; t3 < ntask; t3 += NW) {
      const int m = t3 / (HID / 2), row = 2 * (t3 % (HID / 2)) + half;
      uint4 wv[TOPK], sw;
#pragma unroll
      for (int k = 0; k < TOPK; ++k) wv[k] = ld16(a.w2 + ((size_t)s_ids[m][k] * HID + row) * INTER + kk);
      sw = ld16(a.w_sdown + (size_t)row * INTER + kk);
      float routed = 0.f;
#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        float d = 0.f;
        if constexpr (ACT_LDS) dot8(wv[k], s_actv[((m * TOPK + k) * INTER + kk) / 8], d);
        else dot8(wv[k], ld16(act + (m * TOPK + k) * INTER + kk), d);
        d = hsum<V>(d, lane);
        routed += bfr(d * s_w[m][k]);
      }
      uint4 sa;
      float sig;
      if constexpr (ACT_LDS) {
        sa = s_sa[(m * INTER + kk) / 8];
        sig = s_sig[m];
      } else {
        sa = silu_mul4(ld16(r + m * R_STRIDE + NROUTER + kk), ld16(r + m * R_STRIDE + NROUTER + INTER + kk));
        sig = 1.0f / (1.0f + expf(-gate[m]));
      }
      float sd = 0.f;
      dot8(sw, sa, sd);
      sd = hsum<V>(sd, lane);
      if ((lane & 31) == 0) a.out[(size_t)m * HID + row] = (uint16_t)f2bf(bfr(routed) + sig * bfr(sd));
    }
    stamp(7);
    stamp(8);
  } else {
    // ---- phase B expert-stationary. Units: the s_nu distinct experts, then the shared down.
    // Workgroup block b owns row pairs [p0, p0 + np); its items are (unit, row pair), unit-major,
    // taken round-robin by the B waves 1..3; an item is one 1 KB contiguous piece of w2.
    // Wave-uniform indices go through readfirstlane so the item arithmetic stays scalar.
    constexpr int CB = (V & F_CB4) != 0 ? 4 : hip_decode_moe::CB;
    const int half = lane >> 5, kk = (lane & 31) * 8, bw = ufirst(wave) - 1;
    const int nu = ufirst(s_nu), NUe = nu + 1, G = gridDim.x;
    const int nblk = G > NBLK_MIN ? G : NBLK_MIN;
    int b = blockIdx.x, p0 = 0, np = 0, nit = 0, nch = 0;
    auto set_block = [&]() {
      p0 = (int)((long)b * (HID / 2) / nblk);
      np = (int)((long)(b + 1) * (HID / 2) / nblk) - p0;
      nit = NUe * np;
      const int nwi = (bw >= 0 && nit > bw) ? (nit - bw + NBW - 1) / NBW : 0;
      nch = (nwi + CB - 1) / CB;
    };
    // Lane l < CB describes item l of window c (clamped to the last item): its unit, row pair
    // within the block, slab base, token mask and packed slots (4 bits per token: the slot,
    // TOPK for the shared down, 15 if not routed). The per-item loop reads them with readlane,
    // so it has no divisions and no LDS waits; one dot body serves routed and shared items, so
    // the code stays small (instruction-cache bound otherwise).
    struct ItemInfo {
      int u, pl, valid, mask, kp;
      const uint16_t* base;
    };
    auto item_info = [&](int c, bool slots) {
      const int il0 = bw + NBW * (c * CB + (lane & (CB - 1)));
      const int il = il0 < nit ? il0 : nit - 1;
      ItemInfo f;
      f.u = il / np;
      f.pl = il - f.u * np;
      f.valid = il0 < nit;
      const bool routed = f.u < nu;
      const int e = s_ulist[f.u];
      f.base = (routed ? a.w2 + (size_t)e * HID * INTER : a.w_sdown) + (size_t)(2 * (p0 + f.pl)) * INTER;
      f.mask = 0;
      f.kp = 0;
      if (slots) {
        f.mask = routed ? (int)s_umask[f.u] : (1 << M) - 1;
#pragma unroll
        for (int m = 0; m < M; ++m) {
          const int k = routed ? (int)s_slot[f.u][m] : TOPK;
          f.kp |= (k < 0 ? 15 : k) << (4 * m);
        }
      }
      return f;
    };
    auto rl_addr = [&](const uint16_t* p, int j) {
      const uint64_t v = reinterpret_cast<uint64_t>(p);
      const uint32_t l = (uint32_t)__builtin_amdgcn_readlane((int)(uint32_t)v, j);
      const uint32_t h = (uint32_t)__builtin_amdgcn_readlane((int)(uint32_t)(v >> 32), j);
      return ((uint64_t)h << 32) | l;
    };
    auto load_chunk = [&](int c, uint4 (&buf)[CB]) {
      const ItemInfo f = item_info(c, false);
      const uint32_t off = (uint32_t)(half * INTER + kk) * 2u;
#pragma unroll
      for (int j = 0; j < CB; ++j) buf[j] = ld16g(rl_addr(f.base, j) + off);
    };
    auto comp_chunk = [&](int c, const uint4 (&buf)[CB]) {
      const ItemInfo f = item_info(c, true);
#pragma unroll
      for (int j = 0; j < CB; ++j) {
        if (!__builtin_amdgcn_readlane(f.valid, j)) break;
        const int rl = 2 * __builtin_amdgcn_readlane(f.pl, j) + half;
        unsigned mask = (unsigned)__builtin_amdgcn_readlane(f.mask, j);
        const unsigned kp = (unsigned)__builtin_amdgcn_readlane(f.kp, j);
        while (mask) {
          const int m = __builtin_ctz(mask);
          mask &= mask - 1u;
          const int k = (int)((kp >> (4 * m)) & 15u);
          const bool sh = k == TOPK;
          const uint4 av = sh ? s_sa[(m * INTER + kk) / 8] : s_actv[((m * TOPK + (k & 7)) * INTER + kk) / 8];
          const float w = sh ? 1.0f : s_w[m][k & 7];
          float d = 0.f;
          dot8(buf[j], av, d);
          d = hsum<V>(d, lane);
          // d * 1.0f == d, so the shared term is bf16(sd) exactly as in the per-token form.
          if ((lane & 31) == 0) s_part[(m * NSLOT + k) * RMAX + rl] = bfr(d * w);
        }
      }
    };
    auto combine = [&]() {
      for (int i = tid; i < M * 2 * np; i += NT) {
        const int m = i / (2 * np), rr = i % (2 * np);
        const float* p = s_part + m * NSLOT * RMAX + rr;
        float routed = 0.f;
#pragma unroll
        for (int k = 0; k < TOPK; ++k) routed += p[k * RMAX];
        a.out[(size_t)m * HID + 2 * p0 + rr] = (uint16_t)f2bf(bfr(routed) + s_sig[m] * p[TOPK * RMAX]);
      }
    };
    // Wave 0 stages act, the shared act and the gate; two tokens' loads are issued before any
    // is waited on, so M <= 2 costs one L2 round trip.
    auto stage = [&]() {
      const float gv = gate[lane < M ? lane : 0];
      const int c = (lane & 31) * 8;
#pragma unroll
      for (int m0 = 0; m0 < M; m0 += 2) {
        uint4 av[2][4], g4[2], u4[2];
#pragma unroll
        for (int t = 0; t < 2; ++t) {
          if (m0 + t < M) {
            const int m = m0 + t;
#pragma unroll
            for (int q = 0; q < 4; ++q) av[t][q] = ld16(act + (size_t)(m * TOPK * INTER / 8 + q * 64 + lane) * 8);
            g4[t] = ld16(r + m * R_STRIDE + NROUTER + c);  // lanes 32..63 repeat lanes 0..31
            u4[t] = ld16(r + m * R_STRIDE + NROUTER + INTER + c);
          }
        }
#pragma unroll
        for (int t = 0; t < 2; ++t) {
          if (m0 + t < M) {
            const int m = m0 + t;
#pragma unroll
            for (int q = 0; q < 4; ++q) s_actv[m * TOPK * INTER / 8 + q * 64 + lane] = av[t][q];
            s_sa[m * INTER / 8 + (lane & 31)] = silu_mul4(g4[t], u4[t]);
          }
        }
      }
      if (lane < M) s_sig[lane] = 1.0f / (1.0f + expf(-gv));
    };

    uint4 bA[CB], bB[CB];
    set_block();
    if constexpr (PRE) {
      // Release this wave's act stores now, so neither the arrival nor the barrier below waits
      // for the w2 loads issued next.
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
      if (nch > 0) load_chunk(0, bA);
      if (nch > 1) load_chunk(1, bB);
      stamp(5);
      lds_barrier();
      if (wave == 0) {
        if (lane == 0) grid_arrive_wait(sync);
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        if (blockIdx.x == 0 && tid == 0) __hip_atomic_store(sync + S_ROUTER, 0u, __ATOMIC_RELAXED, AGENT);
        stage();
      }
      lds_barrier();
    } else {
      stamp(5);
      grid_barrier(sync);
      if (blockIdx.x == 0 && tid == 0) __hip_atomic_store(sync + S_ROUTER, 0u, __ATOMIC_RELAXED, AGENT);
      if (wave == 0) stage();
      __syncthreads();
    }
    stamp(6);
    // Window c is computed from bA while window c + 1 loads into bB; then bB moves to bA (after
    // its loads land, which window c + 1 needs next anyway) and window c + 2 starts loading.
    // One compute site keeps the code small.
    for (bool first = true;; first = false) {
      if (!PRE || !first) {
        if (nch > 0) load_chunk(0, bA);
        if (nch > 1) load_chunk(1, bB);
      }
      for (int c = 0; c < nch; ++c) {
        comp_chunk(c, bA);
        if (first && c == 0) stamp(7);
        if (c + 1 < nch) {
#pragma unroll
          for (int j = 0; j < CB; ++j) bA[j] = bB[j];
          if (c + 2 < nch) load_chunk(c + 2, bB);
        }
      }
      lds_barrier();
      combine();
      b += G;
      if (b >= nblk) break;
      lds_barrier();
      set_block();
    }
    stamp(8);
  }
}

template <int M, int V>
int max_grid_for() {
  int per_cu = 0, dev = 0;
  hipDeviceProp_t p;
  if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&p, dev) != hipSuccess) return 0;
  if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, moe_block_kernel<M, V>, NT, 0) != hipSuccess) return 0;
  return per_cu * p.multiProcessorCount;
}

template <int M>
struct VTable {
  template <int V>
  static void launch(dim3 g, dim3 b, hipStream_t s, const Args& a) {
    if constexpr (built_mv(M, V)) hipLaunchKernelGGL((moe_block_kernel<M, V>), g, b, 0, s, a);
  }
  template <int V>
  static int grid() {
    if constexpr (built_mv(M, V)) return max_grid_for<M, V>();
    else return 0;
  }
  template <int... Vs>
  static void launch_v(int v, dim3 g, dim3 b, hipStream_t s, const Args& a, std::integer_sequence<int, Vs...>) {
    ((v == Vs ? (launch<Vs>(g, b, s, a), 0) : 0), ...);
  }
  template <int... Vs>
  static int grid_v(int v, std::integer_sequence<int, Vs...>) {
    int r = 0;
    ((v == Vs ? (r = grid<Vs>(), 0) : 0), ...);
    return r;
  }
};

template <int M>
void launch_m(int v, dim3 g, dim3 b, hipStream_t s, const Args& a) {
  VTable<M>::launch_v(v, g, b, s, a, AllV{});
}
template <int M>
int grid_m(int v) {
  return VTable<M>::grid_v(v, AllV{});
}

}  // namespace hip_decode_moe

// Largest co-resident grid (workgroups) for M rows and a variant; 0 if it cannot be determined
// or the variant is not built for this M.
int64_t hip_decode_moe_block_max_grid(int64_t m, int64_t variant) {
  using namespace hip_decode_moe;
  if (!built(m, variant)) return 0;
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
  TORCH_CHECK(built(M, variant),
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
