// One-launch decode MoE block for Qwen3.5/3.6-MoE on gfx90a (plan 13 stage 1).
//
// For M <= 8 BF16 rows of the post-attention-norm hidden state x [M, 2048] it computes, in one
// persistent launch of one workgroup per CU,
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
// the launch is safe to capture in a graph and replay.
//
// Coherence (gfx90a): agent-scope release fence in every wave before an arrival
// (s_waitcnt vmcnt(0)), glc atomics on the counters, agent-scope acquire fence
// (buffer_wbinvl1_vol) before reading data written by other workgroups.
//
// Numerics: fp32 accumulation with explicit fmaf in a fixed per-lane order, butterfly wave
// reductions, no float atomics; each row's arithmetic depends only on that row, so results
// are bitwise repeatable and batch invariant for M = 1..8 and any grid size.

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <torch/all.h>

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

// Workspace (bytes); zero-initialised once by the caller and kept for the process.
constexpr int WS_SYNC = 0;                                    // unsigned[256]
constexpr int WS_R = 1024;                                    // bf16 [8][768]
constexpr int WS_GATE = WS_R + MAX_M * R_STRIDE * 2;          // f32 [8]
constexpr int WS_ACT = 16384;                                 // bf16 [8][8][256]
constexpr int WS_IDS = WS_ACT + MAX_M * TOPK * INTER * 2;     // i32 [8][8]
constexpr int WS_WTS = WS_IDS + MAX_M * TOPK * 4;             // f32 [8][8]
constexpr int WS_BYTES = 65536;
static_assert(WS_GATE + MAX_M * 4 <= WS_ACT, "workspace layout");
static_assert(WS_WTS + MAX_M * TOPK * 4 <= WS_BYTES, "workspace layout");
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
DEVI float silu(float x) { return x / (1.0f + expf(-x)); }
DEVI uint4 ld16(const uint16_t* p) { return *reinterpret_cast<const uint4*>(p); }
DEVI uint32_t ford(float f) {
  uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
DEVI float iford(uint32_t k) { return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k); }

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

template <int M>
DEVI void gemv_pair(const uint16_t* r0, const uint16_t* r1, const uint4 (&xr)[M][4], int lane,
                    float (&o)[2][M]) {
  uint4 w[2][4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    w[0][j] = ld16(r0 + j * 512 + lane * 8);
    w[1][j] = r1 ? ld16(r1 + j * 512 + lane * 8) : make_uint4(0u, 0u, 0u, 0u);
  }
#pragma unroll
  for (int q = 0; q < 2; ++q)
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float a = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) dot8(w[q][j], xr[m][j], a);
      o[q][m] = wave_sum(a);
    }
}

template <int M>
__global__ void __launch_bounds__(NT) moe_block_kernel(Args a) {
  const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6, tid = threadIdx.x;
  const int gw = blockIdx.x * WPB + wave, NW = gridDim.x * WPB;
  unsigned* sync = reinterpret_cast<unsigned*>(a.ws + WS_SYNC);
  uint16_t* r = reinterpret_cast<uint16_t*>(a.ws + WS_R);
  float* gate = reinterpret_cast<float*>(a.ws + WS_GATE);
  uint16_t* act = reinterpret_cast<uint16_t*>(a.ws + WS_ACT);

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
    gemv_pair<M>(W, W + HID, xr, lane, o);
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
      gemv_pair<M>(W, W + HID, xr, lane, o);
      if (lane == 0)
#pragma unroll
        for (int m = 0; m < M; ++m)
          *reinterpret_cast<uint32_t*>(r + m * R_STRIDE + NROUTER + n) = f2bf(o[0][m]) | (f2bf(o[1][m]) << 16);
    } else {
      gemv_pair<M>(a.w_sgate, nullptr, xr, lane, o);
      if (lane == 0)
#pragma unroll
        for (int m = 0; m < M; ++m) gate[m] = o[0][m];
    }
  }
  if (tid == 0) spin_until_geq(sync + S_ROUTER, gridDim.x, sync + S_ERR);
  __syncthreads();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  // ---- routing, redundantly in every workgroup: softmax top-8 per row (lowest id wins ties),
  // the ascending list of distinct experts and the (expert, row) -> slot table.
  for (int m = wave; m < M; m += WPB) {
    const uint2 lv = *reinterpret_cast<const uint2*>(r + m * R_STRIDE + lane * 4);
    float v[4] = {lo(lv.x), hi(lv.x), lo(lv.y), hi(lv.y)};
    float top[TOPK];
    int tix[TOPK];
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

  // ---- w13 tiles of the distinct experts: task = (expert, intermediate column i)
  {
    const int nu = s_nu;
    for (int t2 = gw; t2 < nu * INTER; t2 += NW) {
      const int u = t2 / INTER, i = t2 % INTER;
      const uint16_t* W = a.w13 + (size_t)s_ulist[u] * NSGU * HID;
      float o[2][M];
      gemv_pair<M>(W + (size_t)i * HID, W + (size_t)(INTER + i) * HID, xr, lane, o);
      if (lane == 0)
#pragma unroll
        for (int m = 0; m < M; ++m) {
          const int k = s_slot[u][m];
          if (k >= 0) act[(m * TOPK + k) * INTER + i] = (uint16_t)f2bf(silu(bfr(o[0][m])) * bfr(o[1][m]));
        }
    }
  }
  grid_barrier(sync);
  if (blockIdx.x == 0 && tid == 0) __hip_atomic_store(sync + S_ROUTER, 0u, __ATOMIC_RELAXED, AGENT);

  // ---- phase B: task = (row pair p, token m); half-wave per output row
  {
    const int half = lane >> 5, kk = (lane & 31) * 8;
    for (int t3 = gw; t3 < M * (HID / 2); t3 += NW) {
      const int m = t3 / (HID / 2), row = 2 * (t3 % (HID / 2)) + half;
      uint4 wv[TOPK], av[TOPK];
#pragma unroll
      for (int k = 0; k < TOPK; ++k) wv[k] = ld16(a.w2 + ((size_t)s_ids[m][k] * HID + row) * INTER + kk);
      const uint4 sw = ld16(a.w_sdown + (size_t)row * INTER + kk);
#pragma unroll
      for (int k = 0; k < TOPK; ++k) av[k] = ld16(act + (m * TOPK + k) * INTER + kk);
      const uint4 g4 = ld16(r + m * R_STRIDE + NROUTER + kk);
      const uint4 u4 = ld16(r + m * R_STRIDE + NROUTER + INTER + kk);
      float routed = 0.f;
#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        float d = 0.f;
        dot8(wv[k], av[k], d);
        d = half_sum(d);
        routed += bfr(d * s_w[m][k]);
      }
      uint4 sa;
      sa.x = f2bf(silu(lo(g4.x)) * lo(u4.x)) | (f2bf(silu(hi(g4.x)) * hi(u4.x)) << 16);
      sa.y = f2bf(silu(lo(g4.y)) * lo(u4.y)) | (f2bf(silu(hi(g4.y)) * hi(u4.y)) << 16);
      sa.z = f2bf(silu(lo(g4.z)) * lo(u4.z)) | (f2bf(silu(hi(g4.z)) * hi(u4.z)) << 16);
      sa.w = f2bf(silu(lo(g4.w)) * lo(u4.w)) | (f2bf(silu(hi(g4.w)) * hi(u4.w)) << 16);
      float sd = 0.f;
      dot8(sw, sa, sd);
      sd = half_sum(sd);
      const float sig = 1.0f / (1.0f + expf(-gate[m]));
      if ((lane & 31) == 0) a.out[(size_t)m * HID + row] = (uint16_t)f2bf(bfr(routed) + sig * bfr(sd));
    }
  }
}

template <int M>
int max_grid_for() {
  int per_cu = 0, dev = 0;
  hipDeviceProp_t p;
  if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&p, dev) != hipSuccess) return 0;
  if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, moe_block_kernel<M>, NT, 0) != hipSuccess) return 0;
  return per_cu * p.multiProcessorCount;
}

}  // namespace hip_decode_moe

// Largest co-resident grid (workgroups) for M rows; 0 if it cannot be determined.
int64_t hip_decode_moe_block_max_grid(int64_t m) {
  using namespace hip_decode_moe;
  switch (m) {
    case 1: return max_grid_for<1>();
    case 2: return max_grid_for<2>();
    case 3: return max_grid_for<3>();
    case 4: return max_grid_for<4>();
    case 5: return max_grid_for<5>();
    case 6: return max_grid_for<6>();
    case 7: return max_grid_for<7>();
    case 8: return max_grid_for<8>();
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
    int64_t grid) {
  using namespace hip_decode_moe;
  const int64_t M = x.size(0);
  TORCH_CHECK(M >= 1 && M <= MAX_M, "hip_decode_moe_block: M must be in 1..8");
  TORCH_CHECK(x.dim() == 2 && x.size(1) == HID && x.is_contiguous(), "x must be [M, 2048] contiguous");
  TORCH_CHECK(out.sizes() == x.sizes() && out.is_contiguous(), "out must match x");
  TORCH_CHECK(w_router.size(0) == NE && w_router.size(1) == HID && w_router.is_contiguous(), "w_router");
  TORCH_CHECK(w_sgu.size(0) == NSGU && w_sgu.size(1) == HID && w_sgu.is_contiguous(), "w_sgu");
  TORCH_CHECK(w_sgate.numel() == HID && w_sgate.is_contiguous(), "w_sgate");
  TORCH_CHECK(w_sdown.size(0) == HID && w_sdown.size(1) == INTER && w_sdown.is_contiguous(), "w_sdown");
  TORCH_CHECK(w13.dim() == 3 && w13.size(0) == NE && w13.size(1) == NSGU && w13.size(2) == HID && w13.is_contiguous(), "w13");
  TORCH_CHECK(w2.dim() == 3 && w2.size(0) == NE && w2.size(1) == HID && w2.size(2) == INTER && w2.is_contiguous(), "w2");
  TORCH_CHECK(ws.numel() * ws.element_size() >= WS_BYTES && ws.is_contiguous(), "workspace too small");
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
  switch (M) {
    case 1: hipLaunchKernelGGL(moe_block_kernel<1>, g, b, 0, stream, a); break;
    case 2: hipLaunchKernelGGL(moe_block_kernel<2>, g, b, 0, stream, a); break;
    case 3: hipLaunchKernelGGL(moe_block_kernel<3>, g, b, 0, stream, a); break;
    case 4: hipLaunchKernelGGL(moe_block_kernel<4>, g, b, 0, stream, a); break;
    case 5: hipLaunchKernelGGL(moe_block_kernel<5>, g, b, 0, stream, a); break;
    case 6: hipLaunchKernelGGL(moe_block_kernel<6>, g, b, 0, stream, a); break;
    case 7: hipLaunchKernelGGL(moe_block_kernel<7>, g, b, 0, stream, a); break;
    default: hipLaunchKernelGGL(moe_block_kernel<8>, g, b, 0, stream, a); break;
  }
  const hipError_t err = hipGetLastError();
  TORCH_CHECK(err == hipSuccess, "hip_decode_moe_block launch failed: ", hipGetErrorString(err));
}
