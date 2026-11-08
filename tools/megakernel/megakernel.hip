// Plan 13 stage 0: persistent-kernel feasibility on gfx90a (one MI250 GCD).
//
// (a) grid barrier and producer/consumer flag latency,
// (b) a persistent skeleton of one Qwen3.6 TP=2 decode layer's GEMV work (qkvz+ba, gdn_out,
//     router + shared gate_up + shared gate, top-8 routing, 16 gathered experts' w13 + SiLU*mul,
//     w2 + shared down with a fixed-order sum) in one launch with 4 grid barriers.
//
// Memory ordering (gfx90a, AMDGPU memory model): L2 is coherent inside one GCD, vL1D is not.
// A producer makes its global stores visible with an agent-scope release fence (s_waitcnt
// vmcnt(0) in *every* wave, which the compiler emits for the fence), the counter/flag uses
// agent-scope atomics (glc, served by L2), and a consumer issues an agent-scope acquire fence
// (buffer_wbinvl1_vol) before it reads data produced elsewhere. buffer_wbl2/invl2 are only
// needed at system scope (peer GCD), which this file does not use.
//
// Numerics: BF16 weights/activations, FP32 accumulation with explicit fmaf (built with
// -ffp-contract=off), fixed per-lane order and a butterfly wave reduction, so a row's result
// depends only on that row: bitwise repeatable and independent of M. No float atomics.

#include <hip/hip_runtime.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__
#define AGENT __HIP_MEMORY_SCOPE_AGENT

constexpr int WPB = 4;           // waves per workgroup
constexpr int NT = WPB * 64;     // threads per workgroup
constexpr int HID = 2048;
constexpr int NQKVZ = 6144, NBA = 32, N1 = NQKVZ + NBA;  // y1 = [qkvz | ba], row stride N1
constexpr int NOUTP = 2048;                              // gdn out_proj rows
constexpr int NROUTER = 256, INTER = 256, NSGU = 2 * INTER, R_STRIDE = 776;  // r = [router|sgu|sgate]
constexpr int SGATE_COL = NROUTER + NSGU;                // 768
constexpr int NE = 256, TOPK = 8;
constexpr unsigned SPIN_MAX = 1u << 22;

// sync words, one per 256-byte line: [0] count, [64] generation, [128] timeout flag, [192] bad
constexpr int S_COUNT = 0, S_GEN = 64, S_ERR = 128, S_BAD = 192;

DEVI float lo(uint32_t v) { return __uint_as_float(v << 16); }
DEVI float hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
DEVI float bf2f(uint16_t v) { return __uint_as_float(((uint32_t)v) << 16); }
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

// ------------------------------------------------------------------------------------------
// Self-resetting grid barrier (count + generation). Every workgroup of the grid must be
// co-resident; the host checks occupancy, and a spin cap turns a deadlock into a flag.
DEVI void grid_barrier(unsigned* s, int sleep) {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");  // each wave: s_waitcnt vmcnt(0)
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
        if (sleep) __builtin_amdgcn_s_sleep(1);
        if (++n > SPIN_MAX) {
          __hip_atomic_store(s + S_ERR, 1u, __ATOMIC_RELAXED, AGENT);
          break;
        }
      }
    }
  }
  __syncthreads();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");  // each wave: buffer_wbinvl1_vol
}

// (a) barrier loop; validate=1 adds a cross-workgroup data check around every barrier.
__global__ void __launch_bounds__(NT) k_barrier(unsigned* s, int iters, int sleep, unsigned* data,
                                                int validate) {
  const unsigned G = gridDim.x, b = blockIdx.x, t = threadIdx.x;
  for (int i = 0; i < iters; ++i) {
    if (validate) data[b * NT + t] = (unsigned)i * 7919u + b * NT + t;
    grid_barrier(s, sleep);
    if (validate) {
      const unsigned nb = (b + 1) % G;
      const unsigned v = data[nb * NT + t];
      if (v != (unsigned)i * 7919u + nb * NT + t) atomicAdd(s + S_BAD, 1u);
      grid_barrier(s, sleep);
    }
  }
}

// (a) producer/consumer flag ping-pong between workgroup 0 and workgroup G-1 (thread 0 each).
// f[0] = ping, f[64] = pong, f[128] = timeout flag. Workgroup 0 resets both at the end.
__global__ void __launch_bounds__(64) k_pingpong(unsigned* f, int iters) {
  if (threadIdx.x != 0) return;
  const unsigned b = blockIdx.x, G = gridDim.x;
  if (b == 0) {
    for (unsigned i = 1; i <= (unsigned)iters; ++i) {
      __hip_atomic_store(f, i, __ATOMIC_RELEASE, AGENT);
      unsigned n = 0;
      while (__hip_atomic_load(f + 64, __ATOMIC_RELAXED, AGENT) != i) {
        if (++n > SPIN_MAX) {
          __hip_atomic_store(f + 128, 1u, __ATOMIC_RELAXED, AGENT);
          i = iters;
          break;
        }
      }
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
    __hip_atomic_store(f, 0u, __ATOMIC_RELAXED, AGENT);
    __hip_atomic_store(f + 64, 0u, __ATOMIC_RELAXED, AGENT);
  } else if (b == G - 1) {
    for (unsigned i = 1; i <= (unsigned)iters; ++i) {
      unsigned n = 0;
      while (__hip_atomic_load(f, __ATOMIC_RELAXED, AGENT) != i) {
        if (++n > SPIN_MAX) {
          __hip_atomic_store(f + 128, 1u, __ATOMIC_RELAXED, AGENT);
          return;
        }
      }
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
      __hip_atomic_store(f + 64, i, __ATOMIC_RELEASE, AGENT);
    }
  }
}

// ------------------------------------------------------------------------------------------
// (b) layer skeleton
struct LayerArgs {
  const uint16_t* x;        // [M, 2048]
  const uint16_t* w_qkvz;   // [6144, 2048]
  const uint16_t* w_ba;     // [32, 2048]
  const uint16_t* w_out;    // [2048, 2048]
  const uint16_t* w_router; // [256, 2048]
  const uint16_t* w_sgu;    // [512, 2048]  gate rows 0..255, up rows 256..511
  const uint16_t* w_sgate;  // [1, 2048]
  const uint16_t* w_sdown;  // [2048, 256]
  const uint16_t* w13;      // [256, 512, 2048]
  const uint16_t* w2;       // [256, 2048, 256]
  uint16_t* y1;             // [M, 6176]
  uint16_t* h;              // [M, 2048]
  uint16_t* r;              // [M, 776]
  uint16_t* act;            // [M, 8, 256]
  uint16_t* out;            // [M, 2048]
  int* ids_out;             // [M, 8]
  float* wts_out;           // [M, 8]
  unsigned* sync;
  int mask;                 // bit p runs phase p+1 (barriers always run)
  int sleep;
};

template <int M>
DEVI void load_x(const uint16_t* x, int ld, int lane, uint4 (&xr)[M][4]) {
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int j = 0; j < 4; ++j) xr[m][j] = ld16(x + (size_t)m * ld + j * 512 + lane * 8);
}

// Dot products of two K=2048 weight rows (r1 may be null) with M activation rows; one wave.
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

DEVI uint32_t ford(float f) {
  uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
DEVI float iford(uint32_t k) { return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k); }

template <int M>
__global__ void __launch_bounds__(NT) k_layer(LayerArgs a) {
  const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6, tid = threadIdx.x;
  const int gw = blockIdx.x * WPB + wave, NW = gridDim.x * WPB;

  __shared__ int s_ids[8][TOPK];
  __shared__ float s_w[8][TOPK];
  __shared__ int s_ulist[64];
  __shared__ int s_uidx[NE];
  __shared__ signed char s_slot[64][8];
  __shared__ int s_wcnt[WPB];
  __shared__ int s_nu;

  // ---- P1: y1 = x @ [w_qkvz; w_ba]^T
  if (a.mask & 1) {
    uint4 xr[M][4];
    load_x<M>(a.x, HID, lane, xr);
    for (int p = gw; p < N1 / 2; p += NW) {
      const int n = 2 * p;
      const uint16_t* W = n < NQKVZ ? a.w_qkvz + (size_t)n * HID : a.w_ba + (size_t)(n - NQKVZ) * HID;
      float o[2][M];
      gemv_pair<M>(W, W + HID, xr, lane, o);
      if (lane == 0)
#pragma unroll
        for (int m = 0; m < M; ++m)
          *reinterpret_cast<uint32_t*>(a.y1 + (size_t)m * N1 + n) = f2bf(o[0][m]) | (f2bf(o[1][m]) << 16);
    }
  }
  grid_barrier(a.sync, a.sleep);

  // ---- P2: h = y1[:, 2048:4096] @ w_out^T
  if (a.mask & 2) {
    uint4 xr[M][4];
    load_x<M>(a.y1 + 2048, N1, lane, xr);
    for (int p = gw; p < NOUTP / 2; p += NW) {
      const int n = 2 * p;
      const uint16_t* W = a.w_out + (size_t)n * HID;
      float o[2][M];
      gemv_pair<M>(W, W + HID, xr, lane, o);
      if (lane == 0)
#pragma unroll
        for (int m = 0; m < M; ++m)
          *reinterpret_cast<uint32_t*>(a.h + (size_t)m * HID + n) = f2bf(o[0][m]) | (f2bf(o[1][m]) << 16);
    }
  }
  grid_barrier(a.sync, a.sleep);

  // ---- P3: r = h @ [w_router; w_sgu; w_sgate]^T
  if (a.mask & 4) {
    uint4 xr[M][4];
    load_x<M>(a.h, HID, lane, xr);
    for (int p = gw; p < (SGATE_COL + 2) / 2; p += NW) {
      const int n = 2 * p;
      const uint16_t *W0, *W1;
      if (n < NROUTER) {
        W0 = a.w_router + (size_t)n * HID;
        W1 = W0 + HID;
      } else if (n < SGATE_COL) {
        W0 = a.w_sgu + (size_t)(n - NROUTER) * HID;
        W1 = W0 + HID;
      } else {
        W0 = a.w_sgate;
        W1 = nullptr;
      }
      float o[2][M];
      gemv_pair<M>(W0, W1, xr, lane, o);
      if (lane == 0)
#pragma unroll
        for (int m = 0; m < M; ++m)
          *reinterpret_cast<uint32_t*>(a.r + (size_t)m * R_STRIDE + n) = f2bf(o[0][m]) | (f2bf(o[1][m]) << 16);
    }
  }
  grid_barrier(a.sync, a.sleep);

  // ---- P4a: every workgroup redundantly computes softmax top-8 (lowest index wins ties) for
  // its rows, the sorted list of distinct experts and the (expert, row) -> slot table.
  for (int m = wave; m < M; m += WPB) {
    const uint2 lv = *reinterpret_cast<const uint2*>(a.r + (size_t)m * R_STRIDE + lane * 4);
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
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int k = 0; k < TOPK; ++k) used |= (s_ids[m][k] == tid);
    const uint64_t bal = __ballot(used);
    const int pre = __popcll(bal & ((1ull << lane) - 1ull));
    if (lane == 0) s_wcnt[wave] = __popcll(bal);
    for (int i = tid; i < 64 * 8; i += NT) (&s_slot[0][0])[i] = -1;
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
    if (tid < M * TOPK) s_slot[s_uidx[s_ids[tid / TOPK][tid % TOPK]]][tid / TOPK] = (signed char)(tid % TOPK);
    if (blockIdx.x == 0 && tid < M * TOPK) {
      a.ids_out[tid] = s_ids[tid / TOPK][tid % TOPK];
      a.wts_out[tid] = s_w[tid / TOPK][tid % TOPK];
    }
    __syncthreads();
  }

  // ---- P4b: act[m, slot] = silu(h @ w13[e, :256]^T) * (h @ w13[e, 256:]^T) per distinct expert
  if (a.mask & 8) {
    uint4 xr[M][4];
    load_x<M>(a.h, HID, lane, xr);
    const int nu = s_nu;
    for (int t = gw; t < nu * INTER; t += NW) {
      const int u = t / INTER, i = t % INTER;
      const uint16_t* W = a.w13 + (size_t)s_ulist[u] * (2 * INTER) * HID;
      float o[2][M];
      gemv_pair<M>(W + (size_t)i * HID, W + (size_t)(INTER + i) * HID, xr, lane, o);
      if (lane == 0)
#pragma unroll
        for (int m = 0; m < M; ++m) {
          const int k = s_slot[u][m];
          if (k >= 0) a.act[((size_t)m * TOPK + k) * INTER + i] = (uint16_t)f2bf(silu(bfr(o[0][m])) * bfr(o[1][m]));
        }
    }
  }
  grid_barrier(a.sync, a.sleep);

  // ---- P5: out = bf16(sum_k bf16(w_k * act_k @ w2[e_k]^T)) + sigmoid(gate) * bf16(sact @ w_sdown^T)
  if (a.mask & 16) {
    const int half = lane >> 5, kk = (lane & 31) * 8;
    for (int p = gw; p < HID / 2; p += NW) {
      const int row = 2 * p + half;
      float res[M];
#pragma unroll
      for (int m = 0; m < M; ++m) {
        uint4 wv[TOPK], av[TOPK];
#pragma unroll
        for (int k = 0; k < TOPK; ++k)
          wv[k] = ld16(a.w2 + ((size_t)s_ids[m][k] * HID + row) * INTER + kk);
        const uint4 sw = ld16(a.w_sdown + (size_t)row * INTER + kk);
#pragma unroll
        for (int k = 0; k < TOPK; ++k) av[k] = ld16(a.act + ((size_t)m * TOPK + k) * INTER + kk);
        const uint4 g4 = ld16(a.r + (size_t)m * R_STRIDE + NROUTER + kk);
        const uint4 u4 = ld16(a.r + (size_t)m * R_STRIDE + NROUTER + INTER + kk);
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
        const float gate = bf2f(a.r[(size_t)m * R_STRIDE + SGATE_COL]);
        const float sig = 1.0f / (1.0f + expf(-gate));
        res[m] = bfr(routed) + bfr(sd) * sig;
      }
      if ((lane & 31) == 0)
#pragma unroll
        for (int m = 0; m < M; ++m) a.out[(size_t)m * HID + row] = (uint16_t)f2bf(res[m]);
    }
  }
}

// ------------------------------------------------------------------------------------------
extern "C" {

int mk_cu_count() {
  hipDeviceProp_t p;
  int dev = 0;
  hipGetDevice(&dev);
  hipGetDeviceProperties(&p, dev);
  return p.multiProcessorCount;
}

int mk_occupancy_barrier() {
  int n = 0;
  hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, k_barrier, NT, 0);
  return n;
}

int mk_occupancy_layer(int M) {
  int n = -1;
  switch (M) {
#define OCC(MM) \
  case MM: hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, k_layer<MM>, NT, 0); break;
    OCC(1) OCC(2) OCC(3) OCC(4) OCC(5) OCC(6) OCC(7) OCC(8)
#undef OCC
    default: break;
  }
  return n;
}

int mk_launch_barrier(int G, unsigned* s, int iters, int sleep, unsigned* data, int validate,
                      hipStream_t st) {
  hipLaunchKernelGGL(k_barrier, dim3(G), dim3(NT), 0, st, s, iters, sleep, data, validate);
  return (int)hipGetLastError();
}

int mk_launch_pingpong(int G, unsigned* f, int iters, hipStream_t st) {
  hipLaunchKernelGGL(k_pingpong, dim3(G), dim3(64), 0, st, f, iters);
  return (int)hipGetLastError();
}

int mk_launch_layer(int M, int G, const LayerArgs* a, hipStream_t st) {
  switch (M) {
#define LAUNCH(MM) \
  case MM: hipLaunchKernelGGL(k_layer<MM>, dim3(G), dim3(NT), 0, st, *a); break;
    LAUNCH(1) LAUNCH(2) LAUNCH(3) LAUNCH(4) LAUNCH(5) LAUNCH(6) LAUNCH(7) LAUNCH(8)
#undef LAUNCH
    default: return -1;
  }
  return (int)hipGetLastError();
}

}  // extern "C"
