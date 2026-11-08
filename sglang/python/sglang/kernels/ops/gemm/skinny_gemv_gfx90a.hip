// Skinny BF16 GEMV y[m, n] = x[m, :] . w[n, :] (+ bias[n]) for M <= 16 rows on gfx90a.
//
// Built at runtime by skinny_gemv_hip.py with hipcc (-ffp-contract=off) and loaded through
// ctypes; launches go on the caller's (torch's current) stream, so they can be graph-captured.
//
// Layout: w is row-major [N, K] bf16. LPR lanes share one row (LPR in {8, 16, 64}), so a wave
// covers 64 / LPR rows per load instruction, each lane reading one dwordx4 (8 bf16) of a
// segment of LPR * 8 elements; K = NSEG * LPR * 8. A wave holds R such row slots, a row group
// is split over P waves along K (P * CH = NSEG segments) and a workgroup holds G row groups.
// Each lane issues all R * CH weight loads before it touches any (no LDS staging), then runs a
// per-lane FP32 fmaf chain, a fixed DPP butterfly across its LPR lanes and a fixed wave-order
// sum of the P parts through LDS.
//
// Numerics and batch invariance: BF16 in, FP32 accumulation with explicit fmaf, BF16 out
// (round to nearest even). Row m's arithmetic is the same instruction sequence whatever M is
// (the M bucket MB only decides how many rows are computed; rows >= M re-read row M-1 and are
// not stored), the schedule depends on (N, K) and the workgroup index only, and there are no
// atomics, so a row's bits do not depend on M or on the run. The optional K stagger rotates
// which segment a workgroup reads first (and the order in which its lanes sum segments) as a
// function of the workgroup index only.

#include <hip/hip_runtime.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

DEVI float lo(uint32_t v) { return __uint_as_float(v << 16); }
DEVI float hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
DEVI float bf2f(uint16_t v) { return __uint_as_float(((uint32_t)v) << 16); }
DEVI uint16_t f2bf(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)0x7fc0u;
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
DEVI void dot8(const u32x4 w, const u32x4 x, float& a) {
  a = fmaf(lo(w.x), lo(x.x), a);
  a = fmaf(hi(w.x), hi(x.x), a);
  a = fmaf(lo(w.y), lo(x.y), a);
  a = fmaf(hi(w.y), hi(x.y), a);
  a = fmaf(lo(w.z), lo(x.z), a);
  a = fmaf(hi(w.z), hi(x.z), a);
  a = fmaf(lo(w.w), lo(x.w), a);
  a = fmaf(hi(w.w), hi(x.w), a);
}

template <int CTRL, int ROW_MASK>
DEVI float dpp(float v) {
  // Lanes outside ROW_MASK (or with an invalid source) get 0.
  return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), CTRL, ROW_MASK, 0xF, false));
}

// Fixed butterfly; lane 63 ends with the sum of all 64 lanes.
DEVI float wave_sum63(float v) {
  v += dpp<0xB1, 0xF>(v);   // quad_perm [1,0,3,2]
  v += dpp<0x4E, 0xF>(v);   // quad_perm [2,3,0,1]
  v += dpp<0x141, 0xF>(v);  // row_half_mirror
  v += dpp<0x140, 0xF>(v);  // row_mirror: every lane holds its 16-lane row sum
  v += dpp<0x142, 0xA>(v);  // row_bcast15 into rows 1, 3
  v += dpp<0x143, 0xC>(v);  // row_bcast31 into rows 2, 3
  return v;
}

// Fixed butterfly inside groups of LPR lanes; returns the group sum in the writer lane
// (lane 63 for LPR = 64, every lane of the group for LPR = 8 or 16).
template <int LPR>
DEVI float group_sum(float v) {
  if constexpr (LPR == 64) {
    return wave_sum63(v);
  } else {
    v += dpp<0xB1, 0xF>(v);
    v += dpp<0x4E, 0xF>(v);
    v += dpp<0x141, 0xF>(v);
    if constexpr (LPR == 16) v += dpp<0x140, 0xF>(v);
    return v;
  }
}

template <bool NT>
DEVI u32x4 ldw(const u32x4* p) {
  if constexpr (NT) {
    return __builtin_nontemporal_load(p);
  } else {
    return *p;
  }
}

template <int MB, int R, int CH, int G, int P, int LPR, bool NT>
__global__ void __launch_bounds__(64 * G * P)
    skg_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
               const uint16_t* __restrict__ bias, uint16_t* __restrict__ y, int M, int N, int sx,
               int sy, int stag) {
  constexpr int NSEG = CH * P;
  constexpr int SW = LPR * 8;          // elements per segment
  constexpr int K = NSEG * SW;
  constexpr int SLOTS = 64 / LPR;      // rows per load instruction
  constexpr int RW = SLOTS * R;        // rows per wave
  __shared__ float red[G][P][MB][RW];
  const int lane = threadIdx.x & 63;
  const int sub = lane % LPR;
  const int rl = lane / LPR;
  const int wave = threadIdx.x >> 6;
  const int g = wave / P;
  const int p = wave - g * P;
  const int n0 = blockIdx.x * (G * RW) + g * RW;
  const int rot = (int)((blockIdx.x * (unsigned)stag) % (unsigned)NSEG);

  u32x4 wv[R][CH];
#pragma unroll
  for (int j = 0; j < CH; ++j) {
    int seg = p * CH + j + rot;
    if (seg >= NSEG) seg -= NSEG;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int n = min(n0 + r * SLOTS + rl, N - 1);
      wv[r][j] = ldw<NT>(reinterpret_cast<const u32x4*>(w + (size_t)n * K + seg * SW + sub * 8));
    }
  }

  // Rows are computed in chunks of MU so that only MU rows of x are live at a time (a fully
  // unrolled MB = 16 would hoist every x load and run one wave per SIMD). The per-row
  // arithmetic is the same in every chunk and bucket, so a row's bits do not depend on M.
  constexpr int MU = MB < 4 ? MB : 4;
  const bool writer = (LPR == 64) ? (lane == 63) : (sub == 0);
#pragma unroll 1
  for (int m0 = 0; m0 < MB; m0 += MU) {
    if (m0 >= M) break;
    float acc[MU][R];
#pragma unroll
    for (int mi = 0; mi < MU; ++mi)
#pragma unroll
      for (int r = 0; r < R; ++r) acc[mi][r] = 0.0f;

#pragma unroll
    for (int j = 0; j < CH; ++j) {
      int seg = p * CH + j + rot;
      if (seg >= NSEG) seg -= NSEG;
#pragma unroll
      for (int mi = 0; mi < MU; ++mi) {
        const int mm = min(m0 + mi, M - 1);
        const u32x4 xv =
            *reinterpret_cast<const u32x4*>(x + (size_t)mm * sx + seg * SW + sub * 8);
#pragma unroll
        for (int r = 0; r < R; ++r) dot8(wv[r][j], xv, acc[mi][r]);
      }
    }

#pragma unroll
    for (int mi = 0; mi < MU; ++mi)
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const float v = group_sum<LPR>(acc[mi][r]);
        if (writer) red[g][p][m0 + mi][r * SLOTS + rl] = v;
      }
  }
  __syncthreads();

  for (int t = threadIdx.x; t < G * MB * RW; t += 64 * G * P) {
    const int slot = t % RW;
    const int m = (t / RW) % MB;
    const int gg = t / (RW * MB);
    float s = red[gg][0][m][slot];
#pragma unroll
    for (int q = 1; q < P; ++q) s += red[gg][q][m][slot];
    const int n = blockIdx.x * (G * RW) + gg * RW + slot;
    if (m < M && n < N) {
      if (bias != nullptr) s += bf2f(bias[n]);
      y[(size_t)m * sy + n] = f2bf(s);
    }
  }
}

typedef int (*skg_fn)(const void*, const void*, const void*, void*, int, int, int, int, int,
                      hipStream_t);

template <int MB, int R, int CH, int G, int P, int LPR, bool NT>
static int skg_launch_t(const void* x, const void* w, const void* b, void* y, int M, int N,
                        int sx, int sy, int stag, hipStream_t s) {
  constexpr int ROWS = G * R * (64 / LPR);
  hipLaunchKernelGGL((skg_kernel<MB, R, CH, G, P, LPR, NT>), dim3((N + ROWS - 1) / ROWS),
                     dim3(64 * G * P), 0, s, (const uint16_t*)x, (const uint16_t*)w,
                     (const uint16_t*)b, (uint16_t*)y, M, N, sx, sy, stag);
  return (int)hipGetLastError();
}

struct SkgEntry {
  int mb, r, p, g, nt, nseg, lpr;
  skg_fn fn;
};

// One config = (R, P, G, NT, NSEG, LPR), instantiated for every M bucket.
#define SKG_E(MB, R, P, G, NT, NSEG, LPR) \
  {MB, R, P, G, NT, NSEG, LPR, &skg_launch_t<MB, R, (NSEG) / (P), G, P, LPR, (NT) != 0>},
#define SKG_C(R, P, G, NT, NSEG, LPR)                                        \
  SKG_E(1, R, P, G, NT, NSEG, LPR) SKG_E(2, R, P, G, NT, NSEG, LPR)          \
      SKG_E(4, R, P, G, NT, NSEG, LPR) SKG_E(8, R, P, G, NT, NSEG, LPR)      \
          SKG_E(16, R, P, G, NT, NSEG, LPR)

#if defined(SKG_SWEEP) && !defined(SKG_NO_SCALAR)
// LPR 64 (K = 2048) and LPR 8 with NSEG 4 (K = 256): R x P x G
#define SKG_RPG(NSEG, LPR)                                                                     \
  SKG_C(1, 1, 1, 1, NSEG, LPR) SKG_C(2, 1, 1, 1, NSEG, LPR) SKG_C(4, 1, 1, 1, NSEG, LPR)      \
  SKG_C(1, 2, 1, 1, NSEG, LPR) SKG_C(2, 2, 1, 1, NSEG, LPR) SKG_C(4, 2, 1, 1, NSEG, LPR)      \
  SKG_C(1, 4, 1, 1, NSEG, LPR) SKG_C(2, 4, 1, 1, NSEG, LPR) SKG_C(4, 4, 1, 1, NSEG, LPR)      \
  SKG_C(1, 1, 2, 1, NSEG, LPR) SKG_C(2, 1, 2, 1, NSEG, LPR) SKG_C(4, 1, 2, 1, NSEG, LPR)      \
  SKG_C(1, 2, 2, 1, NSEG, LPR) SKG_C(2, 2, 2, 1, NSEG, LPR) SKG_C(4, 2, 2, 1, NSEG, LPR)      \
  SKG_C(1, 4, 2, 1, NSEG, LPR) SKG_C(2, 4, 2, 1, NSEG, LPR) SKG_C(4, 4, 2, 1, NSEG, LPR)
#define SKG_TABLE                                                                              \
  SKG_RPG(4, 64) SKG_RPG(4, 8)                                                                 \
  /* LPR 16, K = 2048 (NSEG 16) */                                                             \
  SKG_C(1, 4, 1, 1, 16, 16) SKG_C(2, 4, 1, 1, 16, 16) SKG_C(4, 4, 1, 1, 16, 16)                \
  SKG_C(1, 8, 1, 1, 16, 16) SKG_C(2, 8, 1, 1, 16, 16) SKG_C(4, 8, 1, 1, 16, 16)                \
  SKG_C(1, 16, 1, 1, 16, 16) SKG_C(2, 16, 1, 1, 16, 16) SKG_C(4, 16, 1, 1, 16, 16)             \
  SKG_C(1, 4, 2, 1, 16, 16) SKG_C(2, 4, 2, 1, 16, 16) SKG_C(4, 4, 2, 1, 16, 16)                \
  SKG_C(1, 8, 2, 1, 16, 16) SKG_C(2, 8, 2, 1, 16, 16)                                          \
  /* LPR 8, K = 2048 (NSEG 32) */                                                              \
  SKG_C(1, 4, 1, 1, 32, 8) SKG_C(2, 4, 1, 1, 32, 8) SKG_C(1, 8, 1, 1, 32, 8)                   \
  SKG_C(2, 8, 1, 1, 32, 8) SKG_C(1, 16, 1, 1, 32, 8) SKG_C(2, 16, 1, 1, 32, 8)                 \
  SKG_C(1, 4, 2, 1, 32, 8) SKG_C(2, 4, 2, 1, 32, 8) SKG_C(1, 8, 2, 1, 32, 8)                   \
  SKG_C(2, 8, 2, 1, 32, 8)
#endif

#ifdef SKG_TABLE
static const SkgEntry kTable[] = {SKG_TABLE};
static const int kTableLen = sizeof(kTable) / sizeof(kTable[0]);
#else
static const SkgEntry kTable[1] = {{0, 0, 0, 0, 0, 0, 0, nullptr}};
static const int kTableLen = 0;
#endif

// ------------------------------------------------------------------------------------------
// MFMA variant (skq): y^T tile = x[16, K] x w^T[K, 16 rows] through v_mfma_f32_16x16x16bf16_1k.
// x is always padded to 16 MFMA rows (rows >= M re-read row M-1 and are never stored); the
// MFMA output (m, n) depends only on row m of x and row n of w, and the instruction sequence
// does not depend on M, so a row's bits do not depend on how many rows share the call.
// Each lane loads one dwordx4 (8 consecutive k) of its weight row straight into registers and
// feeds its two halves to two MFMAs; x is loaded with the same k mapping, so the pairing of
// A and B along k is consistent. A wave owns R tiles of 16 rows over K / P; the P parts are
// summed in wave order through LDS. DL selects the C/D lane layout: 0 = (m = 4*(lane/16)+q,
// n = lane%16), 1 = the transpose.

typedef short s16x4 __attribute__((ext_vector_type(4)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef unsigned int u32x2 __attribute__((ext_vector_type(2)));

DEVI s16x4 lo_half(const u32x4 v) { return __builtin_bit_cast(s16x4, (u32x2){v.x, v.y}); }
DEVI s16x4 hi_half(const u32x4 v) { return __builtin_bit_cast(s16x4, (u32x2){v.z, v.w}); }

template <int R, int CH, int G, int P, int DL, bool HB>
__global__ void __launch_bounds__(64 * G * P)
    skq_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
               const uint16_t* __restrict__ bias, uint16_t* __restrict__ y, int M, int N, int sx,
               int sy, int stag) {
  constexpr int NSEG = CH * P;  // segments of 32 k
  constexpr int K = NSEG * 32;
  constexpr int RW = 16 * R;    // rows of w per wave
  __shared__ float red[P > 1 ? G : 1][P > 1 ? P : 1][P > 1 ? 16 * RW : 1];
  const int lane = threadIdx.x & 63;
  const int li = lane & 15;
  const int lg = lane >> 4;
  const int wave = threadIdx.x >> 6;
  const int g = wave / P;
  const int p = wave - g * P;
  const int n0 = blockIdx.x * (G * RW) + g * RW;
  const int rot = (int)((blockIdx.x * (unsigned)stag) % (unsigned)NSEG);

  u32x4 wv[R][CH];
#pragma unroll
  for (int j = 0; j < CH; ++j) {
    int seg = p * CH + j + rot;
    if (seg >= NSEG) seg -= NSEG;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int n = min(n0 + r * 16 + li, N - 1);
      wv[r][j] = ldw<true>(reinterpret_cast<const u32x4*>(w + (size_t)n * K + seg * 32 + lg * 8));
    }
  }
  const int mrow = min(li, M - 1);
  u32x4 xv[CH];
#pragma unroll
  for (int j = 0; j < CH; ++j) {
    int seg = p * CH + j + rot;
    if (seg >= NSEG) seg -= NSEG;
    xv[j] = *reinterpret_cast<const u32x4*>(x + (size_t)mrow * sx + seg * 32 + lg * 8);
  }
  // HB: every load is issued before the first MFMA (otherwise the compiler streams them).
  if constexpr (HB) __builtin_amdgcn_sched_barrier(0);

  f32x4 acc[R];
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r] = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int j = 0; j < CH; ++j)
#pragma unroll
    for (int r = 0; r < R; ++r) {
      acc[r] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(lo_half(xv[j]), lo_half(wv[r][j]), acc[r],
                                                         0, 0, 0);
      acc[r] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(hi_half(xv[j]), hi_half(wv[r][j]), acc[r],
                                                         0, 0, 0);
    }

  if constexpr (P == 1) {
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int m = DL == 0 ? 4 * lg + q : li;
        const int n = n0 + r * 16 + (DL == 0 ? li : 4 * lg + q);
        if (m < M && n < N) {
          float s = acc[r][q];
          if (bias != nullptr) s += bf2f(bias[n]);
          y[(size_t)m * sy + n] = f2bf(s);
        }
      }
  } else {
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int m = DL == 0 ? 4 * lg + q : li;
        const int c = r * 16 + (DL == 0 ? li : 4 * lg + q);
        red[g][p][m * RW + c] = acc[r][q];
      }
    __syncthreads();
    for (int t = threadIdx.x; t < G * 16 * RW; t += 64 * G * P) {
      const int c = t % RW;
      const int m = (t / RW) % 16;
      const int gg = t / (RW * 16);
      const int n = blockIdx.x * (G * RW) + gg * RW + c;
      if (m < M && n < N) {
        float s = red[gg][0][m * RW + c];
#pragma unroll
        for (int q = 1; q < P; ++q) s += red[gg][q][m * RW + c];
        if (bias != nullptr) s += bf2f(bias[n]);
        y[(size_t)m * sy + n] = f2bf(s);
      }
    }
  }
}

template <int R, int CH, int G, int P, int DL, bool HB>
static int skq_launch_t(const void* x, const void* w, const void* b, void* y, int M, int N,
                        int sx, int sy, int stag, hipStream_t s) {
  constexpr int ROWS = G * R * 16;
  hipLaunchKernelGGL((skq_kernel<R, CH, G, P, DL, HB>), dim3((N + ROWS - 1) / ROWS),
                     dim3(64 * G * P), 0, s, (const uint16_t*)x, (const uint16_t*)w,
                     (const uint16_t*)b, (uint16_t*)y, M, N, sx, sy, stag);
  return (int)hipGetLastError();
}

struct SkqEntry {
  int r, p, g, hb, nseg, dl;
  skg_fn fn;
};

// One config = (R, P, G, HB, NSEG, DL); K = 32 * NSEG.
#define SKQ_C(R, P, G, HB, NSEG, DL) \
  {R, P, G, HB, NSEG, DL, &skq_launch_t<R, (NSEG) / (P), G, P, DL, (HB) != 0>},

#ifdef SKG_SWEEP
#define SKQ_DL(R, P, G, NSEG)                                                     \
  SKQ_C(R, P, G, 0, NSEG, 0) SKQ_C(R, P, G, 0, NSEG, 1) SKQ_C(R, P, G, 1, NSEG, 0) \
      SKQ_C(R, P, G, 1, NSEG, 1)
#define SKQ_TABLE                                                                         \
  /* K = 2048 */                                                                          \
  SKQ_DL(1, 4, 1, 64) SKQ_DL(2, 4, 1, 64) SKQ_DL(1, 8, 1, 64) SKQ_DL(2, 8, 1, 64)         \
  SKQ_DL(1, 16, 1, 64) SKQ_DL(2, 16, 1, 64) SKQ_DL(1, 4, 2, 64) SKQ_DL(2, 4, 2, 64)       \
  SKQ_DL(1, 8, 2, 64) SKQ_DL(2, 8, 2, 64)                                                 \
  /* K = 256 */                                                                           \
  SKQ_DL(1, 1, 1, 8) SKQ_DL(2, 1, 1, 8) SKQ_DL(1, 2, 1, 8) SKQ_DL(2, 2, 1, 8)             \
  SKQ_DL(1, 4, 1, 8) SKQ_DL(2, 4, 1, 8) SKQ_DL(1, 1, 2, 8) SKQ_DL(2, 1, 2, 8)             \
  SKQ_DL(1, 2, 2, 8) SKQ_DL(2, 2, 2, 8) SKQ_DL(1, 4, 2, 8) SKQ_DL(2, 4, 2, 8)
#endif

#ifdef SKQ_TABLE
static const SkqEntry kQTable[] = {SKQ_TABLE};
static const int kQTableLen = sizeof(kQTable) / sizeof(kQTable[0]);
#else
static const SkqEntry kQTable[1] = {{0, 0, 0, 0, 0, 0, nullptr}};
static const int kQTableLen = 0;
#endif

// Streaming-read ceiling: every lane issues 8 dwordx4 loads (nontemporal) and folds them.
__global__ void __launch_bounds__(256) skg_read_kernel(const u32x4* __restrict__ w, long long n16,
                                                       unsigned* out) {
  const long long stride = (long long)gridDim.x * 256;
  const long long t = (long long)blockIdx.x * 256 + threadIdx.x;
  unsigned a = 0;
  for (long long i = t; i < n16; i += stride * 8) {
    u32x4 v[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const long long idx = i + u * stride;
      v[u] = idx < n16 ? __builtin_nontemporal_load(w + idx) : u32x4{0, 0, 0, 0};
    }
#pragma unroll
    for (int u = 0; u < 8; ++u) a ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  if (a == 0x9e3779b9u) out[blockIdx.x] = a;  // keeps the loads alive
}

// ------------------------------------------------------------------------------------------
// 4x4x4 MFMA variant (skf): v_mfma_f32_4x4x4bf16_1k runs 16 independent 4x4x4 blocks per
// wave. Block b of every MFMA takes its own k chunk (lanes 4b..4b+3, one dwordx4 = 8 k per
// lane, the two halves feeding two MFMAs), so one load instruction reads 4 weight rows x 256
// contiguous bytes. x rows come in 4 groups of 4 (x padded to 16 rows: a row >= M re-reads row
// M-1 and is never stored); a group whose rows are all >= M is skipped, which leaves every
// other row's arithmetic unchanged. The 16 block partials of an output are summed in a fixed
// order (DPP row_ror 4, 8, then xor 16, 32), the P wave parts in wave order through LDS.
// DL selects the C/D lane layout: 0 = (m = q, n = lane%4), 1 = the transpose.

template <int CTRL>
DEVI float dpp_all(float v) {
  return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), CTRL, 0xF, 0xF, false));
}

DEVI float block_sum(float v) {
  v += dpp_all<0x124>(v);  // row_ror:4
  v += dpp_all<0x128>(v);  // row_ror:8
  v += __shfl_xor(v, 16);
  v += __shfl_xor(v, 32);
  return v;  // lanes 0..3 hold the sums for n = lane
}

template <int R, int CH, int G, int P, int DL, bool HB>
__global__ void __launch_bounds__(64 * G * P)
    skf_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
               const uint16_t* __restrict__ bias, uint16_t* __restrict__ y, int M, int N, int sx,
               int sy, int stag) {
  constexpr int NSEG = CH * P;  // segments of 128 k
  constexpr int K = NSEG * 128;
  constexpr int RW = 4 * R;     // rows of w per wave
  __shared__ float red[P > 1 ? G : 1][P > 1 ? P : 1][P > 1 ? 16 * RW : 1];
  const int lane = threadIdx.x & 63;
  const int j4 = lane & 3;
  const int b = lane >> 2;
  const int wave = threadIdx.x >> 6;
  const int g = wave / P;
  const int p = wave - g * P;
  const int n0 = blockIdx.x * (G * RW) + g * RW;
  const int rot = (int)((blockIdx.x * (unsigned)stag) % (unsigned)NSEG);
  const int ngrp = (M + 3) >> 2;

  u32x4 wv[R][CH];
#pragma unroll
  for (int j = 0; j < CH; ++j) {
    int seg = p * CH + j + rot;
    if (seg >= NSEG) seg -= NSEG;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int n = min(n0 + r * 4 + j4, N - 1);
      wv[r][j] = ldw<true>(reinterpret_cast<const u32x4*>(w + (size_t)n * K + seg * 128 + b * 8));
    }
  }
  if constexpr (HB) __builtin_amdgcn_sched_barrier(0);

  f32x4 acc[4][R];
#pragma unroll
  for (int mg = 0; mg < 4; ++mg) {
#pragma unroll
    for (int r = 0; r < R; ++r) acc[mg][r] = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    if (mg < ngrp) {
      const int mrow = min(4 * mg + j4, M - 1);
#pragma unroll
      for (int j = 0; j < CH; ++j) {
        int seg = p * CH + j + rot;
        if (seg >= NSEG) seg -= NSEG;
        const u32x4 xv =
            *reinterpret_cast<const u32x4*>(x + (size_t)mrow * sx + seg * 128 + b * 8);
#pragma unroll
        for (int r = 0; r < R; ++r) {
          acc[mg][r] = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(lo_half(xv), lo_half(wv[r][j]),
                                                              acc[mg][r], 0, 0, 0);
          acc[mg][r] = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(hi_half(xv), hi_half(wv[r][j]),
                                                              acc[mg][r], 0, 0, 0);
        }
      }
    }
  }

#pragma unroll
  for (int mg = 0; mg < 4; ++mg) {
    if (mg < ngrp) {
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const float v = block_sum(acc[mg][r][q]);
          if (lane < 4) {
            const int m = 4 * mg + (DL == 0 ? q : j4);
            const int c = r * 4 + (DL == 0 ? j4 : q);
            if constexpr (P == 1) {
              const int n = n0 + c;
              if (m < M && n < N) {
                float s = v;
                if (bias != nullptr) s += bf2f(bias[n]);
                y[(size_t)m * sy + n] = f2bf(s);
              }
            } else {
              red[g][p][m * RW + c] = v;
            }
          }
        }
    }
  }
  if constexpr (P > 1) {
    __syncthreads();
    for (int t = threadIdx.x; t < G * 16 * RW; t += 64 * G * P) {
      const int c = t % RW;
      const int m = (t / RW) % 16;
      const int gg = t / (RW * 16);
      const int n = blockIdx.x * (G * RW) + gg * RW + c;
      if (m < M && n < N) {
        float s = red[gg][0][m * RW + c];
#pragma unroll
        for (int q = 1; q < P; ++q) s += red[gg][q][m * RW + c];
        if (bias != nullptr) s += bf2f(bias[n]);
        y[(size_t)m * sy + n] = f2bf(s);
      }
    }
  }
}

template <int R, int CH, int G, int P, int DL, bool HB>
static int skf_launch_t(const void* x, const void* w, const void* b, void* y, int M, int N,
                        int sx, int sy, int stag, hipStream_t s) {
  constexpr int ROWS = G * R * 4;
  hipLaunchKernelGGL((skf_kernel<R, CH, G, P, DL, HB>), dim3((N + ROWS - 1) / ROWS),
                     dim3(64 * G * P), 0, s, (const uint16_t*)x, (const uint16_t*)w,
                     (const uint16_t*)b, (uint16_t*)y, M, N, sx, sy, stag);
  return (int)hipGetLastError();
}

// One config = (R, P, G, HB, NSEG, DL); K = 128 * NSEG.
#define SKF_C(R, P, G, HB, NSEG, DL) \
  {R, P, G, HB, NSEG, DL, &skf_launch_t<R, (NSEG) / (P), G, P, DL, (HB) != 0>},

#ifdef SKG_SWEEP
#define SKF_V(R, P, G, NSEG)                                                      \
  SKF_C(R, P, G, 0, NSEG, 0) SKF_C(R, P, G, 0, NSEG, 1) SKF_C(R, P, G, 1, NSEG, 0) \
      SKF_C(R, P, G, 1, NSEG, 1)
#define SKF_TABLE                                                                      \
  /* K = 2048 */                                                                       \
  SKF_V(1, 4, 1, 16) SKF_V(2, 4, 1, 16) SKF_V(4, 4, 1, 16) SKF_V(1, 8, 1, 16)          \
  SKF_V(2, 8, 1, 16) SKF_V(4, 8, 1, 16) SKF_V(1, 16, 1, 16) SKF_V(2, 16, 1, 16)        \
  SKF_V(4, 16, 1, 16) SKF_V(1, 4, 2, 16) SKF_V(2, 4, 2, 16) SKF_V(4, 4, 2, 16)         \
  SKF_V(1, 8, 2, 16) SKF_V(2, 8, 2, 16) SKF_V(4, 8, 2, 16)                             \
  /* K = 256 */                                                                        \
  SKF_V(1, 1, 1, 2) SKF_V(2, 1, 1, 2) SKF_V(4, 1, 1, 2) SKF_V(1, 2, 1, 2)              \
  SKF_V(2, 2, 1, 2) SKF_V(4, 2, 1, 2) SKF_V(1, 1, 2, 2) SKF_V(2, 1, 2, 2)              \
  SKF_V(4, 1, 2, 2) SKF_V(1, 2, 2, 2) SKF_V(2, 2, 2, 2) SKF_V(4, 2, 2, 2)
#endif

#ifdef SKF_TABLE
static const SkqEntry kFTable[] = {SKF_TABLE};
static const int kFTableLen = sizeof(kFTable) / sizeof(kFTable[0]);
#else
static const SkqEntry kFTable[1] = {{0, 0, 0, 0, 0, 0, nullptr}};
static const int kFTableLen = 0;
#endif

// ------------------------------------------------------------------------------------------
// skf2: the 4x4x4 MFMA GEMV with (A) one fixed-order reduce-scatter across the 16 blocks per
// m-group instead of a 16-way butterfly per value, and (B) m-groups processed one at a time
// (only R x 4 accumulators live), so R can grow to 8 and one x load feeds 2R MFMAs.
// Every output's reduction tree is fixed: stages pair lanes at xor 8, 4, 16, 32 in that order
// (IEEE add is commutative, so either lane of a pair holds the same bits), whatever M is.
// RED = 0 keeps skf's per-value butterfly (for comparison). DIAG (timing only, wrong
// results): 1 = no cross-lane exchange, 2 = one x load reused by every m-group.

template <int D>
DEVI float xchg(float v) {
  if constexpr (D == 8) {
    return dpp_all<0x128>(v);  // row_ror:8 == lane ^ 8 inside a 16-lane row
  } else if constexpr (D == 32) {
    return __shfl_xor(v, 32);
  } else {
    return __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x1F | (D << 10)));
  }
}

// One stage over lane bit D on N live values: halve them (N >= 2) or butterfly (N == 1).
template <int D, int N, bool NOX>
DEVI void rs_stage(float* v, int lane) {
  const bool hi = (lane & D) != 0;
  if constexpr (N >= 2) {
    constexpr int H = N / 2;
#pragma unroll
    for (int i = 0; i < H; ++i) {
      const float send = hi ? v[i] : v[i + H];
      const float keep = hi ? v[i + H] : v[i];
      v[i] = NOX ? keep : keep + xchg<D>(send);
    }
  } else {
    if constexpr (!NOX) v[0] = v[0] + xchg<D>(v[0]);
  }
}

// Reduce-scatter of V values over the 16 blocks (lane bits 2..5). Afterwards v[0..NF) hold the
// sums of original indices idx .. idx + NF - 1; writer is false on the duplicate lanes.
template <int V, bool NOX>
DEVI void reduce_scatter(float* v, int lane, int& idx, bool& writer) {
  constexpr int N1 = V, N2 = V > 1 ? V / 2 : 1, N3 = N2 > 1 ? N2 / 2 : 1, N4 = N3 > 1 ? N3 / 2 : 1;
  rs_stage<8, N1, NOX>(v, lane);
  rs_stage<4, N2, NOX>(v, lane);
  rs_stage<16, N3, NOX>(v, lane);
  rs_stage<32, N4, NOX>(v, lane);
  idx = 0;
  writer = true;
  const int ds[4] = {8, 4, 16, 32};
  const int ns[4] = {N1, N2, N3, N4};
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    if (ns[s] >= 2) {
      if (lane & ds[s]) idx += ns[s] / 2;
    } else if (lane & ds[s]) {
      writer = false;
    }
  }
}

template <int R, int CH, int G, int P, bool HB, int RED, int DIAG>
__global__ void __launch_bounds__(64 * G * P)
    skf2_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                const uint16_t* __restrict__ bias, uint16_t* __restrict__ y, int M, int N, int sx,
                int sy, int stag) {
  constexpr int NSEG = CH * P;  // segments of 128 k
  constexpr int K = NSEG * 128;
  constexpr int RW = 4 * R;     // rows of w per wave
  constexpr int V = 4 * R;      // values per lane per m-group
  constexpr int NF = V >= 16 ? V / 16 : 1;
  __shared__ float red[P > 1 ? G : 1][P > 1 ? P : 1][P > 1 ? 16 * RW : 1];
  const int lane = threadIdx.x & 63;
  const int j4 = lane & 3;
  const int b = lane >> 2;
  const int wave = threadIdx.x >> 6;
  const int g = wave / P;
  const int p = wave - g * P;
  const int n0 = blockIdx.x * (G * RW) + g * RW;
  const int ngrp = (M + 3) >> 2;

  u32x4 wv[R][CH];
#pragma unroll
  for (int j = 0; j < CH; ++j)
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int n = min(n0 + r * 4 + j4, N - 1);
      wv[r][j] = ldw<true>(
          reinterpret_cast<const u32x4*>(w + (size_t)n * K + (p * CH + j) * 128 + b * 8));
    }
  if constexpr (HB) __builtin_amdgcn_sched_barrier(0);
  u32x4 xv0[CH];
  if constexpr (DIAG == 2) {
    const int mrow = min(j4, M - 1);
#pragma unroll
    for (int j = 0; j < CH; ++j)
      xv0[j] = *reinterpret_cast<const u32x4*>(x + (size_t)mrow * sx + (p * CH + j) * 128 + b * 8);
  }

  auto emit = [&](int m, int c, float v) {
    if constexpr (P == 1) {
      const int n = n0 + c;
      if (m < M && n < N) {
        if (bias != nullptr) v += bf2f(bias[n]);
        y[(size_t)m * sy + n] = f2bf(v);
      }
    } else {
      red[g][p][m * RW + c] = v;
    }
  };

#pragma unroll 1
  for (int mg = 0; mg < 4; ++mg) {
    if (mg >= ngrp) break;
    f32x4 acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    const int mrow = min(4 * mg + j4, M - 1);
#pragma unroll
    for (int j = 0; j < CH; ++j) {
      u32x4 xv;
      if constexpr (DIAG == 2) {
        xv = xv0[j];
      } else {
        xv = *reinterpret_cast<const u32x4*>(x + (size_t)mrow * sx + (p * CH + j) * 128 + b * 8);
      }
#pragma unroll
      for (int r = 0; r < R; ++r) {
        acc[r] = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(lo_half(xv), lo_half(wv[r][j]), acc[r],
                                                        0, 0, 0);
        acc[r] = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(hi_half(xv), hi_half(wv[r][j]), acc[r],
                                                        0, 0, 0);
      }
    }
    if constexpr (RED == 0) {
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const float v = DIAG == 1 ? acc[r][q] : block_sum(acc[r][q]);
          if (lane < 4) emit(4 * mg + q, r * 4 + j4, v);
        }
    } else {
      float v[V];
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int q = 0; q < 4; ++q) v[r * 4 + q] = acc[r][q];
      int idx;
      bool writer;
      reduce_scatter<V, DIAG == 1>(v, lane, idx, writer);
      if (writer) {
#pragma unroll
        for (int i = 0; i < NF; ++i) {
          const int vi = idx + i;
          emit(4 * mg + (vi & 3), (vi >> 2) * 4 + j4, v[i]);
        }
      }
    }
  }
  if constexpr (P > 1) {
    __syncthreads();
    for (int t = threadIdx.x; t < G * 16 * RW; t += 64 * G * P) {
      const int c = t % RW;
      const int m = (t / RW) % 16;
      const int gg = t / (RW * 16);
      const int n = blockIdx.x * (G * RW) + gg * RW + c;
      if (m < M && n < N) {
        float s = red[gg][0][m * RW + c];
#pragma unroll
        for (int q = 1; q < P; ++q) s += red[gg][q][m * RW + c];
        if (bias != nullptr) s += bf2f(bias[n]);
        y[(size_t)m * sy + n] = f2bf(s);
      }
    }
  }
}

template <int R, int CH, int G, int P, bool HB, int RED, int DIAG>
static int skf2_launch_t(const void* x, const void* w, const void* b, void* y, int M, int N,
                         int sx, int sy, int stag, hipStream_t s) {
  constexpr int ROWS = G * R * 4;
  hipLaunchKernelGGL((skf2_kernel<R, CH, G, P, HB, RED, DIAG>), dim3((N + ROWS - 1) / ROWS),
                     dim3(64 * G * P), 0, s, (const uint16_t*)x, (const uint16_t*)w,
                     (const uint16_t*)b, (uint16_t*)y, M, N, sx, sy, stag);
  return (int)hipGetLastError();
}

struct Skf2Entry {
  int r, p, g, hb, nseg, red, diag;
  skg_fn fn;
};

// One config = (R, P, G, HB, NSEG, RED, DIAG); K = 128 * NSEG.
#define SKF2_C(R, P, G, HB, NSEG, RED, DIAG) \
  {R, P, G, HB, NSEG, RED, DIAG, &skf2_launch_t<R, (NSEG) / (P), G, P, (HB) != 0, RED, DIAG>},

#ifdef SKG_SWEEP
#define SKF2_RD(R, P, G, NSEG) SKF2_C(R, P, G, 0, NSEG, 0, 0) SKF2_C(R, P, G, 0, NSEG, 1, 0)
#define SKF2_DG(R, P, G, NSEG)                                                          \
  SKF2_C(R, P, G, 0, NSEG, 1, 1) SKF2_C(R, P, G, 0, NSEG, 1, 2) SKF2_C(R, P, G, 0, NSEG, 0, 1) \
      SKF2_C(R, P, G, 0, NSEG, 0, 2)
#define SKF2_TABLE                                                                     \
  /* K = 2048 */                                                                       \
  SKF2_RD(1, 4, 1, 16) SKF2_RD(2, 4, 1, 16) SKF2_RD(4, 4, 1, 16)                        \
  SKF2_RD(1, 4, 2, 16) SKF2_RD(2, 4, 2, 16) SKF2_RD(4, 4, 2, 16)                        \
  SKF2_RD(1, 8, 1, 16) SKF2_RD(2, 8, 1, 16) SKF2_RD(4, 8, 1, 16) SKF2_RD(8, 8, 1, 16)   \
  SKF2_RD(1, 8, 2, 16) SKF2_RD(2, 8, 2, 16) SKF2_RD(4, 8, 2, 16) SKF2_RD(8, 8, 2, 16)   \
  SKF2_RD(1, 16, 1, 16) SKF2_RD(2, 16, 1, 16) SKF2_RD(4, 16, 1, 16) SKF2_RD(8, 16, 1, 16) \
  SKF2_DG(2, 8, 1, 16) SKF2_DG(4, 8, 1, 16) SKF2_DG(8, 8, 1, 16)                        \
  /* K = 256 */                                                                        \
  SKF2_RD(1, 1, 1, 2) SKF2_RD(2, 1, 1, 2) SKF2_RD(4, 1, 1, 2) SKF2_RD(8, 1, 1, 2)       \
  SKF2_RD(1, 2, 1, 2) SKF2_RD(2, 2, 1, 2) SKF2_RD(4, 2, 1, 2) SKF2_RD(8, 2, 1, 2)       \
  SKF2_RD(1, 1, 2, 2) SKF2_RD(2, 1, 2, 2) SKF2_RD(4, 1, 2, 2) SKF2_RD(8, 1, 2, 2)       \
  SKF2_RD(1, 2, 2, 2) SKF2_RD(2, 2, 2, 2) SKF2_RD(4, 2, 2, 2) SKF2_RD(8, 2, 2, 2)
#endif

#ifdef SKF2_TABLE
static const Skf2Entry kF2Table[] = {SKF2_TABLE};
static const int kF2TableLen = sizeof(kF2Table) / sizeof(kF2Table[0]);
#else
static const Skf2Entry kF2Table[1] = {{0, 0, 0, 0, 0, 0, 0, nullptr}};
static const int kF2TableLen = 0;
#endif

extern "C" {

int skf2_count() { return kF2TableLen; }

int skf2_entry(int i, int* out7) {
  if (i < 0 || i >= kF2TableLen) return -1;
  const Skf2Entry& e = kF2Table[i];
  out7[0] = e.r;
  out7[1] = e.p;
  out7[2] = e.g;
  out7[3] = e.hb;
  out7[4] = e.nseg;
  out7[5] = e.red;
  out7[6] = e.diag;
  return 0;
}

int skf2_find(int r, int p, int g, int hb, int nseg, int red, int diag) {
  for (int i = 0; i < kF2TableLen; ++i) {
    const Skf2Entry& e = kF2Table[i];
    if (e.r == r && e.p == p && e.g == g && e.hb == hb && e.nseg == nseg && e.red == red &&
        e.diag == diag)
      return i;
  }
  return -1;
}

int skf2_launch(int idx, const void* x, const void* w, const void* b, void* y, int M, int N,
                int sx, int sy, int stag, void* stream) {
  if (idx < 0 || idx >= kF2TableLen) return -1;
  return kF2Table[idx].fn(x, w, b, y, M, N, sx, sy, stag, (hipStream_t)stream);
}

}  // extern "C"

extern "C" {

int skf_count() { return kFTableLen; }

int skf_entry(int i, int* out6) {
  if (i < 0 || i >= kFTableLen) return -1;
  const SkqEntry& e = kFTable[i];
  out6[0] = e.r;
  out6[1] = e.p;
  out6[2] = e.g;
  out6[3] = e.hb;
  out6[4] = e.nseg;
  out6[5] = e.dl;
  return 0;
}

int skf_find(int r, int p, int g, int hb, int nseg, int dl) {
  for (int i = 0; i < kFTableLen; ++i) {
    const SkqEntry& e = kFTable[i];
    if (e.r == r && e.p == p && e.g == g && e.hb == hb && e.nseg == nseg && e.dl == dl) return i;
  }
  return -1;
}

int skf_launch(int idx, const void* x, const void* w, const void* b, void* y, int M, int N,
               int sx, int sy, int stag, void* stream) {
  if (idx < 0 || idx >= kFTableLen) return -1;
  return kFTable[idx].fn(x, w, b, y, M, N, sx, sy, stag, (hipStream_t)stream);
}

}  // extern "C"

extern "C" {

int skg_count() { return kTableLen; }

int skg_entry(int i, int* out7) {
  if (i < 0 || i >= kTableLen) return -1;
  const SkgEntry& e = kTable[i];
  out7[0] = e.mb;
  out7[1] = e.r;
  out7[2] = e.p;
  out7[3] = e.g;
  out7[4] = e.nt;
  out7[5] = e.nseg;
  out7[6] = e.lpr;
  return 0;
}

int skg_find(int mb, int r, int p, int g, int nt, int nseg, int lpr) {
  for (int i = 0; i < kTableLen; ++i) {
    const SkgEntry& e = kTable[i];
    if (e.mb == mb && e.r == r && e.p == p && e.g == g && e.nt == nt && e.nseg == nseg &&
        e.lpr == lpr)
      return i;
  }
  return -1;
}

int skg_launch(int idx, const void* x, const void* w, const void* b, void* y, int M, int N,
               int sx, int sy, int stag, void* stream) {
  if (idx < 0 || idx >= kTableLen) return -1;
  return kTable[idx].fn(x, w, b, y, M, N, sx, sy, stag, (hipStream_t)stream);
}

int skq_count() { return kQTableLen; }

int skq_entry(int i, int* out6) {
  if (i < 0 || i >= kQTableLen) return -1;
  const SkqEntry& e = kQTable[i];
  out6[0] = e.r;
  out6[1] = e.p;
  out6[2] = e.g;
  out6[3] = e.hb;
  out6[4] = e.nseg;
  out6[5] = e.dl;
  return 0;
}

int skq_find(int r, int p, int g, int hb, int nseg, int dl) {
  for (int i = 0; i < kQTableLen; ++i) {
    const SkqEntry& e = kQTable[i];
    if (e.r == r && e.p == p && e.g == g && e.hb == hb && e.nseg == nseg && e.dl == dl) return i;
  }
  return -1;
}

int skq_launch(int idx, const void* x, const void* w, const void* b, void* y, int M, int N,
               int sx, int sy, int stag, void* stream) {
  if (idx < 0 || idx >= kQTableLen) return -1;
  return kQTable[idx].fn(x, w, b, y, M, N, sx, sy, stag, (hipStream_t)stream);
}

int skg_read(const void* w, long long n16, void* out, int blocks, void* stream) {
  hipLaunchKernelGGL(skg_read_kernel, dim3(blocks), dim3(256), 0, (hipStream_t)stream,
                     (const u32x4*)w, n16, (unsigned*)out);
  return (int)hipGetLastError();
}

}  // extern "C"
