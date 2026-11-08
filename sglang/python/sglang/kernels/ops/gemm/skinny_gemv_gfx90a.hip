// Skinny BF16 GEMV y[m, n] = x[m, :] . w[n, :] (+ bias[n]) for M <= 16 rows on gfx90a.
//
// Built at runtime by skinny_gemv_hip.py with hipcc (-ffp-contract=off) and loaded through
// ctypes; launches go on the caller's (torch's current) stream, so they can be graph-captured.
//
// Layout: w is row-major [N, K] bf16 with K = 512 * NSEG. A workgroup owns G row groups of R
// rows each; every row group is split over P waves along K (P * CH = NSEG segments of 512
// elements, one dwordx4 = 8 bf16 per lane per segment). Each lane issues all R * CH weight
// loads before it touches any (no LDS staging), then runs a per-lane FP32 fmaf chain, a fixed
// DPP butterfly across the 64 lanes and a fixed wave-order sum of the P parts through LDS.
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

template <bool NT>
DEVI u32x4 ldw(const u32x4* p) {
  if constexpr (NT) {
    return __builtin_nontemporal_load(p);
  } else {
    return *p;
  }
}

template <int MB, int R, int CH, int G, int P, bool NT>
__global__ void __launch_bounds__(64 * G * P)
    skg_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
               const uint16_t* __restrict__ bias, uint16_t* __restrict__ y, int M, int N, int sx,
               int sy, int stag) {
  constexpr int NSEG = CH * P;
  constexpr int K = NSEG * 512;
  __shared__ float red[G][P][MB][R];
  const int lane = threadIdx.x & 63;
  const int wave = threadIdx.x >> 6;
  const int g = wave / P;
  const int p = wave - g * P;
  const int n0 = blockIdx.x * (G * R) + g * R;
  const int rot = (int)((blockIdx.x * (unsigned)stag) % (unsigned)NSEG);

  u32x4 wv[R][CH];
#pragma unroll
  for (int j = 0; j < CH; ++j) {
    int seg = p * CH + j + rot;
    if (seg >= NSEG) seg -= NSEG;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int n = min(n0 + r, N - 1);
      wv[r][j] = ldw<NT>(reinterpret_cast<const u32x4*>(w + (size_t)n * K + seg * 512 + lane * 8));
    }
  }

  float acc[MB][R];
#pragma unroll
  for (int m = 0; m < MB; ++m)
#pragma unroll
    for (int r = 0; r < R; ++r) acc[m][r] = 0.0f;

#pragma unroll
  for (int j = 0; j < CH; ++j) {
    int seg = p * CH + j + rot;
    if (seg >= NSEG) seg -= NSEG;
#pragma unroll
    for (int m = 0; m < MB; ++m) {
      const int mm = min(m, M - 1);
      const u32x4 xv = *reinterpret_cast<const u32x4*>(x + (size_t)mm * sx + seg * 512 + lane * 8);
#pragma unroll
      for (int r = 0; r < R; ++r) dot8(wv[r][j], xv, acc[m][r]);
    }
  }

#pragma unroll
  for (int m = 0; m < MB; ++m)
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const float v = wave_sum63(acc[m][r]);
      if (lane == 63) red[g][p][m][r] = v;
    }
  __syncthreads();

  for (int t = threadIdx.x; t < G * MB * R; t += 64 * G * P) {
    const int r = t % R;
    const int m = (t / R) % MB;
    const int gg = t / (R * MB);
    float s = red[gg][0][m][r];
#pragma unroll
    for (int q = 1; q < P; ++q) s += red[gg][q][m][r];
    const int n = blockIdx.x * (G * R) + gg * R + r;
    if (m < M && n < N) {
      if (bias != nullptr) s += bf2f(bias[n]);
      y[(size_t)m * sy + n] = f2bf(s);
    }
  }
}

typedef int (*skg_fn)(const void*, const void*, const void*, void*, int, int, int, int, int,
                      hipStream_t);

template <int MB, int R, int CH, int G, int P, bool NT>
static int skg_launch_t(const void* x, const void* w, const void* b, void* y, int M, int N,
                        int sx, int sy, int stag, hipStream_t s) {
  constexpr int ROWS = G * R;
  hipLaunchKernelGGL((skg_kernel<MB, R, CH, G, P, NT>), dim3((N + ROWS - 1) / ROWS),
                     dim3(64 * G * P), 0, s, (const uint16_t*)x, (const uint16_t*)w,
                     (const uint16_t*)b, (uint16_t*)y, M, N, sx, sy, stag);
  return (int)hipGetLastError();
}

struct SkgEntry {
  int mb, r, p, g, nt, nseg;
  skg_fn fn;
};

#define SKG_E(MB, R, P, G, NT, NSEG) \
  {MB, R, P, G, NT, NSEG, &skg_launch_t<MB, R, (NSEG) / (P), G, P, (NT) != 0>},
#define SKG_MB(R, P, G, NT, NSEG)                                                      \
  SKG_E(1, R, P, G, NT, NSEG) SKG_E(2, R, P, G, NT, NSEG) SKG_E(4, R, P, G, NT, NSEG) \
      SKG_E(8, R, P, G, NT, NSEG) SKG_E(16, R, P, G, NT, NSEG)

#ifdef SKG_SWEEP
#define SKG_NT2(R, P, G) SKG_MB(R, P, G, 0, 4) SKG_MB(R, P, G, 1, 4)
#define SKG_G3(R, P) SKG_NT2(R, P, 1) SKG_NT2(R, P, 2) SKG_NT2(R, P, 4)
#define SKG_P3(R) SKG_G3(R, 1) SKG_G3(R, 2) SKG_G3(R, 4)
#define SKG_TABLE SKG_P3(1) SKG_P3(2) SKG_P3(4)
#endif

#ifndef SKG_TABLE
#error "define SKG_TABLE (production configs) or SKG_SWEEP"
#endif

static const SkgEntry kTable[] = {SKG_TABLE};
static const int kTableLen = sizeof(kTable) / sizeof(kTable[0]);

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

extern "C" {

int skg_count() { return kTableLen; }

int skg_entry(int i, int* out6) {
  if (i < 0 || i >= kTableLen) return -1;
  const SkgEntry& e = kTable[i];
  out6[0] = e.mb;
  out6[1] = e.r;
  out6[2] = e.p;
  out6[3] = e.g;
  out6[4] = e.nt;
  out6[5] = e.nseg;
  return 0;
}

int skg_find(int mb, int r, int p, int g, int nt, int nseg) {
  for (int i = 0; i < kTableLen; ++i) {
    const SkgEntry& e = kTable[i];
    if (e.mb == mb && e.r == r && e.p == p && e.g == g && e.nt == nt && e.nseg == nseg) return i;
  }
  return -1;
}

int skg_launch(int idx, const void* x, const void* w, const void* b, void* y, int M, int N,
               int sx, int sy, int stag, void* stream) {
  if (idx < 0 || idx >= kTableLen) return -1;
  return kTable[idx].fn(x, w, b, y, M, N, sx, sy, stag, (hipStream_t)stream);
}

int skg_read(const void* w, long long n16, void* out, int blocks, void* stream) {
  hipLaunchKernelGGL(skg_read_kernel, dim3(blocks), dim3(256), 0, (hipStream_t)stream,
                     (const u32x4*)w, n16, (unsigned*)out);
  return (int)hipGetLastError();
}

}  // extern "C"
