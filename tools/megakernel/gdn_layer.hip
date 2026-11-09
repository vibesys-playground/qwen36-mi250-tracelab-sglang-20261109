// Plan 13 stage 2 feasibility (gfx90a, one MI250 GCD; TP=2 per-rank shapes of Qwen3.6-35B-A3B).
//
// (1) k_gdn_body<M>: the decode body of one Gated DeltaNet layer in one persistent launch, for
//     M <= 4 rows of the normalized hidden state x [M, 2048]:
//       y      = x @ W_in^T             W_in = packed [q 1024 | k 1024 | v 2048 | z 2048 | b 16 | a 16]
//       qkv    = silu(causal_conv1d_update(y[:, :4096], conv_state[slot], conv_w))   (state shifted)
//       o      = gated delta rule decode step per v-head (l2norm q/k, g, beta; fp32 state [16,128,128])
//       o_n    = rmsnorm_per_head(o) * w_norm * silu(z)
//       out    = o_n @ W_out^T          (before the TP all-reduce)
//     It replaces skinny_linear(in_proj) + fused_qkvzba_causal_conv1d_update + packed recurrent
//     decode + gated RMSNorm + skinny_linear(out_proj): 5 launches.
//     Schedule: phase A = in_proj GEMV quads (group rows q/k/v/b/a of k-head g first, then z),
//     conv in the epilogue; one arrival counter per k-head group; recurrence tiles (m, v-head,
//     8 value rows) wait only on their group; grid barrier; phase C = gated norm prologue (each
//     wave rebuilds o_n for its own columns) + out_proj GEMV quads. The SSM state tile of the
//     first recurrence task and the first out_proj weight quad are prefetched into registers.
// (2) k_ar: in-kernel P2P all-reduce between two GCDs (push model): each workgroup stores its
//     slice into the peer's receive buffer (uncached, fine-grained), system-scope release
//     (buffer_wbl2 + s_waitcnt) and a flag store to the peer; it waits on its own flag,
//     system-scope acquire (buffer_invl2), and sums in rank order 0,1 in fp32 like
//     cross_device_reduce_1stage. `reps` all-reduces back to back in one launch.
//
// Numerics: fp32 accumulation with explicit fmaf (contract off), fixed lane order, butterfly
// reductions, no float atomics; a row's result depends only on that row (M- and grid-invariant).

#include <hip/hip_runtime.h>
#include <stdint.h>

#pragma clang fp contract(off)

#define DEVI __device__ __forceinline__
#define AGENT __HIP_MEMORY_SCOPE_AGENT
#define SYSTEM __HIP_MEMORY_SCOPE_SYSTEM

constexpr int WPB = 4, NT = WPB * 64;
constexpr int HID = 2048, NKH = 8, NVH = 16, DK = 128, DV = 128;
constexpr int QKV = NKH * DK * 2 + NVH * DV;  // 4096
constexpr int ROW_K = NKH * DK, ROW_V = 2 * NKH * DK, ROW_Z = QKV, ROW_B = QKV + NVH * DV, ROW_A = ROW_B + NVH;
constexpr int NIN = ROW_A + NVH;               // 6176
constexpr int GQ = (DK + DK + 2 * DV + 4) / 4; // quads per k-head group: 32 q + 32 k + 64 v + 1 ba = 129
constexpr int NQ_GROUP = NKH * GQ;             // 1032
constexpr int NQ_Z = NVH * DV / 4;             // 512
constexpr int NQ_A = NQ_GROUP + NQ_Z;          // 1544
constexpr int NQ_OUT = HID / 4;                // 512
constexpr int CW = 4;                          // conv width
constexpr unsigned SPIN_MAX = 1u << 24;

// workspace layout (bytes)
constexpr int WS_SYNC = 0;                         // unsigned: [0] count [64] gen [128] err [192+g*64] group counters
constexpr int WS_QKV = 4096;                       // bf16 [4][4096] post-conv
constexpr int WS_Z = WS_QKV + 4 * QKV * 2;         // bf16 [4][2048]
constexpr int WS_BA = WS_Z + 4 * HID * 2;          // bf16 [4][32]
constexpr int WS_O = WS_BA + 4 * 32 * 2 + 256;     // bf16 [4][2048]
constexpr int WS_BYTES = WS_O + 4 * HID * 2;
constexpr int S_COUNT = 0, S_GEN = 64, S_ERR = 128, S_GRP = 192;

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
DEVI float sum16(float v) {
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) v += __shfl_xor(v, o);
  return v;
}
DEVI float silu(float x) { return x / (1.0f + expf(-x)); }
DEVI float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }
DEVI uint4 ld16(const uint16_t* p) { return *reinterpret_cast<const uint4*>(p); }
DEVI void unpack8(const uint4& u, float (&f)[8]) {
  f[0] = lo(u.x); f[1] = hi(u.x); f[2] = lo(u.y); f[3] = hi(u.y);
  f[4] = lo(u.z); f[5] = hi(u.z); f[6] = lo(u.w); f[7] = hi(u.w);
}
DEVI uint4 pack8(const float (&f)[8]) {
  return make_uint4(f2bf(f[0]) | (f2bf(f[1]) << 16), f2bf(f[2]) | (f2bf(f[3]) << 16),
                    f2bf(f[4]) | (f2bf(f[5]) << 16), f2bf(f[6]) | (f2bf(f[7]) << 16));
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

struct GdnArgs {
  const uint16_t* x;        // [M, 2048] bf16 (input-normed hidden)
  const uint16_t* w_in;     // [6176, 2048]
  const uint16_t* conv_w;   // [4096, 4]
  uint16_t* conv_state;     // [slots, 4096, >=3] bf16 (strides below, elements)
  const int* idx;           // [M] cache slot (shared by conv and ssm state); <0 = padding
  const float* A_log;       // [16]
  const uint16_t* dt_bias;  // [16] bf16
  float* ssm;               // [slots, 16, 128, 128] fp32, [hv][v][k]
  const uint16_t* w_norm;   // [128] bf16
  const uint16_t* w_out;    // [2048, 2048]
  uint16_t* out;            // [M, 2048]
  uint8_t* ws;
  long long cs_slot, cs_dim, cs_pos;  // conv_state strides (elements)
  long long ssm_slot;                 // ssm state slot stride (elements)
  int num_slots;
  float scale, eps;
};

// row index of in_proj for group quad (g, i), lane-uniform; r = 0..3 within the quad
DEVI int group_row(int g, int i, int r) {
  if (i < 32) return g * DK + i * 4 + r;                      // q
  if (i < 64) return ROW_K + g * DK + (i - 32) * 4 + r;       // k
  if (i < 128) return ROW_V + g * 2 * DV + (i - 64) * 4 + r;  // v (2 heads)
  return (r < 2) ? ROW_B + 2 * g + r : ROW_A + 2 * g + (r - 2);  // b, a
}

template <int M>
DEVI void gemv_quad(const uint16_t* const (&w)[4], const uint4 (&xr)[M][4], int lane, float (&o)[4][M]) {
  // M >= 3: two passes of two rows, so the weights and M x rows fit 256 VGPRs without spills.
  constexpr int P = (M <= 2) ? 4 : 2;
#pragma unroll
  for (int q0 = 0; q0 < 4; q0 += P) {
    uint4 wr[P][4];
#pragma unroll
    for (int q = 0; q < P; ++q)
#pragma unroll
      for (int j = 0; j < 4; ++j) wr[q][j] = ld16(w[q0 + q] + j * 512 + lane * 8);
#pragma unroll
    for (int q = 0; q < P; ++q)
#pragma unroll
      for (int m = 0; m < M; ++m) {
        float a = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) dot8(wr[q][j], xr[m][j], a);
        o[q0 + q][m] = wave_sum(a);
      }
  }
}

template <int M>
DEVI void dot_quad(const uint4 (&wr)[4][4], const uint4 (&xr)[M][4], float (&o)[4][M]) {
#pragma unroll
  for (int q = 0; q < 4; ++q)
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float a = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) dot8(wr[q][j], xr[m][j], a);
      o[q][m] = wave_sum(a);
    }
}

template <int M>
DEVI float pick(const float (&o)[4][M], int q, int m) {
  float v = 0.f;
#pragma unroll
  for (int qq = 0; qq < 4; ++qq)
#pragma unroll
    for (int mm = 0; mm < M; ++mm)
      if (qq == q && mm == m) v = o[qq][mm];
  return v;
}

template <int M>
__global__ void __launch_bounds__(NT, (M <= 3 ? 2 : 1)) k_gdn_body(GdnArgs a) {
  const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
  const int gw = blockIdx.x * WPB + wave, NW = gridDim.x * WPB;
  unsigned* sync = reinterpret_cast<unsigned*>(a.ws + WS_SYNC);
  uint16_t* ws_qkv = reinterpret_cast<uint16_t*>(a.ws + WS_QKV);
  uint16_t* ws_z = reinterpret_cast<uint16_t*>(a.ws + WS_Z);
  uint16_t* ws_ba = reinterpret_cast<uint16_t*>(a.ws + WS_BA);
  uint16_t* ws_o = reinterpret_cast<uint16_t*>(a.ws + WS_O);

  const int kq = (lane & 15) * 8, rr = lane >> 4;  // recurrence lane layout
  const int ntiles = M * NVH * (DV / 8);

  // prefetch the SSM state of this wave's first recurrence tile (2 value rows x 8 k per lane)
  constexpr bool PF = (M <= 2);  // register prefetches (state tile, first out_proj quad) for M <= 2
  float4 hpf[2][2];
  if (PF) {
    const int t = gw;
    if (t < ntiles) {
      const int m = t / (NVH * 16), hv = (t / 16) % NVH, vt = t % 16;
      const int slot = a.idx[m];
      if (slot >= 0 && slot < a.num_slots) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const float* p = a.ssm + (size_t)slot * a.ssm_slot + ((size_t)hv * DV + vt * 8 + rr + 4 * j) * DK + kq;
          hpf[j][0] = *reinterpret_cast<const float4*>(p);
          hpf[j][1] = *reinterpret_cast<const float4*>(p + 4);
        }
      }
    }
  }

  uint4 xr[M][4];
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int j = 0; j < 4; ++j) xr[m][j] = ld16(a.x + (size_t)m * HID + j * 512 + lane * 8);

  // ---- phase A: in_proj quads; group rows (conv epilogue), then z
  unsigned gcnt = 0;  // 4 bits per group: quads of group g done by this wave (<= 15)
  for (int t = gw; t < NQ_A; t += NW) {
    int rows[4];
    const uint16_t* w[4];
    int g = -1;
    if (t < NQ_GROUP) {
      g = t / GQ;
      const int i = t % GQ;
#pragma unroll
      for (int r = 0; r < 4; ++r) rows[r] = group_row(g, i, r);
    } else {
#pragma unroll
      for (int r = 0; r < 4; ++r) rows[r] = ROW_Z + (t - NQ_GROUP) * 4 + r;
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) w[r] = a.w_in + (size_t)rows[r] * HID;
    float o[4][M];
    gemv_quad<M>(w, xr, lane, o);
    if (lane < 4 * M) {
      const int q = lane / M, m = lane % M;
      int row = rows[0];
#pragma unroll
      for (int r = 1; r < 4; ++r)
        if (q == r) row = rows[r];
      const float y = bfr(pick<M>(o, q, m));  // in_proj output is bf16
      if (row < QKV) {
        const int slot = a.idx[m];
        float res = y;
        if (slot >= 0 && slot < a.num_slots) {
          uint16_t* cs = a.conv_state + (size_t)slot * a.cs_slot + (size_t)row * a.cs_dim;
          const uint16_t* cw = a.conv_w + (size_t)row * CW;
          float s[CW - 1];
#pragma unroll
          for (int p = 0; p < CW - 1; ++p) s[p] = bf2f(cs[p * a.cs_pos]);
          float acc = 0.f;
#pragma unroll
          for (int p = 0; p < CW - 1; ++p) acc += bfr(s[p] * bf2f(cw[p]));
          acc += bfr(y * bf2f(cw[CW - 1]));
          res = silu(acc);
#pragma unroll
          for (int p = 0; p < CW - 2; ++p) cs[p * a.cs_pos] = (uint16_t)f2bf(s[p + 1]);
          cs[(CW - 2) * a.cs_pos] = (uint16_t)f2bf(y);
        }
        ws_qkv[m * QKV + row] = (uint16_t)f2bf(res);
      } else if (row < ROW_B) {
        ws_z[m * HID + (row - ROW_Z)] = (uint16_t)f2bf(y);
      } else {
        ws_ba[m * 32 + (row - ROW_B)] = (uint16_t)f2bf(y);
      }
    }
    if (g >= 0) gcnt += 1u << (4 * g);
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");  // this wave's stores before its arrivals
  if (lane == 0)
#pragma unroll
    for (int g = 0; g < NKH; ++g) {
      const unsigned c = (gcnt >> (4 * g)) & 15u;
      if (c) __hip_atomic_fetch_add(sync + S_GRP + g * 64, c, __ATOMIC_RELAXED, AGENT);
    }

  // prefetch this wave's first out_proj weight quad
  uint4 wpf[4][4];
  if (PF && gw < NQ_OUT) {
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
      for (int j = 0; j < 4; ++j) wpf[q][j] = ld16(a.w_out + (size_t)(gw * 4 + q) * HID + j * 512 + lane * 8);
  }

  // ---- phase B: recurrence tiles (m, hv, 8 value rows)
  for (int t = gw; t < ntiles; t += NW) {
    const int m = t / (NVH * 16), hv = (t / 16) % NVH, vt = t % 16, g = hv / (NVH / NKH);
    const int slot = a.idx[m];
    const int v0 = vt * 8 + rr;
    if (!(slot >= 0 && slot < a.num_slots)) {
      if ((lane & 15) == 0) {
        ws_o[m * HID + hv * DV + v0] = 0;
        ws_o[m * HID + hv * DV + v0 + 4] = 0;
      }
      continue;
    }
    float h[2][8];
    float* hp[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) hp[j] = a.ssm + (size_t)slot * a.ssm_slot + ((size_t)hv * DV + v0 + 4 * j) * DK + kq;
    if (PF && t == gw) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        h[j][0] = hpf[j][0].x; h[j][1] = hpf[j][0].y; h[j][2] = hpf[j][0].z; h[j][3] = hpf[j][0].w;
        h[j][4] = hpf[j][1].x; h[j][5] = hpf[j][1].y; h[j][6] = hpf[j][1].z; h[j][7] = hpf[j][1].w;
      }
    } else {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const float4 u0 = *reinterpret_cast<const float4*>(hp[j]);
        const float4 u1 = *reinterpret_cast<const float4*>(hp[j] + 4);
        h[j][0] = u0.x; h[j][1] = u0.y; h[j][2] = u0.z; h[j][3] = u0.w;
        h[j][4] = u1.x; h[j][5] = u1.y; h[j][6] = u1.z; h[j][7] = u1.w;
      }
    }
    spin_until_geq(sync + S_GRP + g * 64, (unsigned)GQ, sync + S_ERR);
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    float q[8], k[8];
    unpack8(ld16(ws_qkv + m * QKV + g * DK + kq), q);
    unpack8(ld16(ws_qkv + m * QKV + ROW_K + g * DK + kq), k);
    float v[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) v[j] = bf2f(ws_qkv[m * QKV + ROW_V + hv * DV + v0 + 4 * j]);
    const float bv = bf2f(ws_ba[m * 32 + hv]), av = bf2f(ws_ba[m * 32 + NVH + hv]);
    float qs = 0.f, ks = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      qs = fmaf(q[i], q[i], qs);
      ks = fmaf(k[i], k[i], ks);
    }
    qs = sum16(qs);
    ks = sum16(ks);
    const float qn = sqrtf(qs + 1e-6f), kn = sqrtf(ks + 1e-6f);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      q[i] = q[i] / qn * a.scale;
      k[i] = k[i] / kn;
    }
    const float xg = av + bf2f(a.dt_bias[hv]);
    const float sp = (xg <= 20.f) ? logf(1.f + expf(xg)) : xg;
    const float gl = -expf(a.A_log[hv]) * sp;
    const float beta = bfr(sigm(bv));
    const float dec = expf(gl);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      float s = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        h[j][i] *= dec;
        s = fmaf(h[j][i], k[i], s);
      }
      s = sum16(s);
      const float vv = (v[j] - s) * beta;
      float ov = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        h[j][i] = fmaf(vv, k[i], h[j][i]);
        ov = fmaf(h[j][i], q[i], ov);
      }
      ov = sum16(ov);
      if ((lane & 15) == 0) ws_o[m * HID + hv * DV + v0 + 4 * j] = (uint16_t)f2bf(ov);
      *reinterpret_cast<float4*>(hp[j]) = make_float4(h[j][0], h[j][1], h[j][2], h[j][3]);
      *reinterpret_cast<float4*>(hp[j] + 4) = make_float4(h[j][4], h[j][5], h[j][6], h[j][7]);
    }
  }

  grid_barrier(sync);
  if (blockIdx.x == 0 && threadIdx.x < NKH)
    __hip_atomic_store(sync + S_GRP + threadIdx.x * 64, 0u, __ATOMIC_RELAXED, AGENT);

  // ---- phase C: gated RMSNorm per head (each wave, its own 4 x 8 columns), out_proj quads
  if (gw < NQ_OUT) {
    float wn[8];
    unpack8(ld16(a.w_norm + (lane * 8) % DV), wn);
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int col = j * 512 + lane * 8;
        float o8[8], z8[8];
        unpack8(ld16(ws_o + m * HID + col), o8);
        unpack8(ld16(ws_z + m * HID + col), z8);
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) ss = fmaf(o8[i], o8[i], ss);
        ss = sum16(ss);
        const float rstd = 1.0f / sqrtf(ss / (float)DV + a.eps);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float y = o8[i] * rstd * wn[i];
          o8[i] = y * z8[i] * sigm(z8[i]);
        }
        xr[m][j] = pack8(o8);
      }
  }
  for (int t = gw; t < NQ_OUT; t += NW) {
    float o[4][M];
    if (PF && t == gw) {
      dot_quad<M>(wpf, xr, o);
    } else {
      const uint16_t* w[4];
#pragma unroll
      for (int r = 0; r < 4; ++r) w[r] = a.w_out + (size_t)(t * 4 + r) * HID;
      gemv_quad<M>(w, xr, lane, o);
    }
    if (lane < 4 * M) {
      const int q = lane / M, m = lane % M;
      a.out[(size_t)m * HID + t * 4 + q] = (uint16_t)f2bf(pick<M>(o, q, m));
    }
  }
}

// ------------------------------------------------------------------------------------------
// (2) in-kernel P2P all-reduce, two ranks. Each workgroup owns NT*8 bf16 of the vector.
struct ArArgs {
  const uint16_t* in;    // [n] this rank's partial
  uint16_t* out;         // [n]
  uint16_t* my_recv;     // [2][n] in my memory (uncached), written by the peer
  uint16_t* peer_recv;   // [2][n] in the peer's memory (mapped), written by me
  unsigned* my_flag;     // [grid*64] in my memory (uncached), written by the peer
  unsigned* peer_flag;   // [grid*64] peer memory (mapped)
  unsigned* epoch;       // [grid*64] my memory, per-workgroup call counter
  unsigned* err;
  int n, rank, reps, use_wbl2;
};

__global__ void __launch_bounds__(NT) k_ar(ArArgs a) {
  const int b = blockIdx.x, t = threadIdx.x;
  const int i = (b * NT + t) * 8;
  for (int rep = 0; rep < a.reps; ++rep) {
    const unsigned e = a.epoch[b * 64] + 1u + (unsigned)rep;
    const int buf = (int)(e & 1u) * a.n;
    uint4 mine = make_uint4(0, 0, 0, 0);
    if (i < a.n) {
      mine = ld16(a.in + i);
      __builtin_nontemporal_store(mine.x, reinterpret_cast<uint32_t*>(a.peer_recv + buf + i) + 0);
      __builtin_nontemporal_store(mine.y, reinterpret_cast<uint32_t*>(a.peer_recv + buf + i) + 1);
      __builtin_nontemporal_store(mine.z, reinterpret_cast<uint32_t*>(a.peer_recv + buf + i) + 2);
      __builtin_nontemporal_store(mine.w, reinterpret_cast<uint32_t*>(a.peer_recv + buf + i) + 3);
    }
    if (a.use_wbl2)
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");  // system: buffer_wbl2 + s_waitcnt
    else
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    if (t == 0) {
      __hip_atomic_store(a.peer_flag + b * 64, e, __ATOMIC_RELAXED, SYSTEM);
      unsigned n = 0;
      while (__hip_atomic_load(a.my_flag + b * 64, __ATOMIC_RELAXED, SYSTEM) < e) {
        if (++n > SPIN_MAX) {
          __hip_atomic_store(a.err, 1u, __ATOMIC_RELAXED, SYSTEM);
          break;
        }
      }
    }
    __syncthreads();
    if (a.use_wbl2)
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");  // system: buffer_invl2 + buffer_wbinvl1_vol
    else
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    if (i < a.n) {
      uint4 other;
      other.x = __builtin_nontemporal_load(reinterpret_cast<const uint32_t*>(a.my_recv + buf + i) + 0);
      other.y = __builtin_nontemporal_load(reinterpret_cast<const uint32_t*>(a.my_recv + buf + i) + 1);
      other.z = __builtin_nontemporal_load(reinterpret_cast<const uint32_t*>(a.my_recv + buf + i) + 2);
      other.w = __builtin_nontemporal_load(reinterpret_cast<const uint32_t*>(a.my_recv + buf + i) + 3);
      const uint4 x0 = a.rank == 0 ? mine : other, x1 = a.rank == 0 ? other : mine;
      float f0[8], f1[8];
      unpack8(x0, f0);
      unpack8(x1, f1);
#pragma unroll
      for (int k = 0; k < 8; ++k) f0[k] = f0[k] + f1[k];
      *reinterpret_cast<uint4*>(a.out + i) = pack8(f0);
    }
  }
  __syncthreads();
  if (t == 0) a.epoch[b * 64] += (unsigned)a.reps;
}

// ------------------------------------------------------------------------------------------
extern "C" {

int gl_ws_bytes() { return WS_BYTES; }

int gl_occupancy(int M) {
  int n = -1;
  switch (M) {
#define OCC(MM) \
  case MM: (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, k_gdn_body<MM>, NT, 0); break;
    OCC(1) OCC(2) OCC(3) OCC(4)
#undef OCC
    default: break;
  }
  return n;
}

int gl_launch(int M, int G, const GdnArgs* a, hipStream_t st) {
  switch (M) {
#define LAUNCH(MM) \
  case MM: hipLaunchKernelGGL(k_gdn_body<MM>, dim3(G), dim3(NT), 0, st, *a); break;
    LAUNCH(1) LAUNCH(2) LAUNCH(3) LAUNCH(4)
#undef LAUNCH
    default: return -1;
  }
  return (int)hipGetLastError();
}

void* ar_alloc_uncached(size_t bytes) {
  void* p = nullptr;
  if (hipExtMallocWithFlags(&p, bytes, hipDeviceMallocUncached) != hipSuccess) return nullptr;
  if (hipMemset(p, 0, bytes) != hipSuccess) return nullptr;
  (void)hipDeviceSynchronize();
  return p;
}

int ar_ipc_handle(void* p, char* out64) {
  hipIpcMemHandle_t h;
  const hipError_t e = hipIpcGetMemHandle(&h, p);
  for (int i = 0; i < 64; ++i) out64[i] = h.reserved[i];
  return (int)e;
}

void* ar_ipc_open(const char* in64) {
  hipIpcMemHandle_t h;
  for (int i = 0; i < 64; ++i) h.reserved[i] = in64[i];
  void* p = nullptr;
  if (hipIpcOpenMemHandle(&p, h, hipIpcMemLazyEnablePeerAccess) != hipSuccess) return nullptr;
  return p;
}

int ar_launch(int G, const ArArgs* a, hipStream_t st) {
  hipLaunchKernelGGL(k_ar, dim3(G), dim3(NT), 0, st, *a);
  return (int)hipGetLastError();
}

}  // extern "C"
