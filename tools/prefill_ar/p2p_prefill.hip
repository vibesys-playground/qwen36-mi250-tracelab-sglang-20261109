// Plan 31 step 1 microbench kernels (gfx90a, TP=2 pair). Not on any serving path.
//  (c) link probes: peer read / peer push over the in-package XGMI, and hipMemcpyAsync D2D.
//  (d) prototype prefill boundary: push all-reduce + residual add + Gemma RMSNorm in one kernel.
//
// (d) protocol. Each rank pushes its partial rows (bf16, as is) into the peer's receive buffer
// (hipDeviceMallocUncached, IPC-mapped) with non-temporal stores, then sets a per-workgroup flag
// in the peer's control block. Workgroup b owns rows m = b, b + G, ...; the peer's workgroup b
// pushes the same rows, so b waits only for the peer's b. Once the flag shows epoch e, b sums
// rank 0 + rank 1 in fp32, rounds to bf16 (= the custom all-reduce), adds the residual (rounded,
// as the Triton Gemma add+norm with round_sum), and normalizes with a fixed-tree sum of squares.
// Receive buffers are double-buffered by epoch parity (as plan 25 B). FENCE selects how the data
// is ordered before the flag: 0 = s_waitcnt vmcnt(0) only (the data never sits in this GCD's L2:
// it is uncached peer memory), 1 = a system-scope release fence (buffer_wbl2) as well.
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__

static hipStream_t cur_stream() { return c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream(); }

typedef unsigned v4u __attribute__((ext_vector_type(4)));

// ------------------------------------------------------------------------------------------
// memory helpers
int64_t alloc(int64_t bytes, bool uncached) {
  void* p = nullptr;
  hipError_t e = uncached ? hipExtMallocWithFlags(&p, bytes, hipDeviceMallocUncached)
                          : hipMalloc(&p, bytes);
  TORCH_CHECK(e == hipSuccess, "alloc failed");
  TORCH_CHECK(hipMemset(p, 0, bytes) == hipSuccess);
  TORCH_CHECK(hipDeviceSynchronize() == hipSuccess);
  return (int64_t)p;
}

std::vector<int64_t> ipc_handle(int64_t p) {
  hipIpcMemHandle_t h;
  TORCH_CHECK(hipIpcGetMemHandle(&h, (void*)p) == hipSuccess);
  std::vector<int64_t> out(64);
  for (int i = 0; i < 64; ++i) out[i] = (unsigned char)h.reserved[i];
  return out;
}

int64_t ipc_open(std::vector<int64_t> v) {
  hipIpcMemHandle_t h;
  for (int i = 0; i < 64; ++i) h.reserved[i] = (char)v[i];
  void* p = nullptr;
  TORCH_CHECK(hipIpcOpenMemHandle(&p, h, hipIpcMemLazyEnablePeerAccess) == hipSuccess);
  return (int64_t)p;
}

void fill_random(int64_t p, int64_t bytes, int64_t seed) {
  auto t = torch::from_blob((void*)p, {bytes / 2}, torch::TensorOptions().dtype(torch::kInt16).device(torch::kCUDA));
  auto g = torch::randint(-30000, 30000, {bytes / 2}, torch::TensorOptions().dtype(torch::kInt16).device(torch::kCUDA));
  t.copy_(g);
  TORCH_CHECK(hipDeviceSynchronize() == hipSuccess);
}

// ------------------------------------------------------------------------------------------
// (c) link probes
template <int U>
__global__ void __launch_bounds__(512) peer_read_kernel(const v4u* __restrict__ src, v4u* __restrict__ dst, int64_t n16) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (; i + (U - 1) * stride < n16; i += U * stride) {
    v4u v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) v[u] = __builtin_nontemporal_load(src + i + u * stride);
#pragma unroll
    for (int u = 0; u < U; ++u) __builtin_nontemporal_store(v[u], dst + i + u * stride);
  }
  for (; i < n16; i += stride) __builtin_nontemporal_store(__builtin_nontemporal_load(src + i), dst + i);
}

template <int U>
__global__ void __launch_bounds__(512) peer_push_kernel(const v4u* __restrict__ src, v4u* __restrict__ dst, int64_t n16) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (; i + (U - 1) * stride < n16; i += U * stride) {
    v4u v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) v[u] = src[i + u * stride];
#pragma unroll
    for (int u = 0; u < U; ++u) __builtin_nontemporal_store(v[u], dst + i + u * stride);
  }
  for (; i < n16; i += stride) __builtin_nontemporal_store(src[i], dst + i);
}

// mode 0 = read (src remote, dst local), 1 = push (src local, dst remote), 2 = hipMemcpyAsync
void probe(int64_t mode, int64_t dst, int64_t src, int64_t bytes, int64_t blocks) {
  hipStream_t s = cur_stream();
  const int64_t n16 = bytes / 16;
  if (mode == 0)
    hipLaunchKernelGGL(peer_read_kernel<4>, dim3(blocks), dim3(512), 0, s, (const v4u*)src, (v4u*)dst, n16);
  else if (mode == 1)
    hipLaunchKernelGGL(peer_push_kernel<4>, dim3(blocks), dim3(512), 0, s, (const v4u*)src, (v4u*)dst, n16);
  else
    TORCH_CHECK(hipMemcpyAsync((void*)dst, (const void*)src, bytes, hipMemcpyDeviceToDevice, s) == hipSuccess);
  TORCH_CHECK(hipGetLastError() == hipSuccess);
}

// ------------------------------------------------------------------------------------------
// (d) fused push all-reduce + add + Gemma RMSNorm, n = 2048 columns, 256 threads x 8 columns
#define CTL_EPOCH 1024
#define CTL_CNT 1088
#define CTL_ERR 1152
#define SPIN_MAX (1 << 28)

DEVI float bf2f(uint32_t h) { return __uint_as_float(h << 16); }
DEVI uint32_t f2bf(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;
  u += 0x7fffu + ((u >> 16) & 1u);
  return u >> 16;
}
DEVI void unpack8(const v4u& v, float* f) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = __uint_as_float(w[i] << 16);
    f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
  }
}
DEVI v4u pack8(const float* f) {
  v4u v;
  v.x = f2bf(f[0]) | (f2bf(f[1]) << 16);
  v.y = f2bf(f[2]) | (f2bf(f[3]) << 16);
  v.z = f2bf(f[4]) | (f2bf(f[5]) << 16);
  v.w = f2bf(f[6]) | (f2bf(f[7]) << 16);
  return v;
}

struct FArgs {
  const v4u* in;        // [M, 2048] this rank's partial
  const v4u* residual;  // [M, 2048]
  const v4u* weight;    // [2048]
  v4u* out;             // [M, 2048] normed
  v4u* res_out;         // [M, 2048] residual + sum
  v4u* peer_recv;       // peer's receive buffer [2][rows][2048]
  const v4u* my_recv;   // this rank's receive buffer
  unsigned* peer_ctl;     // peer's control block (uncached)
  unsigned* my_ctl;       // this rank's control block (uncached)
  int M, rows, rank;
  float eps;
};

template <int FENCE>
__global__ void __launch_bounds__(256) push_ar_norm_kernel(const FArgs a) {
  __shared__ unsigned s_e;
  __shared__ float sh[4];
  __shared__ unsigned s_last;
  const int t = threadIdx.x, b = blockIdx.x, G = gridDim.x;
  if (t == 0) s_e = __hip_atomic_load(a.my_ctl + CTL_EPOCH, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) + 1u;
  __syncthreads();
  const unsigned e = s_e;
  const size_t par = (size_t)(e & 1u) * a.rows * 256;
  // push
  for (int m = b; m < a.M; m += G)
    __builtin_nontemporal_store(a.in[(size_t)m * 256 + t], a.peer_recv + par + (size_t)m * 256 + t);
  if (FENCE) __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);  // system scope: buffer_wbl2 + waitcnt
  __builtin_amdgcn_s_waitcnt(0);
  __syncthreads();
  if (t == 0) {
    __hip_atomic_store(a.peer_ctl + b, e, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    int n = 0;
    while (__hip_atomic_load(a.my_ctl + b, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < e) {
      if (++n > SPIN_MAX) {
        __hip_atomic_store(a.my_ctl + CTL_ERR, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        break;
      }
    }
  }
  __syncthreads();
  const v4u wv = a.weight[t];
  float w[8];
  unpack8(wv, w);
  for (int m = b; m < a.M; m += G) {
    const size_t o = (size_t)m * 256 + t;
    const uint64_t* rp = reinterpret_cast<const uint64_t*>(a.my_recv + par + o);
    v4u pv;
    {
      const uint64_t lo = __hip_atomic_load(rp, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      const uint64_t hi = __hip_atomic_load(rp + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      pv.x = (uint32_t)lo; pv.y = (uint32_t)(lo >> 32); pv.z = (uint32_t)hi; pv.w = (uint32_t)(hi >> 32);
    }
    float own[8], peer[8], r[8], x[8];
    unpack8(a.in[o], own);
    unpack8(pv, peer);
    unpack8(a.residual[o], r);
    float sq = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float s = a.rank == 0 ? own[i] + peer[i] : peer[i] + own[i];
      x[i] = bf2f(f2bf(bf2f(f2bf(s)) + r[i]));
    }
    sq = x[0] * x[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) sq = sq + x[i] * x[i];
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) sq = sq + __shfl_xor(sq, off, 64);
    if ((t & 63) == 0) sh[t >> 6] = sq;
    __syncthreads();
    const float tot = ((sh[0] + sh[1]) + sh[2]) + sh[3];
    __syncthreads();
    const float rstd = 1.0f / sqrtf(tot / 2048.0f + a.eps);
    a.res_out[o] = pack8(x);
    float y[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) y[i] = (x[i] * rstd) * (1.0f + w[i]);
    a.out[o] = pack8(y);
  }
  // the last workgroup advances the epoch
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  __syncthreads();
  if (t == 0) {
    const unsigned old = __hip_atomic_fetch_add(a.my_ctl + CTL_CNT, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    s_last = old == (unsigned)G - 1 ? 1u : 0u;
    if (s_last) {
      __hip_atomic_store(a.my_ctl + CTL_CNT, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(a.my_ctl + CTL_EPOCH, e, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
  }
}

void push_ar_norm(torch::Tensor in, torch::Tensor residual, torch::Tensor weight, torch::Tensor out,
                  torch::Tensor res_out, int64_t peer_recv, int64_t my_recv, int64_t peer_ctl,
                  int64_t my_ctl, int64_t rows, int64_t rank, double eps, int64_t grid, int64_t fence) {
  TORCH_CHECK(in.size(-1) == 2048 && in.is_contiguous() && residual.is_contiguous());
  FArgs a;
  a.in = (const v4u*)in.data_ptr();
  a.residual = (const v4u*)residual.data_ptr();
  a.weight = (const v4u*)weight.data_ptr();
  a.out = (v4u*)out.data_ptr();
  a.res_out = (v4u*)res_out.data_ptr();
  a.peer_recv = (v4u*)peer_recv;
  a.my_recv = (const v4u*)my_recv;
  a.peer_ctl = (unsigned*)peer_ctl;
  a.my_ctl = (unsigned*)my_ctl;
  a.M = (int)(in.numel() / 2048);
  a.rows = (int)rows;
  a.rank = (int)rank;
  a.eps = (float)eps;
  TORCH_CHECK(a.M <= a.rows && grid <= 1024);
  hipStream_t s = cur_stream();
  if (fence)
    hipLaunchKernelGGL(push_ar_norm_kernel<1>, dim3(grid), dim3(256), 0, s, a);
  else
    hipLaunchKernelGGL(push_ar_norm_kernel<0>, dim3(grid), dim3(256), 0, s, a);
  TORCH_CHECK(hipGetLastError() == hipSuccess);
}

int64_t read_err(int64_t my_ctl) {
  unsigned v = 0;
  TORCH_CHECK(hipMemcpy(&v, (unsigned*)my_ctl + CTL_ERR, 4, hipMemcpyDeviceToHost) == hipSuccess);
  return v;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("alloc", &alloc);
  m.def("ipc_handle", &ipc_handle);
  m.def("ipc_open", &ipc_open);
  m.def("fill_random", &fill_random);
  m.def("probe", &probe);
  m.def("push_ar_norm", &push_ar_norm);
  m.def("read_err", &read_err);
}
