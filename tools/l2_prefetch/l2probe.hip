// Plan 32 stage 2 gate 0: read kernels that leave (or do not leave) bytes in L2.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

template <bool NT>
__global__ void __launch_bounds__(256) touch_kernel(const u32x4* __restrict__ w, long long n16,
                                                    unsigned* out) {
  const long long stride = (long long)gridDim.x * 256;
  const long long t = (long long)blockIdx.x * 256 + threadIdx.x;
  unsigned a = 0;
  for (long long i = t; i < n16; i += stride * 4) {
    u32x4 v[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const long long idx = i + u * stride;
      if (NT) {
        v[u] = idx < n16 ? __builtin_nontemporal_load(w + idx) : u32x4{0, 0, 0, 0};
      } else {
        v[u] = idx < n16 ? w[idx] : u32x4{0, 0, 0, 0};
      }
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) a ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  if (a == 0x9e3779b9u) out[blockIdx.x] = a;  // keeps the loads alive
}

extern "C" int touch(const void* w, long long nbytes, int nt, int blocks, void* out, void* stream) {
  const long long n16 = nbytes / 16;
  if (nt) {
    hipLaunchKernelGGL(touch_kernel<true>, dim3(blocks), dim3(256), 0, (hipStream_t)stream,
                       (const u32x4*)w, n16, (unsigned*)out);
  } else {
    hipLaunchKernelGGL(touch_kernel<false>, dim3(blocks), dim3(256), 0, (hipStream_t)stream,
                       (const u32x4*)w, n16, (unsigned*)out);
  }
  return (int)hipGetLastError();
}
