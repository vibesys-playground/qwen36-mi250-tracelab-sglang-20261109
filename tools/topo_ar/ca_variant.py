"""JIT-build a gfx90a variant of SGLang's HIP custom all-reduce for the microbenchmark only.

The variant is the checkout's custom_all_reduce.hip / custom_all_reduce_hip.cuh with one change
in start_sync: a system-scope release fence before a rank signals that its input is ready, and a
system-scope acquire fence after it has seen every peer's signal. On gfx90a a release at system
scope writes back the GCD's L2 (buffer_wbl2), which is what makes this rank's input buffer,
written by the previous kernel into its local L2, visible to the peer that reads it over XGMI.
`install(ops_module)` swaps the variant's functions into sglang's custom_all_reduce_ops so the
unchanged CustomAllreduce class drives it. Not on any serving path.
"""
import os
import tempfile

SRC = "sglang/python/sglang/kernels/aot/csrc/allreduce"

START_SYNC_OLD = """    __scoped_atomic_store_n(
        &sg.signals[threadIdx.x]->start[blockIdx.x][rank], flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
    // wait until we got true from all ranks
    while (__scoped_atomic_load_n(&self_sg->start[blockIdx.x][threadIdx.x], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) <
           flag)
      ;
"""
START_SYNC_NEW = """    __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    __scoped_atomic_store_n(
        &sg.signals[threadIdx.x]->start[blockIdx.x][rank], flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
    // wait until we got true from all ranks
    while (__scoped_atomic_load_n(&self_sg->start[blockIdx.x][threadIdx.x], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) <
           flag)
      ;
    __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
"""

BINDING = r'''
#include "custom_all_reduce.hip"
#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_custom_ar", &init_custom_ar);
  m.def("all_reduce_reg", &all_reduce_reg);
  m.def("all_reduce_unreg", &all_reduce_unreg);
  m.def("dispose", &dispose);
  m.def("meta_size", &meta_size);
  m.def("register_buffer", &register_buffer);
  m.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
  m.def("register_graph_buffers", &register_graph_buffers);
  m.def("allocate_meta_buffer", &allocate_meta_buffer);
  m.def("get_meta_buffer_ipc_handle", &get_meta_buffer_ipc_handle);
}
'''

FUNCS = ["init_custom_ar", "all_reduce_reg", "all_reduce_unreg", "dispose", "meta_size",
         "register_buffer", "get_graph_buffer_ipc_meta", "register_graph_buffers",
         "allocate_meta_buffer", "get_meta_buffer_ipc_handle"]


def _write(path, text):
    if os.path.exists(path) and open(path).read() == text:
        return
    with open(path, "w") as f:
        f.write(text)


def build(repo_root, build_dir=None, verbose=False):
    from torch.utils.cpp_extension import load

    d = build_dir or tempfile.mkdtemp(prefix="ca_variant_")
    os.makedirs(d, exist_ok=True)
    src = os.path.join(repo_root, SRC)
    cuh = open(os.path.join(src, "custom_all_reduce_hip.cuh")).read()
    assert cuh.count(START_SYNC_OLD) == 1, "start_sync text changed; update ca_variant.py"
    # Write only what changed, so a second call (one per worker) finds the build up to date.
    _write(os.path.join(d, "custom_all_reduce.hip"),
           open(os.path.join(src, "custom_all_reduce.hip")).read())
    _write(os.path.join(d, "custom_all_reduce_hip.cuh"), cuh.replace(START_SYNC_OLD, START_SYNC_NEW))
    _write(os.path.join(d, "binding.hip"), BINDING)
    os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx90a")
    return load(name="ca_variant_ext", sources=[os.path.join(d, "binding.hip")],
                extra_include_paths=[d], build_directory=d, verbose=verbose,
                extra_cflags=["-O3", "-DUSE_ROCM"],
                extra_cuda_cflags=["-O3", "-DUSE_ROCM", "-DNDEBUG", "-DENABLE_BF16",
                                   "--offload-arch=gfx90a"])


def install(ops_module, mod):
    for f in FUNCS:
        setattr(ops_module, f, getattr(mod, f))
