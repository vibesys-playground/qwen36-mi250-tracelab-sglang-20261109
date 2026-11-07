# The candidate's server image. The task (.vibesys/tasks/mi2508x-tracelab-sglang/vibesys.input.toml,
# [candidate]) builds it from the candidate's commit with the repository root as the context
# (.dockerignore at the root narrows it) and runs serving/serve.sh in it from the checkout's root.
# This file, serve.sh and the build context are the candidate's to change.
#
# The starting point installs the vendored upstream SGLang checkout (sglang/) by upstream's own
# ROCm "Install from Source" steps (sglang/docs/docs/hardware-platforms/amd_gpu.mdx), on the
# ROCm 7.2 PyTorch base that upstream's docker/rocm.Dockerfile uses for its ROCm 7.2 flavors,
# pinned by digest. Where those docs leave a step implicit, it follows rocm.Dockerfile: the Rust
# toolchain for SGLang's Rust extension, the ROCm amdsmi Python package, the kernel package's
# ROCm pyproject, and a constraint that keeps the base's ROCm torch and triton.
# Model weights are never baked in: the operator mounts the Hugging Face cache.
FROM rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1@sha256:376bfab5f4f680c8b4b843c6d0c5d1f0a04e5a84ec3e86728db8d11d79a9d1e3

RUN python -m pip install --upgrade pip && pip install setuptools_scm

# amdsmi from the base's ROCm installation (rocm.Dockerfile, ROCm 7.2 flavors).
RUN cd /opt/rocm/share/amd_smi && python3 -m pip install --no-cache-dir .

# Rust toolchain: setuptools-rust builds SGLang's Rust extension during its pip install.
ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 --retry 5 --retry-delay 3 --retry-all-errors -sSf https://sh.rustup.rs | sh -s -- -y \
 && rustc --version && cargo --version

COPY sglang /sgl-workspace/sglang
WORKDIR /sgl-workspace/sglang

# Compile sgl-kernel for MI250 (gfx90a). The build has no GPU, so setup_rocm.py cannot probe the
# architecture and would otherwise default to gfx942, whose code objects do not load on gfx90a.
ENV AMDGPU_TARGET=gfx90a
RUN cd python/sglang/kernels/aot \
 && rm -f pyproject.toml \
 && mv pyproject_rocm.toml pyproject.toml \
 && python setup_rocm.py install

# Install the sglang Python package with its ROCm extras, keeping the base's ROCm torch.
RUN pip list --format=freeze | grep -E '^(torch|triton)' > /tmp/constraints.txt \
 && rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml \
 && pip install --no-cache-dir -c /tmp/constraints.txt -e "python[all_hip]" \
 && python3 -c "import torch; assert torch.version.hip, 'torch is not a ROCm build'; print('torch', torch.__version__, 'hip', torch.version.hip)"

WORKDIR /
ENV PYTHONUNBUFFERED=1
# The operator's Hugging Face cache, mounted read-only as "hf" (the task's [environment] needs).
ENV HF_HOME=/vibesys/mounts/hf HF_HUB_OFFLINE=1
