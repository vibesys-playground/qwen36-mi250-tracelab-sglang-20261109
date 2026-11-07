# Throwaway harness probe: the project's baseline Transformers engine (engine/) behind the task's
# service contract. Same base as the SGLang starting point; adds the engine's pins.
FROM rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1@sha256:376bfab5f4f680c8b4b843c6d0c5d1f0a04e5a84ec3e86728db8d11d79a9d1e3
RUN pip list --format=freeze | grep -E '^(torch|triton)' > /tmp/constraints.txt \
 && pip install --no-cache-dir -c /tmp/constraints.txt transformers==5.22.1 accelerate==1.12.0 \
 && python3 -c "import torch, transformers; assert torch.version.hip; print(torch.__version__, transformers.__version__)"
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/vibesys/mounts/hf HF_HUB_OFFLINE=1
