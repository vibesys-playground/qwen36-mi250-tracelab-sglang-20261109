# Throwaway harness probe: the project's baseline Transformers engine (engine/) behind the task's
# service contract, on the stack the project README gives for the AMD tasks (Python 3.12,
# torch 2.9.1+rocm6.4, requirements-agentx.txt). The ROCm torch wheel carries its own ROCm
# userspace, so a slim Python base suffices.
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
COPY requirements-agentx.txt /tmp/requirements-agentx.txt
RUN pip install --no-cache-dir 'torch==2.9.1+rocm6.4' --index-url https://download.pytorch.org/whl/rocm6.4 \
 && pip install --no-cache-dir -r /tmp/requirements-agentx.txt \
 && python3 -c "import torch, transformers; assert torch.version.hip; print(torch.__version__, transformers.__version__)"
ENV PYTHONUNBUFFERED=1 HOME=/tmp
ENV HF_HOME=/vibesys/mounts/hf HF_HUB_OFFLINE=1
