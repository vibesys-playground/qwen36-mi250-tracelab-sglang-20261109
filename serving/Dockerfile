# Throwaway harness probe: the project's baseline Transformers engine (engine/) behind the task's
# service contract, on the stack the project README gives for the AMD tasks (Python 3.12,
# torch 2.9.1+rocm6.4, requirements-agentx.txt), over the ROCm 6.4 base the vibesys example uses.
FROM rocm/pytorch:rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1@sha256:f444357218627704168ef31d991cb6827999208d38906b9c22383c6a829f54e9
COPY requirements-agentx.txt /tmp/requirements-agentx.txt
RUN python3 -m pip install --no-cache-dir 'torch==2.9.1+rocm6.4' --index-url https://download.pytorch.org/whl/rocm6.4 \
 && python3 -m pip install --no-cache-dir -r /tmp/requirements-agentx.txt \
 && python3 -c "import torch, transformers; assert torch.version.hip; print(torch.__version__, transformers.__version__)"
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/vibesys/mounts/hf HF_HUB_OFFLINE=1
