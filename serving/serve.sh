#!/usr/bin/env bash
# The candidate's server, started by the task from the checkout's root in the image
# serving/Dockerfile builds. The task passes the contract's arguments: the checkpoint and its
# revision (read offline from the mounted Hugging Face cache), --host 127.0.0.1, --port, and
# --context-length 262144. This script is the candidate's to change.
#
# The starting point is upstream SGLang's standard server with only what loading the
# checkpoint across the node's eight GPUs needs.
set -euo pipefail
# The checkout's root holds the vendored sglang/ tree, which Python would import from the
# working directory as a namespace package in place of the package the image installed.
cd /
# Triton, the JIT kernels and torch write caches under $HOME; a job may run as a user whose HOME
# is missing or read-only inside the image.
if ! { mkdir -p "${HOME:-/nonexistent}" && [ -w "$HOME" ]; } 2>/dev/null; then
  export HOME
  HOME="$(mktemp -d)"
fi
# MI250 (gfx90a) adaptations, none of which changes the model's math:
# - The HIP default attention backend is aiter, which the image does not install and which
#   targets gfx942/gfx950; Triton attention runs on gfx90a. GDN layers already use Triton.
# - Custom all-reduce is tuned for gfx942's all-to-all XGMI; MI250's 8 GCDs are not one hop
#   apart, so every collective goes through RCCL.
# - The chat default is non-thinking (per-request chat_template_kwargs still override it).
# - Usage reports cached prompt tokens, which the replay's prefix-cache accounting reads.
# - Speculative decoding with the checkpoint's own MTP head (mtp.* tensors, one layer): it
#   drafts 2 tokens per step as a topk=1 chain and the target verifies all 3 positions in one
#   forward. Temperature-0 requests verify greedily (accept only the target's argmax), so the
#   emitted tokens are the target's greedy tokens.
exec python3 -m sglang.launch_server "$@" \
  --tp-size 8 \
  --attention-backend triton \
  --disable-custom-all-reduce \
  --enable-cache-report \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 2 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 3
