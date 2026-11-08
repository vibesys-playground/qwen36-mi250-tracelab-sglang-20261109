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
# Parallel layout: four TP=2 replicas, one per MI250 card. DP rank r takes HIP devices 2r and
# 2r+1, which are the two GCDs of one package (measured, plan 4 step 1), so every all-reduce
# stays on the in-package link. prefix_affinity sends each session's later rounds to the
# replica whose radix and recurrent-state cache holds its prefix; new sessions go to the
# least-loaded replica.
# Speculative decoding with the checkpoint's own MTP head (mtp.* tensors, one layer): it
#   drafts 2 tokens per step as a topk=1 chain and the target verifies all 3 positions in one
#   forward. Temperature-0 requests verify greedily (accept only the target's argmax), so the
#   emitted tokens are the target's greedy tokens.
# Deterministic (batch-invariant) inference: fixed attention split tiles, one MoE kernel
#   config and greedy verify, so a token's logits do not depend on what else is in its batch
#   or on how much of its prompt came from the radix cache. Without it, MTP made a repeated
#   temperature-0 prompt diverge in 3 of 20 sends when the cache was kept (diagnostic j48).
#   On ROCm it requires TP <= 2 when custom all-reduce is off, which this layout has.
exec python3 -m sglang.launch_server "$@" \
  --tp-size 2 \
  --dp-size 4 \
  --load-balance-method prefix_affinity \
  --attention-backend triton \
  --disable-custom-all-reduce \
  --enable-cache-report \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 2 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 3 \
  --enable-deterministic-inference
