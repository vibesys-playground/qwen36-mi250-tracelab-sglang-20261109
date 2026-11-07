#!/usr/bin/env bash
# The candidate's server, started by the task from the checkout's root in the image
# serving/Dockerfile builds. The task passes the contract's arguments: the checkpoint and its
# revision (read offline from the mounted Hugging Face cache), --host 127.0.0.1, --port, and
# --context-length 262144. This script is the candidate's to change.
#
# The starting point is upstream SGLang's standard server with only what loading the
# checkpoint across the node's eight GPUs needs.
set -euo pipefail
exec python3 -m sglang.launch_server "$@" --tp-size 8
