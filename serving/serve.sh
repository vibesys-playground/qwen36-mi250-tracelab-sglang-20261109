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
exec python3 -m sglang.launch_server "$@" --tp-size 8
