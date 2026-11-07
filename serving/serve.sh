#!/usr/bin/env bash
# Throwaway harness probe: map the task's serve arguments onto the baseline engine's.
set -euo pipefail
args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --model-path) args+=(--model "$2"); shift 2 ;;
    --context-length) args+=(--max-model-len "$2"); shift 2 ;;
    *) args+=("$1"); shift ;;
  esac
done
exec python3 -m engine.server "${args[@]}"
