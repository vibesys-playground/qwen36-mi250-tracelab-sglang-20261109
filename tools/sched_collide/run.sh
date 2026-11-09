#!/usr/bin/env bash
# Plan 30 step 1: one TP=2 replica (GCDs 0,1), idle vs loaded prefill timing under
# three scheduler settings. Optional extra env for every server via $EXTRA_ENV_ALL.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cp "$HERE/client.py" /tmp/sc_client.py
cd /
export HOME="$(mktemp -d)"
export HIP_VISIBLE_DEVICES=0,1
P=30188; U=http://127.0.0.1:$P

serve() {  # $1 tag, rest: extra args
  local tag=$1; shift
  echo "######## server $tag @ $(date +%T) args: $*"
  python3 -m sglang.launch_server --model-path Qwen/Qwen3.6-35B-A3B \
    --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 --host 127.0.0.1 --port $P \
    --context-length 262144 --tp-size 2 --attention-backend triton \
    --cuda-graph-backend-prefill breakable --cuda-graph-max-bs-prefill 2816 \
    --enable-cache-report --mamba-track-interval 64 \
    --default-chat-template-kwargs '{"enable_thinking": false}' "$@" > /tmp/srv_$tag.log 2>&1 &
  local pid=$! t0=$(date +%s) ok=0
  for i in $(seq 1 480); do
    if python3 -c "import requests,sys; sys.exit(0 if requests.get('$U/health',timeout=5).status_code==200 else 1)" 2>/dev/null; then ok=1; break; fi
    kill -0 $pid 2>/dev/null || break; sleep 5
  done
  echo "health_ok=$ok after $(( $(date +%s)-t0 )) s"
  if [ $ok = 1 ]; then timeout 900 python3 -u /tmp/sc_client.py "$U" "$tag"; echo "client exit $?"; fi
  kill $pid 2>/dev/null; sleep 8; pkill -9 -f "sglang.launch_server" 2>/dev/null; pkill -9 -f "sglang::" 2>/dev/null; sleep 5
  grep -nE "Traceback|Error" /tmp/srv_$tag.log | grep -v "Ignore import error" | head -20
  echo "----- $tag prefill/decode lines (tail) -----"
  grep -E "Prefill batch|Decode batch" /tmp/srv_$tag.log | tail -n 12
}

for mode in ${MODES:-overlap nooverlap afterprefill}; do
  case $mode in
    overlap) serve overlap ;;
    nooverlap) serve nooverlap --disable-overlap-schedule ;;
    afterprefill) ( export SGLANG_DISABLE_OVERLAP_AFTER_PREFILL=1; serve afterprefill ) ;;
    defer) ( export SGLANG_DEFER_DECODE_LAUNCH=1; serve defer ) ;;
    deferap) ( export SGLANG_DEFER_DECODE_LAUNCH=1 SGLANG_DISABLE_OVERLAP_AFTER_PREFILL=1; serve deferap ) ;;
  esac
done
echo "######## done @ $(date +%T)"
