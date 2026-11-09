#!/bin/bash
# Plan 25: one TP=2 replica (GCD 0,1) with the decode boundary fusion on, then off (j138 flags).
S=$(cd "$(dirname "$0")/../.." && pwd)
cd /
export HOME=$(mktemp -d)
OUT=/tmp/p25prof; mkdir -p $OUT
export HIP_VISIBLE_DEVICES=0,1
P=30177; U=http://127.0.0.1:$P
for MODE in on off; do
  if [ $MODE = on ]; then export SGLANG_HIP_DECODE_BOUNDARY_FUSION=1; else export SGLANG_HIP_DECODE_BOUNDARY_FUSION=0; fi
  echo "######## MODE $MODE @ $(date +%T)"
  python3 -m sglang.launch_server --model-path Qwen/Qwen3.6-35B-A3B --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 \
    --host 127.0.0.1 --port $P --context-length 262144 --tp-size 2 --attention-backend triton \
    --cuda-graph-backend-prefill breakable --cuda-graph-max-bs-prefill 2816 --mamba-track-interval 64 \
    --enable-cache-report --default-chat-template-kwargs '{"enable_thinking": false}' > $OUT/srv_$MODE.log 2>&1 &
  pid=$!; t0=$(date +%s); ok=0
  for i in $(seq 1 480); do
    if python3 -c "import requests,sys; sys.exit(0 if requests.get('$U/health',timeout=5).status_code==200 else 1)" 2>/dev/null; then ok=1; break; fi
    kill -0 $pid 2>/dev/null || break; sleep 5
  done
  echo "PROFILE $MODE health_ok=$ok after $(( $(date +%s)-t0 )) s"
  if [ $ok = 1 ]; then timeout 1200 python3 -u $S/tools/decode_boundary/profile_client.py "$U" $OUT $MODE; echo "client exit $?"; fi
  kill $pid 2>/dev/null; sleep 10; kill -9 $pid 2>/dev/null; pkill -9 -f sglang.launch_server; sleep 5
  echo "===== server log $MODE ====="
  grep -nE "P2P|skinny GEMV|Traceback|Error|error|deferred out_proj" $OUT/srv_$MODE.log | grep -v "Ignore import error" | head -40
  tail -n 15 $OUT/srv_$MODE.log
done
python3 - <<'PY'
import json
a = json.load(open("/tmp/p25prof/greedy_on.json")); b = json.load(open("/tmp/p25prof/greedy_off.json"))
for i, (x, y) in enumerate(zip(a, b)):
    d = next((k for k, (u, v) in enumerate(zip(x, y)) if u != v), None)
    print(f"GREEDY on-vs-off prompt{i}: equal={x == y} first_diff={d}")
PY
