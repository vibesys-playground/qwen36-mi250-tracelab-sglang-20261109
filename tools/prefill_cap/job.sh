#!/usr/bin/env bash
# Plan 16 A/B: serve.sh with the prefill-graph cap swapped, then tools/prefill_cap/client.py.
# usage: job.sh <cap> [<cap>...]   (run from the checkout root; one full 4x TP=2 server at a time)
CK=$PWD
export HOME=$(mktemp -d)
P=30177; U=http://127.0.0.1:$P
for CAP in "$@"; do
  echo "################ CAP=$CAP $(date +%T)"
  sed "s/--cuda-graph-max-bs-prefill [0-9]*/--cuda-graph-max-bs-prefill $CAP/" $CK/serving/serve.sh > /tmp/serve_$CAP.sh
  grep -n "max-bs-prefill" /tmp/serve_$CAP.sh
  bash /tmp/serve_$CAP.sh --model-path Qwen/Qwen3.6-35B-A3B --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 \
    --host 127.0.0.1 --port $P --context-length 262144 > /tmp/srv_$CAP.log 2>&1 &
  pid=$!; t0=$(date +%s); ok=0
  for i in $(seq 1 720); do
    if python3 -c "import requests,sys; sys.exit(0 if requests.get('$U/health',timeout=5).status_code==200 else 1)" 2>/dev/null; then ok=1; break; fi
    kill -0 $pid 2>/dev/null || break; sleep 5
  done
  echo "CAP=$CAP health_ok=$ok after $(( $(date +%s)-t0 )) s"
  echo "----- capture/memory log CAP=$CAP -----"
  grep -nE "Capture|capture|avail mem|Load weight|KV Cache|max_total_num_tokens|Traceback|Error|out of memory" /tmp/srv_$CAP.log | grep -v "Ignore import error" | head -80
  if [ $ok = 1 ]; then timeout 2400 python3 -u $CK/tools/prefill_cap/client.py "$U" "cap$CAP"; echo "client exit $?"; fi
  pkill -P $pid 2>/dev/null; kill $pid 2>/dev/null; sleep 10; pkill -9 -f "sglang.launch_server" 2>/dev/null; pkill -9 -f "sglang::" 2>/dev/null; sleep 5
  echo "----- prefill batch lines CAP=$CAP -----"
  grep -cE "Prefill batch" /tmp/srv_$CAP.log
done
