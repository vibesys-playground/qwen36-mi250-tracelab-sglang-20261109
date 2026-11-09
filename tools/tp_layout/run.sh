#!/usr/bin/env bash
# Plan 26 feasibility (diagnostic only, not on the serving path; serving/serve.sh is untouched).
#  1. Fenced HIP custom all-reduce on the TP=4 and TP=8 groups (tools/topo_ar, TOPO_AR_MODE=ca):
#     latency p50 in graph mode at 8 KB-2.3 MB and 500-iteration correctness vs an fp32 sum.
#  2. The head server re-laid as tp8/dp1, tp4/dp2 and tp2/dp4 (serve.sh's flags otherwise):
#     - the full-node layout: a cold replay of all 48 recorded sessions with 8 slots;
#     - one replica of each TP degree (HIP devices 0..t-1): the j138 profile harness
#       (TTFT floor sweep, decode inter-token vs batch, decode kernel breakdown at bs = t,
#       prefill CPU step vs GPU busy).
# Layout switching is by flags here only. LAYOUTS / SKIP_AR select parts for a rerun.
set -u
S=$(cd "$(dirname "$0")/../.." && pwd)
D=$S/tools/tp_layout
cd /
export HOME=$(mktemp -d)
echo "GPUS: $(python3 -c 'import torch;print(torch.cuda.device_count(), torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).gcnArchName)')"
echo "start $(date +%T)"

if [ "${SKIP_AR:-0}" != 1 ]; then
  echo "######## CUSTOM AR (fenced HIP kernel of the checkout) @ $(date +%T)"
  TOPO_AR_MODE=ca TOPO_AR_NO_VARIANT=1 TOPO_AR_CA_TIMEOUT=1500 \
    TOPO_AR_CA_SETS=pair01,quad0123,2quads_concurrent,all8 \
    timeout 1600 python3 "$S/tools/topo_ar/topo_ar.py" > /tmp/ar.log 2>&1
  echo "topo_ar exit $?"
  grep -E "^(devices|CA class|CAINIT|RESULT|fenced)|Error|Traceback" /tmp/ar.log | head -80
  tail -n 5 /tmp/ar.log
fi

MODEL=(--model-path Qwen/Qwen3.6-35B-A3B --revision 995ad96eacd98c81ed38be0c5b274b04031597b0
       --host 127.0.0.1 --context-length 262144)
# serve.sh's flags at 5215874, minus the layout pair (and --enable-metrics as j138 had).
FLAGS=(--attention-backend triton --cuda-graph-backend-prefill breakable --cuda-graph-max-bs-prefill 2816
       --enable-cache-report --mamba-track-interval 64
       --default-chat-template-kwargs '{"enable_thinking": false}' --enable-metrics)
PORT=30177
SPID=""

start_server() {  # name hip tp dp
  local name=$1 hip=$2 tp=$3 dp=$4
  PORT=$((PORT + 1)); U=http://127.0.0.1:$PORT
  local lay=(--tp-size "$tp")
  [ "$dp" -gt 1 ] && lay+=(--dp-size "$dp" --load-balance-method prefix_affinity)
  echo "######## SERVER $name: HIP $hip tp $tp dp $dp port $PORT @ $(date +%T)"
  HIP_VISIBLE_DEVICES=$hip python3 -m sglang.launch_server "${MODEL[@]}" --port "$PORT" \
    "${lay[@]}" "${FLAGS[@]}" > "/tmp/srv_$name.log" 2>&1 &
  SPID=$!
  local t0=$(date +%s) ok=0
  for i in $(seq 1 240); do
    if python3 -c "import requests,sys; sys.exit(0 if requests.get('$U/health',timeout=5).status_code==200 else 1)" 2>/dev/null; then ok=1; break; fi
    kill -0 "$SPID" 2>/dev/null || break
    sleep 5
  done
  echo "SERVER $name health_ok=$ok after $(( $(date +%s) - t0 )) s"
  [ $ok = 1 ]
}

stop_server() {  # name
  local name=$1
  # The server's whole process tree (tokenizer, DP controller, schedulers), then any leftovers.
  python3 - "$SPID" <<'EOF_KILL'
import sys, time, psutil
try:
    p = psutil.Process(int(sys.argv[1])); ps = p.children(recursive=True) + [p]
except psutil.NoSuchProcess:
    ps = []
for x in ps:
    try: x.terminate()
    except psutil.NoSuchProcess: pass
_, alive = psutil.wait_procs(ps, timeout=15)
for x in alive:
    try: x.kill()
    except psutil.NoSuchProcess: pass
for x in psutil.process_iter(["cmdline", "name"]):
    c = " ".join(x.info["cmdline"] or []) + " " + (x.info["name"] or "")
    if ("sglang.launch_server" in c or "sglang::" in c) and x.pid != psutil.Process().pid:
        try: x.kill()
        except psutil.NoSuchProcess: pass
EOF_KILL
  sleep 8
  echo "===== server log $name ====="
  grep -nE "skinny|custom.all.reduce|CustomAllreduce|AR\]|disabled|max_total_num_tokens|Capture cuda graph end|Load weight end|Traceback|Error|error" \
    "/tmp/srv_$name.log" | grep -v "Ignore import error" | head -40
  echo "----- decode batch lines $name (sample) -----"
  grep -E "Decode batch" "/tmp/srv_$name.log" | awk 'NR%40==1' | head -30
  echo "----- tail $name -----"; tail -n 15 "/tmp/srv_$name.log"
}

run_layout() {  # tp dp hip_all hip_one
  local tp=$1 dp=$2 hall=$3 hone=$4
  if [ "$dp" = 1 ]; then
    if start_server "tp${tp}" "$hall" "$tp" 1; then
      PROF_OUT=/tmp/prof/tp$tp timeout 2400 python3 -u "$D/client.py" "$U" both "$tp"; echo "client exit $?"
    fi
    stop_server "tp${tp}"
    return
  fi
  if start_server "tp${tp}dp${dp}" "$hall" "$tp" "$dp"; then
    timeout 1200 python3 -u "$D/client.py" "$U" replay "$tp"; echo "client exit $?"
  fi
  stop_server "tp${tp}dp${dp}"
  if start_server "tp${tp}one" "$hone" "$tp" 1; then
    PROF_OUT=/tmp/prof/tp$tp timeout 1500 python3 -u "$D/client.py" "$U" profile "$tp"; echo "client exit $?"
  fi
  stop_server "tp${tp}one"
}

for L in ${LAYOUTS:-tp8 tp4 tp2}; do
  case $L in
    tp8) run_layout 8 1 0,1,2,3,4,5,6,7 0,1,2,3,4,5,6,7 ;;
    tp4) run_layout 4 2 0,1,2,3,4,5,6,7 0,1,2,3 ;;
    tp2) run_layout 2 4 0,1,2,3,4,5,6,7 0,1 ;;
  esac
done
echo "end $(date +%T)"
