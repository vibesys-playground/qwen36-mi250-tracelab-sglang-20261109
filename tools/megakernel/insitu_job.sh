#!/bin/bash
# Plan 29 step 1 (one MI250 job): (a) MoE block bench on GCD 0 with the sibling GCD 1 idle or
# loaded, NLAYERS 8 and 40; (b) one TP=2 replica on GCD 0,1 (plan 25's j168 recipe) per
# megakernel mode "variant@WG/CU": bs=2 unprofiled inter-token p50 and a 30-step trace.
S=$(cd "$(dirname "$0")/../.." && pwd)
cd /
export HOME=$(mktemp -d)
OUT=/tmp/p29; mkdir -p $OUT
export P29_TMP=$OUT
echo "######## BENCH @ $(date +%T)"
rocm-smi --showtopo 2>/dev/null | grep -A12 -i "weight" | head -14
rocm-smi --showclocks --showpower 2>/dev/null | grep -E "GPU\[[01]\]" | head
HIP_VISIBLE_DEVICES=0 NLAYERS_LIST=8,40 timeout 2400 python3 -u $S/tools/megakernel/moe_block_bench.py --mi250
echo "bench exit $?"
pkill -9 -f moe_block_bench.py; sleep 3
export HIP_VISIBLE_DEVICES=0,1
P=30179; U=http://127.0.0.1:$P
for MODE in 21@2 21@1 0@1; do
  export SGLANG_HIP_DECODE_MK_VARIANT=${MODE%@*} SGLANG_HIP_DECODE_MK_WG_PER_CU=${MODE#*@}
  TAG=v${MODE%@*}w${MODE#*@}
  echo "######## MODE $MODE ($TAG) @ $(date +%T)"
  python3 -m sglang.launch_server --model-path Qwen/Qwen3.6-35B-A3B --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 \
    --host 127.0.0.1 --port $P --context-length 262144 --tp-size 2 --attention-backend triton \
    --cuda-graph-backend-prefill breakable --cuda-graph-max-bs-prefill 2816 --mamba-track-interval 64 \
    --enable-cache-report --default-chat-template-kwargs '{"enable_thinking": false}' > $OUT/srv_$TAG.log 2>&1 &
  pid=$!; t0=$(date +%s); ok=0
  for i in $(seq 1 480); do
    if python3 -c "import requests,sys; sys.exit(0 if requests.get('$U/health',timeout=5).status_code==200 else 1)" 2>/dev/null; then ok=1; break; fi
    kill -0 $pid 2>/dev/null || break; sleep 5
  done
  echo "PROFILE $MODE health_ok=$ok after $(( $(date +%s)-t0 )) s"
  if [ $ok = 1 ]; then
    ( sleep 40; rocm-smi --showclocks --showpower 2>/dev/null | grep -E "GPU\[[01]\]" | sed "s/^/SMI $TAG /" ) &
    timeout 1200 python3 -u $S/tools/megakernel/insitu_client.py "$U" $OUT $TAG; echo "client exit $?"
  fi
  kill $pid 2>/dev/null; sleep 10; kill -9 $pid 2>/dev/null; pkill -9 -f sglang.launch_server; sleep 5
  echo "===== server log $MODE ====="
  grep -nE "megakernel|Traceback|Error|error|timed out" $OUT/srv_$TAG.log | grep -v "Ignore import error" | head -20
  tail -n 8 $OUT/srv_$TAG.log
done
python3 - <<'PY'
import json, os
tags = ["v21w2", "v21w1", "v0w1"]
g = {t: json.load(open(f"/tmp/p29/greedy_{t}.json")) for t in tags if os.path.exists(f"/tmp/p29/greedy_{t}.json")}
for t in g:
    if t != "v21w2" and "v21w2" in g:
        print(f"GREEDY {t} vs v21w2: equal per prompt {[a == b for a, b in zip(g[t], g['v21w2'])]}")
PY
