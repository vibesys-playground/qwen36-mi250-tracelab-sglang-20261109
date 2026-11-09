#!/usr/bin/env bash
# Plan 27 stage 0 (feasibility, no serving change). Run from the checkout root.
#  A. the head's own server (serving/serve.sh, 4x TP=2): full TraceLab replay with nonce 1, then
#     round 0 of every session again with nonce 2; prompts and greedy output ids saved.
#  B. one eager TP=2 replica with plan 21's routing dump: teacher-forced prompt+output prefills
#     give each output token's real top-8 routing per layer.
#  C. offline: prompt-lookup a(k), verify-window expert overlap, projected S(k), kill gates.
CK=$PWD
cd /
export HOME=$(mktemp -d)
W=/tmp/p27; rm -rf $W; mkdir -p $W/route
P=30177; U=http://127.0.0.1:$P
ARGS="--model-path Qwen/Qwen3.6-35B-A3B --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 --host 127.0.0.1 --port $P --context-length 262144"

wait_health() {  # pid
  local t0=$(date +%s) ok=0
  for i in $(seq 1 720); do
    if python3 -c "import requests,sys; sys.exit(0 if requests.get('$U/health',timeout=5).status_code==200 else 1)" 2>/dev/null; then ok=1; break; fi
    kill -0 $1 2>/dev/null || break; sleep 5
  done
  echo "health_ok=$ok after $(( $(date +%s)-t0 )) s"; [ $ok = 1 ]
}
stop_server() {  # pid
  pkill -P $1 2>/dev/null; kill $1 2>/dev/null; sleep 10
  pkill -9 -f "sglang.launch_server" 2>/dev/null; pkill -9 -f "sglang::" 2>/dev/null; sleep 8
}

echo "######## A: head server + replay @ $(date +%T)"
bash $CK/serving/serve.sh $ARGS > $W/srvA.log 2>&1 &
pid=$!
if wait_health $pid; then
  timeout 1800 python3 -u $CK/tools/ngram_lookup/replay.py $U $W/replay1.json; echo "replay1 exit $?"
  timeout 600 python3 -u $CK/tools/ngram_lookup/replay.py $U $W/replay2.json --rounds first; echo "replay2 exit $?"
else
  tail -n 40 $W/srvA.log
fi
stop_server $pid

echo "######## B: eager TP=2 replica + routing dump @ $(date +%T)"
if [ -s $W/replay1.json ]; then
  SGLANG_MOE_ROUTE_DUMP_DIR=$W/route HIP_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server $ARGS \
    --tp-size 2 --attention-backend triton --chunked-prefill-size 8192 --disable-cuda-graph \
    > $W/srvB.log 2>&1 &
  pid=$!
  if wait_health $pid; then
    timeout 1800 python3 -u $CK/tools/ngram_lookup/teacher.py $U $W/replay1.json $W/route $W/routes.npz --max-req 160
    echo "teacher exit $?"
  else
    tail -n 40 $W/srvB.log
  fi
  stop_server $pid
fi

echo "######## C: analysis @ $(date +%T)"
if [ -s $W/replay1.json ] && [ -s $W/routes.npz ]; then
  R2=""; [ -s $W/replay2.json ] && R2=$W/replay2.json
  timeout 1800 python3 -u $CK/tools/ngram_lookup/analyze.py $W/replay1.json $W/routes.npz $R2
  echo "analyze exit $?"
fi
echo "######## done @ $(date +%T)"
