#!/usr/bin/env bash
# Plan 21 step 1: record real prefill routing on one TP=2 replica, then run route_tune.py on one GCD.
CK=$PWD
cd /
export HOME=$(mktemp -d)
P=30177; U=http://127.0.0.1:$P
export SGLANG_MOE_ROUTE_DUMP_DIR=/tmp/route
rm -rf /tmp/route; mkdir -p /tmp/route
export HIP_VISIBLE_DEVICES=0,1
python3 -m sglang.launch_server --model-path Qwen/Qwen3.6-35B-A3B --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --host 127.0.0.1 --port $P --context-length 262144 --tp-size 2 --attention-backend triton \
  --chunked-prefill-size 8192 --disable-cuda-graph \
  --default-chat-template-kwargs '{"enable_thinking": false}' > /tmp/srv.log 2>&1 &
pid=$!; t0=$(date +%s); ok=0
for i in $(seq 1 600); do
  if python3 -c "import requests,sys; sys.exit(0 if requests.get('$U/health',timeout=5).status_code==200 else 1)" 2>/dev/null; then ok=1; break; fi
  kill -0 $pid 2>/dev/null || break; sleep 5
done
echo "health_ok=$ok after $(( $(date +%s)-t0 )) s"
if [ $ok = 1 ]; then timeout 1500 python3 -u $CK/tools/moe_tune/route_client.py "$U"; echo "client exit $?"; fi
pkill -P $pid 2>/dev/null; kill $pid 2>/dev/null; sleep 10; pkill -9 -f "sglang.launch_server" 2>/dev/null; pkill -9 -f "sglang::" 2>/dev/null; sleep 8
echo "dump files: $(ls /tmp/route | wc -l)"; ls /tmp/route | sed -E 's/.*_M//' | sort | uniq -c | sort -k2 -n | head -30
tail -n 15 /tmp/srv.log
export HIP_VISIBLE_DEVICES=0
python3 -u $CK/tools/moe_tune/route_tune.py --dump /tmp/route
