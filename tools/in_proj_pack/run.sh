#!/usr/bin/env bash
# Plan 32 stage 1 hardware proof: bitwise microbench, then one TP=2 replica (GCDs 0,1) with the
# packed in_proj off (= head behaviour) and on: greedy outputs equal, decode-graph kernel count.
S=$PWD; cd /; export HOME=$(mktemp -d); O=/tmp/inproj; mkdir -p $O
export HIP_VISIBLE_DEVICES=0,1
echo "===== BITWISE ====="
timeout 900 python3 $S/tools/in_proj_pack/bitwise.py; bw=$?
echo "bitwise exit $bw"
P=30179; U=http://127.0.0.1:$P
for packed in 0 1; do
  tag=packed$packed
  echo "===== SERVER $tag ====="
  SGLANG_QWEN35_PACKED_IN_PROJ=$packed python3 -m sglang.launch_server --model-path Qwen/Qwen3.6-35B-A3B --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 --host 127.0.0.1 --port $P --context-length 262144 --tp-size 2 --attention-backend triton --cuda-graph-backend-prefill breakable --cuda-graph-max-bs-prefill 2816 --mamba-track-interval 64 --enable-cache-report --default-chat-template-kwargs '{"enable_thinking": false}' > $O/srv_$tag.log 2>&1 &
  pid=$!; ok=0; t0=$(date +%s)
  for i in $(seq 1 480); do if python3 -c "import requests,sys; sys.exit(0 if requests.get('$U/health',timeout=5).status_code==200 else 1)" 2>/dev/null; then ok=1; break; fi; kill -0 $pid 2>/dev/null || break; sleep 5; done
  echo "health_ok=$ok after $(( $(date +%s)-t0 )) s"
  if [ $ok = 1 ]; then timeout 1500 python3 -u $S/tools/in_proj_pack/client.py "$U" $tag $O; echo "client exit $?"; fi
  kill $pid 2>/dev/null; sleep 8; kill -9 $pid 2>/dev/null; sleep 5
  grep -nE "Packed BF16|skinny GEMM|Traceback|Error|GDN decode" $O/srv_$tag.log | grep -v "Ignore import error" | head -30
  if [ $ok != 1 ]; then tail -n 40 $O/srv_$tag.log; fi
done
echo "===== GREEDY COMPARE ====="
python3 - <<'PY'
import json
a = json.load(open("/tmp/inproj/greedy_packed0.json")); b = json.load(open("/tmp/inproj/greedy_packed1.json"))
same = {k: a[k] == b.get(k) for k in a}
print("GREEDY packed0 vs packed1 equal per prompt:", same)
print("GREEDY_ALL_EQUAL", all(same.values()))
PY
exit $bw
