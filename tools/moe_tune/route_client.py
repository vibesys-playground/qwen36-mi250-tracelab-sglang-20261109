"""Plan 21: send exact-length generic prompts (python source and prose) so the server's eager prefill
dumps its MoE routing. usage: route_client.py URL"""
import glob, os, sys, time
import requests
from transformers import AutoTokenizer

U = sys.argv[1]
MODEL, REV = "Qwen/Qwen3.6-35B-A3B", "995ad96eacd98c81ed38be0c5b274b04031597b0"
SIZES = [184, 617, 1652, 2587]
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)


def gather(paths, want):
    ids = []
    for p in paths:
        try:
            ids += tok(open(p, errors="ignore").read(), add_special_tokens=False)["input_ids"]
        except Exception:
            continue
        if len(ids) >= want:
            break
    return ids


import torch, sympy  # noqa: F401  (only to locate source trees)
code = sorted(glob.glob(os.path.dirname(torch.__file__) + "/nn/modules/*.py")) + \
       sorted(glob.glob(os.path.dirname(sympy.__file__) + "/core/*.py"))
prose = sorted(glob.glob("/usr/share/doc/*/copyright")) + sorted(glob.glob("/usr/share/common-licenses/*")) + \
        sorted(glob.glob("/opt/rocm*/share/doc/*/*")) + sorted(glob.glob(os.path.dirname(torch.__file__) + "/../*.md")) + \
        sorted(glob.glob("/usr/lib/python3*/**/*.rst", recursive=True))[:200]
print("code files", len(code), "prose files", len(prose), flush=True)
cids, pids = gather(code, 60000), gather(prose, 20000)
print("code tokens", len(cids), "prose tokens", len(pids), flush=True)
if len(pids) < 12000:
    pids = gather(code[::-1], 20000)  # fall back to more code from the other end


def send(ids, tag):
    t = time.time()
    r = requests.post(U + "/generate", json={"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0}}, timeout=1200)
    print(tag, len(ids), r.status_code, f"{time.time()-t:.2f}s", flush=True)


# warm-up (small, below the dump threshold of 128 tokens... sent at 64 tokens)
send(cids[:64], "warm")
LC = len(cids)
for k, M in enumerate(SIZES):  # distinct offsets: no prefix-cache hit between requests
    send(cids[1000 + k * 3000:1000 + k * 3000 + M], f"M{M}-code-a")
    o = LC // 2 + k * 3000
    send(cids[o:o + M], f"M{M}-code-b")
    send(pids[500 + k * 3000:500 + k * 3000 + M], f"M{M}-prose")
