"""Plan 27 stage 0, phase B: real top-8 routing of each request's output tokens.

Against one eager TP=2 replica started with SGLANG_MOE_ROUTE_DUMP_DIR (plan 21's hook in
fused_moe.py, which saves the (tokens, top_k) expert ids of every eager prefill MoE call of
>= 128 tokens), send prompt + output as one prefill (max_new_tokens=1) after a cache flush.
A token's routing depends only on the tokens before it, so prefill row p carries the same
routing that token p gets as a decode or verify input. Rows of the last len(out) positions
are the output tokens' routing, per layer.

usage: teacher.py URL REPLAY.json DUMP_DIR OUT.npz [--max-req N]
"""
import argparse, glob, json, os, re, time

import numpy as np
import requests

NL = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("replay")
    ap.add_argument("dump")
    ap.add_argument("out")
    ap.add_argument("--max-req", type=int, default=160)
    a = ap.parse_args()
    recs = [r for r in json.load(open(a.replay))["recs"] if len(r["out"]) >= 16]
    # spread over sessions and rounds: round-robin by session
    by_s = {}
    for r in recs:
        by_s.setdefault(r["sid"], []).append(r)
    pick = []
    while len(pick) < a.max_req and any(by_s.values()):
        for s in sorted(by_s):
            if by_s[s] and len(pick) < a.max_req:
                pick.append(by_s[s].pop(0))
    keys, routes, bad = [], [], 0
    t0 = time.time()
    for i, r in enumerate(pick):
        for f in glob.glob(a.dump + "/*.npy"):
            os.remove(f)
        requests.post(a.url + "/flush_cache", timeout=120)
        seq = r["prompt"] + r["out"]
        resp = requests.post(a.url + "/generate", json={
            "input_ids": seq, "sampling_params": {"max_new_tokens": 1, "temperature": 0}}, timeout=1200)
        files = sorted(glob.glob(a.dump + "/*.npy"))
        pids = sorted({re.search(r"/p(\d+)_", f).group(1) for f in files})
        if resp.status_code != 200 or not pids:
            bad += 1
            print("teacher miss", r["sid"], r["round"], resp.status_code, len(files), flush=True)
            continue
        mine = sorted([f for f in files if f"/p{pids[0]}_" in f])
        if len(mine) % NL:
            bad += 1
            print("teacher odd file count", r["sid"], r["round"], len(mine), flush=True)
            continue
        arrs = [np.load(f) for f in mine]
        nch = len(arrs) // NL
        per_layer = [np.concatenate([arrs[c * NL + l] for c in range(nch)], 0) for l in range(NL)]
        T = len(r["out"])
        if per_layer[0].shape[0] < T:
            bad += 1
            print("teacher short", r["sid"], r["round"], per_layer[0].shape, T, flush=True)
            continue
        rt = np.stack([pl[-T:] for pl in per_layer], 0)  # (NL, T, topk)
        keys.append((r["sid"], r["round"]))
        routes.append(rt.astype(np.int16))
        if i < 3 or i % 20 == 0:
            print(f"teacher {i}: sess {r['sid']} round {r['round']} seq {len(seq)} chunks {nch} "
                  f"route {rt.shape} ranks {len(pids)} elapsed {time.time() - t0:.0f}s", flush=True)
    for f in glob.glob(a.dump + "/*.npy"):
        os.remove(f)
    print(f"teacher done: {len(routes)} requests with routing, {bad} misses, {time.time() - t0:.0f}s", flush=True)
    np.savez(a.out, keys=np.array(keys, dtype=np.int32),
             **{f"r{i}": x for i, x in enumerate(routes)})


if __name__ == "__main__":
    main()
