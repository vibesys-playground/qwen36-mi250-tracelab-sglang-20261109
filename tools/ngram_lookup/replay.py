"""Plan 27 stage 0, phase A: replay the TraceLab sessions against the running head server and
capture every round's prompt and greedy output token ids.

The prompts are built the way Request Factory's session_runner builds them (see the task's
tracelab README): a synthetic coding-text corpus with a per-run nonce, tokenized line by line
into a 1M-token pool; session i reads the pool from offset i*9973; round r's prompt is the
previous round's prompt + the model's real output + input_len fresh pool tokens. Requests go
to /generate as token ids (temperature 0, ignore_eos, exact max_new_tokens), 8 sessions at a time.
Tool waits are skipped: they change timing, not tokens.

This file reads the trace (it has to, to replay it). The offline lookup simulation in
analyze.py reads only each request's own prompt and output ids.

usage: replay.py URL OUT.json [--nonce HEX] [--rounds all|first] [--slots 8]
"""
import argparse, csv, json, os, secrets, threading, time

import requests
from transformers import AutoTokenizer

MODEL, REV = "Qwen/Qwen3.6-35B-A3B", "995ad96eacd98c81ed38be0c5b274b04031597b0"
HERE = os.path.dirname(os.path.abspath(__file__))
TRACE = os.path.join(HERE, "..", "prefill_cap", "sessions.csv")  # = recorded_sessions.csv rows
POOL_LIMIT = 1_000_000


def build_pool(nonce):
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV).backend_tokenizer
    lines = []
    for i in range(100_000):
        lines.append(f"def case_{i}(items):")
        lines.append(f"    return sorted(set(items))  # {i} {nonce}")
    pool = []
    for s in range(0, len(lines), 4096):
        for e in tok.encode_batch(lines[s:s + 4096], add_special_tokens=False):
            pool.extend(e.ids)
        if len(pool) >= POOL_LIMIT:
            break
    return pool[:POOL_LIMIT]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--nonce", default=None)
    ap.add_argument("--rounds", default="all", choices=["all", "first"])
    ap.add_argument("--slots", type=int, default=8)
    a = ap.parse_args()
    nonce = a.nonce or secrets.token_hex(16)
    t0 = time.time()
    pool = build_pool(nonce)
    print(f"pool {len(pool)} tokens, nonce {nonce}, built in {time.time() - t0:.1f}s", flush=True)

    sess = {}
    for row in csv.DictReader(open(TRACE)):
        sess.setdefault(int(row["session_id"]), []).append(row)
    order = sorted(sess)
    q = list(enumerate(order))
    lock = threading.Lock()
    recs, errs = [], []

    def take(cur, n):
        out = [pool[(cur + j) % len(pool)] for j in range(n)]
        return (cur + n) % len(pool), out

    def slot():
        while True:
            with lock:
                if not q:
                    return
                ordinal, sid = q.pop(0)
            cur = (ordinal * 9973) % len(pool)
            ctx = []
            rows = sorted(sess[sid], key=lambda r: int(r["round_idx"]))
            if a.rounds == "first":
                rows = rows[:1]
            for row in rows:
                plen = min(int(row["prefix_len"]), len(ctx))
                need = int(row["input_len"]) + (int(row["prefix_len"]) - plen)
                cur, fresh = take(cur, need)
                prompt = ctx[:plen] + fresh
                n_out = int(row["output_len"])
                body = {"input_ids": prompt,
                        "sampling_params": {"max_new_tokens": n_out, "temperature": 0, "ignore_eos": True}}
                try:
                    r = requests.post(a.url + "/generate", json=body, timeout=1800).json()
                    out = list(r["output_ids"])
                    assert len(out) == n_out, (len(out), n_out)
                except Exception as e:  # keep going; the analysis skips missing rounds
                    with lock:
                        errs.append((sid, row["round_idx"], repr(e)[:300]))
                    out = []
                with lock:
                    recs.append({"sid": sid, "round": int(row["round_idx"]), "prompt": prompt, "out": out})
                ctx = prompt + out

    ths = [threading.Thread(target=slot) for _ in range(a.slots)]
    t1 = time.time()
    [t.start() for t in ths]
    [t.join() for t in ths]
    dur = time.time() - t1
    ntok = sum(len(r["out"]) for r in recs)
    print(f"replay rounds={a.rounds}: {len(recs)} requests, {ntok} output tokens in {dur:.1f}s "
          f"({ntok / max(dur, 1e-9):.1f} tok/s, no tool waits), errors {len(errs)}", flush=True)
    for e in errs[:10]:
        print("ERR", e, flush=True)
    recs.sort(key=lambda r: (r["sid"], r["round"]))
    json.dump({"nonce": nonce, "recs": recs}, open(a.out, "w"))


if __name__ == "__main__":
    main()
