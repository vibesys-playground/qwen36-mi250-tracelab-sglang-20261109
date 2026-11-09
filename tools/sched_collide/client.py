"""Plan 30 step 1: does a prefill that arrives during a decode wait behind an already queued step?

Against one TP=2 replica (--dp-size 1), measure a 3000-token cached prefix + 561 new tokens:
  idle   : nothing else running
  loaded : one background session decoding (ignore_eos), arrivals spread uniformly over a step
Prints per-request TTFT and the server's meta_info phases, then mean/p50 of each.
Usage: client.py URL TAG
"""
import json
import random
import statistics
import sys
import threading
import time

import requests

U, TAG = sys.argv[1], sys.argv[2]
N = 30
RNG = random.Random(30)


def ids(n):
    return [RNG.randrange(1000, 60000) for _ in range(n)]


def gen(input_ids, max_new, rec=None, stop_evt=None):
    body = {
        "input_ids": input_ids,
        "stream": True,
        "sampling_params": {"max_new_tokens": max_new, "temperature": 0, "ignore_eos": True},
    }
    t0 = time.perf_counter()
    r = {"t0": t0, "ttft": None, "times": [], "meta": {}}
    if rec is not None:
        rec.append(r)
    with requests.post(U + "/generate", json=body, stream=True, timeout=1200) as resp:
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            s = line[6:].strip()
            if s == "[DONE]":
                break
            c = json.loads(s)
            now = time.perf_counter()
            if r["ttft"] is None:
                r["ttft"] = now - t0
            r["times"].append(now)
            r["meta"] = c.get("meta_info", r["meta"])
            if stop_evt is not None and stop_evt.is_set():
                break
    return r


def pct(v, p):
    v = sorted(v)
    if not v:
        return float("nan")
    k = (len(v) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


KEYS = ("ttft", "prefill_latency", "before_queue", "queue_time", "after_prefill")


def val(r, k):
    x = r["ttft"] if k == "ttft" else r["meta"].get(k)
    return None if x is None else 1000 * x


def report(mode, rs):
    print(f"REQ {TAG} {mode} i cached ttft prefill_latency before_queue queue_time after_prefill", flush=True)
    for i, r in enumerate(rs):
        print(f"REQ {TAG} {mode} {i} {r['meta'].get('cached_tokens')} "
              + " ".join(f"{val(r, k):.2f}" if val(r, k) is not None else "nan" for k in KEYS), flush=True)
    for k in KEYS:
        v = [val(r, k) for r in rs if val(r, k) is not None]
        if v:
            print(f"SUMMARY {TAG} {mode} {k}: n={len(v)} mean {statistics.mean(v):.2f} p50 {pct(v, 50):.2f} "
                  f"p10 {pct(v, 10):.2f} p90 {pct(v, 90):.2f}", flush=True)


def main():
    base = ids(3000)
    # Warm the prefix and the 561-token graph bucket.
    gen(base, 1)
    for _ in range(4):
        gen(base + ids(561), 8)
    time.sleep(1)

    idle = []
    for _ in range(N):
        idle.append(gen(base + ids(561), 8))
        time.sleep(0.3)
    report("idle", idle)

    rec, stop = [], threading.Event()
    bg = threading.Thread(target=gen, args=(ids(2000), 4000, rec, stop))
    bg.start()
    while not rec or rec[0]["ttft"] is None:
        time.sleep(0.05)
    time.sleep(1.0)
    loaded, windows = [], []
    for _ in range(N):
        time.sleep(0.5 + RNG.uniform(0, 0.0068))
        r = gen(base + ids(561), 8)
        loaded.append(r)
        windows.append((r["t0"], r["t0"] + (r["ttft"] or 0) + 0.2))
    w1 = time.perf_counter()
    alive = bg.is_alive()
    stop.set()
    bg.join()
    report("loaded", loaded)
    ts = [t for t in rec[0]["times"] if t <= w1]
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    # Background steps not disturbed by a probe prefill.
    clean = [b - a for a, b in zip(ts, ts[1:]) if not any(s <= b and a <= e for s, e in windows)]
    print(f"BG {TAG}: alive_through_window={alive} tokens={len(ts)} inter-token all p50 {1000 * pct(gaps, 50):.2f} "
          f"mean {1000 * statistics.mean(gaps):.2f} | undisturbed p50 {1000 * pct(clean, 50):.2f} "
          f"mean {1000 * statistics.mean(clean):.2f} p90 {1000 * pct(clean, 90):.2f}", flush=True)
    for k in ("ttft", "prefill_latency", "before_queue", "after_prefill"):
        a = [val(r, k) for r in idle if val(r, k) is not None]
        b = [val(r, k) for r in loaded if val(r, k) is not None]
        if a and b:
            print(f"DELTA {TAG} loaded-idle {k}: mean {statistics.mean(b) - statistics.mean(a):+.2f} "
                  f"p50 {pct(b, 50) - pct(a, 50):+.2f} ms", flush=True)


if __name__ == "__main__":
    main()
