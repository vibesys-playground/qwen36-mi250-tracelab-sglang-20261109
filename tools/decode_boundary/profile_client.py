"""Plan 25 server profile client: bs=2 decode inter-token time, a 30-step GPU trace (analyzed
with analyze_trace.py) and temperature-0 outputs for a cross-mode comparison."""
import glob
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time

import requests

U, OUT, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
HERE = os.path.dirname(os.path.abspath(__file__))
RNG = random.Random(4321)


def ids(n, rng=RNG):
    return [rng.randrange(1000, 60000) for _ in range(n)]


def gen(input_ids, max_new, rec=None, stop=None):
    body = {"input_ids": input_ids, "stream": True,
            "sampling_params": {"max_new_tokens": max_new, "temperature": 0, "ignore_eos": True}}
    r = {"times": [], "out": [], "ttft": None}
    if rec is not None:
        rec.append(r)
    t0 = time.perf_counter()
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
            if "output_ids" in c:
                r["out"] = c["output_ids"]
            if stop is not None and stop.is_set():
                break
    return r


def pct(v, p):
    v = sorted(v)
    k = (len(v) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def main():
    # temperature-0 outputs: same prompts in every mode (fixed seed), each run twice
    prng = random.Random(99)
    prompts = [ids(n, prng) for n in (200, 1500, 3000)]
    outs = []
    for p in prompts:
        a = gen(p, 64)["out"]
        b = gen(p, 64)["out"]
        outs.append(a)
        print(f"GREEDY {TAG} len {len(p)} repeat_equal={a == b} sha={hashlib.sha1(json.dumps(a).encode()).hexdigest()[:12]}", flush=True)
    with open(f"{OUT}/greedy_{TAG}.json", "w") as f:
        json.dump(outs, f)
    # concurrent bs=2 batch: both requests decode together
    rs = []
    th = [threading.Thread(target=lambda p=p: rs.append((p, gen(p, 64)["out"]))) for p in prompts[:2]]
    [t.start() for t in th]
    [t.join() for t in th]
    for p, o in rs:
        i = prompts.index(p)
        print(f"GREEDY {TAG} bs2 prompt{i} equal_to_bs1={o == outs[i]}", flush=True)

    rec, stop = [], threading.Event()
    th = [threading.Thread(target=gen, args=(ids(3000), 2400, rec, stop)) for _ in range(2)]
    [t.start() for t in th]
    while len(rec) < 2 or any(r["ttft"] is None for r in rec):
        time.sleep(0.2)
    time.sleep(3)
    w0 = time.perf_counter()
    time.sleep(6)
    w1 = time.perf_counter()
    gaps = []
    for r in rec:
        ts = [t for t in r["times"] if w0 <= t <= w1]
        gaps += [b - a for a, b in zip(ts, ts[1:])]
    print(f"DECODE {TAG} bs=2 unprofiled 6s: inter-token p50 {1000 * pct(gaps, 50):.3f} "
          f"p90 {1000 * pct(gaps, 90):.3f} mean {1000 * statistics.mean(gaps):.3f} ms (n={len(gaps)})", flush=True)
    d = f"{OUT}/dec2_{TAG}"
    j = {"output_dir": d, "num_steps": 30, "activities": ["GPU"], "with_stack": False,
         "record_shapes": False, "profile_id": f"dec2{TAG}"}
    print("PROF", requests.post(U + "/start_profile", json=j, timeout=600).status_code, flush=True)
    t, last, fs = time.time(), None, []
    while time.time() - t < 300:
        fs = sorted(glob.glob(f"{d}/*TP-*.trace.json.gz"))
        sz = sum(os.path.getsize(f) for f in fs)
        if len(fs) >= 2 and sz == last:
            break
        last = sz
        time.sleep(3)
    stop.set()
    [t.join() for t in th]
    for f in fs:
        if "TP-0" in f:
            print(f"===== ANALYZE {TAG} =====", flush=True)
            subprocess.run([sys.executable, "-I", f"{HERE}/analyze_trace.py", f, f"decode-bs2-{TAG}"], check=False)


if __name__ == "__main__":
    main()
