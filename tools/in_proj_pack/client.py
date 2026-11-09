"""Plan 32 server probe: greedy outputs (fixed prompts, sequential, twice) and a bs=2 decode
GPU profile. Usage: client.py <url> <tag> <outdir>."""
import glob, json, os, random, statistics, subprocess, sys, threading, time

import requests

U, TAG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)


def ids(n, rng):
    return [rng.randrange(1000, 60000) for _ in range(n)]


def gen(input_ids, max_new, rec=None, stop_evt=None):
    body = {"input_ids": input_ids, "stream": True,
            "sampling_params": {"max_new_tokens": max_new, "temperature": 0, "ignore_eos": True}}
    r = {"times": [], "out": [], "meta": {}}
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
            r["times"].append(time.perf_counter())
            r["meta"] = c.get("meta_info", r["meta"])
            if "output_ids" in c:
                r["out"] = c["output_ids"]
            if stop_evt is not None and stop_evt.is_set():
                break
    return r


def pct(v, p):
    v = sorted(v); k = (len(v) - 1) * p / 100; f = int(k); c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def main():
    rng = random.Random(1234)
    prompts = [ids(n, rng) for n in (7, 50, 600, 3000, 1500)]
    outs = {}
    for rep in range(2):
        for i, p in enumerate(prompts):
            r = gen(p, 200)
            outs[f"{rep}:{i}"] = r["out"]
    rep_ok = all(outs[f"0:{i}"] == outs[f"1:{i}"] for i in range(len(prompts)))
    print(f"GREEDY {TAG}: repeat-equal {rep_ok}; lens {[len(outs[f'0:{i}']) for i in range(len(prompts))]}", flush=True)
    json.dump(outs, open(f"{OUT}/greedy_{TAG}.json", "w"))

    rng = random.Random(99)
    rec = []; stop = threading.Event()
    ths = [threading.Thread(target=gen, args=(ids(3000, rng), 2400, rec, stop)) for _ in range(2)]
    [t.start() for t in ths]
    while len(rec) < 2 or any(not r["times"] for r in rec):
        time.sleep(0.2)
    time.sleep(3)
    w0 = time.perf_counter(); time.sleep(4); w1 = time.perf_counter()
    gaps = []
    for r in rec:
        ts = [t for t in r["times"] if w0 <= t <= w1]
        gaps += [b - a for a, b in zip(ts, ts[1:])]
    print(f"DECODE {TAG} bs=2 unprofiled 4s: inter-token p50 {1000*pct(gaps,50):.3f} p90 {1000*pct(gaps,90):.3f} mean {1000*statistics.mean(gaps):.3f} ms n {len(gaps)}", flush=True)
    d = f"{OUT}/dec2_{TAG}"
    x = requests.post(U + "/start_profile", json={"output_dir": d, "num_steps": 30, "activities": ["GPU"],
                                                  "with_stack": False, "record_shapes": False, "profile_id": TAG}, timeout=600)
    print("start_profile", x.status_code, x.text.strip()[:80], flush=True)
    t = time.time(); last = None; fs = []
    while time.time() - t < 300:
        fs = sorted(glob.glob(f"{d}/*TP-*.trace.json.gz")); sz = sum(os.path.getsize(f) for f in fs)
        if len(fs) >= 2 and sz == last:
            break
        last = sz; time.sleep(3)
    stop.set(); [th.join() for th in ths]
    for f in fs:
        if "TP-0" in f:
            print(f"===== ANALYZE decode-bs2 {TAG} =====", flush=True)
            subprocess.run([sys.executable, "-I", f"{HERE}/analyze.py", f, f"decode-bs2-{TAG}"], check=False)


if __name__ == "__main__":
    main()
