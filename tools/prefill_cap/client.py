"""A/B client for the breakable prefill-graph cap (plan 16): idle sweep + loaded replay."""
import csv, json, os, random, statistics, sys, threading, time
import requests

U = sys.argv[1]
TAG = sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
RNG = random.Random(4321)
NSLOTS = int(os.environ.get("NSLOTS", "8"))


def ids(n, rng=RNG):
    return [rng.randrange(1000, 60000) for _ in range(n)]


def gen(input_ids, max_new):
    body = {"input_ids": input_ids, "stream": True,
            "sampling_params": {"max_new_tokens": max_new, "temperature": 0, "ignore_eos": True}}
    t0 = time.perf_counter()
    r = {"t0": t0, "ttft": None, "times": [], "meta": {}, "out": []}
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
            if "output_ids" in c:
                r["out"] = c["output_ids"]
    return r


def pct(v, p):
    v = sorted(v)
    if not v:
        return float("nan")
    k = (len(v) - 1) * p / 100
    f = int(k); c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def ms(x):
    return "nan" if x is None else f"{1000 * x:.1f}"


def mean(v):
    return statistics.mean(v) if v else float("nan")


SESS = {}
for row in csv.DictReader(open(os.path.join(HERE, "sessions.csv"))):
    SESS.setdefault(int(row["session_id"]), []).append(row)

BUCKETS = ((0, 1024), (1025, 1536), (1537, 2048), (2049, 99999))


def replay(tag, seed):
    rng = random.Random(seed)
    q = sorted(SESS); lock = threading.Lock(); allrec = []
    T0 = time.perf_counter()

    def slot(si):
        while True:
            with lock:
                if not q:
                    return
                sid = q.pop(0)
            prompt = []
            for row in sorted(SESS[sid], key=lambda r: int(r["round_idx"])):
                prompt = prompt + ids(int(row["input_len"]), rng)
                r = gen(prompt, int(row["output_len"]))
                r.update(sess=sid, round=int(row["round_idx"]),
                         new=len(prompt) - (r["meta"].get("cached_tokens") or 0))
                with lock:
                    allrec.append(r)
                outi = r["out"] if len(r["out"]) == r["meta"].get("completion_tokens") else ids(int(row["output_len"]), rng)
                prompt = prompt + list(outi)
                time.sleep(int(row["tool_wait_after_ms"]) / 1000)

    ths = [threading.Thread(target=slot, args=(i,)) for i in range(NSLOTS)]
    [t.start() for t in ths]; [t.join() for t in ths]
    dur = time.perf_counter() - T0
    toks = sum(r["meta"].get("completion_tokens", 0) for r in allrec)
    print(f"REPLAY {tag}: sessions {len(SESS)} reqs {len(allrec)} in {dur:.1f}s, {toks/dur:.1f} out tok/s", flush=True)
    print("REQ tag sess round prompt cached new ttft_ms prefill_lat_ms queue_ms tpot_ms")
    for r in sorted(allrec, key=lambda r: r["t0"]):
        m = r["meta"]; ts = r["times"]
        tp = (ts[-1] - ts[0]) / (len(ts) - 1) if len(ts) > 1 else None
        print(f"REQ {tag} {r['sess']} {r['round']} {m.get('prompt_tokens')} {m.get('cached_tokens')} {r['new']} {ms(r['ttft'])} {ms(m.get('prefill_latency'))} {ms(m.get('queue_time'))} {ms(tp)}")
    tt = [r["ttft"] for r in allrec]
    tp = [(r["times"][-1] - r["times"][0]) / (len(r["times"]) - 1) for r in allrec if len(r["times"]) > 1]
    print(f"SUMMARY {tag} ttft_ms p50 {ms(pct(tt,50))} p90 {ms(pct(tt,90))} p99 {ms(pct(tt,99))} mean {ms(mean(tt))} | tpot p50 {ms(pct(tp,50))} p90 {ms(pct(tp,90))} | out_tok_s {toks/dur:.1f}", flush=True)
    for lo, hi in BUCKETS:
        b = [r for r in allrec if lo <= r["new"] <= hi]
        if b:
            pf = [r["meta"].get("prefill_latency") or 0 for r in b]
            tb = [r["ttft"] for r in b]
            print(f"BUCKET {tag} new[{lo},{hi}] n={len(b)} prefill mean {ms(mean(pf))} p50 {ms(pct(pf,50))} p90 {ms(pct(pf,90))} | ttft mean {ms(mean(tb))} p50 {ms(pct(tb,50))} p90 {ms(pct(tb,90))}", flush=True)
    return allrec


def main():
    phase = lambda s: print(f"\n######## {TAG} {s} @ {time.strftime('%H:%M:%S')}", flush=True)
    phase("WARMUP")
    for n in (300, 700, 1100, 1700, 2600):
        base = ids(3000); gen(base, 2); gen(base + ids(n), 2)
    phase("IDLE SWEEP")
    for P in (3000,):
        base = ids(P); gen(base, 1)
        for new in (1024, 1152, 1280, 1536, 1652, 1792, 2048, 2304, 2587):
            rs = [gen(base + ids(new), 1) for _ in range(3)]
            print(f"FLOOR {TAG} prefix {P} new {new}: prefill_ms {[ms(r['meta'].get('prefill_latency')) for r in rs]} ttft_ms {[ms(r['ttft']) for r in rs]} cached {[r['meta'].get('cached_tokens') for r in rs]}", flush=True)
    for new in (1700, 2587):
        rs = [gen(ids(new), 1) for _ in range(3)]
        print(f"FLOOR {TAG} cold new {new}: prefill_ms {[ms(r['meta'].get('prefill_latency')) for r in rs]} ttft_ms {[ms(r['ttft']) for r in rs]}", flush=True)
    phase("REPLAY 1 (all sessions, 8 slots)")
    replay(TAG + "-r1", 11)
    phase("REPLAY 2 (fresh ids)")
    replay(TAG + "-r2", 22)


if __name__ == "__main__":
    main()
