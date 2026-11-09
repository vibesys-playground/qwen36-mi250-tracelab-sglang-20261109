import csv, json, os, random, sys, threading, time, glob, statistics, subprocess
import requests

U = sys.argv[1]
MODE = sys.argv[2]            # replay | profile | both
NRANK = int(sys.argv[3])      # TP degree: trace files per profile, and the decode batch
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("PROF_OUT", "/tmp/prof")
RNG = random.Random(4321)
TRITON_CACHE = os.environ.get("TRITON_CACHE_DIR") or os.path.join(os.environ.get("HOME", "/tmp"), ".triton", "cache")


def ids(n, rng=RNG):
    return [rng.randrange(1000, 60000) for _ in range(n)]


def gen(input_ids, max_new, rec=None, stop_evt=None):
    body = {"input_ids": input_ids, "stream": True,
            "sampling_params": {"max_new_tokens": max_new, "temperature": 0, "ignore_eos": True}}
    t0 = time.perf_counter()
    r = {"t0": t0, "w0": time.time(), "ttft": None, "times": [], "meta": {}, "out": []}
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
                r["wf"] = time.time()
            r["times"].append(now)
            r["meta"] = c.get("meta_info", r["meta"])
            if "output_ids" in c:
                r["out"] = c["output_ids"]
            if stop_evt is not None and stop_evt.is_set():
                break
    r["t1"] = time.perf_counter()
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


def triton_entries():
    out = {}
    for root in (TRITON_CACHE,):
        for d in glob.glob(root + "/*"):
            try:
                out[d] = os.path.getmtime(d)
            except OSError:
                pass
    return out


def wait_trace(d, pid, n=2, timeout=300):
    t = time.time(); last = None
    while time.time() - t < timeout:
        fs = sorted(glob.glob(f"{d}/*{pid}*TP-*.trace.json.gz"))
        sz = sum(os.path.getsize(f) for f in fs)
        if len(fs) >= n and sz == last:
            return fs
        last = sz
        time.sleep(3)
    return sorted(glob.glob(f"{d}/*{pid}*TP-*.trace.json.gz"))


def start_profile(d, pid, steps, acts):
    j = {"output_dir": d, "num_steps": steps, "activities": acts, "with_stack": False,
         "record_shapes": False, "profile_id": pid}
    t = time.time()
    x = requests.post(U + "/start_profile", json=j, timeout=600)
    print(f"PROF start {pid} {acts} steps={steps} -> {x.status_code} {x.text.strip()[:80]} in {time.time()-t:.2f}s", flush=True)


ROWS = list(csv.DictReader(open(os.path.join(HERE, "sessions.csv"))))
SESS = {}
for row in ROWS:
    SESS.setdefault(int(row["session_id"]), []).append(row)


def replay(tag, sids, nslots=2):
    """Replay the recorded sessions sids with nslots concurrent slots (one replica's share of the
    benchmark's 8 slots over 4 replicas), fresh token ids, outputs fed back as the next prefix."""
    rng = random.Random(hash(tag) & 0xffff)
    q = list(sids); lock = threading.Lock(); allrec = []
    before = triton_entries()
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
                r.update(sess=sid, round=int(row["round_idx"]), slot=si,
                         new=len(prompt) - (r["meta"].get("cached_tokens") or 0))
                with lock:
                    allrec.append(r)
                outi = r["out"] if len(r["out"]) == r["meta"].get("completion_tokens") else ids(int(row["output_len"]), rng)
                prompt = prompt + list(outi)
                time.sleep(int(row["tool_wait_after_ms"]) / 1000)

    ths = [threading.Thread(target=slot, args=(i,)) for i in range(nslots)]
    [t.start() for t in ths]; [t.join() for t in ths]
    dur = time.perf_counter() - T0
    after = triton_entries()
    newc = sorted(m for d, m in after.items() if d not in before)
    toks = sum(r["meta"].get("completion_tokens", 0) for r in allrec)
    print(f"REPLAY {tag}: sessions {len(sids)} reqs {len(allrec)} in {dur:.1f}s, {toks/dur:.1f} out tok/s; new triton cache entries during replay {len(newc)}", flush=True)
    print(f"REQ {tag} sess round slot prompt cached new ttft_ms queue_time_ms prefill_lat_ms before_q_ms other_q_ms after_pf_ms tpot_ms jit_entries_in_ttft")
    for r in sorted(allrec, key=lambda r: r["t0"]):
        m = r["meta"]; ts = r["times"]
        tp = (ts[-1] - ts[0]) / (len(ts) - 1) if len(ts) > 1 else None
        jit = sum(1 for t in newc if r["w0"] - 0.05 <= t <= r.get("wf", r["w0"]) + 0.05)
        r["jit"] = jit
        print(f"REQ {tag} {r['sess']} {r['round']} {r['slot']} {m.get('prompt_tokens')} {m.get('cached_tokens')} {r['new']} {ms(r['ttft'])} {ms(m.get('queue_time'))} {ms(m.get('prefill_latency'))} {ms(m.get('before_queue'))} {ms(m.get('other_queue'))} {ms(m.get('after_prefill'))} {ms(tp)} {jit}")

    def summ(name, vals):
        vals = [v for v in vals if v is not None]
        print(f"SUMMARY {tag} {name}: n={len(vals)} p50 {ms(pct(vals,50))} p90 {ms(pct(vals,90))} mean {ms(statistics.mean(vals)) if vals else 'nan'}", flush=True)
    summ("ttft", [r["ttft"] for r in allrec])
    for k in ("queue_time", "prefill_latency", "before_queue", "other_queue", "after_prefill"):
        summ(k, [r["meta"].get(k) for r in allrec])
    nojit = [r for r in allrec if r["jit"] == 0]
    summ("ttft(no JIT in window)", [r["ttft"] for r in nojit])
    summ("ttft(JIT in window)", [r["ttft"] for r in allrec if r["jit"] > 0])
    summ("new_tokens(/1000)", [r["new"] / 1000 for r in allrec])
    summ("tpot", [(r["times"][-1] - r["times"][0]) / (len(r["times"]) - 1) for r in allrec if len(r["times"]) > 1])
    ig = []
    for r in allrec:
        ig += [b - a for a, b in zip(r["times"], r["times"][1:])]
    summ("inter-token", ig)
    # client TTFT minus server prefill_latency minus queue: residual (tokenize, dispatch, detokenize, stream)
    res = [r["ttft"] - (r["meta"].get("prefill_latency") or 0) - (r["meta"].get("queue_time") or 0) for r in allrec]
    summ("ttft - prefill_latency - queue_time", res)
    # TTFT buckets by new tokens
    for lo, hi in ((0, 256), (256, 600), (600, 1100), (1100, 1700), (1700, 9999)):
        b = [r for r in allrec if lo <= r["new"] < hi]
        if b:
            print(f"BUCKET {tag} new[{lo},{hi}) n={len(b)} ttft p50 {ms(pct([r['ttft'] for r in b],50))} prefill p50 {ms(pct([r['meta'].get('prefill_latency') or 0 for r in b],50))} queue p50 {ms(pct([r['meta'].get('queue_time') or 0 for r in b],50))} jit-reqs {sum(1 for r in b if r['jit'])}", flush=True)
    return allrec


def pfrun(tag, prefix_len, new, acts=("CPU", "GPU")):
    base = ids(prefix_len) if prefix_len else []
    if base:
        gen(base, 1); time.sleep(0.5)
    r0 = gen(base + ids(new), 1)
    time.sleep(0.5)
    start_profile(OUT + "/" + tag, tag, 3, list(acts))
    time.sleep(0.5)
    r = gen(base + ids(new), 8)
    fs = wait_trace(OUT + "/" + tag, tag, n=NRANK, timeout=40)
    if len(fs) < NRANK:
        try:
            requests.post(U + "/stop_profile", timeout=300)
        except Exception as ex:
            print("stop_profile error", ex)
        fs = wait_trace(OUT + "/" + tag, tag, n=NRANK, timeout=150)
    print(f"PF {tag}: prefix {prefix_len} new {new}: unprofiled ttft {ms(r0['ttft'])} prefill_latency {ms(r0['meta'].get('prefill_latency'))} | profiled ttft {ms(r['ttft'])} prefill_latency {ms(r['meta'].get('prefill_latency'))} cached {r['meta'].get('cached_tokens')}", flush=True)
    for f in fs:
        if "TP-0" in f:
            subprocess.run([sys.executable, "-I", os.path.join(HERE, "pfan.py"), f, tag], check=False)


def decode_gaps(bs, secs=4.0, prefix=3000, keep=False):
    """bs concurrent long decodes; inter-token gaps over a window once all are decoding."""
    rec = []; stop = threading.Event()
    ths = [threading.Thread(target=gen, args=(ids(prefix), 2400, rec, stop)) for _ in range(bs)]
    [t.start() for t in ths]
    while len(rec) < bs or any(r["ttft"] is None for r in rec):
        time.sleep(0.2)
    time.sleep(2)
    w0 = time.perf_counter(); time.sleep(secs); w1 = time.perf_counter()
    gaps = []
    for r in rec:
        ts = [t for t in r["times"] if w0 <= t <= w1]
        gaps += [b - a for a, b in zip(ts, ts[1:])]
    print(f"DECODE bs={bs} unprofiled {secs:.0f}s: inter-token p50 {ms(pct(gaps,50))} p90 {ms(pct(gaps,90))} "
          f"mean {ms(statistics.mean(gaps))} ms -> {bs/statistics.mean(gaps):.1f} tok/s per replica", flush=True)
    if keep:
        return ths, stop
    stop.set(); [t.join() for t in ths]
    time.sleep(1)


def main():
    phase = lambda s: print(f"\n######## {s} @ {time.strftime('%H:%M:%S')}", flush=True)
    print(f"mode {MODE} nrank {NRANK} out {OUT}; triton cache dir {TRITON_CACHE}: {len(triton_entries())} entries at ready", flush=True)
    if MODE in ("replay", "both"):
        # The whole recorded sample, 8 slots, cold server (the benchmark runs no warmup).
        phase("REPLAY FULL (cold server, all 48 sessions, 8 slots)")
        replay("FULL", sorted(SESS), nslots=8)
    if MODE not in ("profile", "both"):
        return

    phase("TTFT floor sweep (idle server)")
    for P in (3000,):
        base = ids(P); gen(base, 1)
        for new in (16, 128, 256, 561, 1024, 1652, 2587):
            rs = [gen(base + ids(new), 1) for _ in range(3)]
            print(f"FLOOR prefix {P} new {new}: ttft {[ms(r['ttft']) for r in rs]} prefill_latency {[ms(r['meta'].get('prefill_latency')) for r in rs]} cached {[r['meta'].get('cached_tokens') for r in rs]}", flush=True)
    for new in (900, 2048):
        rs = [gen(ids(new), 1) for _ in range(3)]
        print(f"FLOOR cold new {new}: ttft {[ms(r['ttft']) for r in rs]} prefill_latency {[ms(r['meta'].get('prefill_latency')) for r in rs]}", flush=True)

    phase("DECODE inter-token vs batch (unprofiled)")
    for bs in (1, 2, 4, 8):
        decode_gaps(bs)

    phase(f"DECODE PROFILE bs={NRANK}")
    ths, stop = decode_gaps(NRANK, keep=True)
    start_profile(OUT + "/dec", "dec", 30, ["GPU"])
    time.sleep(3)
    fs_dec = wait_trace(OUT + "/dec", "dec", n=NRANK)
    start_profile(OUT + "/deccpu", "deccpu", 20, ["CPU", "GPU"])
    time.sleep(3)
    fs_cpu = wait_trace(OUT + "/deccpu", "deccpu", n=NRANK)
    print(f"DECODE still running: {sum(1 for t in ths if t.is_alive())}", flush=True)
    stop.set(); [t.join() for t in ths]
    time.sleep(2)

    phase("PREFILL PROFILES")
    pfrun("pf128", 3000, 128)
    pfrun("pf561", 3000, 561)
    pfrun("pf1652", 3000, 1652)
    pfrun("pfcold2048", 0, 2048)

    phase("DECODE ANALYSIS")
    for fs, lab, extra in ((fs_dec, f"decode-bs{NRANK}", ()), (fs_cpu, f"decode-bs{NRANK}-CPU+GPU", ("cpu",))):
        print(f"FILES {lab}: {[os.path.basename(f)+':'+str(os.path.getsize(f)) for f in fs]}")
        pick = [f for f in fs if "TP-0" in f][:1]
        for f in pick:
            print(f"===== ANALYZE {lab} =====", flush=True)
            subprocess.run([sys.executable, "-I", os.path.join(HERE, "analyze.py"), f, lab, *extra], check=False)


if __name__ == "__main__":
    main()
