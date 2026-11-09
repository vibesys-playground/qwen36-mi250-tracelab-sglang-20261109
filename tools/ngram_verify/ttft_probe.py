"""Plan 27 stage 2 step 1: where does TTFT go with speculation on vs off? One command job.

Four single TP=2 servers (gate.py's flags plus --enable-metrics), one per GCD pair:
  nospec        the head's decode path
  k2            prompt-lookup drafts, k=2, decode-shaped verify (serve.sh)
  nospec_noovl  nospec with --disable-overlap-schedule
  k2_noovl      k2 with --disable-overlap-schedule
Each replays all TraceLab sessions (tools/prefill_cap/sessions.csv, the benchmark's trace)
at 2 concurrent sessions, which is the per-replica load of the benchmark (8 over 4
replicas), twice with fresh text. Prompt text comes from a synthetic token pool built like
the benchmark's corpus (def case_i lines with a nonce); each round appends the round's new
input tokens to the previous prompt and output, and the tool wait is honoured.
Per request it records the client TTFT (first streamed chunk) and the server's TTFT
breakdown from meta_info: before_queue, queue_time, prefill_latency (forward entry to
prefill result processed) and after_prefill (to the first output). This tool reads the
trace to shape requests; the server's drafter does not.
"""
import asyncio
import csv
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate import BASE_FLAGS, MODEL, REV, SPEC, SPEC_ENV, log, wait_ready  # noqa: E402

HERE = Path(__file__).resolve().parent
TRACE = HERE.parent / "prefill_cap" / "sessions.csv"
OUT = Path(os.environ.get("P27_OUT", "/tmp/p27probe"))
K2 = SPEC + ["--speculative-num-draft-tokens", "3"]
NOOVL = ["--disable-overlap-schedule"]
VARIANTS = {
    "nospec": ([], {}),
    "k2": (K2, SPEC_ENV),
    "nospec_noovl": (NOOVL, {}),
    "k2_noovl": (K2 + NOOVL, SPEC_ENV),
}
CONCURRENCY = 2
KEYS = ("ttft", "first_token_latency", "before_queue", "queue_time", "prefill_latency", "after_prefill")


def build_pool(nonce):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    lines = [f"def case_{i}(items):\n    return sorted(set(items))  # {i} {nonce}\n" for i in range(40000)]
    pool = []
    for e in tok.backend_tokenizer.encode_batch(lines, add_special_tokens=False):
        pool.extend(e.ids)
    return pool


def sessions():
    sess = {}
    for row in csv.DictReader(open(TRACE)):
        sess.setdefault(int(row["session_id"]), []).append(row)
    return {k: sorted(v, key=lambda r: int(r["round_idx"])) for k, v in sess.items()}


async def gen(client, url, ids, max_new):
    body = {"input_ids": ids, "stream": True,
            "sampling_params": {"max_new_tokens": max_new, "temperature": 0, "ignore_eos": True}}
    t0 = time.perf_counter()
    rec = {"ttft": None, "times": [], "meta": {}, "out": []}
    async with client.stream("POST", url + "/generate", json=body) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            s = line[5:].strip()
            if s == "[DONE]":
                break
            c = json.loads(s)
            now = time.perf_counter()
            if rec["ttft"] is None:
                rec["ttft"] = now - t0
            rec["times"].append(now)
            rec["meta"] = c.get("meta_info", rec["meta"])
            if "output_ids" in c:
                rec["out"] = c["output_ids"]
    return rec


async def replay(name, url, pool, seed):
    rng = random.Random(seed)
    sess = sessions()
    queue = sorted(sess)
    recs = []
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=1800) as client:
        async def slot():
            while queue:
                sid = queue.pop(0)
                cur = rng.randrange(len(pool))
                prompt = []
                for row in sess[sid]:
                    n = int(row["input_len"])
                    prompt = prompt + [pool[(cur + j) % len(pool)] for j in range(n)]
                    cur += n
                    r = await gen(client, url, prompt, int(row["output_len"]))
                    m = r["meta"]
                    r.update(sess=sid, round=int(row["round_idx"]),
                             new=len(prompt) - (m.get("cached_tokens") or 0))
                    recs.append(r)
                    prompt = prompt + list(r["out"])
                    await asyncio.sleep(int(row["tool_wait_after_ms"]) / 1000)

        await asyncio.gather(*(slot() for _ in range(CONCURRENCY)))
    dur = time.perf_counter() - t0
    toks = sum(r["meta"].get("completion_tokens", 0) for r in recs)
    log(f"[{name}] replay seed {seed}: {len(recs)} requests, {toks} tokens in {dur:.1f}s = {toks / dur:.1f} tok/s")
    return recs, toks / dur


def pct(v, p):
    v = sorted(x for x in v if x is not None)
    if not v:
        return float("nan")
    k = (len(v) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def summarize(name, recs, rate):
    lines = [f"{name:13s} tok/s {rate:.1f}  n={len(recs)}"]
    for key in KEYS:
        v = [(r["ttft"] if key == "ttft" else r["meta"].get(key)) for r in recs]
        v = [x * 1000 for x in v if x is not None]
        if v:
            lines.append(f"{name:13s}   {key:20s} mean {statistics.mean(v):8.2f}  p50 {pct(v, 50):8.2f}  p90 {pct(v, 90):8.2f} ms (n={len(v)})")
    tp = [(r["times"][-1] - r["times"][0]) / (len(r["times"]) - 1) * 1000 for r in recs if len(r["times"]) > 1]
    # Spec streams several tokens per chunk: per-token time is span / (tokens - 1).
    tpt = [(r["times"][-1] - r["times"][0]) / max(1, r["meta"].get("completion_tokens", 1) - 1) * 1000
           for r in recs if len(r["times"]) > 1]
    lines.append(f"{name:13s}   per-chunk gap p50 {pct(tp, 50):.2f} ms, per-token p50 {pct(tpt, 50):.2f} ms")
    for lo, hi in ((0, 256), (257, 1024), (1025, 4096), (4097, 10 ** 9)):
        b = [r for r in recs if lo <= r["new"] <= hi]
        if b:
            def col(k):
                return [(r["ttft"] if k == "ttft" else r["meta"].get(k) or 0) * 1000 for r in b]
            lines.append(f"{name:13s}   new[{lo},{hi}] n={len(b):3d} ttft p50 {pct(col('ttft'), 50):7.2f}  queue p50 {pct(col('queue_time'), 50):6.2f}  "
                         f"prefill p50 {pct(col('prefill_latency'), 50):7.2f}  after p50 {pct(col('after_prefill'), 50):6.2f}  before p50 {pct(col('before_queue'), 50):6.2f}")
    return "\n".join(lines)


async def one_variant(name, gpus, port, pools):
    flags, env_extra = VARIANTS[name]
    env = dict(os.environ, HIP_VISIBLE_DEVICES=gpus, **env_extra)
    logf = OUT / f"server_{name}.log"
    cmd = [sys.executable, "-m", "sglang.launch_server", *BASE_FLAGS, "--enable-metrics",
           "--port", str(port), *flags]
    proc = subprocess.Popen(cmd, env=env, stdout=open(logf, "w"), stderr=subprocess.STDOUT,
                            cwd="/", start_new_session=True)
    url = f"http://127.0.0.1:{port}"
    t0 = time.time()
    try:
        ok = await wait_ready(url, proc)
        log(f"[{name}] ready={ok} after {time.time() - t0:.0f}s")
        if not ok:
            log(f"[{name}] server log tail:\n" + "".join(open(logf, errors="replace").readlines()[-60:]))
            return None
        async with httpx.AsyncClient(timeout=600) as c:  # warm the prefill graphs
            for n in (300, 900, 2000):
                await gen(c, url, pools[0][:n], 4)
        out = []
        for seed, pool in enumerate(pools):
            await httpx.AsyncClient().post(url + "/flush_cache")
            out.append(await replay(name, url, pool, 100 + seed))
        return out
    except Exception as e:  # noqa: BLE001
        log(f"[{name}] error {type(e).__name__}: {e}")
        log(f"[{name}] server log tail:\n" + "".join(open(logf, errors="replace").readlines()[-60:]))
        return None
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=60)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pools = [build_pool("a1b2c3d4e5f60718293a4b5c6d7e8f90"), build_pool("0f9e8d7c6b5a49382716051f2e3d4c5b")]
    names = list(VARIANTS)
    gpus = ["0,1", "2,3", "4,5", "6,7"]
    rs = await asyncio.gather(*(one_variant(n, gpus[i], 31200 + 10 * i, pools) for i, n in enumerate(names)))
    log("=== TTFT breakdown (ms); per pass, then both passes pooled ===")
    for name, r in zip(names, rs):
        if r is None:
            log(f"{name}: NO RESULT")
            continue
        for i, (recs, rate) in enumerate(r):
            log(summarize(f"{name}#{i}", recs, rate))
        log(summarize(f"{name}*", [x for recs, _ in r for x in recs], statistics.mean(rate for _, rate in r)))
    for name in names:
        text = (OUT / f"server_{name}.log").read_text(errors="replace")
        acc = [float(x.split("accept len: ")[1].split(",")[0]) for x in text.splitlines() if "accept len: " in x]
        if acc:
            log(f"{name}: mean logged accept len {statistics.mean(acc):.2f} over {len(acc)} lines")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
