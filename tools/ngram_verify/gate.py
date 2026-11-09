"""Plan 27 gates, one command job on an MI250 node (stage 2: GPU drafts, adaptive width).

First the GPU drafter unit check (unit_gpu_drafts.py) on one GCD. Then four single TP=2
servers start at once, one per GCD pair, with serve.sh's flags (--dp-size 1):
  nospec   no speculation (the head's decode path)
  k2       prompt-lookup drafts on the GPU, k=2, adaptive verify width (serve.sh)
  k2c      k2 with 30% of the drafts replaced by random ids (debug switch)
  k2fixed  k2 without the adaptive width (always T = 3)
Each gets the same requests (temperature 0, ignore_eos, exact lengths, token-id prompts):
  * 8 round-0 prompts of the TraceLab sessions (synthetic coding-text pool, fixed nonce)
    and 4 generic text prompts;
  * seq1..seq3: one request at a time, /flush_cache before each (bs=1, no radix hit);
  * pair, pair2: two at a time, twice; conc: all 12 at once; kept: one at a time warm;
  * stream: /v1/completions streaming like session_runner (ids per chunk vs usage and
    max_tokens, no chunk that restates the output so far);
  * speed: bs=1 and bs=2 decode of 512 tokens, wall tok/s.
Gate (c): every spec variant's seq1..3 outputs equal nospec seq1 token for token, and at
pair concurrency k2 equals k2c (lossless whatever the drafts), against the noise floor of
nospec pair vs nospec pair2. This tool reads the trace to build prompts; the server's
drafter does not.
"""
import asyncio
import collections
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

MODEL, REV = "Qwen/Qwen3.6-35B-A3B", "995ad96eacd98c81ed38be0c5b274b04031597b0"
HERE = Path(__file__).resolve().parent
TRACE = HERE.parent / "prefill_cap" / "sessions.csv"
OUT = Path(os.environ.get("P27_OUT", "/tmp/p27gate"))
GENERIC = [
    "def merge_sorted(a, b):\n    \"\"\"Merge two sorted lists into one sorted list.\"\"\"\n",
    "The history of the printing press begins",
    "class LRUCache:\n    def __init__(self, capacity):\n",
    "Write a short explanation of how TCP congestion control works.\n\n",
]
BASE_FLAGS = [
    "--model-path", MODEL, "--revision", REV, "--host", "127.0.0.1",
    "--context-length", "262144", "--tp-size", "2", "--attention-backend", "triton",
    "--cuda-graph-backend-prefill", "breakable", "--cuda-graph-max-bs-prefill", "2816",
    "--enable-cache-report", "--mamba-track-interval", "64",
    "--default-chat-template-kwargs", '{"enable_thinking": false}',
    "--max-running-requests", "8", "--cuda-graph-max-bs-decode", "8", "--enable-metrics",
]
SPEC = ["--speculative-algorithm", "NGRAM", "--speculative-ngram-max-bfs-breadth", "1"]
SPEC_ENV = {"SGLANG_NGRAM_PROMPT_LOOKUP": "1", "SGLANG_NGRAM_DECODE_VERIFY": "1"}
K2 = SPEC + ["--speculative-num-draft-tokens", "3"]
NOOVL = ["--disable-overlap-schedule"]
VARIANTS = {
    "nospec": ([], {}),
    "k2": (K2, dict(SPEC_ENV)),
    "k2c": (K2, dict(SPEC_ENV, SGLANG_SPEC_DEBUG_CORRUPT_DRAFT_PCT="30")),
    "k2fixed": (K2, dict(SPEC_ENV, SGLANG_NGRAM_ADAPTIVE_WIDTH="0")),
    "k2_noovl": (K2 + NOOVL, dict(SPEC_ENV)),
    "k2fixed_noovl": (K2 + NOOVL, dict(SPEC_ENV, SGLANG_NGRAM_ADAPTIVE_WIDTH="0")),
    "nospec2": ([], {}),
}
# Phase 1 is the exactness gate; phase 2 adds the overlap-off variants (exactness at
# bs=1 and pair) and a second nospec as the replay timing noise floor.
PHASES = [["nospec", "k2", "k2c", "k2fixed"], ["k2_noovl", "k2fixed_noovl", "nospec2"]]
FULL = {"nospec", "k2", "k2c", "k2fixed"}


def log(*a):
    print(*a, flush=True)


def build_prompts():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    nonce = "5f1e0c9a2b7d4e8f90a1b2c3d4e5f607"
    lines = []
    for i in range(20000):
        lines.append(f"def case_{i}(items):")
        lines.append(f"    return sorted(set(items))  # {i} {nonce}")
    pool = []
    for e in tok.backend_tokenizer.encode_batch(lines, add_special_tokens=False):
        pool.extend(e.ids)
    sess = {}
    for row in csv.DictReader(open(TRACE)):
        sess.setdefault(int(row["session_id"]), []).append(row)
    prompts = []
    for ordinal, sid in enumerate(sorted(sess)[:8]):
        row = min(sess[sid], key=lambda r: int(r["round_idx"]))
        n_in = min(int(row["input_len"]) + int(row["prefix_len"]), 6000)
        cur = (ordinal * 9973) % len(pool)
        ids = [pool[(cur + j) % len(pool)] for j in range(n_in)]
        prompts.append({"name": f"s{sid}r0", "ids": ids, "max": min(int(row["output_len"]), 256)})
    for i, text in enumerate(GENERIC):
        prompts.append({"name": f"g{i}", "ids": tok(text, add_special_tokens=False).input_ids, "max": 160})
    return prompts


async def generate(client, url, p, max_new=None):
    body = {"input_ids": p["ids"], "sampling_params": {
        "max_new_tokens": max_new or p["max"], "temperature": 0, "ignore_eos": True}}
    r = await client.post(url + "/generate", json=body)
    r.raise_for_status()
    j = r.json()
    return list(j["output_ids"]), j.get("meta_info", {})


async def stream(client, url, p):
    body = {"model": MODEL, "prompt": p["ids"], "max_tokens": p["max"], "temperature": 0,
            "stream": True, "return_token_ids": True, "ignore_eos": True,
            "stream_options": {"include_usage": True}}
    chunks, usage, restated, finish = [], None, 0, None
    acc = []
    async with client.stream("POST", url + "/v1/completions", json=body) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                ids = ch.get("token_ids") or []
                if ids:
                    if acc and len(ids) > len(acc) and ids[: len(acc)] == acc:
                        restated += 1
                    acc = acc + list(ids)
                    chunks.append(len(ids))
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    return {"ids": acc, "chunks": chunks, "usage": usage, "restated": restated, "finish": finish}


async def run_client(name, url, prompts, logfile):
    res = {}
    async with httpx.AsyncClient(timeout=1800) as client:
        async def flush():
            await client.post(url + "/flush_cache")

        for rep in ((1, 2, 3) if name in FULL else (1,)):
            outs = []
            for p in prompts:
                await flush()
                outs.append((await generate(client, url, p))[0])
            res[f"seq{rep}"] = outs
            log(f"[{name}] seq{rep} done")
        await flush()
        outs = [None] * len(prompts)
        for i in range(0, len(prompts), 2):
            pair = await asyncio.gather(*(generate(client, url, prompts[j]) for j in range(i, min(i + 2, len(prompts)))))
            for j, (o, _) in enumerate(pair):
                outs[i + j] = o
        res["pair"] = outs
        await flush()
        outs = [None] * len(prompts)
        for i in range(0, len(prompts), 2):
            pair = await asyncio.gather(*(generate(client, url, prompts[j]) for j in range(i, min(i + 2, len(prompts)))))
            for j, (o, _) in enumerate(pair):
                outs[i + j] = o
        res["pair2"] = outs
        await flush()
        if name not in FULL:
            res["stream"] = [await stream(client, url, p) for p in prompts[:4] + prompts[8:10]]
            return res
        res["conc"] = [o for o, _ in await asyncio.gather(*(generate(client, url, p) for p in prompts))]
        res["kept"] = [(await generate(client, url, p))[0] for p in prompts]
        log(f"[{name}] pair/conc/kept done")
        res["stream"] = [await stream(client, url, p) for p in prompts[:4] + prompts[8:10]]
        # speed: 512 new tokens, bs=1 then bs=2
        speed = {}
        for bs in (1, 2):
            await flush()
            t0 = time.perf_counter()
            outs = await asyncio.gather(*(generate(client, url, prompts[j], 512) for j in range(bs)))
            dt = time.perf_counter() - t0
            n = sum(len(o) for o, _ in outs)
            speed[f"bs{bs}"] = round(n / dt, 1)
        res["speed"] = speed
        log(f"[{name}] speed {speed}")
    return res


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(results, prompts, logs):
    base = results.get("nospec")
    lines = []
    lines.append("=== gate (c) exactness: outputs vs nospec seq1 (prompt: first differing token, '-' = equal) ===")
    for name, r in results.items():
        if r is None:
            lines.append(f"{name}: NO RESULT")
            continue
        for mode in ("seq1", "seq2", "seq3", "pair", "pair2", "conc", "kept"):
            if mode not in r:
                continue
            ref = base["seq1"] if base else None
            diffs = []
            for i, p in enumerate(prompts):
                d = first_diff(r[mode][i], ref[i]) if ref else "?"
                if len(r[mode][i]) != p["max"]:
                    d = f"LEN{len(r[mode][i])}"
                diffs.append("-" if d is None else str(d))
            ok = all(x == "-" for x in diffs)
            lines.append(f"{name:7s} {mode:5s} {'EQUAL' if ok else 'DIFF '} {' '.join(diffs)}")
    lines.append("--- repeatability within a variant (seq1/seq2/seq3 identical; pair/conc vs nospec's own pair/conc) ---")
    for name, r in results.items():
        if r is None:
            continue
        if "seq2" not in r:
            continue
        rep = all(r["seq1"][i] == r["seq2"][i] == r["seq3"][i] for i in range(len(prompts)))
        pc = ""
        if base:
            pc = " pair==nospec.pair %d/%d conc==nospec.conc %d/%d" % (
                sum(r["pair"][i] == base["pair"][i] for i in range(len(prompts))), len(prompts),
                sum(r["conc"][i] == base["conc"][i] for i in range(len(prompts))), len(prompts))
        lines.append(f"{name:7s} seq repeatable={rep}{pc}")
    lines.append("--- pair concurrency (2 requests at once) ---")
    n = len(prompts)

    def same(a, b):
        return sum(a[i] == b[i] for i in range(n))

    if base:
        lines.append(f"noise floor: nospec pair vs nospec pair2 {same(base['pair'], base['pair2'])}/{n}")
    for name in ("k2", "k2fixed", "k2_noovl", "k2fixed_noovl"):
        r, c = results.get(name), results.get("k2c")
        if r:
            lines.append(f"{name} pair vs pair2 {same(r['pair'], r['pair2'])}/{n}"
                         + (f"; vs nospec pair {same(r['pair'], base['pair'])}/{n}" if base else ""))
        if r and c:
            lines.append(f"GATE pair: {name} pair vs k2c pair {same(r['pair'], c['pair'])}/{n}, "
                         f"{name} pair2 vs k2c pair2 {same(r['pair2'], c['pair2'])}/{n}")
    lines.append("=== stream integrity (session_runner checks) ===")
    for name, r in results.items():
        if r is None:
            continue
        for k, s in enumerate(r["stream"]):
            p = (prompts[:4] + prompts[8:10])[k]
            u = s["usage"] or {}
            # Counts are strict. The streams run warm (prefix-cache hits), so their ids are
            # compared with nospec's own streams rather than the cold seq1 outputs.
            ok = (len(s["ids"]) == p["max"] == u.get("completion_tokens") and s["restated"] == 0
                  and max(s["chunks"] or [0]) >= 1 and s["finish"] == "length")
            ref = results.get("nospec")
            same = "n/a" if not ref else str(s["ids"] == ref["stream"][k]["ids"])
            lines.append(f"{name:7s} {p['name']:6s} {'OK  ' if ok else 'FAIL'} ids==nospec.stream {same} ids={len(s['ids'])} usage={u.get('completion_tokens')} "
                         f"prompt_tokens={u.get('prompt_tokens')} chunks={len(s['chunks'])} max_chunk={max(s['chunks'] or [0])} restated={s['restated']} finish={s['finish']}")
    lines.append("=== replay (full trace, 2 sessions per server, one pass): TTFT breakdown in ms ===")
    for name, r in results.items():
        if r is not None and r.get("replay"):
            lines.append(r["replay"])
    lines.append("=== speed (wall tok/s, 512 new tokens) ===")
    for name, r in results.items():
        if r is not None and "speed" in r:
            lines.append(f"{name:7s} {r['speed']}{'  (EAGER: graph capture failed)' if r.get('eager') else ''}")
    lines.append("=== server logs ===")
    for name, f in logs.items():
        text = Path(f).read_text(errors="replace")
        accept = [float(m.group(1)) for m in re.finditer(r"accept len: ([0-9.]+)", text)]
        if accept:
            lines.append(f"{name:7s} server log mean accept len {sum(accept) / len(accept):.2f} over {len(accept)} lines")
        for l in text.splitlines():
            if "NGRAM drafts:" in l or "adaptive width" in l or "matched n-gram lengths" in l:
                lines.append(f"{name:7s} LOG: {l[:300]}")
        errs = [l for l in text.splitlines() if re.search(r"Traceback|Error|error:", l) and "Ignore import error" not in l and "server_args=" not in l]
        for l in errs[:15]:
            lines.append(f"{name:7s} LOG: {l[:300]}")
    return "\n".join(lines)


async def wait_ready(url, proc, timeout=3600):
    t0 = time.time()
    async with httpx.AsyncClient(timeout=5) as c:
        while time.time() - t0 < timeout:
            if proc.poll() is not None:
                return False
            try:
                if (await c.get(url + "/health")).status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(5)
    return False


async def one_variant(name, gpus, port, prompts):
    """Run one variant; a spec variant that fails to start is retried once eagerly
    (--disable-cuda-graph) so the exactness data still comes back (gate (d) then has
    no graph-replay number for it)."""
    r = await _one_variant(name, gpus, port, prompts, [])
    if r == "not_ready" and name != "nospec":
        log(f"[{name}] retrying with --disable-cuda-graph (exactness only)")
        r = await _one_variant(name, gpus, port, prompts, ["--disable-cuda-graph"])
        if isinstance(r, dict):
            r["eager"] = True
    return r if isinstance(r, dict) else None


async def _one_variant(name, gpus, port, prompts, extra):
    flags, env_extra = VARIANTS[name]
    flags = flags + extra
    env = dict(os.environ, HIP_VISIBLE_DEVICES=gpus, **env_extra)
    logf = OUT / f"server_{name}.log"
    cmd = [sys.executable, "-m", "sglang.launch_server", *BASE_FLAGS, "--port", str(port), *flags]
    proc = subprocess.Popen(cmd, env=env, stdout=open(logf, "w"), stderr=subprocess.STDOUT, cwd="/", start_new_session=True)
    url = f"http://127.0.0.1:{port}"
    t0 = time.time()
    try:
        ok = await wait_ready(url, proc)
        log(f"[{name}] ready={ok} after {time.time() - t0:.0f}s")
        if not ok:
            log(f"[{name}] server log tail:\n" + "".join(open(logf, errors="replace").readlines()[-60:]))
            return "not_ready"
        res = await run_client(name, url, prompts, logf)
        if name != "k2c" and POOL:
            from ttft_probe import replay, summarize

            async with httpx.AsyncClient() as c:
                await c.post(url + "/flush_cache")
            recs, rate = await replay(name, url, POOL[0], 100)
            res["replay"] = summarize(name, recs, rate)
            log(res["replay"])
        return res
    except Exception as e:  # noqa: BLE001
        log(f"[{name}] client error {type(e).__name__}: {e}")
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


POOL = []


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    unit = subprocess.run([sys.executable, str(HERE / "unit_gpu_drafts.py")], cwd="/",
                          env=dict(os.environ, HIP_VISIBLE_DEVICES="0"),
                          capture_output=True, text=True, timeout=1800)
    log("=== unit: GPU drafts vs host drafts ===\n" + unit.stdout[-4000:] + unit.stderr[-3000:])
    prompts = build_prompts()
    log("prompts:", [(p["name"], len(p["ids"]), p["max"]) for p in prompts])
    from ttft_probe import build_pool

    POOL.append(build_pool("a1b2c3d4e5f60718293a4b5c6d7e8f90"))
    gpus = ["0,1", "2,3", "4,5", "6,7"]
    results = {}
    for names in PHASES:
        rs = await asyncio.gather(*(one_variant(n, gpus[i], 31100 + 10 * i, prompts) for i, n in enumerate(names)))
        results.update(zip(names, rs))
    names = [n for ph in PHASES for n in ph]
    json.dump(results, open(OUT / "results.json", "w"))
    try:
        log(compare(results, prompts, {n: OUT / f"server_{n}.log" for n in names}))
    except Exception as e:  # noqa: BLE001
        # OUT dies with the job, so the raw outputs go to the log for offline comparison.
        log(f"compare failed: {type(e).__name__}: {e}")
        log("RESULTS_JSON " + json.dumps({n: (r and {k: v for k, v in r.items() if k != "replay"}) for n, r in results.items()}))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
