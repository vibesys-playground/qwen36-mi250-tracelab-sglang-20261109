"""Diagnose MTP speculative decoding: stream integrity and greedy repeatability.

Run from the checkout's root inside the candidate image. For each server variant (no MTP; MTP
with the 2-stage or the unified verify attention; unified verify with deliberately corrupted
drafts, a debug-only server switch) it starts serving/serve.sh with the variant's flags and
environment, then:

* determinism: four short generic greedy prompts (of the kind the service checker sends) sent
  N times in a row, once with the radix cache kept and once with /flush_cache before every
  request; prints how many distinct outputs each prompt produced, where the first divergence
  is, and each request's cached prompt tokens and speculative verify count;
* replay: the recorded TraceLab session lengths and topology at concurrency 8, with generic
  source code as payload (not the benchmark's token pool), prompts as token ids, streaming with
  return_token_ids / ignore_eos / include_usage like Request Factory's session_runner; each
  round is checked the way session_runner checks it (a chunk that restates the accumulated
  output, streamed ids vs usage.completion_tokens, completion_tokens vs max_tokens), and every
  failing round is printed with its chunk sizes. Wall time and client-side TTFT/TPOT are
  printed as hints;
* the server log: accept length, decode-batch throughput by running requests, graph capture.

Diagnostic only; it is not the task's benchmark and its timings are not a measurement.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import csv
import json
import os
import re
import secrets
import statistics
import sysconfig
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

MODEL = "Qwen/Qwen3.6-35B-A3B"
REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
# Short, open greedy completions of the same kind the service checker uses (generic text,
# not the checker's own prompts).
DET_PROMPTS = [
    "The largest ocean on Earth is",
    "It was a cold winter morning, and the old lighthouse keeper",
    "def binary_search(items, target):\n    # return the index of target, or -1\n",
    "Here is a list of the seven continents:",
]
CSV_CANDIDATES = [
    ".vibesys/tasks/mi2508x-tracelab-sglang/tracelab/recorded_sessions.csv",
    ".vibesys/shared/tracelab/recorded_sessions.csv",
]


def log(*args):
    print(*args, flush=True)


def stop_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=60)
    except Exception:  # noqa: BLE001
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass
    time.sleep(5)


async def wait_ready(url: str, proc: subprocess.Popen, timeout: float) -> bool:
    deadline = time.time() + timeout
    async with httpx.AsyncClient(timeout=5) as client:
        while time.time() < deadline:
            if proc.poll() is not None:
                return False
            try:
                if (await client.get(url + "/health")).status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(3)
    return False


async def stream_completion(client, url, body, timing=None):
    """Return (text, token_id_chunks, usage, finish_reason, error, sglext)."""
    text, chunks, usage, finish, error, ext = [], [], None, None, None, {}
    t0 = time.perf_counter()
    try:
        async with client.stream("POST", url + "/v1/completions", json=body) as resp:
            if resp.status_code != 200:
                return "", [], None, None, f"HTTP {resp.status_code}: {(await resp.aread())[:300]!r}", ext
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if "error" in event:
                    error = str(event["error"])[:300]
                if event.get("usage"):
                    usage = event["usage"]
                if event.get("sglext"):
                    ext.update({k: v for k, v in event["sglext"].items() if v})
                for choice in event.get("choices") or []:
                    text.append(choice.get("text") or "")
                    ids = choice.get("token_ids")
                    if ids:
                        if timing is not None:
                            timing.append(time.perf_counter() - t0)
                        chunks.append(list(ids))
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    return "".join(text), chunks, usage, finish, error, ext


async def determinism(url: str, repeats: int, max_tokens: int, results: dict) -> int:
    """Return the number of prompts (x cache modes) with more than one distinct output."""
    diffs = 0
    async with httpx.AsyncClient(timeout=120) as client:
        for flush in (False, True):
            mode = "flush" if flush else "kept"
            log(f"-- determinism, {'flush_cache before each request' if flush else 'radix cache kept'}, n={repeats}, max_tokens={max_tokens}")
            for prompt in DET_PROMPTS:
                outs, cached, verify = [], [], []
                for _ in range(repeats):
                    if flush:
                        await client.post(url + "/flush_cache")
                    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                            "temperature": 0.0, "stream": True, "return_token_ids": True,
                            "stream_options": {"include_usage": True},
                            "return_spec_tokens_details": True}
                    text, chunks, usage, _, err, ext = await stream_completion(client, url, body)
                    ids = [t for c in chunks for t in c]
                    outs.append(err or (text + " " + str(ids)))
                    cached.append(((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"))
                    verify.append((ext.get("spec_tokens_details") or {}).get("spec_verify_ct"))
                counts = collections.Counter(outs)
                verdict = "OK " if len(counts) == 1 else "DIFF"
                diffs += len(counts) > 1
                results[(mode, prompt)] = counts.most_common(1)[0][0]
                log(f"  {verdict} distinct={len(counts)} {prompt[:40]!r}")
                log(f"     cached_tokens {cached}")
                log(f"     verify_ct     {verify}")
                if len(counts) > 1:
                    variants = [o for o, _ in counts.most_common()]
                    base = variants[0]
                    for other in variants[1:]:
                        i = next((k for k, (a, b) in enumerate(zip(base, other)) if a != b),
                                 min(len(base), len(other)))
                        log(f"     x{counts[base]} {base[:400]!r}")
                        log(f"     x{counts[other]} {other[:400]!r}  (first diff at char {i})")
                    log(f"     sequence: {[variants.index(o) for o in outs]}")
    return diffs


def restates(accumulated, incoming):  # session_runner's integrity.rs check
    return bool(accumulated) and len(incoming) > len(accumulated) and incoming[:len(accumulated)] == accumulated


async def replay(url: str, root: Path, limit_rounds: int) -> None:
    from transformers import AutoTokenizer

    path = next((root / p for p in CSV_CANDIDATES if (root / p).is_file()), None)
    if path is None:
        log("replay: no recorded_sessions.csv in the checkout; skipped")
        return
    rows = list(csv.DictReader(path.open()))
    sessions = collections.defaultdict(list)
    for row in rows:
        sessions[row["session_id"]].append(row)
    for s in sessions.values():
        s.sort(key=lambda r: int(r["round_idx"]))
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    nonce = secrets.token_hex(16)
    # Payload: generic source code (the image's Python standard library, in sorted file order),
    # not the benchmark's token pool. Only the recorded lengths and topology come from the trace.
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    parts, size = [], 0
    for f in sorted(stdlib.glob("*.py")) + sorted(stdlib.glob("*/*.py")):
        try:
            parts.append(f.read_text(errors="replace"))
        except OSError:
            continue
        size += len(parts[-1])
        if size > 6_000_000:
            break
    pool = tok.encode("".join(parts), add_special_tokens=False)
    log(f"replay: {len(rows)} rounds, {len(sessions)} sessions, token pool {len(pool)}")
    offset = [0]
    stats = collections.Counter()
    first_chunk_sizes = collections.Counter()
    failures = []
    ttft, ttft_cold, tpot = [], [], []
    queue: asyncio.Queue = asyncio.Queue()
    for sid in sessions:
        queue.put_nowait(sid)
    done_rounds = [0]

    def take(n):
        start = offset[0] % (len(pool) - n - 1)
        offset[0] += n
        return pool[start:start + n]

    async def worker(client):
        while not queue.empty() and done_rounds[0] < limit_rounds:
            sid = queue.get_nowait()
            history: list[int] = []
            for row in sessions[sid]:
                prompt = history + take(int(row["input_len"]))
                want = int(row["output_len"])
                body = {"model": MODEL, "rid": f"diag_{sid}_{row['round_idx']}_{nonce[:6]}",
                        "prompt": prompt, "max_tokens": want, "temperature": 0.0, "stream": True,
                        "ignore_eos": True, "return_token_ids": True,
                        "stream_options": {"include_usage": True}}
                timing = []
                text, chunks, usage, finish, err, _ = await stream_completion(client, url, body, timing)
                done_rounds[0] += 1
                if timing:
                    ttft.append(timing[0])
                    if int(row["round_idx"]) == 0:
                        ttft_cold.append(timing[0])
                    if want > 1:
                        tpot.append((timing[-1] - timing[0]) / (want - 1))
                ids, problems = [], []
                for chunk in chunks:
                    if restates(ids, chunk):
                        problems.append(f"chunk restates accumulated output ({len(ids)} ids -> chunk {chunk[:6]})")
                        break
                    ids.extend(chunk)
                ct = (usage or {}).get("completion_tokens")
                pt = (usage or {}).get("prompt_tokens")
                if err:
                    problems.append(f"error {err}")
                if ct != want:
                    problems.append(f"completion_tokens {ct} != max_tokens {want}")
                if ct is not None and len(ids) != ct and not problems:
                    problems.append(f"streamed ids {len(ids)} != completion_tokens {ct}")
                if pt is not None and pt != len(prompt):
                    problems.append(f"prompt_tokens {pt} != sent {len(prompt)}")
                if finish != "length":
                    problems.append(f"finish_reason {finish}")
                for c in chunks[:2]:
                    first_chunk_sizes[len(c)] += 1
                stats["rounds"] += 1
                stats["chunks"] += len(chunks)
                stats["ids"] += len(ids)
                if problems:
                    stats["failed"] += 1
                    failures.append((sid, row["round_idx"], problems, [len(c) for c in chunks[:8]],
                                     [c for c in chunks[:3]]))
                    break  # session_runner stops a session at a failed round
                history = prompt + ids

    t0 = time.time()
    async with httpx.AsyncClient(timeout=600) as client:
        await asyncio.gather(*(worker(client) for _ in range(8)))
    log(f"replay: {dict(stats)} in {time.time() - t0:.1f}s (diagnostic timing, not a measurement)")
    def q(xs, f):
        xs = sorted(xs)
        return 1000 * xs[min(len(xs) - 1, int(f * len(xs)))] if xs else float("nan")
    log(f"replay: TTFT ms p50 {q(ttft, .5):.1f} p90 {q(ttft, .9):.1f} mean {1000 * statistics.fmean(ttft or [0]):.1f}; "
        f"round-0 (cold) TTFT ms p50 {q(ttft_cold, .5):.1f} mean {1000 * statistics.fmean(ttft_cold or [0]):.1f} n={len(ttft_cold)}; "
        f"TPOT ms p50 {q(tpot, .5):.2f} p90 {q(tpot, .9):.2f} (client-side, diagnostic)")
    log(f"replay: mean ids per chunk {stats['ids'] / max(stats['chunks'], 1):.2f}; sizes of first two chunks {dict(first_chunk_sizes)}")
    for sid, rnd, problems, sizes, head in failures:
        log(f"  FAILED session {sid} round {rnd}: {problems}; chunk sizes {sizes}; first chunks {head}")


def summarize_server_log(logfile: Path) -> None:
    text = logfile.read_text(errors="replace")
    accept = [float(m.group(1)) for m in re.finditer(r"accept len: ([0-9.]+)", text)]
    if accept:
        log(f"server: {len(accept)} decode log lines, mean accept len {sum(accept) / len(accept):.2f}")
    gen = collections.defaultdict(list)
    for m in re.finditer(r"#running-req: (\d+).*?gen throughput \(token/s\): ([0-9.]+)", text):
        gen[int(m.group(1))].append(float(m.group(2)))
    for bs in sorted(gen)[:6]:
        xs = sorted(gen[bs])
        log(f"server: Decode batch running-req={bs}: n={len(xs)} gen tok/s median {xs[len(xs) // 2]:.1f}")
    shown = 0
    for line in text.splitlines():
        if re.search(r"(?i)breakable|capture cuda graph (begin|end)|prefill.*graph|graph.*prefill", line) and shown < 12:
            log("  server:", line[:240])
            shown += 1
    for line in text.splitlines():
        if re.search(r"Traceback|Error|error:|AssertionError|deterministic|CORRUPT", line) and "Ignore import error" not in line and "server_args=" not in line:
            log("  server:", line[:300])


VARIANTS = {
    # name: (speculative flags kept, extra env)
    "nospec": (False, {}),
    "spec": (True, {"SGLANG_TRITON_UNIFIED_VERIFY": "0"}),
    "spec-uv": (True, {"SGLANG_TRITON_UNIFIED_VERIFY": "1"}),
    "spec-uv-corrupt": (True, {"SGLANG_TRITON_UNIFIED_VERIFY": "1",
                               "SGLANG_SPEC_DEBUG_CORRUPT_DRAFT": "0.3"}),
}


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--replay-rounds", type=int, default=303)
    parser.add_argument("--variants", default="nospec,spec-uv,spec-uv-corrupt,spec")
    args = parser.parse_args()
    root = Path.cwd()
    logdir = Path(tempfile.mkdtemp(prefix="mtp-diag-"))
    serve = (root / "serving/serve.sh").read_text()
    lines = [l for l in serve.splitlines() if "--speculative-" not in l]
    while lines and not lines[-1].strip():
        lines.pop()
    lines[-1] = lines[-1].rstrip().removesuffix("\\").rstrip()
    nospec_script = logdir / "serve_nospec.sh"
    nospec_script.write_text("\n".join(lines) + "\n")
    outputs, summary = {}, []
    t_job = time.time()
    for name in args.variants.split(","):
        spec, env_extra = VARIANTS[name]
        log(f"\n===== variant {name} (t={time.time() - t_job:.0f}s) env={env_extra} =====")
        script = root / "serving/serve.sh" if spec else nospec_script
        logfile = logdir / f"diag-{name}.log"
        argv = ["bash", str(script), "--model-path", MODEL, "--revision", REVISION, "--host",
                "127.0.0.1", "--port", str(args.port), "--context-length", "262144"]
        env = {k: v for k, v in os.environ.items()
               if k not in ("SGLANG_TRITON_UNIFIED_VERIFY", "SGLANG_SPEC_DEBUG_CORRUPT_DRAFT")}
        env.update(env_extra)
        proc = subprocess.Popen(argv, stdout=logfile.open("w"), stderr=subprocess.STDOUT,
                                start_new_session=True, env=env)
        t0 = time.time()
        url = f"http://127.0.0.1:{args.port}"
        try:
            if not await wait_ready(url, proc, 1200):
                log(f"server did not become ready (exit {proc.poll()}); tail:")
                for line in logfile.read_text(errors="replace").splitlines()[-40:]:
                    log("  |", line[:300])
                summary.append(f"{name}: server not ready")
                continue
            log(f"ready after {time.time() - t0:.0f}s")
            outputs[name] = {}
            diffs = await determinism(url, args.repeats, args.max_tokens, outputs[name])
            summary.append(f"{name}: determinism groups with >1 distinct output: {diffs} of {2 * len(DET_PROMPTS)}")
            await replay(url, root, args.replay_rounds)
        finally:
            stop_server(proc)
            summarize_server_log(logfile)
    log("\n===== summary =====")
    for line in summary:
        log(line)
    names = list(outputs)
    for key in (outputs[names[0]] if names else {}):
        same = [n for n in names if outputs[n].get(key) == outputs[names[0]].get(key)]
        log(f"cross-variant {key[0]} {key[1][:30]!r}: same as {names[0]}: {same}")
    log(f"total {time.time() - t_job:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
