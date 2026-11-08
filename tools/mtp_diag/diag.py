"""Diagnose MTP speculative decoding: stream integrity and greedy repeatability.

Run from the checkout's root inside the candidate image. For each server variant it starts
serving/serve.sh (plus the variant's extra flags), then:

* determinism: each of the service checker's four greedy prompts sent N times in a row, once
  with the radix cache kept and once with /flush_cache before every request; prints how many
  distinct outputs each prompt produced and where the first divergence is;
* replay: the recorded TraceLab sessions at concurrency 8, prompts as token ids, streaming with
  return_token_ids / ignore_eos / include_usage like Request Factory's session_runner; each
  round is checked the way session_runner checks it (a chunk that restates the accumulated
  output, streamed ids vs usage.completion_tokens, completion_tokens vs max_tokens), and every
  failing round is printed with its chunk sizes.

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
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

MODEL = "Qwen/Qwen3.6-35B-A3B"
REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
DET_PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a land far away,",
    "def fibonacci(n):\n    # return the n-th Fibonacci number\n",
    "The following is a list of the planets in the Solar System:",
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


async def stream_completion(client, url, body):
    """Return (text, token_id_chunks, usage, finish_reason, error)."""
    text, chunks, usage, finish, error = [], [], None, None, None
    try:
        async with client.stream("POST", url + "/v1/completions", json=body) as resp:
            if resp.status_code != 200:
                return "", [], None, None, f"HTTP {resp.status_code}: {(await resp.aread())[:300]!r}"
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
                for choice in event.get("choices") or []:
                    text.append(choice.get("text") or "")
                    ids = choice.get("token_ids")
                    if ids:
                        chunks.append(list(ids))
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    return "".join(text), chunks, usage, finish, error


async def determinism(url: str, repeats: int) -> None:
    async with httpx.AsyncClient(timeout=120) as client:
        for flush in (False, True):
            log(f"-- determinism, {'flush_cache before each request' if flush else 'radix cache kept'}, n={repeats}")
            for prompt in DET_PROMPTS:
                outs = []
                for _ in range(repeats):
                    if flush:
                        await client.post(url + "/flush_cache")
                    body = {"model": MODEL, "prompt": prompt, "max_tokens": 32,
                            "temperature": 0.0, "stream": True}
                    text, _, _, _, err = await stream_completion(client, url, body)
                    outs.append(err or text)
                counts = collections.Counter(outs)
                verdict = "OK " if len(counts) == 1 else "DIFF"
                log(f"  {verdict} distinct={len(counts)} {prompt[:40]!r}")
                if len(counts) > 1:
                    variants = [o for o, _ in counts.most_common()]
                    base = variants[0]
                    for other in variants[1:]:
                        i = next((k for k, (a, b) in enumerate(zip(base, other)) if a != b),
                                 min(len(base), len(other)))
                        log(f"     x{counts[base]} {base!r}")
                        log(f"     x{counts[other]} {other!r}  (first diff at char {i})")
                    log(f"     sequence: {[variants.index(o) for o in outs]}")


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
    corpus = "".join(f"def case_{i}(items):\n    return sorted(set(items))  # {i} {nonce}\n"
                     for i in range(20000))
    pool = tok.encode(corpus, add_special_tokens=False)
    log(f"replay: {len(rows)} rounds, {len(sessions)} sessions, token pool {len(pool)}")
    offset = [0]
    stats = collections.Counter()
    first_chunk_sizes = collections.Counter()
    failures = []
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
                text, chunks, usage, finish, err = await stream_completion(client, url, body)
                done_rounds[0] += 1
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
    log(f"replay: mean ids per chunk {stats['ids'] / max(stats['chunks'], 1):.2f}; sizes of first two chunks {dict(first_chunk_sizes)}")
    for sid, rnd, problems, sizes, head in failures:
        log(f"  FAILED session {sid} round {rnd}: {problems}; chunk sizes {sizes}; first chunks {head}")


def summarize_server_log(logfile: Path) -> None:
    accept = [float(m.group(1)) for m in re.finditer(r"accept len: ([0-9.]+)", logfile.read_text(errors="replace"))]
    if accept:
        log(f"server: {len(accept)} decode log lines, mean accept len {sum(accept) / len(accept):.2f}")
    for line in logfile.read_text(errors="replace").splitlines():
        if re.search(r"Traceback|Error|error:|AssertionError|deterministic", line) and "Ignore import error" not in line:
            log("  server:", line[:300])


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--replay-rounds", type=int, default=303)
    parser.add_argument("--variants", default="spec,spec-det,nospec")
    args = parser.parse_args()
    root = Path.cwd()
    logdir = Path(tempfile.mkdtemp(prefix="mtp-diag-"))
    serve = (root / "serving/serve.sh").read_text()
    variants = {
        "spec": [],
        "spec-det": ["--enable-deterministic-inference"],
    }
    for name in args.variants.split(","):
        log(f"\n===== variant {name} =====")
        if name == "nospec":
            # The same server without the speculative flags.
            lines = [l for l in serve.splitlines() if "--speculative-" not in l]
            while lines and not lines[-1].strip():
                lines.pop()
            lines[-1] = lines[-1].rstrip().removesuffix("\\").rstrip()
            script = logdir / "serve_nospec.sh"
            script.write_text("\n".join(lines) + "\n")
            extra = []
        else:
            extra, script = variants[name], root / "serving/serve.sh"
        logfile = logdir / f"diag-{name}.log"
        argv = ["bash", str(script), "--model-path", MODEL, "--revision", REVISION, "--host",
                "127.0.0.1", "--port", str(args.port), "--context-length", "262144", *extra]
        log("server argv:", " ".join(argv[1:]))
        proc = subprocess.Popen(argv, stdout=logfile.open("w"), stderr=subprocess.STDOUT,
                                start_new_session=True)
        t0 = time.time()
        url = f"http://127.0.0.1:{args.port}"
        try:
            if not await wait_ready(url, proc, 900):
                log(f"server did not become ready (exit {proc.poll()}); tail:")
                for line in logfile.read_text(errors="replace").splitlines()[-40:]:
                    log("  |", line[:300])
                continue
            log(f"ready after {time.time() - t0:.0f}s")
            await determinism(url, args.repeats)
            if name != "nospec":
                await replay(url, root, args.replay_rounds)
        finally:
            stop_server(proc)
            summarize_server_log(logfile)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
