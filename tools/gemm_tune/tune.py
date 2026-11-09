"""Plan 22 step 1: default hipBLASLt pick vs PyTorch TunableOp pick for one TP=2 prefill forward's dense GEMMs.

Phases (each its own process, so the results file is read fresh):
  tune   : TunableOp tuning on, run each (M, shape) eagerly once (tunes + writes CSV), hash outputs x20 repeats.
  eval   : tuning off, read CSV. Per (M, shape): default vs tuned in a HIP graph (distinct weights per call,
           as in the model), median over 50 replays; kernel names eager vs graph replay; output hash x20.
Usage: tune.py [--buckets 256,640,...] [--dir /tmp/x]   (runs both phases as subprocesses and compares hashes)
"""
import argparse, hashlib, json, os, statistics, subprocess, sys

# (name, N, K, calls per forward); TP=2 per-rank shapes of Qwen3.6-35B-A3B (40 layers, every 4th attention).
SHAPES = [
    ("gdn_qkvz", 6144, 2048, 30),
    ("gdn_ba", 32, 2048, 30),
    ("gdn_out", 2048, 2048, 30),
    ("attn_qkv", 4608, 2048, 10),
    ("attn_o", 2048, 2048, 10),
    ("moe_router", 256, 2048, 40),
    ("shared_gate_up", 512, 2048, 40),
    ("shared_down", 2048, 256, 40),
]
DEFAULT_BUCKETS = [256, 640, 1024, 2048, 2816]


def mk(M, N, K, cnt, torch):
    g = torch.Generator(device="cuda").manual_seed(N * 7 + K)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, generator=g)
    ws = [(torch.randn(N, K, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02) for _ in range(cnt)]
    return x, ws


def h(t):
    return hashlib.sha1(t.view(torch.int16).cpu().numpy().tobytes()).hexdigest()[:12]


def kernels(fn, torch):
    from torch.profiler import profile, ProfilerActivity
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn(); torch.cuda.synchronize()
    return sorted({e.name for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA and "Cijk" in e.name or
                   (e.device_type == torch.autograd.DeviceType.CUDA and ("gemm" in e.name.lower() or "Cijk" in e.name))})


def main_phase(phase, buckets, csv):
    import torch
    import torch.nn.functional as F
    import torch.cuda.tunable as tu
    tu.set_filename(csv, insert_device_ordinal=False)
    tu.enable(True)
    tu.tuning_enable(phase == "tune")
    out = {}
    for M in buckets:
        for name, N, K, cnt in SHAPES:
            key = f"{M}:{name}"
            x, ws = mk(M, N, K, cnt, torch)
            r = {}
            if phase == "tune":
                y = F.linear(x, ws[0]); torch.cuda.synchronize()
                hs = {h(F.linear(x, ws[0])) for _ in range(20)}
                r["hashes"] = sorted(hs)
                r["tuned_hash"] = h(y)
            else:
                for mode in ("default", "tuned"):
                    tu.enable(mode == "tuned")
                    ys = [None] * cnt
                    s = torch.cuda.Stream()
                    with torch.cuda.stream(s):
                        for i in range(cnt): ys[i] = F.linear(x, ws[i])
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        for i in range(cnt): ys[i] = F.linear(x, ws[i])
                    for _ in range(5): g.replay()
                    torch.cuda.synchronize()
                    ts = []
                    for _ in range(50):
                        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
                        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
                        ts.append(a.elapsed_time(b) * 1000)
                    r[mode + "_us_per_forward"] = statistics.median(ts)
                    r[mode + "_hash"] = h(ys[0])
                    r[mode + "_hash_repeat"] = sorted({h((g.replay(), ys[0])[1]) for _ in range(20)})
                    try:
                        r[mode + "_kern_graph"] = kernels(g.replay, torch)
                        r[mode + "_kern_eager"] = kernels(lambda: F.linear(x, ws[0]), torch)
                    except Exception as e:
                        r[mode + "_kern_err"] = repr(e)
                tu.enable(True)
            out[key] = r
            del x, ws
            print(json.dumps({key: r}), flush=True)
    if phase == "tune":
        tu.write_file()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", default=",".join(map(str, DEFAULT_BUCKETS)))
    ap.add_argument("--dir", default="/tmp/gemm_tune")
    ap.add_argument("--phase")
    a = ap.parse_args()
    buckets = [int(b) for b in a.buckets.split(",")]
    os.makedirs(a.dir, exist_ok=True)
    csv = os.path.join(a.dir, "tunableop_results0.csv")
    if a.phase:
        res = main_phase(a.phase, buckets, csv)
        json.dump(res, open(os.path.join(a.dir, a.phase + ".json"), "w"))
        return
    env = dict(os.environ, PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE="512",
               PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS="100", PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS="200",
               PYTORCH_TUNABLEOP_VERBOSE="1")
    for ph in ("tune", "eval"):
        print("######## phase", ph, flush=True)
        subprocess.run([sys.executable, __file__, "--phase", ph, "--buckets", a.buckets, "--dir", a.dir], env=env, check=True)
    t = json.load(open(os.path.join(a.dir, "tune.json"))); e = json.load(open(os.path.join(a.dir, "eval.json")))
    print("######## SUMMARY (us per forward = sum over calls of one forward, HIP graph, rotating weights)")
    for M in buckets:
        dsum = tsum = ssum = 0.0
        for name, N, K, cnt in SHAPES:
            k = f"{M}:{name}"; r = e[k]
            d, u = r["default_us_per_forward"], r["tuned_us_per_forward"]
            stable = len(t[k]["hashes"]) == 1 and len(r["tuned_hash_repeat"]) == 1 and r["tuned_hash"] == t[k]["tuned_hash"]
            same_graph_eager = r.get("tuned_kern_graph") == r.get("tuned_kern_eager")
            differs = r.get("tuned_kern_graph") != r.get("default_kern_graph")
            dsum += d; tsum += u; ssum += u if stable else d
            print(f"M={M:5d} {name:15s} N={N:5d} K={K:5d} x{cnt:2d} default {d:8.1f} tuned {u:8.1f} saving {d-u:7.1f} us "
                  f"stable={stable} graph==eager={same_graph_eager} tuned!=default_kernel={differs} bitwise_vs_default={r['tuned_hash']==r['default_hash']}")
            print("      kernels default:", r.get("default_kern_graph"), "tuned:", r.get("tuned_kern_graph"), r.get("tuned_kern_err", ""))
        print(f"== M={M}: default {dsum/1000:.2f} ms, tuned {tsum/1000:.2f} ms (saving {(dsum-tsum)/1000:.2f}), stable-only {ssum/1000:.2f} ms (saving {(dsum-ssum)/1000:.2f})", flush=True)
    print("######## CSV"); print(open(csv).read())


if __name__ == "__main__":
    main()
