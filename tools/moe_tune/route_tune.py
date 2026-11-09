"""Plan 21: prefill fused-MoE under real routing, on one GCD (TP=2 shard shapes, E=256).

Reads the expert ids recorded by SGLANG_MOE_ROUTE_DUMP_DIR (one .npy of shape (M, 8) per MoE call),
then for each recorded prefill size M:
  - routing stats (touched experts, tokens per expert),
  - the head config's fused_experts time under that routing,
  - a pure-read floor: one kernel that only streams the touched experts' w13 + w2 bytes,
  - a wider config search (BLOCK_M up to 128, GROUP_SIZE_M up to 32) under the same routing.
Times are per layer, whole fused_experts call (align + up GEMM + act + down GEMM + sum), graph replayed;
"ms/prefill" multiplies by 40 layers. Prints ROW/JSON lines so job logs carry the result.
"""
import argparse, glob, itertools, json, os, re, sys, time, traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import triton
import triton.language as tl

import tune  # noqa: E402  (installs the aiter stub, defines bench and the runtime patches)

LAYERS = 40
E, TOPK, H, I_FULL, TP = 256, 8, 2048, 512, 2
N = I_FULL // TP


def log(*a):
    print(*a, flush=True)


def load_groups(d, only=None):
    """-> {M: [list of 40 id arrays, ...]} from the first pid's consecutive same-M runs of 40 files."""
    byp = {}
    for f in sorted(glob.glob(os.path.join(d, "p*_n*_M*.npy"))):
        m = re.match(r"p(\d+)_n(\d+)_M(\d+)\.npy", os.path.basename(f))
        if only is not None and int(m.group(3)) not in only:
            continue
        byp.setdefault(m.group(1), []).append((int(m.group(2)), int(m.group(3)), f))
    out = {}
    if not byp:
        return out
    pid = sorted(byp, key=lambda p: -len(byp[p]))[0]
    seq = sorted(byp[pid])
    i = 0
    while i < len(seq):
        M = seq[i][1]
        j = i
        while j < len(seq) and seq[j][1] == M:
            j += 1
        run = seq[i:j]
        for k in range(0, len(run) - LAYERS + 1, LAYERS):
            out.setdefault(M, []).append([np.load(r[2]).astype(np.int32) for r in run[k:k + LAYERS]])
        i = j
    return out


def stats(layers_ids):
    t, mx, mean = [], [], []
    for ids in layers_ids:
        c = np.bincount(ids.ravel(), minlength=E)
        nz = c[c > 0]
        t.append(len(nz))
        mx.append(int(c.max()))
        mean.append(float(nz.mean()))
    return dict(touched_mean=round(float(np.mean(t)), 1), touched_min=int(min(t)), touched_max=int(max(t)),
                max_tok_mean=round(float(np.mean(mx)), 1), mean_tok_per_touched=round(float(np.mean(mean)), 1),
                max_tok_max=int(max(mx)))


@triton.jit
def _read_kernel(w_ptr, ids_ptr, out_ptr, per_expert, NCH: tl.constexpr, BLOCK: tl.constexpr, CH: tl.constexpr):
    pe = tl.program_id(0)
    pc = tl.program_id(1)
    e = tl.load(ids_ptr + pe).to(tl.int64)
    base = w_ptr + e * per_expert + pc * CH
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, CH, BLOCK):
        acc += tl.load(base + off + tl.arange(0, BLOCK)).to(tl.float32)
    tl.store(out_ptr + pe * NCH + pc, tl.sum(acc))


def read_floor_fn(w1, w2, touched, variant):
    BLOCK, CH = variant
    ids = torch.tensor(touched, dtype=torch.int32, device="cuda")
    n = len(touched)
    pe1, pe2 = w1[0].numel(), w2[0].numel()
    nc1, nc2 = pe1 // CH, pe2 // CH
    o1 = torch.empty(n * nc1, dtype=torch.float32, device="cuda")
    o2 = torch.empty(n * nc2, dtype=torch.float32, device="cuda")

    def run():
        _read_kernel[(n, nc1)](w1, ids, o1, pe1, NCH=nc1, BLOCK=BLOCK, CH=CH)
        _read_kernel[(n, nc2)](w2, ids, o2, pe2, NCH=nc2, BLOCK=BLOCK, CH=CH)
    return run


def smem_ok(c, ns=2):
    return (c["BLOCK_SIZE_M"] * c["BLOCK_SIZE_K"] + c["BLOCK_SIZE_K"] * c["BLOCK_SIZE_N"]) * 2 * ns <= 65536


def coarse(M, out_dim, kdim):
    bms = [16, 32, 64] if M <= 256 else [32, 64, 128]
    res = []
    for bm, bn, bk, nw in itertools.product(bms, (64, 128, 256), (32, 64, 128, 256), (4, 8)):
        if bn > out_dim or bk > kdim:
            continue
        c = dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=1, num_warps=nw,
                 num_stages=2, waves_per_eu=0)
        if smem_ok(c):
            res.append(c)
    return res


def refine(c):
    out = []
    for g, wpe in itertools.product((1, 4, 8, 16, 32), (0, 2)):
        if g == c["GROUP_SIZE_M"] and wpe == c["waves_per_eu"]:
            continue
        out.append(dict(c, GROUP_SIZE_M=g, waves_per_eu=wpe))
    return out


class Ctx:
    """Everything one worker needs for one M: weights, real routing, the patched config lookup."""

    def __init__(self, dump, M, search_layers, final_layers):
        self.fm, self.fc = tune._patch_runtime()
        from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig

        self.M = M
        groups = load_groups(dump, only={M})
        self.grp = groups[M]
        torch.manual_seed(0)
        self.w1 = (torch.randn(E, 2 * N, H, device="cuda") * 0.02).to(torch.bfloat16)
        self.w2 = (torch.randn(E, H, N, device="cuda") * 0.02).to(torch.bfloat16)
        self.rcfg = MoeRunnerConfig(inplace=False)
        self.x = torch.randn(M, H, device="cuda").to(torch.bfloat16)
        self.wts = torch.softmax(torch.randn(M, TOPK, device="cuda"), dim=1).float()
        real_try = self.fc.try_get_optimal_moe_config
        self.cur, self.head_cfg = {}, {}

        def patched(w1s, w2s, top_k, dtype, M_, **kw):
            if "up" in self.cur:
                return dict(self.cur["up"]), (dict(self.cur["down"]), None)
            r = real_try(w1s, w2s, top_k, dtype, M_, **kw)
            self.head_cfg["up"], self.head_cfg["down"] = dict(r[0]), dict(r[1][0])
            return r
        self.fm.try_get_optimal_moe_config = patched
        self.S4, self.S8 = self.layer_sets(search_layers), self.layer_sets(final_layers)

    def layer_sets(self, n):
        ids_all = self.grp[0]
        step = LAYERS // n
        return [torch.from_numpy(ids_all[i]).to("cuda", torch.int32) for i in range(step // 2, LAYERS, step)][:n]

    def run_fn(self, sets):
        def run():
            o = None
            for ids in sets:
                o = self.fm.fused_experts(self.x, self.w1, self.w2, (self.wts, ids, None), self.rcfg)
            return o
        return run

    def timed(self, up, down, sets, iters=2):
        if up is None:
            self.cur.clear()
        else:
            self.cur["up"], self.cur["down"] = up, down
        try:
            return tune.bench(self.run_fn(sets), iters=iters, reps=3) / len(sets)
        finally:
            self.cur.clear()

    def head(self):
        self.timed(None, None, self.S4[:1])
        return self.head_cfg["up"], self.head_cfg["down"]


def jl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())


def worker(a):
    ctx = Ctx(a.dump, a.M, a.search_layers, a.final_layers)
    out = a.out
    hu, hd = ctx.head()
    if a.mode == "measure":
        # head time, read floor, then (if given) the best config back to back
        th = min(ctx.timed(None, None, ctx.S8, iters=4) for _ in range(2))
        floors = []
        for ids in ctx.S8:
            touched = sorted(set(ids.cpu().numpy().ravel().tolist()))
            best = float("inf")
            for variant in ((1024, 8192), (2048, 16384), (4096, 32768)):
                try:
                    best = min(best, tune.bench(read_floor_fn(ctx.w1, ctx.w2, touched, variant), iters=3, reps=3))
                except Exception as e:
                    log("floor variant failed", variant, type(e).__name__, str(e)[:120])
            floors.append((best, len(touched) * (ctx.w1[0].numel() + ctx.w2[0].numel()) * 2))
        res = dict(head_us=th, floor_us=float(np.mean([f[0] for f in floors])),
                   gb=float(np.mean([f[1] for f in floors])) / 1e9, head_up=hu, head_down=hd)
        if a.best:
            bu, bd = json.loads(a.best)
            tb = min(ctx.timed(bu, bd, ctx.S8, iters=4) for _ in range(2))
            th2 = ctx.timed(None, None, ctx.S8, iters=4)
            ctx.cur["up"], ctx.cur["down"] = bu, bd
            ob = ctx.run_fn(ctx.S8[:1])().float().clone()
            ctx.cur.clear()
            oh = ctx.run_fn(ctx.S8[:1])().float().clone()
            res.update(best_us=tb, head_us=min(th, th2),
                       relerr=((ob - oh).abs().max() / (oh.abs().max() + 1e-6)).item())
        jl(out, dict(kind="measure", **res))
        return
    cands = json.load(open(a.cands))
    other = json.loads(a.other) if a.other else (hd if a.stage == "up" else hu)
    done = {r["i"] for r in map(json.loads, open(out))} if os.path.exists(out) else set()
    for i, c in enumerate(cands):
        if i in done:
            continue
        jl(out, dict(i=i, start=True))
        up, down = (c, other) if a.stage == "up" else (other, c)
        try:
            t = ctx.timed(up, down, ctx.S4)
        except Exception as e:  # compile / config errors that leave the context usable
            torch.cuda.synchronize()
            jl(out, dict(i=i, err=type(e).__name__ + ": " + (str(e).splitlines() or [""])[0][:160]))
            continue
        jl(out, dict(i=i, t=t))


def run_worker(args_list, out):
    """Run worker subprocesses until every candidate has a result; a crash marks the candidate that was running."""
    import subprocess
    for attempt in range(40):
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker"] + args_list)
        rows = [json.loads(l) for l in open(out)] if os.path.exists(out) else []
        if p.returncode == 0:
            return rows
        started = [r["i"] for r in rows if r.get("start")]
        finished = {r["i"] for r in rows if "t" in r or "err" in r}
        bad = [i for i in started if i not in finished]
        log(f"WORKER crashed rc={p.returncode}; candidate(s) {bad} marked CRASH")
        for i in bad:
            jl(out, dict(i=i, err="CRASH(illegal access or abort)"))
        if not bad:
            return rows
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--ms", default="184,617,1652,2587")
    ap.add_argument("--search-layers", type=int, default=4)
    ap.add_argument("--final-layers", type=int, default=8)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--mode", default="cands")
    ap.add_argument("--M", type=int, default=0)
    ap.add_argument("--stage", default="up")
    ap.add_argument("--cands", default="")
    ap.add_argument("--other", default="")
    ap.add_argument("--best", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--work", default="/tmp/rtwork")
    a = ap.parse_args()
    if a.worker:
        worker(a)
        return

    groups = load_groups(a.dump)
    log("DUMP groups per M:", {M: len(v) for M, v in sorted(groups.items())})
    os.makedirs(a.work, exist_ok=True)
    summary = []
    for M in [int(x) for x in a.ms.split(",")]:
        try:
            if M not in groups:
                log(f"M={M}: no dump; have {sorted(groups)}")
                continue
            for gi, g in enumerate(groups[M]):
                log(f"STATS M={M} group{gi}", json.dumps(stats(g)))
            base = ["--dump", a.dump, "--M", str(M), "--search-layers", str(a.search_layers),
                    "--final-layers", str(a.final_layers)]
            mo = f"{a.work}/m{M}_head.jsonl"
            m0 = run_worker(base + ["--mode", "measure", "--out", mo], mo)
            m0 = [r for r in m0 if r.get("kind") == "measure"][-1]
            hu, hd = m0["head_up"], m0["head_down"]
            log(f"M={M} head up={json.dumps(hu)} down={json.dumps(hd)} head8={m0['head_us']:.1f}us/layer")
            log(f"M={M} read floor {m0['floor_us']:.1f}us/layer ({m0['gb']:.3f} GB/layer, "
                f"{m0['gb'] * 1e9 / (m0['floor_us'] * 1e-6) / 1e12:.2f} TB/s)")
            t0 = time.time()
            best = {"up": hu, "down": hd}
            for stage, (out_dim, kdim) in (("up", (2 * N, H)), ("down", (H, N))):
                def sweep(cands, tag):
                    cf, of = f"{a.work}/m{M}_{stage}_{tag}.json", f"{a.work}/m{M}_{stage}_{tag}.jsonl"
                    json.dump(cands, open(cf, "w"))
                    other = best["down"] if stage == "up" else best["up"]
                    rows = run_worker(base + ["--mode", "cands", "--stage", stage, "--cands", cf,
                                              "--other", json.dumps(other), "--out", of], of)
                    res = {r["i"]: r for r in rows if "t" in r or "err" in r}
                    errs = [(cands[i], r["err"]) for i, r in res.items() if "err" in r]
                    for c, e in errs[:6]:
                        log(f"M={M} {stage} CANDIDATE-ERROR {e} {json.dumps(c)}")
                    return sorted(((r["t"], cands[i]) for i, r in res.items() if "t" in r), key=lambda r: r[0])
                cl = coarse(M, out_dim, kdim)
                res = sweep(cl, "coarse")
                res2 = list(res[:3])
                rc = []
                for _, c in res[:2]:
                    rc += refine(c)
                if rc:
                    res2 += sweep(rc, "refine")
                res2.sort(key=lambda r: r[0])
                log(f"M={M} {stage} ({len(cl)} coarse, {len(rc)} refine) top5:",
                    json.dumps([(round(t, 1), c) for t, c in res2[:5]]))
                if res2:
                    best[stage] = res2[0][1]
            search_s = time.time() - t0
            mf = f"{a.work}/m{M}_final.jsonl"
            f = run_worker(base + ["--mode", "measure", "--best", json.dumps([best["up"], best["down"]]),
                                   "--out", mf], mf)
            f = [r for r in f if r.get("kind") == "measure"][-1]
            th, tb, fl = f["head_us"], f["best_us"], f["floor_us"]
            row = dict(M=M, head_us=round(th, 1), best_us=round(tb, 1), floor_us=round(fl, 1),
                       head_ms_prefill=round(th * LAYERS / 1000, 2), best_ms_prefill=round(tb * LAYERS / 1000, 2),
                       floor_ms_prefill=round(fl * LAYERS / 1000, 2),
                       head_minus_floor_ms=round((th - fl) * LAYERS / 1000, 2),
                       saving_ms=round((th - tb) * LAYERS / 1000, 2), relerr=round(f["relerr"], 5),
                       up=best["up"], down=best["down"], head_up=hu, head_down=hd, search_s=round(search_s))
            summary.append(row)
            log("ROW", json.dumps(row))
        except Exception:
            traceback.print_exc()
            sys.stdout.flush()
    log("SUMMARY")
    for r in summary:
        log(f"M={r['M']:5d} head {r['head_ms_prefill']:7.2f} ms  best {r['best_ms_prefill']:7.2f} ms  "
            f"floor {r['floor_ms_prefill']:7.2f} ms  head-floor {r['head_minus_floor_ms']:6.2f}  saving {r['saving_ms']:6.2f}  relerr {r['relerr']}")
    log("===JSON route_best===")
    log(json.dumps({str(r["M"]): {"up": r["up"], "down": r["down"]} for r in summary}, indent=2))
    log("===END===")


if __name__ == "__main__":
    main()
