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


def load_groups(d):
    """-> {M: [list of 40 id arrays, ...]} from the first pid's consecutive same-M runs of 40 files."""
    byp = {}
    for f in sorted(glob.glob(os.path.join(d, "p*_n*_M*.npy"))):
        m = re.match(r"p(\d+)_n(\d+)_M(\d+)\.npy", os.path.basename(f))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--ms", default="184,617,1652,2587")
    ap.add_argument("--search-layers", type=int, default=4)
    ap.add_argument("--final-layers", type=int, default=8)
    a = ap.parse_args()

    fm, fc = tune._patch_runtime()
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig

    groups = load_groups(a.dump)
    log("DUMP groups per M:", {M: len(v) for M, v in sorted(groups.items())})
    torch.manual_seed(0)
    w1 = (torch.randn(E, 2 * N, H, device="cuda") * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(E, H, N, device="cuda") * 0.02).to(torch.bfloat16)
    rcfg = MoeRunnerConfig(inplace=False)

    real_try = fc.try_get_optimal_moe_config
    cur, head_cfg = {}, {}

    def patched(w1s, w2s, top_k, dtype, M, **kw):
        if "up" in cur:
            return dict(cur["up"]), (dict(cur["down"]), None)
        r = real_try(w1s, w2s, top_k, dtype, M, **kw)
        head_cfg["up"], head_cfg["down"] = dict(r[0]), dict(r[1][0])
        return r
    fm.try_get_optimal_moe_config = patched

    errs = {}
    summary = []
    for M in [int(x) for x in a.ms.split(",")]:
        try:
            if M not in groups:
                log(f"M={M}: no dump; have {sorted(groups)}")
                continue
            grp = groups[M]
            for gi, g in enumerate(grp):
                log(f"STATS M={M} group{gi}", json.dumps(stats(g)))
            ids_all = grp[0]  # first request (generic code text); the others are reported as stats
            x = torch.randn(M, H, device="cuda").to(torch.bfloat16)
            wts = torch.softmax(torch.randn(M, TOPK, device="cuda"), dim=1).float()

            def layer_sets(n):
                step = LAYERS // n
                return [torch.from_numpy(ids_all[i]).to("cuda", torch.int32) for i in range(step // 2, LAYERS, step)][:n]

            def make_run(sets):
                def run():
                    o = None
                    for ids in sets:
                        o = fm.fused_experts(x, w1, w2, (wts, ids, None), rcfg)
                    return o
                return run

            def timed(up, down, sets, iters=2):
                if up is None:
                    cur.clear()
                else:
                    cur["up"], cur["down"] = up, down
                try:
                    return tune.bench(make_run(sets), iters=iters, reps=3) / len(sets)
                except Exception as e:
                    torch.cuda.synchronize()
                    msg = type(e).__name__ + ": " + (str(e).splitlines() or [""])[0][:160]
                    errs[msg] = errs.get(msg, 0) + 1
                    if errs[msg] == 1 and len(errs) <= 10:
                        log("CANDIDATE-ERROR", msg, up if up is not None else "", down if down is not None else "")
                    return float("inf")
                finally:
                    cur.clear()

            S4, S8 = layer_sets(a.search_layers), layer_sets(a.final_layers)
            t_head4 = timed(None, None, S4)
            hu, hd = head_cfg["up"], head_cfg["down"]
            log(f"M={M} head up={json.dumps(hu)} down={json.dumps(hd)} head4={t_head4:.1f}us/layer")

            # read floor over the touched experts of the final layers
            floors = []
            for ids in S8:
                touched = sorted(set(ids.cpu().numpy().ravel().tolist()))
                best = float("inf")
                for variant in ((1024, 8192), (2048, 16384), (4096, 32768)):
                    try:
                        best = min(best, tune.bench(read_floor_fn(w1, w2, touched, variant), iters=3, reps=3))
                    except Exception as e:
                        log("floor variant failed", variant, type(e).__name__, str(e)[:120])
                bytes_ = len(touched) * (w1[0].numel() + w2[0].numel()) * 2
                floors.append((best, bytes_))
            floor_us = float(np.mean([f[0] for f in floors]))
            gb = float(np.mean([f[1] for f in floors])) / 1e9
            log(f"M={M} read floor {floor_us:.1f}us/layer ({gb:.3f} GB/layer, {gb * 1e9 / (floor_us * 1e-6) / 1e12:.2f} TB/s)")

            # search: up with down at head, then down with best up
            t0 = time.time()
            up_best, down_best = hu, hd
            for stage, (out_dim, kdim) in (("up", (2 * N, H)), ("down", (H, N))):
                def t_of(c):
                    return timed(c, down_best, S4) if stage == "up" else timed(up_best, c, S4)
                res = sorted(((t_of(c), c) for c in coarse(M, out_dim, kdim)), key=lambda r: r[0])
                res = [r for r in res if r[0] < float("inf")]
                res2 = list(res[:3])
                for _, c in res[:2]:
                    res2 += [(t_of(d), d) for d in refine(c)]
                res2 = sorted([r for r in res2 if r[0] < float("inf")], key=lambda r: r[0])
                log(f"M={M} {stage} top5:", [(round(t, 1), {k: v for k, v in c.items()}) for t, c in res2[:5]])
                if res2:
                    if stage == "up":
                        up_best = res2[0][1]
                    else:
                        down_best = res2[0][1]
            search_s = time.time() - t0

            # final back-to-back on 8 layers: head, best, plus the up-only / down-only swaps
            th = timed(None, None, S8, iters=4)
            tb = timed(up_best, down_best, S8, iters=4)
            th2 = timed(None, None, S8, iters=4)
            th = min(th, th2)
            tb2 = timed(up_best, down_best, S8, iters=4)
            tb = min(tb, tb2)
            cur["up"], cur["down"] = up_best, down_best
            ob = make_run(S8[:1])().float().clone()
            cur.clear()
            oh = make_run(S8[:1])().float().clone()
            relerr = ((ob - oh).abs().max() / (oh.abs().max() + 1e-6)).item()
            row = dict(M=M, head_us=round(th, 1), best_us=round(tb, 1), floor_us=round(floor_us, 1),
                       head_ms_prefill=round(th * LAYERS / 1000, 2), best_ms_prefill=round(tb * LAYERS / 1000, 2),
                       floor_ms_prefill=round(floor_us * LAYERS / 1000, 2),
                       head_minus_floor_ms=round((th - floor_us) * LAYERS / 1000, 2),
                       saving_ms=round((th - tb) * LAYERS / 1000, 2), relerr=round(relerr, 5),
                       up=up_best, down=down_best, head_up=hu, head_down=hd, search_s=round(search_s))
            summary.append(row)
            log("ROW", json.dumps(row))
        except Exception:
            traceback.print_exc()
            sys.stdout.flush()
    log("ERROR-COUNTS", json.dumps(errs))
    log("SUMMARY")
    for r in summary:
        log(f"M={r['M']:5d} head {r['head_ms_prefill']:7.2f} ms  best {r['best_ms_prefill']:7.2f} ms  "
            f"floor {r['floor_ms_prefill']:7.2f} ms  head-floor {r['head_minus_floor_ms']:6.2f}  saving {r['saving_ms']:6.2f}  relerr {r['relerr']}")
    log("===JSON route_best===")
    log(json.dumps({str(r["M"]): {"up": r["up"], "down": r["down"]} for r in summary}, indent=2))
    log("===END===")


if __name__ == "__main__":
    main()
