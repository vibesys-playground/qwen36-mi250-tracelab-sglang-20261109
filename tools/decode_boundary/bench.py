"""Plan 25 step 1: decode boundaries between layer bodies on gfx90a (GCD 0 and 1, HIP graphs).

Chain (one "layer", 40 per graph, 200 replays), per-rank TP=2 shapes:
  skf2 out_proj (2048x2048) -> [boundary A] -> 54 MB GEMV + 2048x2048 GEMV (MoE stand-in)
  -> [boundary B] -> skf2 in_proj (6176x2048) -> next layer's out_proj input (a column slice)
Variants: h (head: custom AR + Triton Gemma add+norm on both boundaries), A (out_proj with the
P2P all-reduce + add + norm epilogue), B (P2P AR + add + norm kernel on the MoE side), B19
(plan 19's fused custom-AR kernel on the MoE side), AB, AB19.
Also: per-kernel durations (torch profiler over graph replays, rank 0), correctness of A and B
(500 GEMV-written iterations, ranks, replays, rows), lm_head hipBLASLt vs skf2 configs, and the
logits gather RCCL vs P2P. Prints RESULT lines and GATE lines. Not on any serving path.
"""
import contextlib
import json
import os
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HID = 2048
EPS = 1e-6
VOCAB_SHARD = 124160
CHAIN = 40
REPLAYS = 200
CORR_ITERS = 500
MS = (1, 2, 4)


def log(*a):
    print(*a, flush=True)


def bits_hash(t):
    v = t.contiguous().view(torch.int16).to(torch.int64).flatten()
    w = (torch.arange(v.numel(), device=v.device, dtype=torch.int64) % 65521) + 1
    return int((v * w).sum().item())


def ulp_err(out, ref):
    ref = ref.double()
    a = ref.abs().clamp_min(2.0 ** -126)
    ulp = torch.exp2(torch.floor(torch.log2(a)) - 7)
    e = (out.double() - ref).abs() / ulp
    return e.max().item(), (e <= 1.0).double().mean().item()


def capture(fn, ctx, warm=2):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warm):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with ctx():
        with torch.cuda.graph(g, stream=s):
            res = fn()
    torch.cuda.synchronize()
    return g, res


def time_graph(g, pg, reps=REPLAYS):
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    dist.barrier(group=pg)
    t0 = time.perf_counter()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e6


class Ctx:
    pass


def setup(rank):
    c = Ctx()
    c.rank = rank
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    from sglang.srt.distributed.device_communicators import custom_all_reduce as car
    import sglang.srt.distributed.device_communicators.custom_all_reduce_ops as ops
    from sglang.kernels.ops.gemm import skinny_gemv_hip as G
    from sglang.kernels.ops.gemm.skinny_p2p_hip import HipP2PBoundary
    from sglang.kernels.ops.layernorm.minimax_m3_rmsnorm import gemma_fused_add_rmsnorm

    c.pg = dist.new_group([0, 1], backend="gloo")
    c.nccl = dist.new_group([0, 1], backend="nccl")
    c.ca = car.CustomAllreduce(group=c.pg, device=rank)
    c.has19 = bool(getattr(ops, "HAS_ALL_REDUCE_ADD_RMSNORM", False))
    c.G = G
    c.lib = G.build()  # SKG_SWEEP: every K=2048 skf2 config
    c.gemma = gemma_fused_add_rmsnorm

    def ago(obj):
        out = [None, None]
        dist.all_gather_object(out, obj, group=c.pg)
        return out

    c.p2p = HipP2PBoundary(c.lib, rank, ago, gather_cols=VOCAB_SHARD)
    c.cfg = {
        "o": G.F2Config(2, 8, 1),
        "big": G.F2Config(4, 16, 1),
        "in": G.F2Config(4, 16, 1),
    }
    c.idx = {k: c.lib.q_find(v, HID, "f2") for k, v in c.cfg.items()}
    if rank == 0:
        log(f"device {torch.cuda.get_device_name()} {torch.cuda.get_device_properties(0).gcnArchName} "
            f"ca.disabled={c.ca.disabled} has_plan19_op={c.has19} idx={c.idx}")
    return c


def gemv(c, key, x, w):
    return c.G.mfma_gemv(c.lib, x, w, None, c.cfg[key], c.idx[key], "f2")


# ------------------------------------------------------------------------------------------
# chain timing
# ------------------------------------------------------------------------------------------
def make_weights(rank):
    g = torch.Generator(device="cuda")
    g.manual_seed(1000 + rank)
    W = {
        "o": (torch.randn(HID, HID, generator=g, device="cuda") * 0.02).to(torch.bfloat16),
        "big": (torch.randn(13184, HID, generator=g, device="cuda") * 0.02).to(torch.bfloat16),
        "down": (torch.randn(HID, HID, generator=g, device="cuda") * 0.02).to(torch.bfloat16),
        "in": (torch.randn(6176, HID, generator=g, device="cuda") * 0.02).to(torch.bfloat16),
    }
    g.manual_seed(77)  # norm weights replicated
    W["n1"] = (torch.randn(HID, generator=g, device="cuda") * 0.3).to(torch.bfloat16)
    W["n2"] = (torch.randn(HID, generator=g, device="cuda") * 0.3).to(torch.bfloat16)
    return W


def layer(c, W, x, res, variant):
    if "A" in variant:
        h, res = c.p2p.out_proj_add_norm(x, W["o"], res, W["n1"], EPS)
    else:
        y = gemv(c, "o", x, W["o"])
        h, res = c.gemma(c.ca.custom_all_reduce(y), res, W["n1"], EPS, round_sum=True)
    big = gemv(c, "big", h, W["big"])
    p = gemv(c, "o", big[:, :HID], W["down"])
    if variant.endswith("B19"):
        r = c.ca.custom_all_reduce_add_rmsnorm(p, res, W["n2"], EPS, 1.0)
        if r is None:
            raise RuntimeError("plan 19 op declined")
        h2, res = r
    elif variant.endswith("B"):
        h2, res = c.p2p.ar_add_norm(p, res, W["n2"], EPS)
    else:
        h2, res = c.gemma(c.ca.custom_all_reduce(p), res, W["n2"], EPS, round_sum=True)
    q = gemv(c, "in", h2, W["in"])
    return q[:, :HID], res


def chain_part(c):
    W = make_weights(c.rank)
    variants = ["h", "A", "B", "AB"] + (["B19", "AB19"] if c.has19 else [])
    out = {}
    graphs = {}
    for m in MS:
        torch.manual_seed(5 + c.rank)
        x0 = torch.randn(m, 6176, device="cuda").to(torch.bfloat16)[:, :HID]
        r0 = torch.ones(m, HID, device="cuda", dtype=torch.bfloat16)
        for v in variants:
            def run(v=v):
                x, res = x0, r0
                for _ in range(CHAIN):
                    x, res = layer(c, W, x, res, v)
                return x

            g, _ = capture(run, c.ca.capture)
            out[(m, v)] = time_graph(g, c.pg) / CHAIN
            if m == 2 and v in ("h", "AB", "AB19", "A", "B"):
                graphs[v] = g
            else:
                del g
    prof = {}
    for v, g in graphs.items():
        prof[v] = profile_graph(c, g)
    return {"us": {f"{m}:{v}": t for (m, v), t in out.items()}, "prof": prof,
            "err": c.p2p.errors()}


def profile_graph(c, g, reps=20):
    from torch.profiler import ProfilerActivity, profile

    dist.barrier(group=c.pg)
    if c.rank != 0:
        for _ in range(reps):
            g.replay()
        torch.cuda.synchronize()
        dist.barrier(group=c.pg)
        return None
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(reps):
            g.replay()
        torch.cuda.synchronize()
    dist.barrier(group=c.pg)
    agg = {}
    for e in p.events():
        dt = getattr(e, "device_type", None)
        if dt is None or "CUDA" not in str(dt):
            continue
        name = e.name
        dur = e.device_time if hasattr(e, "device_time") else e.cuda_time
        n, t = agg.get(name, (0, 0.0))
        agg[name] = (n + 1, t + dur)
    rows = []
    for name, (n, t) in agg.items():
        per_layer_count = n / (reps * CHAIN)
        rows.append((name[:90], round(per_layer_count, 3), round(t / n, 3), round(t / (reps * CHAIN), 3)))
    rows.sort(key=lambda r: -r[3])
    return rows[:14]


# ------------------------------------------------------------------------------------------
# correctness of A and B against the head
# ------------------------------------------------------------------------------------------
def corr_part(c):
    W = make_weights(c.rank)
    res = {}
    for m in (1, 2, 4, 16):
        inp = torch.zeros(m, HID, dtype=torch.bfloat16, device="cuda")
        resid = torch.zeros(m, HID, dtype=torch.bfloat16, device="cuda")

        def head():
            y = gemv(c, "o", inp, W["o"])
            return c.gemma(c.ca.custom_all_reduce(y), resid, W["n1"], EPS, round_sum=True)

        def fa():
            return c.p2p.out_proj_add_norm(inp, W["o"], resid, W["n1"], EPS)

        def fb():
            return c.p2p.ar_add_norm(gemv(c, "o", inp, W["o"]), resid, W["n1"], EPS)

        gh, (ho, hr) = capture(head, c.ca.capture)
        ga, (ao, ar) = capture(fa, contextlib.nullcontext)
        gb, (bo, br) = capture(fb, contextlib.nullcontext)
        st = {"A_res_bad": 0, "B_res_bad": 0, "A_max_ulp": 0.0, "B_max_ulp": 0.0,
              "A_within1_min": 1.0, "B_within1_min": 1.0, "eager_bad": 0}
        hashes = []
        g = torch.Generator(device="cuda")
        for it in range(CORR_ITERS):
            g.manual_seed(7919 * it + 31 * m + c.rank)
            inp.copy_(torch.randn(m, HID, generator=g, device="cuda").to(torch.bfloat16))
            g.manual_seed(104729 * it + m)
            resid.copy_(torch.randn(m, HID, generator=g, device="cuda").to(torch.bfloat16))
            dist.barrier(group=c.pg)
            gh.replay()
            ga.replay()
            gb.replay()
            torch.cuda.synchronize()
            for k, o, r in (("A", ao, ar), ("B", bo, br)):
                st[f"{k}_res_bad"] += int((r.view(torch.int16) != hr.view(torch.int16)).sum().item())
                mx, fr = ulp_err(o, ho)
                st[f"{k}_max_ulp"] = max(st[f"{k}_max_ulp"], mx)
                st[f"{k}_within1_min"] = min(st[f"{k}_within1_min"], fr)
            hashes.append((bits_hash(ao), bits_hash(ar), bits_hash(bo), bits_hash(br)))
            if it < 10:  # eager calls match the graph replays
                eo, er = c.p2p.out_proj_add_norm(inp, W["o"], resid, W["n1"], EPS)
                fo, fr_ = c.p2p.ar_add_norm(gemv(c, "o", inp, W["o"]), resid, W["n1"], EPS)
                torch.cuda.synchronize()
                st["eager_bad"] += int((eo.view(torch.int16) != ao.view(torch.int16)).sum().item())
                st["eager_bad"] += int((fo.view(torch.int16) != bo.view(torch.int16)).sum().item())
        first = (ao.clone(), bo.clone())
        rep_bad = 0
        for _ in range(200):
            ga.replay()
            gb.replay()
            torch.cuda.synchronize()
            rep_bad += int(not torch.equal(ao, first[0])) + int(not torch.equal(bo, first[1]))
        st["rep_bad"] = rep_bad
        st["hashes"] = hashes
        res[m] = st
        del gh, ga, gb
    # row invariance: row 0 of the same 16-row inputs for M = 1..4, eager
    g = torch.Generator(device="cuda")
    g.manual_seed(4242 + c.rank)
    X = torch.randn(16, HID, generator=g, device="cuda").to(torch.bfloat16)
    g.manual_seed(4243)
    R = torch.randn(16, HID, generator=g, device="cuda").to(torch.bfloat16)
    rows = {"A": [], "B": []}
    for m in (1, 2, 3, 4, 8, 16):
        dist.barrier(group=c.pg)
        o, r = c.p2p.out_proj_add_norm(X[:m], W["o"], R[:m], W["n1"], EPS)
        o2, r2 = c.p2p.ar_add_norm(gemv(c, "o", X[:m], W["o"]), R[:m], W["n1"], EPS)
        torch.cuda.synchronize()
        rows["A"].append(torch.cat([o[0], r[0]]))
        rows["B"].append(torch.cat([o2[0], r2[0]]))
    row_ok = {k: all(torch.equal(v[0], t) for t in v[1:]) for k, v in rows.items()}
    return {"per_m": res, "rows_ok": row_ok, "err": c.p2p.errors()}


# ------------------------------------------------------------------------------------------
# lm_head and the logits gather
# ------------------------------------------------------------------------------------------
LM_CFGS = [(2, 8, 1), (4, 8, 1), (8, 8, 1), (2, 16, 1), (4, 16, 1), (8, 16, 1), (4, 8, 2),
           (8, 8, 2), (4, 4, 1), (8, 4, 1), (4, 4, 2)]


def lm_part(c):
    G = c.G
    g = torch.Generator(device="cuda")
    g.manual_seed(900 + c.rank)
    Wl = (torch.randn(VOCAB_SHARD, HID, generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    X = torch.randn(16, HID, generator=g, device="cuda").to(torch.bfloat16)
    nbytes = Wl.numel() * 2
    t = {}
    cfgs = []
    for r, p, gg in LM_CFGS:
        cfg = G.F2Config(r, p, gg)
        i = c.lib.q_find(cfg, HID, "f2")
        if i >= 0:
            cfgs.append((cfg, i))
    for m in (1, 2, 4, 8, 16):
        x = X[:m]
        gr, _ = capture(lambda: [torch.matmul(x, Wl.T) for _ in range(5)], contextlib.nullcontext)
        t[f"{m}:blas"] = time_graph(gr, c.pg, 40) / 5
        del gr
        for cfg, i in cfgs:
            gr, _ = capture(lambda: [G.mfma_gemv(c.lib, x, Wl, None, cfg, i, "f2") for _ in range(5)],
                            contextlib.nullcontext)
            t[f"{m}:{cfg.rows},{cfg.kparts},{cfg.groups}"] = time_graph(gr, c.pg, 40) / 5
            del gr
    # best config at M=2
    best = min(((k, v) for k, v in t.items() if k.startswith("2:") and "blas" not in k),
               key=lambda kv: kv[1])[0].split(":")[1]
    r, p, gg = (int(v) for v in best.split(","))
    cfg = G.F2Config(r, p, gg)
    i = c.lib.q_find(cfg, HID, "f2")
    # row invariance and repeatability
    ref = G.mfma_gemv(c.lib, X[:1], Wl, None, cfg, i, "f2")[0].clone()
    row_ok = all(torch.equal(G.mfma_gemv(c.lib, X[:m], Wl, None, cfg, i, "f2")[0], ref)
                 for m in range(1, 17))
    rep_ok = all(torch.equal(G.mfma_gemv(c.lib, X, Wl, None, cfg, i, "f2"),
                             G.mfma_gemv(c.lib, X, Wl, None, cfg, i, "f2")) for _ in range(20))
    # argmax agreement with hipBLASLt over 10k rows
    flips = near = 0
    worst = 0.0
    g.manual_seed(31337 + c.rank)
    for _ in range(625):
        xb = torch.randn(16, HID, generator=g, device="cuda").to(torch.bfloat16)
        lb = torch.matmul(xb, Wl.T)
        ls = G.mfma_gemv(c.lib, xb, Wl, None, cfg, i, "f2")
        ab, as_ = lb.argmax(-1), ls.argmax(-1)
        bad = (ab != as_).nonzero().flatten()
        for j in bad.tolist():
            flips += 1
            top = lb[j].float()
            v1 = top[ab[j]].item()
            v2 = top[as_[j]].item()
            ulp = 2.0 ** (torch.floor(torch.log2(torch.tensor(abs(v1)))).item() - 7)
            gap = abs(v1 - v2) / ulp
            worst = max(worst, gap)
            near += int(gap <= 2.0)
    maxdiff = (torch.matmul(X, Wl.T).float() - G.mfma_gemv(c.lib, X, Wl, None, cfg, i, "f2").float()).abs().max().item()
    return {"us": t, "gbps_best_m2": nbytes / t[f"2:{best}"] / 1e3, "best": best,
            "row_ok": row_ok, "rep_ok": rep_ok, "flips": flips, "near_tie_flips": near,
            "worst_gap_ulp": worst, "maxabs_vs_blas": maxdiff}


def gather_part(c):
    t = {}
    ok = True
    for m in (1, 2, 4, 16):
        torch.manual_seed(50 + c.rank)
        x = torch.randn(m, VOCAB_SHARD, device="cuda").to(torch.bfloat16)

        def rccl():
            out = torch.empty((2 * m, VOCAB_SHARD), dtype=x.dtype, device="cuda")
            dist.all_gather_into_tensor(out, x, group=c.nccl)
            return out.view(2, m, VOCAB_SHARD).movedim(0, 1).reshape(m, 2 * VOCAB_SHARD)

        def p2p():
            return c.p2p.gather(x)

        for _ in range(3):
            rccl()
        torch.cuda.synchronize()
        dist.barrier(group=c.pg)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        ev[0].record()
        for _ in range(50):
            rccl()
        ev[1].record()
        torch.cuda.synchronize()
        t[f"{m}:rccl_eager"] = ev[0].elapsed_time(ev[1]) * 1000 / 50
        gp, po = capture(lambda: [p2p() for _ in range(10)], contextlib.nullcontext)
        t[f"{m}:p2p"] = time_graph(gp, c.pg, 50) / 10
        for _ in range(20):
            x.copy_(torch.randn_like(x, dtype=torch.float32).to(torch.bfloat16))
            dist.barrier(group=c.pg)
            ro = rccl()
            gp.replay()
            torch.cuda.synchronize()
            ok = ok and all(torch.equal(ro.view(torch.int16), b.view(torch.int16)) for b in po)
        del gp
    return {"us": t, "bitwise": ok, "err": c.p2p.errors()}


# ------------------------------------------------------------------------------------------
def worker(rank, world):
    import faulthandler

    faulthandler.dump_traceback_later(2700, exit=True)
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method="tcp://127.0.0.1:29625")
    c = setup(rank)
    out = {}
    for name, fn in (("chain", chain_part), ("corr", corr_part), ("lm", lm_part),
                     ("gather", gather_part)):
        try:
            out[name] = fn(c)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            out[name] = f"ERR {type(e).__name__}: {e}"
        dist.barrier(group=c.pg)
        log(f"rank {rank} done {name}")
        allr = [None, None]
        dist.all_gather_object(allr, out, group=c.pg)
        if rank == 0:
            try:
                report(allr)
            except Exception:  # noqa: BLE001
                traceback.print_exc()
    dist.barrier(group=c.pg)
    faulthandler.cancel_dump_traceback_later()
    os._exit(0)


def report(r):
    log("RAW " + json.dumps(r, default=str)[:4000])
    v = {}
    ch = [x.get("chain") for x in r]
    if all(isinstance(x, dict) for x in ch):
        log("RESULT chain: us per layer (max over ranks), chain of 40 in one graph, 200 replays")
        keys = sorted({k.split(":")[1] for k in ch[0]["us"]})
        log("RESULT  M  " + "  ".join(f"{k:>7}" for k in keys))
        us = {k: max(x["us"][k] for x in ch) for k in ch[0]["us"]}
        for m in MS:
            log(f"RESULT {m:2d}  " + "  ".join(f"{us[f'{m}:{k}']:7.2f}" for k in keys))
        log(f"RESULT chain p2p errors {[x['err'] for x in ch]}")
        prof = ch[0]["prof"]
        for k, rows in prof.items():
            log(f"RESULT prof M=2 variant {k}: (kernel, per-layer count, avg us, us per layer)")
            for row in rows or []:
                log(f"RESULT   {row}")

        def avg(rows, pat, excl=None):
            hits = [x for x in rows if pat in x[0] and (excl is None or excl not in x[0])]
            return hits[0][2] if hits else float("nan")

        hrows = prof.get("h") or []
        ar_us = avg(hrows, "cross_device_reduce_1stage", "rmsnorm")
        nm_us = avg(hrows, "gemma_fused_add_rmsnorm")
        v["calib"] = ar_us >= 6.3 and nm_us >= 3.8
        log(f"GATE calibration (h, M=2): AR {ar_us:.2f} us (>=6.3), add+norm {nm_us:.2f} us (>=3.8) "
            f"-> {'PASS' if v['calib'] else 'FAIL (bench not in-situ-like)'}")
        sa = us["2:h"] - us["2:A"]
        sb = us["2:h"] - us["2:B"]
        sab = us["2:h"] - us["2:AB"]
        v["A"] = sa >= 4.0
        log(f"GATE A saves {sa:.2f} us/layer at M=2 (>=4) -> {'PASS' if v['A'] else 'FAIL'}")
        log(f"GATE B saves {sb:.2f} us/layer at M=2; AB saves {sab:.2f}")
        if "2:B19" in us:
            log(f"GATE B19 saves {us['2:h'] - us['2:B19']:.2f} us/layer at M=2; AB19 "
                f"{us['2:h'] - us['2:AB19']:.2f}")
        v["save_ab"] = 40 * sab
    co = [x.get("corr") for x in r]
    if all(isinstance(x, dict) for x in co):
        log("RESULT correctness (500 GEMV-written iterations per M, graph mode, vs head)")
        okall = True
        for m in co[0]["per_m"]:
            s = [x["per_m"][m] for x in co]
            ranks_eq = s[0]["hashes"] == s[1]["hashes"]
            for k in ("A", "B"):
                rb = sum(y[f"{k}_res_bad"] for y in s)
                mu = max(y[f"{k}_max_ulp"] for y in s)
                w1 = min(y[f"{k}_within1_min"] for y in s)
                log(f"RESULT {k} M={m}: residual!=head {rb}, out max_ulp {mu:.2f}, within1ulp_min "
                    f"{w1:.6f}")
                okall = okall and rb == 0 and w1 >= 0.999
            rep = sum(y["rep_bad"] for y in s)
            eb = sum(y["eager_bad"] for y in s)
            log(f"RESULT M={m}: ranks_identical={ranks_eq} replay_mismatch={rep}/400 eager_vs_graph_bad={eb}")
            okall = okall and ranks_eq and rep == 0 and eb == 0
        rows = [x["rows_ok"] for x in co]
        log(f"RESULT rows invariant {rows}; p2p errors {[x['err'] for x in co]}")
        okall = okall and all(all(x.values()) for x in rows) and all(not any(x["err"]) for x in co)
        v["corr"] = okall
        log(f"GATE A/B correctness -> {'PASS' if okall else 'FAIL'}")
    lm = [x.get("lm") for x in r]
    if all(isinstance(x, dict) for x in lm):
        log("RESULT lm_head us (rank 0 / rank 1), N=124160 K=2048")
        for k in lm[0]["us"]:
            log(f"RESULT lm {k:>10}: {lm[0]['us'][k]:8.1f} {lm[1]['us'][k]:8.1f}  "
                f"({508.6e3 / lm[0]['us'][k]:.0f} GB/s)")
        for x in lm:
            log(f"RESULT lm best={x['best']} GB/s={x['gbps_best_m2']:.0f} row_ok={x['row_ok']} "
                f"rep_ok={x['rep_ok']} argmax flips={x['flips']}/10000 near_tie={x['near_tie_flips']} "
                f"worst_gap_ulp={x['worst_gap_ulp']:.2f} maxabs={x['maxabs_vs_blas']:.4g}")
        best = lm[0]["best"]
        bm2 = max(x["us"][f"2:{best}"] for x in lm)
        blas2 = max(x["us"]["2:blas"] for x in lm)
        never_slower = all(max(x["us"][f"{m}:{best}"] for x in lm) <= max(x["us"][f"{m}:blas"] for x in lm)
                           for m in (1, 2, 4, 8, 16))
        v["C_lm"] = bm2 <= 480 and never_slower and all(x["row_ok"] and x["rep_ok"] for x in lm)
        v["save_lm"] = blas2 - bm2 if v["C_lm"] else 0.0
        log(f"GATE lm_head skf2 {best}: {bm2:.1f} us vs hipBLASLt {blas2:.1f} at M=2 (<=480, never "
            f"slower={never_slower}) -> {'PASS' if v['C_lm'] else 'FAIL'}")
    ga = [x.get("gather") for x in r]
    if all(isinstance(x, dict) for x in ga):
        for k in ga[0]["us"]:
            log(f"RESULT gather {k:>8}: {max(x['us'][k] for x in ga):7.2f} us")
        okg = all(x["bitwise"] for x in ga) and all(not any(x["err"]) for x in ga)
        # RCCL in situ (j138 profile): 39.7 us at bs=2; the eager number includes launch gaps
        sg = 39.7 - max(x["us"]["2:p2p"] for x in ga)
        v["C_g"] = okg and sg > 0
        v["save_g"] = sg if v["C_g"] else 0.0
        log(f"GATE gather bitwise={okg} saves {sg:.1f} us at M=2 -> {'PASS' if v['C_g'] else 'FAIL'}")
    tot = v.get("save_ab", 0.0) + v.get("save_lm", 0.0) + v.get("save_g", 0.0)
    log(f"GATE projected per bs=2 step: 40 x AB = {v.get('save_ab', 0):.0f} us, lm_head "
        f"{v.get('save_lm', 0):.0f} us, gather {v.get('save_g', 0):.0f} us; total {tot:.0f} us "
        f"(>=300) -> {'PASS' if tot >= 300 else 'FAIL'}")


def main():
    log("devices:", torch.cuda.device_count())
    mp.spawn(worker, args=(2,), nprocs=2, join=True)


if __name__ == "__main__":
    main()
