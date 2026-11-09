"""Plan 13 stage 1 / plan 23: the one-launch decode MoE block (sgl_kernel hip_decode_moe_block) against
the head's per-op MoE sequence for the same block, on one gfx90a GCD.

Head sequence (as Qwen2MoeSparseMoeBlock.forward runs it on HIP decode): router linear and the
shared expert's gate_up / down through skinny_linear, SiluAndMul, fused_topk (the small-M
Triton router), fused_experts (tuned gfx90a configs), fused_gate_sigmoid_mul_add.
"Old head" re-runs it with plan 2/9/11's switches turned back (AOT topk_softmax, align +
count_sort, default MoE configs, two-kernel split-K reduce).

Times HIP-graph replays over NLAYERS blocks with distinct weights (cold L2), for M in 1, 2, 4, 8.
Checks: error against an fp32 reference, routing ids against the head router on the kernel's
own logits (including forced ties), end-to-end id agreement with the head, bitwise
repeatability, graph == eager, batch invariance (M = 1..8) and grid invariance.
Prints one JSON line starting with "RESULT ".

Plan 28 (the default): the expert-stationary phase B variants (bits 128, 256, 512; see the .hip)
against the plan-23 default (21, chunks of <= 2 rows). Checks: bitwise equality to 21 at M <= 16,
rows independent of M, of the split and of the grid (half, 77, 13 workgroups), 50x repeatability,
graph == eager, error vs the fp32 reference, counters back at 0. Graph replay us at
M = 1, 2, 3, 4, 8, 16 (three interleaved repeats; the per-op head once), and per-phase times from
the stamped builds (s_memrealtime per wave at phase boundaries, 25 MHz). ``--stage1`` runs the
plan 13 comparison against the per-op head.
"""

import contextlib
import importlib
import json
import math
import sys
import traceback
import types

import torch
import torch.nn.functional as F

DEV = torch.device("cuda", 0)
HID, NE, INTER, TOPK = 2048, 256, 256, 8
NLAYERS = 8
MS = (1, 2, 3, 4, 5, 6, 7, 8)
RES = {}


def log(*a):
    print(*a, flush=True)


def bf(t):
    return t.to(torch.bfloat16)


# ------------------------------------------------------------------------------------ timing
def ev_time(fn, iters, reps=5):
    fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(iters):
            fn()
        en.record()
        torch.cuda.synchronize()
        best = min(best, st.elapsed_time(en) * 1000.0 / iters)
    return best


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def graph_time(fn, iters=20):
    g = capture(fn)
    t = ev_time(g.replay, iters)
    del g
    return t


# ------------------------------------------------------------------------------------ head ops
def setup_head():
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as fm

    # Single process, no TP group: unsymmetric output allocation, no fused moe_sum+all-reduce.
    fm.get_parallel = lambda: types.SimpleNamespace(tp_group=None)
    fm.use_symmetric_memory = lambda *a, **k: contextlib.nullcontext()
    fm.is_allocation_symmetric = lambda: False
    real_get_exec = fm.get_exec

    def get_exec():
        try:
            e = real_get_exec()
            e.moe.enable_fused_moe_sum_all_reduce  # noqa: B018
            return e
        except Exception:  # noqa: BLE001
            return types.SimpleNamespace(moe=types.SimpleNamespace(enable_fused_moe_sum_all_reduce=False))

    fm.get_exec = get_exec
    from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul_add
    from sglang.kernels.ops.gemm.skinny_gemm_hip import skinny_linear
    from sglang.srt.layers.activation import SiluAndMul
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
    from sglang.srt.layers.moe.topk import StandardTopKOutput, fused_topk

    act = SiluAndMul()
    cfg = MoeRunnerConfig(inplace=False)

    def head(L, x):
        sh = skinny_linear(act(skinny_linear(x, L["sgu"], None)), L["sdown"], None)
        logits = skinny_linear(x, L["router"], None)
        tw, ti = fused_topk(x, logits, TOPK, True)
        routed = fm.fused_experts(x, L["w13"], L["w2"], StandardTopKOutput(tw, ti, logits), cfg)
        fused_gate_sigmoid_mul_add(x, L["sgate"], sh, routed)
        return {"out": routed, "ids": ti, "wts": tw, "logits": logits}

    def route(logits):
        return fused_topk(logits, logits, TOPK, True)

    return head, route


@contextlib.contextmanager
def old_head_switches():
    """Plan 2 (configs), 9 (router/align) and 11 (fused split-K reduce) turned back."""
    topk_mod = importlib.import_module("sglang.srt.layers.moe.topk")
    mab = importlib.import_module("sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size")
    cfgm = importlib.import_module("sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config")
    sk = importlib.import_module("sglang.kernels.ops.gemm.skinny_gemm_hip")
    saved = (topk_mod._use_hip_small_m_triton_router, mab._use_small_numel_align, cfgm._config_device_name,
             sk._fused_reduce_ok)
    topk_mod._use_hip_small_m_triton_router = False
    mab._use_small_numel_align = False
    cfgm._config_device_name = lambda: "AMD_Instinct_MI250X/MI250"
    cfgm.get_moe_configs.cache_clear()
    sk._fused_reduce_ok = False
    try:
        yield
    finally:
        (topk_mod._use_hip_small_m_triton_router, mab._use_small_numel_align, cfgm._config_device_name,
         sk._fused_reduce_ok) = saved
        cfgm.get_moe_configs.cache_clear()


# ------------------------------------------------------------------------------------ data
def make_block(gen, ties=False):
    def w(*shape):
        return bf(torch.randn(*shape, device=DEV, generator=gen) * (1.0 / math.sqrt(shape[-1])))

    L = {"router": w(NE, HID), "sgu": w(2 * INTER, HID), "sgate": w(1, HID).view(-1), "sdown": w(HID, INTER)}
    if ties:  # experts 100..131 duplicate 0..31: exact logit ties, lowest id must win
        L["router"][100:132] = L["router"][0:32]
        L["router"] = L["router"].contiguous()
    w13 = torch.empty(NE, 2 * INTER, HID, dtype=torch.bfloat16, device=DEV)
    w2 = torch.empty(NE, HID, INTER, dtype=torch.bfloat16, device=DEV)
    for e0 in range(0, NE, 32):
        w13[e0:e0 + 32] = w(32, 2 * INTER, HID)
        w2[e0:e0 + 32] = w(32, HID, INTER)
    L["w13"], L["w2"] = w13, w2
    return L


def reference(L, x, ids, wts):
    r = lambda t: t.to(torch.bfloat16).float()  # noqa: E731
    f = lambda t: t.float()  # noqa: E731
    xf = f(x)
    sgu = r(xf @ f(L["sgu"]).t())
    sh = r(r(F.silu(sgu[:, :INTER]) * sgu[:, INTER:]) @ f(L["sdown"]).t())
    gate = xf @ f(L["sgate"])
    out = torch.zeros(x.shape[0], HID, device=DEV)
    for m in range(x.shape[0]):
        acc = torch.zeros(HID, device=DEV)
        for k in range(TOPK):
            e = int(ids[m, k])
            w13 = f(L["w13"][e])
            g, u = r(w13[:INTER] @ xf[m]), r(w13[INTER:] @ xf[m])
            acc += r((f(L["w2"][e]) @ r(F.silu(g) * u)) * float(wts[m, k]))
        out[m] = r(r(acc) + torch.sigmoid(gate[m]) * sh[m])
    return out


def rel_err(a, ref):
    return ((a.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-12)).item()


# ------------------------------------------------------------------------------------ plan 28
REF = 21  # plan 23 default (per-token phase B), launched in chunks of <= 2 rows as the head does
VARIANTS = [  # (name, variant bits, workgroups per CU, max M built); bits in hip_decode_moe_block.hip
    ("es/2", 133, 2, 2), ("es+pre/2", 389, 2, 8), ("es+pre+cb4/2", 1413, 2, 2), ("es+pre/3", 389, 3, 8),
]
TIME_MS = (1, 2, 3, 4, 8, 16)
# s_memrealtime advances 25 ticks per us here (plan 23, j144/j151).
TICK_US = 0.04


def phases(st, G):
    """Stamps per wave: 0 start, 1 phase A, 2 router wait, 3 routing, 4 w13 done, 5 first w2
    loads issued, 6 barrier + staging done, 7 first w2 chunk computed (B waves), 8 end."""
    s = st[:G].reshape(G * 4, 16)[:, :9].double()
    t0 = s[:, 0][s[:, 0] > 0].min()
    t = (s - t0) * TICK_US
    valid = s > 0

    def med(i):
        return t[:, i][valid[:, i]].median().item() if valid[:, i].any() else float("nan")

    def mx(i):
        return t[:, i][valid[:, i]].max().item() if valid[:, i].any() else float("nan")

    return {"A": med(1), "wait": med(2) - med(1), "route": med(3) - med(2), "w13_med": med(4) - med(3),
            "w13_tail": mx(4) - med(4), "issue": med(5) - med(4), "barrier+stage": med(6) - mx(5),
            "chunk0": med(7) - med(6), "B_end": mx(8) - med(6), "span": mx(8)}


def plan28_main():
    props = torch.cuda.get_device_properties(0)
    cu = props.multi_processor_count
    log("device:", props.name, props.gcnArchName, "CUs", cu)
    from sglang.kernels.ops.moe import hip_decode_moe_block as mk

    log("platform_ok", mk.platform_ok())
    co = {}
    for m in range(1, 9):
        for v in (0, REF, REF | 8, 133, 389, 397, 1413):
            g = torch.ops.sgl_kernel.hip_decode_moe_block_max_grid(m, v)
            if g:
                co[(m, v)] = g
    log("co-resident grid by (M, variant):", co)
    RES["co_grid"] = {f"{m},{v}": g for (m, v), g in co.items()}
    head, route = setup_head()
    gen = torch.Generator(device=DEV).manual_seed(4321)
    blocks = [make_block(gen, ties=(i == 1)) for i in range(NLAYERS)]
    torch.cuda.synchronize()
    mk._prepare(DEV)
    ws = mk._workspaces[0]
    views = mk.workspace_views(DEV)

    def rows_of(M, v, chunk):
        if v in (REF, REF | 8):
            chunk = 2
        if chunk:
            return [min(chunk, M - r) for r in range(0, M, chunk)]
        return mk.launch_rows(M)

    def kern(L, x, v, wpc, out=None, grid=None, chunk=None):
        if out is None:
            out = torch.empty_like(x)
        r0 = 0
        for m in rows_of(x.shape[0], v, chunk):
            g = grid or min(co[(m, v)], wpc * cu)
            torch.ops.sgl_kernel.hip_decode_moe_block(x[r0:r0 + m], L["router"], L["sgu"], L["sgate"], L["sdown"],
                                                      L["w13"], L["w2"], out[r0:r0 + m], ws, g, v)
            r0 += m
        return out

    # ---- checks per variant (all against the plan-23 default, which must match bitwise)
    x16 = bf(torch.randn(16, HID, device=DEV, generator=gen))
    xc = [bf(torch.randn(2, HID, device=DEV, generator=gen)) for _ in range(4)]
    y16_ref = kern(blocks[0], x16, REF, 2).clone()
    yc_ref = [kern(blocks[i], xc[i], REF, 2).clone() for i in range(4)]
    ok_by = {}
    for name, v, wpc, mmax in VARIANTS:
        if (1, v) not in co or co[(1, v)] < wpc * cu:
            log("skip", name, "not co-resident at M=1:", co.get((1, v)))
            continue
        try:
            Mx = min(16, mmax) if mmax == 8 else mmax
            c = {"grid_by_M": {m: min(co[(m, v)], wpc * cu) for m in range(1, mmax + 1)}}
            xm = x16[:Mx].contiguous()
            y = kern(blocks[0], xm, v, wpc).clone()
            c["bitwise_eq_ref"] = bool(torch.equal(y, y16_ref[:Mx])) and all(
                torch.equal(kern(blocks[i], xc[i], v, wpc), yc_ref[i]) for i in range(4))
            errs, ids_ok = [], True
            for i in range(4):
                yy = kern(blocks[i], xc[i], v, wpc)
                torch.cuda.synchronize()
                ids, wts, logits = views["ids"][:2].clone(), views["wts"][:2].clone(), views["logits"][:2].clone()
                tw, ti = route(logits.contiguous())
                ids_ok &= bool(torch.equal(ids, ti.to(torch.int32)))
                h = head(blocks[i], xc[i])
                errs.append((rel_err(yy, reference(blocks[i], xc[i], ids, wts)),
                             rel_err(h["out"], reference(blocks[i], xc[i], h["ids"], h["wts"]))))
            c["ids_equal_head_router_same_logits"] = ids_ok
            c["relerr_kern_head"] = errs
            c["relerr_ok"] = all(k <= max(2 * hh, 1e-2) for k, hh in errs)
            c["repeatable_50"] = all(torch.equal(kern(blocks[0], xm, v, wpc), y) for _ in range(50)) and all(
                torch.equal(kern(blocks[1], xc[1], v, wpc), kern(blocks[1], xc[1], v, wpc)) for _ in range(20))
            inv = {m: bool(torch.equal(kern(blocks[0], xm[:m].contiguous(), v, wpc), y[:m])) for m in range(1, Mx)}
            inv["rows_alone"] = all(torch.equal(kern(blocks[0], xm[m:m + 1].contiguous(), v, wpc), y[m:m + 1])
                                    for m in range(Mx))
            for g in (cu // 2, 77, 13):
                inv[f"grid_{g}"] = bool(torch.equal(kern(blocks[0], xm, v, wpc, grid=g), y))
            c["batch_invariant"] = inv
            out_buf = torch.empty_like(xm)
            gph = capture(lambda: kern(blocks[0], xm, v, wpc, out=out_buf))
            out_buf.zero_()
            gph.replay()
            torch.cuda.synchronize()
            c["graph_equals_eager"] = bool(torch.equal(out_buf, y))
            for _ in range(5):
                gph.replay()
            torch.cuda.synchronize()
            c["graph_replay_stable"] = bool(torch.equal(out_buf, y))
            del gph
            for _ in range(600):
                mk.poll_error()
            torch.cuda.synchronize()
            c["sync_err"] = int(views["sync"][128].item())
            c["sync_after"] = [int(views["sync"][i].item()) for i in (0, 192)]
            c["ok"] = (c["bitwise_eq_ref"] and ids_ok and c["relerr_ok"] and c["repeatable_50"]
                       and all(inv.values()) and c["graph_equals_eager"] and c["graph_replay_stable"]
                       and c["sync_err"] == 0 and c["sync_after"] == [0, 0])
            ok_by[name] = c["ok"]
            RES.setdefault("checks", {})[name] = c
        except Exception:  # noqa: BLE001
            log("check failed", name, traceback.format_exc()[-2500:])
            continue
        log("check", name, f"v={v} wpc={wpc}", c)

    # ---- timing: graph replay per block; three interleaved repeats
    cands = [("head_default", REF, 2, None)]  # what 5b273e5 ships: chunks of 2 rows, M <= 4
    for name, v, wpc, mmax in VARIANTS:
        if ok_by.get(name):
            cands.append((name, v, wpc, None))
            if name == "es+pre/2":
                cands.append(("es+pre/2 chunk2", v, wpc, 2))
    xs_by_m = {M: [bf(torch.randn(M, HID, device=DEV, generator=gen)) for _ in range(NLAYERS)] for M in TIME_MS}
    tim = {}
    for rep in range(3):
        for M in TIME_MS:
            xs = xs_by_m[M]
            outs = [torch.empty_like(x) for x in xs]
            for n, v, w, ch in cands:
                if n == "head_default" and M > 4:
                    continue
                vm = next((mm for nn, vv, ww, mm in VARIANTS if nn == n.split(" ")[0]), 8)
                if M > vm:
                    continue
                if ch and M <= ch:
                    continue
                try:
                    t = graph_time(lambda: [kern(L, x, v, w, out=o, chunk=ch) for L, x, o in zip(blocks, xs, outs)])
                except Exception:  # noqa: BLE001
                    log("timing failed", n, M, traceback.format_exc()[-800:])
                    continue
                tim.setdefault((n, M), []).append(t / NLAYERS)
            if rep == 0:
                try:
                    tim[("per_op_head", M)] = [graph_time(lambda: [head(L, x) for L, x in zip(blocks, xs)]) / NLAYERS]
                except Exception:  # noqa: BLE001
                    log("per-op head failed", M, traceback.format_exc()[-800:])
    ne = {M: sum(int(torch.unique(head(L, x)["ids"]).numel()) for L, x in zip(blocks, xs_by_m[M])) / NLAYERS
          for M in TIME_MS}
    mb = {M: ((NE + 2 * INTER + 1) * HID * 2 + HID * INTER * 2 + ne[M] * (2 * INTER * HID + HID * INTER) * 2) / 1e6
          for M in TIME_MS}
    table = []
    for n in [c[0] for c in cands] + ["per_op_head"]:
        row = {"variant": n}
        for M in TIME_MS:
            ts = tim.get((n, M))
            if ts:
                row[f"M{M}_us"] = round(min(ts), 2)
                row[f"M{M}_reps"] = [round(t, 2) for t in ts]
                row[f"M{M}_TBps"] = round(mb[M] / min(ts), 3)
        table.append(row)
        log("timing", row)
    RES["timing"] = table
    RES["MB"] = mb
    RES["unique_experts"] = ne

    # ---- per-phase times from the stamped builds (last layer of an 8-layer graph, 30 replays)
    ph = {}
    for n, vs, w, Ms in (("head_default", REF | 8, 2, (1, 2)), ("es+pre/2", 397, 2, (1, 2, 3, 8))):
        for M in Ms:
            try:
                xs = xs_by_m[M]
                outs = [torch.empty_like(x) for x in xs]
                G = min(co[(M, vs)], w * cu)
                g = capture(lambda: [kern(L, x, vs, w, out=o) for L, x, o in zip(blocks, xs, outs)])
                rows = []
                for _ in range(30):
                    views["stamps"].zero_()
                    g.replay()
                    torch.cuda.synchronize()
                    rows.append(phases(views["stamps"].clone(), min(G, 256)))
                t_graph = ev_time(g.replay, 20) / NLAYERS
                del g
                agg = {k: round(sorted(r[k] for r in rows)[len(rows) // 2], 2) for k in rows[0]}
                agg["graph_us_per_block"] = round(t_graph, 2)
                agg["grid"] = G
                ph[f"{n} M={M}"] = agg
                log("phases", n, f"M={M}", agg)
            except Exception:  # noqa: BLE001
                log("phases failed", n, M, traceback.format_exc()[-1500:])
    RES["phases"] = ph

    # ---- gates (plan 28): absolute MI210 j151 numbers, and scaled to this device's head
    hd = {r["variant"]: r for r in table}.get("head_default", {})
    scale = {M: hd.get(f"M{M}_us", base) / base for M, base in ((1, 34.51), (2, 53.46))}
    good = [r for r in table if ok_by.get(r["variant"].split(" ")[0]) and "chunk2" not in r["variant"]
            and "M1_us" in r and "M2_us" in r]
    gate = {"head_default": {M: hd.get(f"M{M}_us") for M in (1, 2, 3, 4)}, "scale": scale}
    if good:
        best = min(good, key=lambda r: r["M2_us"] + r["M1_us"])
        m1, m2, m3 = best["M1_us"], best["M2_us"], best.get("M3_us", 1e9)
        s1, s2 = scale[1], scale[2]
        gate.update(best=best["variant"], M1=m1, M2=m2, M3=m3,
                    go=(m1 <= 31.5 * s1 and m2 <= 48.5 * s2) or (m1 <= 33 * s1 and m2 <= 50 * s2 and m3 <= 72 * s2),
                    kill=(m2 > 50.5 * s2 and m1 > 32.5 * s1))
    # dispatch per M: fastest correct option among native, chunk2 and the per-op head
    disp = {}
    for M in TIME_MS:
        opts = {r["variant"]: r[f"M{M}_us"] for r in table if f"M{M}_us" in r
                and (r["variant"] in ("per_op_head", "head_default") or ok_by.get(r["variant"].split(" ")[0]))}
        if opts:
            disp[M] = min(opts.items(), key=lambda kv: kv[1])
    gate["fastest_by_M"] = disp
    RES["gate"] = gate
    log("GATE", gate)
    log("RESULT " + json.dumps(RES, default=str))


# ------------------------------------------------------------------------------------ main
def main():
    props = torch.cuda.get_device_properties(0)
    log("device:", props.name, props.gcnArchName, "CUs", props.multi_processor_count)
    from sglang.kernels.ops.moe import hip_decode_moe_block as mk

    log("platform_ok", mk.platform_ok(), "env_enabled", mk.env_enabled())
    log("max grid by M:", {m: torch.ops.sgl_kernel.hip_decode_moe_block_max_grid(m, 0) for m in range(1, 9)})
    head, route = setup_head()
    gen = torch.Generator(device=DEV).manual_seed(4321)
    blocks = [make_block(gen, ties=(i == 1)) for i in range(NLAYERS)]
    torch.cuda.synchronize()
    log(f"{NLAYERS} blocks, {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    def kern(L, x, out=None, grid=None):
        return mk.decode_moe_block(x, L["router"], L["sgu"], L["sgate"], L["sdown"], L["w13"], L["w2"],
                                   out=out, grid=grid)

    # ---- correctness
    corr = []
    agree_rows = agree_bad = 0
    for M in (2, 8):
        for i, L in enumerate(blocks[:4]):
            x = bf(torch.randn(M, HID, device=DEV, generator=gen))
            y = kern(L, x)
            torch.cuda.synchronize()
            v = mk.workspace_views(DEV)
            ids, wts, logits = v["ids"][:M].clone(), v["wts"][:M].clone(), v["logits"][:M].clone()
            tw, ti = route(logits.contiguous())
            h = head(L, x)
            c = {"M": M, "block": i, "ties": i == 1,
                 "ids_equal_head_router_same_logits": bool(torch.equal(ids, ti.to(torch.int32))),
                 "wts_maxdiff_same_logits": (wts - tw.float()).abs().max().item(),
                 "ids_equal_head_e2e": bool(torch.equal(ids, h["ids"].to(torch.int32))),
                 "logits_maxdiff_head": (logits.float() - h["logits"].float()).abs().max().item(),
                 "kern_relerr": rel_err(y, reference(L, x, ids, wts)),
                 "head_relerr": rel_err(h["out"], reference(L, x, h["ids"], h["wts"])),
                 "kern_vs_head_relerr": rel_err(y, h["out"].float())}
            corr.append(c)
            log("correctness", c)
    # end-to-end routing agreement over many rows (logits differ only by GEMV summation order)
    for _ in range(64):
        x = bf(torch.randn(8, HID, device=DEV, generator=gen))
        kern(blocks[0], x)
        ids = mk.workspace_views(DEV)["ids"][:8].clone()
        h = head(blocks[0], x)
        agree_rows += 8
        agree_bad += int((ids != h["ids"].to(torch.int32)).any(1).sum().item())
    RES["routing_e2e_rows_differ"] = [agree_bad, agree_rows]
    log(f"e2e routing: {agree_bad}/{agree_rows} rows differ from the head (logit rounding only)")
    RES["correctness"] = corr

    # ---- determinism
    x8 = bf(torch.randn(8, HID, device=DEV, generator=gen))
    y8 = kern(blocks[0], x8).clone()
    rep = all(torch.equal(kern(blocks[0], x8), y8) for _ in range(100))
    inv = {m: bool(torch.equal(kern(blocks[0], x8[:m].contiguous()), y8[:m])) for m in range(1, 8)}
    cu = props.multi_processor_count
    inv["grid_half"] = bool(torch.equal(kern(blocks[0], x8, grid=cu // 2), y8))
    out_buf = torch.empty_like(x8)
    g = capture(lambda: kern(blocks[0], x8, out=out_buf))
    out_buf.zero_()
    g.replay()
    torch.cuda.synchronize()
    graph_eq = bool(torch.equal(out_buf, y8))
    del g
    sync = mk.workspace_views(DEV)["sync"]
    RES["determinism"] = {"repeatable_100": rep, "batch_invariant": inv, "graph_equals_eager": graph_eq,
                          "sync_err": int(sync[128].item()), "router_counter_after": int(sync[192].item()),
                          "barrier_count_after": int(sync[0].item())}
    log("determinism", RES["determinism"])

    # ---- timing (graph replay, per block)
    tim = {}
    for M in MS:
        xs = [bf(torch.randn(M, HID, device=DEV, generator=gen)) for _ in range(NLAYERS)]
        outs = [torch.empty_like(x) for x in xs]
        tim[("kernel", M)] = graph_time(lambda: [kern(L, x, out=o) for L, x, o in zip(blocks, xs, outs)]) / NLAYERS
        tim[("new_head", M)] = graph_time(lambda: [head(L, x) for L, x in zip(blocks, xs)]) / NLAYERS
        with old_head_switches():
            try:
                tim[("old_head", M)] = graph_time(lambda: [head(L, x) for L, x in zip(blocks, xs)]) / NLAYERS
            except Exception:  # noqa: BLE001
                log("old head failed:", traceback.format_exc()[-1500:])
        # bytes: dense + distinct experts, from the head's routing of these inputs
        ne = sum(int(torch.unique(head(L, x)["ids"]).numel()) for L, x in zip(blocks, xs)) / NLAYERS
        nbytes = (NE + 2 * INTER + 1) * HID * 2 + HID * INTER * 2 + ne * (2 * INTER * HID + HID * INTER) * 2
        row = {k: tim.get((k, M)) for k in ("old_head", "new_head", "kernel")}
        row.update(M=M, experts=ne, MB=nbytes / 1e6,
                   kernel_TBps=nbytes / (tim[("kernel", M)] * 1e-6) / 1e12,
                   new_head_TBps=nbytes / (tim[("new_head", M)] * 1e-6) / 1e12,
                   speedup_vs_new=tim[("new_head", M)] / tim[("kernel", M)],
                   speedup_vs_old=(tim[("old_head", M)] / tim[("kernel", M)]) if ("old_head", M) in tim else None)
        RES.setdefault("timing", []).append(row)
        log("timing us/block", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()})
    try:
        from collections import Counter

        from torch.profiler import ProfilerActivity, profile

        x = bf(torch.randn(2, HID, device=DEV, generator=gen))
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            head(blocks[0], x)
            torch.cuda.synchronize()
        names = [e.name for e in prof.events() if e.device_type.name in ("CUDA", "PrivateUse1")]
        RES["new_head_kernels_per_block"] = len(names)
        log("new head kernels per block:", len(names), Counter(n[:50] for n in names).most_common(12))
    except Exception:  # noqa: BLE001
        log("kernel count failed:", traceback.format_exc()[-500:])

    m2 = next(r for r in RES["timing"] if r["M"] == 2)
    ok = (all(c["ids_equal_head_router_same_logits"] and c["kern_relerr"] <= max(2 * c["head_relerr"], 1e-2)
              for c in corr)
          and rep and all(inv.values()) and graph_eq and RES["determinism"]["sync_err"] == 0)
    RES["gate"] = {"correct": ok, "speedup_M2_vs_new_head": m2["speedup_vs_new"],
                   "verdict": "GO" if ok and m2["speedup_vs_new"] >= 1.4 else "KILL"}
    log("GATE", RES["gate"])
    log("RESULT " + json.dumps(RES, default=str))


if __name__ == "__main__":
    sys.exit(main() if "--stage1" in sys.argv else plan28_main())
