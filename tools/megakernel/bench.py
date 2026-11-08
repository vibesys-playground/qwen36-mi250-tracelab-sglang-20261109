"""Plan 13 stage 0: is a persistent per-layer decode kernel feasible on gfx90a?

Builds tools/megakernel/megakernel.hip with hipcc (loaded through ctypes; launches go on torch's
current stream, so they are captured by torch.cuda.graph) and measures on one GCD:

(a) grid-barrier latency for G in {CU, 2 CU, 4 CU} workgroups of 256 threads, eager and in a
    HIP graph, with a cross-workgroup data check; producer/consumer flag ping-pong latency.
(b) a persistent skeleton of one TP=2 decode layer's GEMV work at M=2 against the head's kernel
    sequence for the same work (skinny GEMM + ordered reduce, topk_softmax, silu_and_mul,
    fused_experts, fused_gate_sigmoid_mul_add), both replayed from HIP graphs over 4 layers
    with distinct weights; correctness against an fp32 torch reference, bitwise repeatability
    and bitwise row invariance for M in 1..8.

Prints a gate summary and one JSON line starting with "RESULT ".
"""

import ctypes
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = torch.device("cuda", 0)
RES = {}

HID, NQKVZ, NBA, NOUTP = 2048, 6144, 32, 2048
NROUTER, INTER, NE, TOPK = 256, 256, 256, 8
N1, R_STRIDE = NQKVZ + NBA, 776
NLAYERS = 4


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------------------------------ build
def build():
    hipcc = shutil.which("hipcc") or ("/opt/rocm/bin/hipcc" if os.path.exists("/opt/rocm/bin/hipcc") else None)
    log("hipcc:", hipcc, "| ninja:", shutil.which("ninja"))
    RES["hipcc"] = hipcc
    if hipcc is None:
        raise SystemExit("NO_HIPCC: hipcc is not in the job image; stage 0 needs the AOT setup_rocm.py path")
    src = os.path.join(HERE, "megakernel.hip")
    digest = hashlib.sha256(open(src, "rb").read()).hexdigest()[:12]
    out_dir = "/tmp/megakernel_build"
    os.makedirs(out_dir, exist_ok=True)
    so = os.path.join(out_dir, f"megakernel_{digest}.so")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    cmd = [hipcc, "-O3", f"--offload-arch={arch}", "-ffp-contract=off", "-fPIC", "-shared", "-std=c++17",
           "-Wno-unused-value", "-Wno-unused-result", "-o", so, src]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    log(f"hipcc rc={r.returncode} in {time.time() - t0:.1f}s")
    if r.returncode != 0:
        log(r.stdout[-4000:], r.stderr[-4000:])
        raise SystemExit("hipcc failed")
    lib = ctypes.CDLL(so)
    vp, ci = ctypes.c_void_p, ctypes.c_int
    lib.mk_launch_barrier.argtypes = [ci, vp, ci, ci, vp, ci, vp]
    lib.mk_launch_pingpong.argtypes = [ci, vp, ci, vp]
    lib.mk_launch_layer.argtypes = [ci, ci, vp, vp]
    lib.mk_occupancy_layer.argtypes = [ci]
    return lib


class LayerArgs(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in (
        "x", "w_qkvz", "w_ba", "w_out", "w_router", "w_sgu", "w_sgate", "w_sdown", "w13", "w2",
        "y1", "h", "r", "act", "out", "ids_out", "wts_out", "sync")] + [("mask", ctypes.c_int), ("sleep", ctypes.c_int)]


def stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def check(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what}: hip error {rc}")


# ------------------------------------------------------------------------------------------ timing
def ev_time(fn, iters, reps=5):
    """Best-of-reps us per call of fn() (eager)."""
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


def capture(fn, warm=2):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warm):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def graph_time(fn, iters=20, reps=5):
    g = capture(fn)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    t = ev_time(g.replay, iters, reps)
    del g
    return t


# ------------------------------------------------------------------------------------------ (a)
def test_barrier(lib, cu):
    log("\n=== (a) grid barrier / flag latency ===")
    occ = lib.mk_occupancy_barrier()
    log("barrier kernel occupancy (WG/CU):", occ)
    sync = torch.zeros(256, dtype=torch.int32, device=DEV)
    data = torch.zeros(4 * cu * 256 * 2, dtype=torch.int32, device=DEV)
    out = {"occupancy": occ, "rows": []}
    N = 10000
    for mult in (1, 2, 4):
        G = mult * cu
        if mult > occ:
            log(f"G={G}: skipped, occupancy {occ}")
            continue
        # correctness: 2000 iterations of write / barrier / read neighbour / barrier
        sync.zero_()
        check(lib.mk_launch_barrier(G, sync.data_ptr(), 2000, 1, data.data_ptr(), 1, stream()), "barrier validate")
        torch.cuda.synchronize()
        bad, err = int(sync[192].item()), int(sync[128].item())
        for sleep in (0, 1):
            def run(iters, sleep=sleep, G=G):
                check(lib.mk_launch_barrier(G, sync.data_ptr(), iters, sleep, data.data_ptr(), 0, stream()), "barrier")
            t0e = ev_time(lambda: run(0), 50)
            tNe = ev_time(lambda: run(N), 3, reps=3)
            t0g = graph_time(lambda: run(0), iters=50)
            tNg = graph_time(lambda: run(N), iters=3, reps=3)
            err |= int(sync[128].item())
            row = {"G": G, "sleep": sleep, "launch_eager_us": t0e, "launch_graph_us": t0g,
                   "barrier_eager_us": (tNe - t0e) / N, "barrier_graph_us": (tNg - t0g) / N,
                   "data_mismatches": bad, "timeouts": err}
            out["rows"].append(row)
            log(f"G={G:4d} sleep={sleep}: barrier eager {row['barrier_eager_us']:.3f} us, graph "
                f"{row['barrier_graph_us']:.3f} us | empty launch eager {t0e:.2f} us graph {t0g:.2f} us | "
                f"data mismatches {bad} timeouts {err}")
    # ping-pong between WG 0 and WG CU-1
    f = torch.zeros(256, dtype=torch.int32, device=DEV)
    P = 10000
    def pp(iters):
        check(lib.mk_launch_pingpong(cu, f.data_ptr(), iters, stream()), "pingpong")
    t0e, tPe = ev_time(lambda: pp(0), 50), ev_time(lambda: pp(P), 3, reps=3)
    t0g, tPg = graph_time(lambda: pp(0), iters=50), graph_time(lambda: pp(P), iters=3, reps=3)
    out["flag_one_way_eager_us"] = (tPe - t0e) / (2 * P)
    out["flag_one_way_graph_us"] = (tPg - t0g) / (2 * P)
    out["flag_timeouts"] = int(f[128].item())
    log(f"flag ping-pong one-way: eager {out['flag_one_way_eager_us']:.3f} us, graph "
        f"{out['flag_one_way_graph_us']:.3f} us, timeouts {out['flag_timeouts']}")
    RES["barrier"] = out
    return out


# ------------------------------------------------------------------------------------------ (b)
def bf(t):
    return t.to(torch.bfloat16)


def make_layer(gen, idx):
    def w(*shape):
        return bf(torch.randn(*shape, device=DEV, generator=gen) * (1.0 / math.sqrt(shape[-1])))
    L = {
        "w_qkvz": w(NQKVZ, HID), "w_ba": w(NBA, HID), "w_out": w(NOUTP, HID),
        "w_router": w(NROUTER, HID), "w_sgu": w(2 * INTER, HID), "w_sgate": w(1, HID),
        "w_sdown": w(HID, INTER),
    }
    w13 = torch.empty(NE, 2 * INTER, HID, dtype=torch.bfloat16, device=DEV)
    w2 = torch.empty(NE, HID, INTER, dtype=torch.bfloat16, device=DEV)
    for e0 in range(0, NE, 32):
        w13[e0:e0 + 32] = w(32, 2 * INTER, HID)
        w2[e0:e0 + 32] = w(32, HID, INTER)
    L["w13"], L["w2"] = w13, w2
    L["idx"] = idx
    return L


def skel_bufs(M):
    z = lambda *s, dt=torch.bfloat16: torch.zeros(*s, dtype=dt, device=DEV)  # noqa: E731
    return {"y1": z(M, N1), "h": z(M, HID), "r": z(M, R_STRIDE), "act": z(M, TOPK, INTER), "out": z(M, HID),
            "ids": z(M, TOPK, dt=torch.int32), "wts": z(M, TOPK, dt=torch.float32)}


class Skeleton:
    def __init__(self, lib, sync):
        self.lib, self.sync = lib, sync
        self.keep = []

    def args(self, L, x, B, mask=31, sleep=1):
        a = LayerArgs()
        for k in ("w_qkvz", "w_ba", "w_out", "w_router", "w_sgu", "w_sgate", "w_sdown", "w13", "w2"):
            setattr(a, k, L[k].data_ptr())
        a.x = x.data_ptr()
        for k in ("y1", "h", "r", "act", "out"):
            setattr(a, k, B[k].data_ptr())
        a.ids_out, a.wts_out, a.sync = B["ids"].data_ptr(), B["wts"].data_ptr(), self.sync.data_ptr()
        a.mask, a.sleep = mask, sleep
        self.keep.append(a)
        return a

    def launch(self, M, G, a):
        check(self.lib.mk_launch_layer(M, G, ctypes.byref(a), stream()), "layer")


def head_ops():
    """The head's decode kernels for this work, as qwen2_moe/qwen3_5 call them on gfx90a."""
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    try:
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    except Exception:  # noqa: BLE001
        log("set_global_server_args failed:", traceback.format_exc()[-800:])
    from sglang.kernels.ops.gemm.skinny_gemm_hip import skinny_linear
    from sglang.srt.layers.moe.topk import StandardTopKOutput, fused_topk
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts
    from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul_add
    from sglang.srt.layers.activation import SiluAndMul
    act = SiluAndMul()
    silu_mul = act
    cfg = MoeRunnerConfig(inplace=False)

    def layer(L, x):
        y1a = skinny_linear(x, L["w_qkvz"], None)
        skinny_linear(x, L["w_ba"], None)
        hh = skinny_linear(y1a[:, 2048:4096], L["w_out"], None)
        logits = skinny_linear(hh, L["w_router"], None)
        sh = skinny_linear(silu_mul(skinny_linear(hh, L["w_sgu"], None)), L["w_sdown"], None)
        tw, ti = fused_topk(hh, logits, TOPK, True)
        routed = fused_experts(hh, L["w13"], L["w2"], StandardTopKOutput(tw, ti, logits), cfg)
        fused_gate_sigmoid_mul_add(hh, L["w_sgate"].squeeze(0), sh, routed)
        return {"y1": y1a, "h": hh, "logits": logits, "ids": ti, "wts": tw, "out": routed}

    def route(logits):
        return fused_topk(logits, logits, TOPK, True)

    return layer, route


def reference(L, x, ids, wts):
    """fp32 math with bf16 activations at the head's storage points; routing given."""
    r = lambda t: t.to(torch.bfloat16).float()  # noqa: E731
    f = lambda t: t.float()  # noqa: E731
    xf = f(x)
    y1 = r(xf @ f(L["w_qkvz"]).t())
    hh = r(y1[:, 2048:4096] @ f(L["w_out"]).t())
    sgu = r(hh @ f(L["w_sgu"]).t())
    sh = r(r(F.silu(sgu[:, :INTER]) * sgu[:, INTER:]) @ f(L["w_sdown"]).t())
    gate = hh @ f(L["w_sgate"]).t()
    out = torch.zeros(x.shape[0], HID, device=DEV)
    for m in range(x.shape[0]):
        acc = torch.zeros(HID, device=DEV)
        for k in range(TOPK):
            e = int(ids[m, k])
            w13 = f(L["w13"][e])
            g, u = r(w13[:INTER] @ hh[m]), r(w13[INTER:] @ hh[m])
            a = r(F.silu(g) * u)
            acc += r((f(L["w2"][e]) @ a) * float(wts[m, k]))
        out[m] = r(r(acc) + torch.sigmoid(gate[m]) * sh[m])
    return {"y1": y1, "h": hh, "out": out}


def rel_err(a, ref):
    return ((a.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-12)).item()


def test_skeleton(lib, cu):
    log("\n=== (b) persistent layer skeleton vs head kernel sequence ===")
    out = {}
    occ = {m: lib.mk_occupancy_layer(m) for m in range(1, 9)}
    log("layer kernel occupancy (WG/CU) by M:", occ)
    out["occupancy"] = occ
    gen = torch.Generator(device=DEV).manual_seed(1234)
    layers = [make_layer(gen, i) for i in range(NLAYERS)]
    torch.cuda.synchronize()
    log(f"weights: {NLAYERS} layers, {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    M = 2
    xs = [bf(torch.randn(M, HID, device=DEV, generator=gen)) for _ in range(NLAYERS)]
    sync = torch.zeros(256, dtype=torch.int32, device=DEV)
    sk = Skeleton(lib, sync)
    Gs = [k * cu for k in (1, 2, 3, 4) if k <= occ[M]]

    # ---- head
    head = route = None
    try:
        head, route = head_ops()
        hres = [head(L, x) for L, x in zip(layers, xs)]
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        log("HEAD SEQUENCE FAILED:", traceback.format_exc()[-3000:])
        hres = None

    # ---- skeleton eager, correctness at M=2
    bufs = [skel_bufs(M) for _ in range(NLAYERS)]
    G0 = Gs[-1]
    sargs = [sk.args(L, x, B) for L, x, B in zip(layers, xs, bufs)]
    for a in sargs:
        sk.launch(M, G0, a)
    torch.cuda.synchronize()
    corr = []
    for i, (L, x, B) in enumerate(zip(layers, xs, bufs)):
        c = {"layer": i}
        if hres is not None:
            hr = hres[i]
            # routing kernel check: topk_softmax on the skeleton's own bf16 logits
            tw, ti = route(B["r"][:, :NROUTER].contiguous())
            c["ids_equal_topk_softmax"] = bool(torch.equal(B["ids"], ti.to(torch.int32)))
            c["wts_maxdiff_topk_softmax"] = (B["wts"] - tw.float()).abs().max().item()
            c["ids_equal_head_e2e"] = bool(torch.equal(B["ids"], hr["ids"].to(torch.int32)))
            c["logits_maxdiff_head"] = (B["r"][:, :NROUTER].float() - hr["logits"].float()).abs().max().item()
            href = reference(L, x, hr["ids"], hr["wts"])
            c["head_out_relerr"] = rel_err(hr["out"], href["out"])
            c["skel_vs_head_out_relerr"] = rel_err(B["out"], hr["out"].float())
        ref = reference(L, x, B["ids"], B["wts"])
        c["skel_y1_relerr"] = rel_err(B["y1"][:, :NQKVZ], ref["y1"])
        c["skel_out_relerr"] = rel_err(B["out"], ref["out"])
        c["n_experts"] = int(torch.unique(B["ids"]).numel())
        corr.append(c)
        log("correctness", c)
    out["correctness"] = corr

    # ---- repeatability (100 eager launches + graph replays) and row invariance (M=1..8)
    ref_out = bufs[0]["out"].clone()
    rep_ok = True
    for _ in range(100):
        sk.launch(M, G0, sargs[0])
        if not torch.equal(bufs[0]["out"], ref_out):
            rep_ok = False
    torch.cuda.synchronize()
    out["repeatable_100"] = rep_ok
    x8 = bf(torch.randn(8, HID, device=DEV, generator=gen))
    inv = {}
    B8 = skel_bufs(8)
    sk.launch(8, cu, sk.args(layers[0], x8, B8))
    torch.cuda.synchronize()
    for m in range(1, 8):
        Bm = skel_bufs(m)
        G = cu * min(occ[m], 2)
        sk.launch(m, G, sk.args(layers[0], x8[:m].contiguous(), Bm))
        torch.cuda.synchronize()
        inv[m] = bool(torch.equal(Bm["out"], B8["out"][:m]) and torch.equal(Bm["ids"], B8["ids"][:m]))
    # same M, different grid size must agree too (schedule-independent rows)
    if len(Gs) > 1:
        Bg = skel_bufs(M)
        sk.launch(M, Gs[0], sk.args(layers[0], xs[0], Bg))
        torch.cuda.synchronize()
        inv["M2_G_change"] = bool(torch.equal(Bg["out"], ref_out))
    out["row_invariant"] = inv
    log("repeatable over 100 launches:", rep_ok, "| row invariance vs M=8:", inv)

    # ---- timing (graph of 4 layers, distinct weights; per layer)
    def skel_fn(G, mask=31):
        aa = [sk.args(L, x, B, mask) for L, x, B in zip(layers, xs, bufs)]
        def fn():
            for a in aa:
                sk.launch(M, G, a)
        return fn

    tim = {}
    for G in Gs:
        tim[f"skel_graph_G{G}"] = graph_time(skel_fn(G)) / NLAYERS
        tim[f"skel_eager_G{G}"] = ev_time(skel_fn(G), 20) / NLAYERS
        log(f"skeleton G={G}: graph {tim[f'skel_graph_G{G}']:.1f} us/layer, eager {tim[f'skel_eager_G{G}']:.1f}")
    # graph replay output must equal eager output
    g = capture(skel_fn(G0))
    for B in bufs:
        B["out"].zero_()
    g.replay()
    torch.cuda.synchronize()
    out["graph_equals_eager"] = bool(torch.equal(bufs[0]["out"], ref_out))
    del g
    bestG = min(Gs, key=lambda G: tim[f"skel_graph_G{G}"])
    # phase breakdown at the best G (other phases skipped; 4 barriers always run)
    for name, mask in (("barriers_only", 0), ("P1_qkvz_ba", 1), ("P2_gdn_out", 2), ("P3_router_shared", 4),
                       ("P4_w13", 8), ("P5_w2_shared_down", 16)):
        tim[f"phase_{name}"] = graph_time(skel_fn(bestG, mask)) / NLAYERS
    log("phase times (graph, us/layer):", {k: round(v, 1) for k, v in tim.items() if k.startswith("phase_")})
    # restore full outputs
    skel_fn(bestG)()
    torch.cuda.synchronize()

    if hres is not None:
        def head_fn():
            for L, x in zip(layers, xs):
                head(L, x)
        tim["head_graph"] = graph_time(head_fn) / NLAYERS
        tim["head_eager"] = ev_time(head_fn, 10) / NLAYERS
        log(f"head sequence: graph {tim['head_graph']:.1f} us/layer, eager {tim['head_eager']:.1f}")
        try:
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                head_fn()
                torch.cuda.synchronize()
            names = [e.name for e in prof.events() if e.device_type.name in ("CUDA", "PrivateUse1")]
            out["head_kernels_per_layer"] = len(names) / NLAYERS
            from collections import Counter
            log("head kernels per layer:", len(names) / NLAYERS, Counter(n[:60] for n in names).most_common(20))
        except Exception:  # noqa: BLE001
            log("kernel count failed:", traceback.format_exc()[-600:])
    out["timing_us_per_layer"] = tim

    # ---- bytes and bandwidth
    dense = (N1 + NOUTP + NROUTER + 2 * INTER + 1) * HID * 2 + HID * INTER * 2
    per_exp = (2 * INTER * HID + HID * INTER) * 2
    n_exp = [c["n_experts"] for c in corr]
    bytes_layer = dense + per_exp * sum(n_exp) / len(n_exp)
    out["bytes_per_layer"] = bytes_layer
    skel_us = tim[f"skel_graph_G{bestG}"]
    out["best_G"] = bestG
    out["skel_us"] = skel_us
    out["skel_TBps"] = bytes_layer / (skel_us * 1e-6) / 1e12
    if "head_graph" in tim:
        out["head_us"] = tim["head_graph"]
        out["head_TBps"] = bytes_layer / (tim["head_graph"] * 1e-6) / 1e12
        out["speedup"] = tim["head_graph"] / skel_us
    log(f"bytes/layer {bytes_layer / 1e6:.1f} MB (experts {n_exp}) | skeleton {skel_us:.1f} us = "
        f"{out['skel_TBps']:.3f} TB/s | head {out.get('head_us', float('nan')):.1f} us = "
        f"{out.get('head_TBps', float('nan')):.3f} TB/s | speedup {out.get('speedup', float('nan')):.2f}x")
    out["timeouts"] = int(sync[128].item())
    RES["skeleton"] = out
    return out


# ------------------------------------------------------------------------------------------ gate
def gate():
    b, s = RES.get("barrier", {}), RES.get("skeleton", {})
    reasons = []
    rows = [r for r in b.get("rows", []) if r["sleep"] == 1]
    G = s.get("best_G")
    brow = next((r for r in rows if r["G"] == G), rows[0] if rows else None)
    if brow is None:
        reasons.append("no barrier number")
    else:
        bmax = max(brow["barrier_eager_us"], brow["barrier_graph_us"])
        if bmax > 3.0:
            reasons.append(f"barrier {bmax:.2f} us > 3 us")
        if brow["data_mismatches"] or brow["timeouts"]:
            reasons.append("barrier data check failed")
    sp = s.get("speedup")
    if sp is None:
        reasons.append("no head comparison")
    elif sp < 1.6:
        reasons.append(f"speedup {sp:.2f}x < 1.6x")
    if s.get("skel_TBps", 0) < 0.8:
        reasons.append(f"bandwidth {s.get('skel_TBps', 0):.3f} TB/s < 0.8")
    if not s.get("repeatable_100") or not s.get("graph_equals_eager"):
        reasons.append("not repeatable")
    if not all(s.get("row_invariant", {0: False}).values()):
        reasons.append("not row-invariant")
    corr = s.get("correctness", [])
    for c in corr:
        lim = max(2 * c.get("head_out_relerr", 0.0), 1e-2)
        if c["skel_out_relerr"] > lim:
            reasons.append(f"layer {c['layer']} rel err {c['skel_out_relerr']:.3g} > {lim:.3g}")
        if c.get("ids_equal_topk_softmax") is False:
            reasons.append(f"layer {c['layer']} routing ids differ from topk_softmax")
    if s.get("timeouts"):
        reasons.append("skeleton barrier timeout")
    if reasons:
        verdict = "PARK"
    elif sp >= 2.0:
        verdict = "PASS"
    else:
        verdict = "JUDGMENT (1.6x..2x)"
    RES["gate"] = {"verdict": verdict, "reasons": reasons}
    log("\n=== GATE:", verdict, reasons)


def main():
    props = torch.cuda.get_device_properties(0)
    cu = props.multi_processor_count
    log("device:", torch.cuda.get_device_name(0), props.gcnArchName, "CUs", cu)
    log("torch", torch.__version__, "hip", torch.version.hip)
    RES["device"] = {"name": torch.cuda.get_device_name(0), "arch": props.gcnArchName, "cu": cu}
    lib = build()
    assert lib.mk_cu_count() == cu
    for part in (test_barrier, test_skeleton):
        try:
            part(lib, cu)
        except Exception:  # noqa: BLE001
            log(f"{part.__name__} FAILED:", traceback.format_exc()[-3000:])
    gate()
    log("RESULT " + json.dumps(RES, default=str))


if __name__ == "__main__":
    sys.exit(main())
