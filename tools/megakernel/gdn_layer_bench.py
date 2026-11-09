"""Plan 13 stage 2 feasibility: one-launch decode GDN layer body and in-kernel P2P all-reduce.

Builds tools/megakernel/gdn_layer.hip with hipcc (ctypes; launches on torch's current stream, so
they are captured by torch.cuda.graph) and measures on MI250 GCDs:

(1) GDN body, one GCD, TP=2 per-rank shapes, HIP-graph replay over NLAYERS layers with distinct
    weights and states (cold L2), M = 1..4:
      head  = skinny_linear(packed in_proj) + fused_qkvzba_causal_conv1d_update_contiguous +
              fused_recurrent_gated_delta_rule_packed_decode + gated RMSNorm (layernorm_fn) +
              skinny_linear(out_proj)         (what Qwen3_5GatedDeltaNet.forward runs on decode)
      fused = k_gdn_body (one launch)
    plus the input GemmaRMSNorm (fused add) the layer runs before the body. Checks: output and
    new conv/SSM state against the head and an fp32 reference, bitwise repeatability (100
    runs), batch invariance (M = 1..4), grid invariance, graph == eager.
(2) all-reduce of M x 2048 bf16 between GCD 0 and 1 (two processes): in-kernel push (k_ar,
    uncached receive buffers, buffer_wbl2 / buffer_invl2) as a loop inside one launch and as
    one launch per all-reduce in a graph, against sglang's CustomAllreduce
    (cross_device_reduce_1stage) in a graph.

Kill gate (plan 13 stage 2 brief): fused body at M=2 <= 60% of the head body.
Prints one JSON line starting with "RESULT ".
"""

import contextlib
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

HERE = os.path.dirname(os.path.abspath(__file__))
HID, NKH, NVH, DK, DV = 2048, 8, 16, 128, 128
QKV, NIN = 4096, 6176
NLAYERS, SLOTS = 8, 8
EPS = 1e-6
RES = {}


def log(*a):
    print(*a, flush=True)


def build():
    hipcc = shutil.which("hipcc") or "/opt/rocm/bin/hipcc"
    src = os.path.join(HERE, "gdn_layer.hip")
    digest = hashlib.sha256(open(src, "rb").read()).hexdigest()[:12]
    os.makedirs("/tmp/megakernel_build", exist_ok=True)
    so = f"/tmp/megakernel_build/gdn_layer_{digest}.so"
    if not os.path.exists(so):
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        cmd = [hipcc, "-O3", f"--offload-arch={arch}", "-ffp-contract=off", "-fPIC", "-shared", "-std=c++17",
               "-Wno-unused-value", "-Wno-unused-result", "-o", so, src]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        log(f"hipcc rc={r.returncode} in {time.time() - t0:.1f}s")
        if r.returncode != 0:
            log(r.stdout[-4000:], r.stderr[-4000:])
            raise SystemExit("hipcc failed")
    return so


def load(so):
    lib = ctypes.CDLL(so)
    vp, ci = ctypes.c_void_p, ctypes.c_int
    lib.gl_launch.argtypes = [ci, ci, vp, vp]
    lib.gl_occupancy.argtypes = [ci]
    lib.ar_alloc_uncached.argtypes = [ctypes.c_size_t]
    lib.ar_alloc_uncached.restype = vp
    lib.ar_ipc_handle.argtypes = [vp, ctypes.c_char_p]
    lib.ar_ipc_open.argtypes = [ctypes.c_char_p]
    lib.ar_ipc_open.restype = vp
    lib.ar_launch.argtypes = [ci, vp, vp]
    return lib


class GdnArgs(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in (
        "x", "w_in", "conv_w", "conv_state", "idx", "A_log", "dt_bias", "ssm", "w_norm", "w_out", "out", "ws")] + [
        ("cs_slot", ctypes.c_longlong), ("cs_dim", ctypes.c_longlong), ("cs_pos", ctypes.c_longlong),
        ("ssm_slot", ctypes.c_longlong), ("num_slots", ctypes.c_int), ("scale", ctypes.c_float),
        ("eps", ctypes.c_float)]


class ArArgs(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in (
        "inp", "out", "my_recv", "peer_recv", "my_flag", "peer_flag", "epoch", "err")] + [
        (n, ctypes.c_int) for n in ("n", "rank", "reps", "use_wbl2")]


def stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


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


# ================================================================================ (1) GDN body
def make_layer(gen, dev):
    def w(*shape, s=None):
        s = s if s is not None else 1.0 / math.sqrt(shape[-1])
        return (torch.randn(*shape, device=dev, generator=gen) * s).to(torch.bfloat16)

    L = {"w_in": w(NIN, HID), "w_out": w(HID, HID), "conv_w": w(QKV, 4, s=0.5),
         "A_log": torch.log(torch.rand(NVH, device=dev, generator=gen) * 15 + 1).float(),
         "dt_bias": w(NVH, s=1.0), "w_norm": (1.0 + 0.1 * torch.randn(DV, device=dev, generator=gen)).to(torch.bfloat16)}
    L["conv_state"] = w(SLOTS, QKV, 3, s=1.0)
    L["ssm"] = torch.randn(SLOTS, NVH, DV, DK, device=dev, generator=gen) * 0.05
    return L


def setup_head():
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    from sglang.kernels.ops.attention.fla.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode
    from sglang.kernels.ops.attention.fla.layernorm_gated import layernorm_fn
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import fused_qkvzba_causal_conv1d_update_contiguous
    from sglang.kernels.ops.gemm import skinny_gemm_hip as sk

    log("head GEMV active: in_proj", sk.gemv_active(NIN, HID), "out_proj", sk.gemv_active(HID, HID))
    RES["head_gemv_active"] = [sk.gemv_active(NIN, HID), sk.gemv_active(HID, HID)]

    def head(L, x, idx):
        M = x.shape[0]
        y = sk.skinny_linear(x, L["w_in"], None)
        qkv, z, b, a = fused_qkvzba_causal_conv1d_update_contiguous(
            y[:, :6144], y[:, 6144:], L["conv_state"], L["conv_w"], None, idx,
            qkv_dim=QKV, v_dim=NVH * DV, num_v_heads=NVH, head_v_dim=DV, activation="silu")
        o = qkv.new_empty(M, 1, NVH, DV)
        fused_recurrent_gated_delta_rule_packed_decode(qkv, a, b, L["A_log"], L["dt_bias"], DK ** -0.5, L["ssm"], o,
                                                        idx, use_qk_l2norm_in_kernel=True)
        on = layernorm_fn(o.reshape(M * NVH, DV), L["w_norm"], None, z=z.reshape(M * NVH, DV), eps=EPS,
                          group_size=None, norm_before_gate=True, is_rms_norm=True, activation="swish")
        return sk.skinny_linear(on.reshape(M, HID), L["w_out"], None)

    return head


def reference(L, x, idx, conv_state, ssm):
    """fp32 reference with bf16 rounding where the head stores bf16; mutates the state copies."""
    r = lambda t: t.to(torch.bfloat16).float()  # noqa: E731
    M = x.shape[0]
    y = r(x.float() @ L["w_in"].float().t())
    out = torch.empty(M, HID, device=x.device)
    for m in range(M):
        s = int(idx[m])
        cs = conv_state[s].float()  # [4096, 3]
        cw = L["conv_w"].float()
        acc = r(cs[:, 0] * cw[:, 0]) + r(cs[:, 1] * cw[:, 1]) + r(cs[:, 2] * cw[:, 2]) + r(y[m, :QKV] * cw[:, 3])
        qkv = r(acc * torch.sigmoid(acc))
        conv_state[s] = torch.stack([cs[:, 1], cs[:, 2], y[m, :QKV]], 1).to(conv_state.dtype)
        q = qkv[:1024].view(NKH, DK)
        k = qkv[1024:2048].view(NKH, DK)
        v = qkv[2048:].view(NVH, DV)
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * DK ** -0.5
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        bb, aa = y[m, 6144:6160], y[m, 6160:6176]
        xg = aa + L["dt_bias"].float()
        g = -torch.exp(L["A_log"]) * torch.where(xg <= 20, torch.log1p(torch.exp(xg)), xg)
        beta = r(torch.sigmoid(bb))
        o = torch.empty(NVH, DV, device=x.device)
        for hv in range(NVH):
            h = ssm[s, hv] * torch.exp(g[hv])  # [V, K]
            kk, qq = k[hv // 2], q[hv // 2]
            vv = (v[hv] - h @ kk) * beta[hv]
            h = h + vv[:, None] * kk[None, :]
            ssm[s, hv] = h
            o[hv] = h @ qq
        o = r(o)
        z = y[m, 4096:6144].view(NVH, DV)
        on = o * torch.rsqrt((o * o).mean(-1, keepdim=True) + EPS) * L["w_norm"].float()
        on = r(on * z * torch.sigmoid(z)).view(HID)
        out[m] = r(on @ L["w_out"].float().t())
    return out


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-12)).item()


def gdn_part(lib):
    dev = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(0)
    cu = props.multi_processor_count
    occ = {m: lib.gl_occupancy(m) for m in (1, 2, 3, 4)}
    log("device", props.name, props.gcnArchName, "CUs", cu, "occupancy (WG/CU) by M", occ)
    RES["occupancy"] = occ
    head = setup_head()
    gen = torch.Generator(device=dev).manual_seed(1234)
    layers = [make_layer(gen, dev) for _ in range(NLAYERS)]
    ws_bytes = lib.gl_ws_bytes()
    ws = torch.zeros(ws_bytes, dtype=torch.uint8, device=dev)

    def fused(L, x, idx, out=None, grid=None, ws_=None):
        M = x.shape[0]
        out = out if out is not None else torch.empty(M, HID, dtype=torch.bfloat16, device=dev)
        cs, ssm = L["conv_state"], L["ssm"]
        a = GdnArgs(x.data_ptr(), L["w_in"].data_ptr(), L["conv_w"].data_ptr(), cs.data_ptr(), idx.data_ptr(),
                    L["A_log"].data_ptr(), L["dt_bias"].data_ptr(), ssm.data_ptr(), L["w_norm"].data_ptr(),
                    L["w_out"].data_ptr(), out.data_ptr(), (ws_ if ws_ is not None else ws).data_ptr(),
                    cs.stride(0), cs.stride(1), cs.stride(2), ssm.stride(0), cs.shape[0], DK ** -0.5, EPS)
        g = grid or occ[M] * cu
        rc = lib.gl_launch(M, g, ctypes.byref(a), stream())
        if rc != 0:
            raise RuntimeError(f"gl_launch rc={rc}")
        return out

    def snap(L):
        return L["conv_state"].clone(), L["ssm"].clone()

    def restore(L, s):
        L["conv_state"].copy_(s[0])
        L["ssm"].copy_(s[1])

    idx_all = torch.tensor([3, 5, 1, 6], dtype=torch.int32, device=dev)
    sync = ws[:4096].view(torch.int32)

    # ---- correctness
    corr = []
    for M in (1, 2, 4):
        for li in range(2):
            L = layers[li]
            x = (torch.randn(M, HID, device=dev, generator=gen)).to(torch.bfloat16)
            idx = idx_all[:M].contiguous()
            s0 = snap(L)
            yh = head(L, x, idx)
            sh = snap(L)
            restore(L, s0)
            yf = fused(L, x, idx)
            torch.cuda.synchronize()
            sf = snap(L)
            cs_ref, ssm_ref = s0[0].clone(), s0[1].clone()
            yr = reference(L, x, idx, cs_ref, ssm_ref)
            restore(L, s0)
            c = {"M": M, "layer": li, "fused_vs_ref": rel(yf, yr), "head_vs_ref": rel(yh, yr), "fused_vs_head": rel(yf, yh),
                 "conv_state_equal_head": bool(torch.equal(sf[0], sh[0])),
                 "conv_state_maxdiff_ref": (sf[0].float() - cs_ref.float()).abs().max().item(),
                 "ssm_maxdiff_head": (sf[1] - sh[1]).abs().max().item(),
                 "ssm_maxdiff_ref": (sf[1] - ssm_ref).abs().max().item(),
                 "ssm_untouched_slots_equal": bool(torch.equal(sf[1][[0, 2, 4, 7]], s0[1][[0, 2, 4, 7]])),
                 "sync_err": int(sync[128].item())}
            corr.append(c)
            log("correctness", c)
    RES["correctness"] = corr

    # ---- determinism (state restored before every call)
    L = layers[0]
    x4 = torch.randn(4, HID, device=dev, generator=gen).to(torch.bfloat16)
    s0 = snap(L)
    y4 = fused(L, x4, idx_all).clone()
    st4 = snap(L)
    rep = True
    for _ in range(100):
        restore(L, s0)
        y = fused(L, x4, idx_all)
        rep &= bool(torch.equal(y, y4)) and bool(torch.equal(L["ssm"], st4[1])) and bool(torch.equal(L["conv_state"], st4[0]))
    inv = {}
    for m in (1, 2, 3):
        restore(L, s0)
        inv[m] = bool(torch.equal(fused(L, x4[:m].contiguous(), idx_all[:m].contiguous()), y4[:m]))
    for gname, gsz in (("grid_cu", cu), ("grid_half", cu // 2)):
        restore(L, s0)
        inv[gname] = bool(torch.equal(fused(L, x4, idx_all, grid=gsz), y4))
    restore(L, s0)
    out_buf = torch.empty_like(x4)
    g = capture(lambda: fused(L, x4, idx_all, out=out_buf))
    restore(L, s0)
    out_buf.zero_()
    g.replay()
    torch.cuda.synchronize()
    graph_eq = bool(torch.equal(out_buf, y4)) and bool(torch.equal(L["ssm"], st4[1]))
    del g
    restore(L, s0)
    RES["determinism"] = {"repeatable_100": rep, "invariance": inv, "graph_equals_eager": graph_eq,
                          "sync_err": int(sync[128].item()), "barrier_count_after": int(sync[0].item()),
                          "group_counters_after": [int(sync[192 + 64 * i].item()) for i in range(NKH)]}
    log("determinism", RES["determinism"])

    # ---- timing (graph replay per layer)
    try:
        from sglang.srt.layers.layernorm import GemmaRMSNorm

        norm = GemmaRMSNorm(HID, eps=EPS).to(dev).to(torch.bfloat16)
    except Exception:  # noqa: BLE001
        norm = None
        log("GemmaRMSNorm unavailable:", traceback.format_exc()[-600:])
    tim = []
    for M in (1, 2, 3, 4):
        xs = [torch.randn(M, HID, device=dev, generator=gen).to(torch.bfloat16) for _ in range(NLAYERS)]
        outs = [torch.empty_like(x) for x in xs]
        idx = idx_all[:M].contiguous()
        row = {"M": M}
        row["head"] = graph_time(lambda: [head(L_, x, idx) for L_, x in zip(layers, xs)]) / NLAYERS
        row["fused"] = graph_time(lambda: [fused(L_, x, idx, out=o) for L_, x, o in zip(layers, xs, outs)]) / NLAYERS
        if M == 4:
            o2 = [torch.empty(2, HID, dtype=torch.bfloat16, device=dev) for _ in xs]
            row["fused_2x2"] = graph_time(lambda: [
                (fused(L_, x[:2], idx[:2], out=o[:2]), fused(L_, x[2:], idx[2:], out=o2_))
                for L_, x, o, o2_ in zip(layers, xs, outs, o2)]) / NLAYERS
        if norm is not None:
            res = [torch.randn(M, HID, device=dev, generator=gen).to(torch.bfloat16) for _ in range(NLAYERS)]
            try:
                row["input_norm"] = graph_time(lambda: [norm(x, r) for x, r in zip(xs, res)]) / NLAYERS
            except Exception:  # noqa: BLE001
                log("norm timing failed:", traceback.format_exc()[-600:])
        nbytes = (NIN * HID + HID * HID) * 2 + M * NVH * DV * DK * 4 * 2 + M * QKV * 3 * 2 * 2
        row["MB"] = nbytes / 1e6
        row["fused_TBps"] = nbytes / (row["fused"] * 1e-6) / 1e12
        row["head_TBps"] = nbytes / (row["head"] * 1e-6) / 1e12
        row["ratio"] = row["fused"] / row["head"]
        if "input_norm" in row:
            row["ratio_incl_norm"] = row["fused"] / (row["head"] + row["input_norm"])
        tim.append(row)
        log("timing us/layer", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()})
    RES["timing"] = tim
    # per-kernel head breakdown and kernel count at M=2
    try:
        from collections import defaultdict

        from torch.profiler import ProfilerActivity, profile

        x = torch.randn(2, HID, device=dev, generator=gen).to(torch.bfloat16)
        idx = idx_all[:2].contiguous()
        for _ in range(3):
            head(layers[0], x, idx)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for L_ in layers:
                head(L_, x, idx)
            torch.cuda.synchronize()
        evs = [e for e in prof.events() if e.device_type.name in ("CUDA", "PrivateUse1")]
        agg = defaultdict(lambda: [0, 0.0])
        for e in evs:
            agg[e.name[:60]][0] += 1
            agg[e.name[:60]][1] += e.device_time if hasattr(e, "device_time") else e.cuda_time
        RES["head_kernels_per_layer"] = len(evs) / NLAYERS
        RES["head_kernel_breakdown_us_per_layer"] = {k: round(v[1] / NLAYERS, 2) for k, v in agg.items()}
        log("head kernels per layer:", len(evs) / NLAYERS, RES["head_kernel_breakdown_us_per_layer"])
    except Exception:  # noqa: BLE001
        log("profile failed:", traceback.format_exc()[-800:])


# ================================================================================ (2) all-reduce
def ar_worker(rank, so, port, outq):
    import torch.distributed as dist

    try:
        torch.cuda.set_device(rank)
        dev = torch.device("cuda", rank)
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
        lib = load(so)
        res = {"rank": rank, "peer_access": torch.cuda.can_device_access_peer(rank, 1 - rank)}
        NMAX = 4 * HID
        GMAX = 64
        recv = lib.ar_alloc_uncached(2 * NMAX * 2)
        flag = lib.ar_alloc_uncached(GMAX * 64 * 4)
        hbuf = ctypes.create_string_buffer(64)
        hs = []
        for p in (recv, flag):
            if lib.ar_ipc_handle(p, hbuf) != 0:
                raise RuntimeError("ipc handle")
            hs.append(bytes(hbuf.raw))
        allh = [None, None]
        dist.all_gather_object(allh, hs)
        peer_recv = lib.ar_ipc_open(allh[1 - rank][0])
        peer_flag = lib.ar_ipc_open(allh[1 - rank][1])
        if not peer_recv or not peer_flag:
            raise RuntimeError("ipc open failed")
        epoch = torch.zeros(GMAX * 64, dtype=torch.int32, device=dev)
        err = torch.zeros(64, dtype=torch.int32, device=dev)
        rows = []
        for M in (1, 2, 4):
            n = M * HID
            G = max(1, n // (256 * 8))
            inp = torch.empty(n, dtype=torch.bfloat16, device=dev)
            out = torch.empty_like(inp)

            def run(reps, wbl2=1, inp=inp, out=out, n=n, G=G):
                a = ArArgs(inp.data_ptr(), out.data_ptr(), recv, peer_recv, flag, peer_flag, epoch.data_ptr(),
                           err.data_ptr(), n, rank, reps, wbl2)
                rc = lib.ar_launch(G, ctypes.byref(a), stream())
                if rc != 0:
                    raise RuntimeError(f"ar_launch rc={rc}")

            # correctness with fresh inputs
            bad = 0
            gen = torch.Generator(device=dev).manual_seed(100 + rank)
            for it in range(200):
                inp.copy_((torch.randn(n, device=dev, generator=gen)).to(torch.bfloat16))
                torch.cuda.synchronize()
                both = [None, None]
                dist.all_gather_object(both, inp.cpu())
                run(1)
                torch.cuda.synchronize()
                ref = (both[0].float() + both[1].float()).to(torch.bfloat16)
                bad += int((out.cpu() != ref).sum().item())
            row = {"M": M, "bytes": n * 2, "grid": G, "mismatches_200": bad}
            # in-kernel loop: per all-reduce cost inside one persistent launch
            for wbl2 in (1, 0):
                dist.barrier()
                torch.cuda.synchronize()
                st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                st.record()
                run(2000, wbl2)
                en.record()
                torch.cuda.synchronize()
                row[f"inkernel_loop_us{'' if wbl2 else '_nowbl2'}"] = st.elapsed_time(en) * 1000.0 / 2000
            # one launch per all-reduce, 20 per graph
            g = capture(lambda: [run(1) for _ in range(20)])
            dist.barrier()
            torch.cuda.synchronize()
            best = float("inf")
            for _ in range(5):
                dist.barrier()
                st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                st.record()
                for _ in range(20):
                    g.replay()
                en.record()
                torch.cuda.synchronize()
                best = min(best, st.elapsed_time(en) * 1000.0 / 400)
            row["push_launch_graph_us"] = best
            del g
            rows.append(row)
        res["push"] = rows
        res["err"] = int(err[0].item())
        # baseline: sglang CustomAllreduce (cross_device_reduce_1stage) in a graph
        try:
            from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

            set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
            from sglang.srt.distributed.device_communicators.custom_all_reduce import CustomAllreduce

            ca = CustomAllreduce(dist.group.WORLD, dev)
            res["ca_disabled"] = ca.disabled
            base = []
            if not ca.disabled:
                for M in (1, 2, 4):
                    x = torch.randn(M, HID, device=dev).to(torch.bfloat16)
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        ca.custom_all_reduce(x)
                    torch.cuda.current_stream().wait_stream(s)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with ca.capture():
                        with torch.cuda.graph(g):
                            ys = [ca.custom_all_reduce(x) for _ in range(20)]
                    torch.cuda.synchronize()
                    best = float("inf")
                    for _ in range(5):
                        dist.barrier()
                        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        st.record()
                        for _ in range(20):
                            g.replay()
                        en.record()
                        torch.cuda.synchronize()
                        best = min(best, st.elapsed_time(en) * 1000.0 / 400)
                    both = [None, None]
                    dist.all_gather_object(both, x.cpu())
                    ok = bool(torch.equal(ys[-1].cpu(), (both[0].float() + both[1].float()).to(torch.bfloat16)))
                    base.append({"M": M, "ca_graph_us": best, "ca_correct": ok})
                    del g
            res["custom_ar"] = base
        except Exception:  # noqa: BLE001
            res["custom_ar_error"] = traceback.format_exc()[-1500:]
        outq.put(res)
    except Exception:  # noqa: BLE001
        outq.put({"rank": rank, "error": traceback.format_exc()[-3000:]})
    finally:
        with contextlib.suppress(Exception):
            dist.destroy_process_group()


def ar_part(so):
    import torch.multiprocessing as mp

    if torch.cuda.device_count() < 2:
        log("AR: fewer than 2 devices, skipped")
        return
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29000 + os.getpid() % 1000
    ps = [ctx.Process(target=ar_worker, args=(r, so, port, q)) for r in range(2)]
    for p in ps:
        p.start()
    out = []
    for _ in ps:
        try:
            out.append(q.get(timeout=900))
        except Exception:  # noqa: BLE001
            out.append({"error": "timeout"})
    for p in ps:
        p.join(timeout=60)
    out.sort(key=lambda r: r.get("rank", 9))
    RES["allreduce"] = out
    for r in out:
        log("allreduce", json.dumps(r)[:3000])


def main():
    try:
        subprocess.run(["bash", "-c", "rocm-smi --showtopotype 2>/dev/null | head -20"], check=False)
    except Exception:  # noqa: BLE001
        pass
    so = build()
    lib = load(so)
    try:
        gdn_part(lib)
    except Exception:  # noqa: BLE001
        log("GDN part failed:", traceback.format_exc()[-3000:])
        RES["gdn_error"] = traceback.format_exc()[-3000:]
    try:
        if os.environ.get("SKIP_AR") != "1":
            ar_part(so)
    except Exception:  # noqa: BLE001
        log("AR part failed:", traceback.format_exc()[-3000:])
    t2 = next((r for r in RES.get("timing", []) if r["M"] == 2), None)
    if t2:
        RES["gate"] = {"fused_over_head_M2": t2["ratio"], "fused_over_head_plus_norm_M2": t2.get("ratio_incl_norm"),
                       "pass_60pct": t2["ratio"] <= 0.60}
        log("GATE", RES["gate"])
    log("RESULT " + json.dumps(RES, default=str))


if __name__ == "__main__":
    sys.exit(main())
