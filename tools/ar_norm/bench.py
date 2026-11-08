"""Plan 19 step 1: decode launches outside the MoE block on gfx90a (GCD 0 and 1, HIP graphs).

A  fused all-reduce + residual add + Gemma RMSNorm (CustomAllreduce.custom_all_reduce_add_rmsnorm)
   against the head's ca.custom_all_reduce -> Triton gemma_fused_add_rmsnorm(round_sum=True) and
   against the all-reduce alone; correctness over 500 iterations with GEMM-written inputs: the
   residual bitwise equal to the head, the output within 1 bf16 ulp of the head and of an fp64
   reference, ranks bitwise identical, repeatable over 100 replays, rows independent of M.
B  one launch for in_proj_qkvz + in_proj_ba (skinny_gemm_packed_hip.skinny_linear_pair) against
   the two skinny GEMMs, each followed by the fused unpack + Conv1D, bitwise.
C  fused_qk_gemma_rmsnorm_rope_gate on HIP against fused_qk_gemma_rmsnorm_with_gate +
   triton_mrope_fused, at the per-GCD TP=2 attention shapes.
Not on any serving path. Prints RESULT lines and a GATE summary.
"""
import json
import os
import sys
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HIDDEN = 2048
EPS = 1e-6
A_ROWS = (1, 2, 4, 8, 16, 32)
CHAIN = 40
REPLAYS = 200
CORR_ITERS = 500
MODEL = "Qwen/Qwen3.6-35B-A3B"
REV = "995ad96eacd98c81ed38be0c5b274b04031597b0"
TP = 2


def log(*a):
    print(*a, flush=True)


def load_text_config():
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(MODEL, "config.json", revision=REV, local_files_only=True)
        with open(path) as f:
            raw = json.load(f)
        return raw.get("text_config", raw)
    except Exception as e:  # noqa: BLE001
        log(f"WARNING: config not found ({e!r}); using plan shapes")
        return None


def graph_us(step, x0, ops, ctx, pg=None, warm=True):
    """us per step of `ops` chained step(x) -> x calls captured in one HIP graph."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    if warm:
        with torch.cuda.stream(s):
            x = x0
            for _ in range(3):
                x = step(x)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with ctx():
        with torch.cuda.graph(g, stream=s):
            x = x0
            for _ in range(ops):
                x = step(x)
    torch.cuda.synchronize()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    if pg is not None:
        dist.barrier(group=pg)
    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        g.replay()
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / (REPLAYS * ops) * 1e6
    del g
    return us


def ulp_err(out, ref64):
    """Error in units of one bf16 ulp of the reference; fraction within 1 ulp."""
    refb = ref64.to(torch.bfloat16).double()
    a = refb.abs().clamp_min(2.0 ** -126)
    ulp = torch.exp2(torch.floor(torch.log2(a)) - 7)
    e = (out.double() - ref64).abs() / ulp
    return e.max().item(), (e <= 1.0).double().mean().item()


def bits_hash(t):
    v = t.contiguous().view(torch.int16).to(torch.int64).flatten()
    w = (torch.arange(v.numel(), device=v.device, dtype=torch.int64) % 65521) + 1
    return int((v * w).sum().item())


# ------------------------------------------------------------------------------------------
# A: fused AR + add + norm, two ranks
# ------------------------------------------------------------------------------------------
def part_a(rank, pg, ca):
    import contextlib

    from sglang.kernels.ops.layernorm.minimax_m3_rmsnorm import gemma_fused_add_rmsnorm

    gen = torch.Generator(device="cuda")
    gen.manual_seed(1234)  # same weights on both ranks
    w_norm = (torch.randn(HIDDEN, generator=gen, device="cuda") * 0.3).to(torch.bfloat16)
    W = (torch.randn(HIDDEN, HIDDEN, generator=gen, device="cuda") * 0.02).to(torch.bfloat16)

    def head(x, res):
        h = ca.custom_all_reduce(x)
        return gemma_fused_add_rmsnorm(h, res, w_norm, EPS, round_sum=True)

    def fused(x, res):
        r = ca.custom_all_reduce_add_rmsnorm(x, res, w_norm, EPS, 1.0)
        if r is None:
            raise RuntimeError("fused AR+norm declined")
        return r

    res_tbl = {}
    row0_ref = None
    rows_ok = True
    for m in A_ROWS:
        out = {}
        torch.manual_seed(100 + rank)
        x0 = torch.randn(m, HIDDEN, device="cuda").to(torch.bfloat16)
        r0 = torch.randn(m, HIDDEN, device="cuda").to(torch.bfloat16)
        cap = ca.capture
        # timing: chains whose every AR input is the previous kernel's output
        out["ar_us"] = graph_us(lambda v: ca.custom_all_reduce(v), x0, CHAIN, cap, pg)
        st = {"res": r0}

        def head_step(v):
            o, st["res"] = head(v, st["res"])
            return o

        def fused_step(v):
            o, st["res"] = fused(v, st["res"])
            return o

        out["head_us"] = graph_us(head_step, x0, CHAIN, cap, pg)
        out["fused_us"] = graph_us(fused_step, x0, CHAIN, cap, pg)
        out["norm_us"] = graph_us(
            lambda v: gemma_fused_add_rmsnorm(v, r0, w_norm, EPS, round_sum=True)[0],
            x0, CHAIN, contextlib.nullcontext)

        # correctness: static inputs refreshed per iteration; a GEMM writes the AR input
        inp = torch.zeros(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
        resid = torch.zeros(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = inp @ W
        torch.cuda.synchronize()
        gh, gf = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with ca.capture():
            with torch.cuda.graph(gh, stream=s):
                hsum = ca.custom_all_reduce(inp @ W)
                h_out, h_res = gemma_fused_add_rmsnorm(hsum, resid, w_norm, EPS, round_sum=True)
            with torch.cuda.graph(gf, stream=s):
                f_out, f_res = fused(inp @ W, resid)
        torch.cuda.synchronize()
        res_bad = 0
        out_1ulp_min = 1.0
        head_max = ref_max = 0.0
        ref_1ulp_min = 1.0
        eager_res_bad = 0
        eager_out_maxdiff = 0.0
        hashes = []
        g = torch.Generator(device="cuda")
        for it in range(CORR_ITERS):
            g.manual_seed(7919 * it + 31 * m + rank)
            inp.copy_(torch.randn(m, HIDDEN, generator=g, device="cuda").to(torch.bfloat16))
            g.manual_seed(104729 * it + m)  # residual replicated across ranks, like the model's
            resid.copy_(torch.randn(m, HIDDEN, generator=g, device="cuda").to(torch.bfloat16))
            dist.barrier(group=pg)
            gh.replay()
            gf.replay()
            torch.cuda.synchronize()
            res_bad += int((h_res.view(torch.int16) != f_res.view(torch.int16)).sum().item())
            mh, fr = ulp_err(f_out, h_out.double())
            head_max, out_1ulp_min = max(head_max, mh), min(out_1ulp_min, fr)
            # fp64 reference from the same rounding points (bf16 AR sum, bf16 residual sum)
            sref = f_res.double()
            var = (sref * sref).mean(dim=-1, keepdim=True)
            ref = sref / torch.sqrt(var + EPS) * (1.0 + w_norm.double())
            mr, frr = ulp_err(f_out, ref)
            ref_max, ref_1ulp_min = max(ref_max, mr), min(ref_1ulp_min, frr)
            hashes.append(bits_hash(f_out) ^ (bits_hash(f_res) << 1))
            if it < 20:  # eager (unregistered) path on the same inputs
                hx = inp @ W
                eo, er = fused(hx, resid)
                ho, hr = head(hx, resid)
                torch.cuda.synchronize()
                eager_res_bad += int((er.view(torch.int16) != hr.view(torch.int16)).sum().item())
                eager_out_maxdiff = max(eager_out_maxdiff, (eo.float() - ho.float()).abs().max().item())
        # repeatability: same inputs, 100 replays
        first = (f_out.clone(), f_res.clone())
        rep_bad = 0
        for _ in range(100):
            dist.barrier(group=pg)
            gf.replay()
            torch.cuda.synchronize()
            rep_bad += int(not (torch.equal(f_out, first[0]) and torch.equal(f_res, first[1])))
        # row invariance: rows [0, m) of a fixed 32-row input give the same bits for every m
        g.manual_seed(555 + rank)
        big = torch.randn(32, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
        g.manual_seed(556)
        bigr = torch.randn(32, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
        dist.barrier(group=pg)
        ro, rr = fused(big[:m].contiguous(), bigr[:m].contiguous())
        torch.cuda.synchronize()
        if row0_ref is None:
            row0_ref = (ro[0].clone(), rr[0].clone())
        rows_ok &= torch.equal(ro[0], row0_ref[0]) and torch.equal(rr[0], row0_ref[1])
        if m == 32:
            out["row_prefix"] = (ro.clone(), rr.clone())
        del gh, gf
        out.update(res_bad=res_bad, head_max_ulp=head_max, head_within1=out_1ulp_min,
                   ref_max_ulp=ref_max, ref_within1=ref_1ulp_min, eager_res_bad=eager_res_bad,
                   eager_out_maxdiff=eager_out_maxdiff, rep_bad=rep_bad,
                   hash=hash(tuple(hashes)))
        res_tbl[m] = out
    # every m: rows equal to the 32-row call's first m rows
    full = res_tbl[32].pop("row_prefix")
    for m in A_ROWS:
        dist.barrier(group=pg)
        ro, rr = fused(big[:m].contiguous(), bigr[:m].contiguous())
        torch.cuda.synchronize()
        rows_ok &= torch.equal(ro, full[0][:m]) and torch.equal(rr, full[1][:m])
    for m in A_ROWS:
        res_tbl[m]["rows_ok"] = rows_ok
    return res_tbl


# ------------------------------------------------------------------------------------------
# B: packed qkvz + ba projection, one GCD
# ------------------------------------------------------------------------------------------
def part_b(tc):
    import contextlib

    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        fused_qkvzba_causal_conv1d_update_contiguous,
    )
    from sglang.kernels.ops.gemm import skinny_gemm_hip as sk
    from sglang.kernels.ops.gemm import skinny_gemm_packed_hip as pk

    if tc is not None:
        NK = tc["linear_num_key_heads"] // TP
        NV = tc["linear_num_value_heads"] // TP
        DK, DV, KW = tc["linear_key_head_dim"], tc["linear_value_head_dim"], tc["linear_conv_kernel_dim"]
    else:
        NK, NV, DK, DV, KW = 8, 16, 128, 128, 4
    QKV, VD = 2 * NK * DK + NV * DV, NV * DV
    N1, N2 = QKV + VD, 2 * NV
    log(f"B shapes: qkvz N={N1} ba N={N2} K={HIDDEN} qkv_dim={QKV} conv_width={KW} "
        f"cfg qkvz={sk._config_for(N1, HIDDEN)} ba={sk._config_for(N2, HIDDEN)} "
        f"packed-as-one={sk._config_for(N1 + N2, HIDDEN)}")
    LAYERS = 30
    gen = torch.Generator(device="cuda")
    gen.manual_seed(42)
    packed = [(torch.randn(N1 + N2, HIDDEN, generator=gen, device="cuda") * 0.02).to(torch.bfloat16)
              for _ in range(LAYERS)]
    conv_w = [(torch.randn(QKV, KW, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
              for _ in range(LAYERS)]
    slots = 16
    res = {}
    # the model aliases in_proj_qkvz / in_proj_ba to these row slices of the packed weight
    def today(x, l):
        return sk.skinny_linear(x, packed[l][:N1], None), sk.skinny_linear(x, packed[l][N1:], None)

    def pair(x, l):
        y = pk.skinny_linear_pair(x, packed[l], N1)
        return y[:, :N1], y[:, N1:]

    def one(x, l):
        y = sk.skinny_linear(x, packed[l], None)
        return y[:, :N1], y[:, N1:]

    def conv(qkvz, ba, l, state, idx):
        return fused_qkvzba_causal_conv1d_update_contiguous(
            qkvz, ba, state, conv_w[l], None, idx, qkv_dim=QKV, v_dim=VD,
            num_v_heads=NV, head_v_dim=DV, activation="silu")

    for m in (1, 2, 4, 8, 16):
        r = {}
        x0 = (torch.randn(m, HIDDEN, generator=gen, device="cuda")).to(torch.bfloat16)
        assert pk.applicable(x0, packed[0], N1)
        idx = torch.arange(m, dtype=torch.int32, device="cuda")
        state = torch.randn(slots, QKV, KW - 1, generator=gen, device="cuda").to(torch.bfloat16)
        for name, fn in (("today", today), ("pair", pair), ("one_cfg", one)):
            ctr = {"l": 0}

            def step(v, fn=fn, ctr=ctr):
                l = ctr["l"] % LAYERS
                ctr["l"] += 1
                q, b = fn(v, l)
                conv(q, b, l, state, idx)
                return v
            # gemm-only chain (each call reads its own layer's weight; L2 cold like the model)
            def gstep(v, fn=fn, ctr=ctr):
                l = ctr["l"] % LAYERS
                ctr["l"] += 1
                fn(v, l)
                return v
            r[f"{name}_us"] = graph_us(gstep, x0, LAYERS, contextlib.nullcontext)
            r[f"{name}_conv_us"] = graph_us(step, x0, LAYERS, contextlib.nullcontext)
        # bitwise: GEMM outputs and the conv outputs and state
        bad = {"qkvz": 0, "ba": 0, "conv": 0, "state": 0}
        one_bad = {"qkvz": 0, "ba": 0}
        for it in range(50):
            l = it % LAYERS
            x = torch.randn(m, HIDDEN, generator=gen, device="cuda").to(torch.bfloat16)
            st0 = torch.randn(slots, QKV, KW - 1, generator=gen, device="cuda").to(torch.bfloat16)
            q1, b1 = today(x, l)
            q2, b2 = pair(x, l)
            q3, b3 = one(x, l)
            bad["qkvz"] += int((q1 != q2).sum().item())
            bad["ba"] += int((b1 != b2).sum().item())
            one_bad["qkvz"] += int((q1 != q3).sum().item())
            one_bad["ba"] += int((b1 != b3).sum().item())
            s1, s2 = st0.clone(), st0.clone()
            o1 = conv(q1, b1, l, s1, idx)
            o2 = conv(q2, b2, l, s2, idx)
            bad["conv"] += sum(int((u.view(torch.int16) != w.view(torch.int16)).sum().item())
                               for u, w in zip(o1, o2))
            bad["state"] += int((s1.view(torch.int16) != s2.view(torch.int16)).sum().item())
        r["bad"], r["one_cfg_bad"] = bad, one_bad
        res[m] = r
    return res


# ------------------------------------------------------------------------------------------
# C: fused QK norm + mrope + gate, one GCD
# ------------------------------------------------------------------------------------------
def part_c(tc):
    import contextlib

    from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
        fused_qk_gemma_rmsnorm_rope_gate,
    )
    from sglang.kernels.ops.attention.rotary_triton import triton_mrope_fused
    from sglang.srt.models.utils import fused_qk_gemma_rmsnorm_with_gate

    if tc is not None:
        NH, NKV = tc["num_attention_heads"] // TP, max(1, tc["num_key_value_heads"] // TP)
        HD = tc.get("head_dim", 256)
        rp = tc.get("rope_parameters") or tc.get("rope_scaling") or {}
        RD = int(HD * rp.get("partial_rotary_factor", tc.get("partial_rotary_factor", 0.25)))
        sec = rp.get("mrope_section", [11, 11, 10])
        inter = bool(rp.get("mrope_interleaved", True))
        base = rp.get("rope_theta", tc.get("rope_theta", 10_000_000))
    else:
        NH, NKV, HD, RD, sec, inter, base = 8, 1, 256, 64, [11, 11, 10], True, 10_000_000
    half = RD // 2
    # axis map exactly as MRotaryEmbedding._build_axis_map
    if inter:
        axes = [0] * half
        for axis in (1, 2):
            for lane in range(axis, min(3 * sec[axis], half), 3):
                axes[lane] = axis
    else:
        axes = [a for a, n in enumerate(sec) for _ in range(n)]
    axis_map = torch.tensor(axes, dtype=torch.long, device="cuda")
    maxpos = 65536
    inv = 1.0 / (base ** (torch.arange(0, RD, 2, dtype=torch.float32, device="cuda") / RD))
    fr = torch.outer(torch.arange(maxpos, dtype=torch.float32, device="cuda"), inv)
    cache = torch.cat((fr.cos(), fr.sin()), dim=-1)  # fp32 on HIP
    log(f"C shapes: q_heads={NH} kv_heads={NKV} head_dim={HD} rotary={RD} mrope_section={sec} "
        f"interleaved={inter} theta={base}")
    gen = torch.Generator(device="cuda")
    gen.manual_seed(7)
    qw = (torch.randn(HD, generator=gen, device="cuda") * 0.3).to(torch.bfloat16)
    kw = (torch.randn(HD, generator=gen, device="cuda") * 0.3).to(torch.bfloat16)
    q_size, kv = NH * HD, NKV * HD

    def today(qkv, pos):
        T = qkv.shape[0]
        q_gate, k, _ = qkv.split([q_size * 2, kv, kv], dim=-1)
        qf, kf, gf = fused_qk_gemma_rmsnorm_with_gate(q_gate, k, qw, kw, EPS, HD, NH)
        q, kk = qf.view(T, -1), kf.view(T, -1)
        triton_mrope_fused(q, kk, cache, pos, sec, HD, RD, inter, False, True, None)
        return q, kk, gf.view(T, -1)

    def fused(qkv, pos):
        T = qkv.shape[0]
        q_gate, k, _ = qkv.split([q_size * 2, kv, kv], dim=-1)
        qo, ko, go = fused_qk_gemma_rmsnorm_rope_gate(
            q_gate, k, qw, kw, cache, pos, EPS, NH, NKV, HD, RD, has_gate=True,
            mrope_axis_map=axis_map)
        return qo.view(T, -1), ko.view(T, -1), go.view(T, -1)

    res = {}
    for m in (1, 2, 4, 8):
        r = {}
        qkv = torch.randn(m, q_size * 2 + 2 * kv, generator=gen, device="cuda").to(torch.bfloat16)
        maxd = {"q": 0.0, "k": 0.0, "gate": 0.0}
        neq = {"q": 0, "k": 0, "gate": 0}
        for it in range(200):
            qkv.copy_(torch.randn(m, q_size * 2 + 2 * kv, generator=gen, device="cuda"))
            if it % 2 == 0:  # text: three equal rows
                p = torch.randint(0, maxpos, (m,), generator=gen, device="cuda")
                pos = p.unsqueeze(0).expand(3, m).contiguous()
            else:
                pos = torch.randint(0, maxpos, (3, m), generator=gen, device="cuda")
            a, b = today(qkv, pos), fused(qkv, pos)
            for n, u, w in zip(("q", "k", "gate"), a, b):
                maxd[n] = max(maxd[n], (u.float() - w.float()).abs().max().item())
                neq[n] += int((u.view(torch.int16) != w.view(torch.int16)).sum().item())
        pos = torch.arange(m, device="cuda").unsqueeze(0).expand(3, m).contiguous()
        r["today_us"] = graph_us(lambda v: (today(qkv, pos), v)[1], qkv, 10, contextlib.nullcontext)
        r["fused_us"] = graph_us(lambda v: (fused(qkv, pos), v)[1], qkv, 10, contextlib.nullcontext)
        r["maxdiff"], r["neq"] = maxd, neq
        res[m] = r
    return res


def worker(rank, world, tc):
    import faulthandler

    faulthandler.dump_traceback_later(1500, exit=True)
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method="tcp://127.0.0.1:29611")
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    from sglang.srt.distributed.device_communicators import custom_all_reduce as car
    import sglang.srt.distributed.device_communicators.custom_all_reduce_ops as ops

    pg = dist.new_group([0, 1], backend="gloo")
    ca = car.CustomAllreduce(group=pg, device=rank)
    if rank == 0:
        log(f"device {torch.cuda.get_device_name()} arch "
            f"{torch.cuda.get_device_properties(0).gcnArchName} ca.disabled={ca.disabled} "
            f"deterministic={ca.use_amd_deterministic_impl} "
            f"has_fused_op={ops.HAS_ALL_REDUCE_ADD_RMSNORM}")
    out = {}
    try:
        out["A"] = part_a(rank, pg, ca)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        out["A"] = f"ERR {type(e).__name__}: {e}"
    dist.barrier()
    gathered = [None] * world
    dist.all_gather_object(gathered, out["A"])
    if rank == 0:
        report_a(gathered)
        for part, fn in (("B", part_b), ("C", part_c)):
            try:
                out[part] = fn(tc)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                out[part] = f"ERR {type(e).__name__}: {e}"
        report_bc(out["B"], out["C"])
        gate(gathered[0], out["B"], out["C"], gathered)
    dist.barrier()
    faulthandler.cancel_dump_traceback_later()
    dist.destroy_process_group()


def report_a(g):
    if any(isinstance(v, str) for v in g):
        log("RESULT A", [v for v in g if isinstance(v, str)])
        return
    log("RESULT A  us per boundary, max over ranks; chain of 40 in one HIP graph, 200 replays")
    log("RESULT A  M   ar_only  head(ar+norm)  fused  fused-ar  head-fused  norm_alone")
    for m in A_ROWS:
        t = {k: max(r[m][k] for r in g) for k in ("ar_us", "head_us", "fused_us", "norm_us")}
        log(f"RESULT A {m:3d} {t['ar_us']:8.2f} {t['head_us']:13.2f} {t['fused_us']:7.2f} "
            f"{t['fused_us'] - t['ar_us']:8.2f} {t['head_us'] - t['fused_us']:10.2f} "
            f"{t['norm_us']:10.2f}")
    log(f"RESULT A  correctness ({CORR_ITERS} GEMM-written iterations per M, graph mode)")
    for m in A_ROWS:
        rs = [r[m] for r in g]
        log(f"RESULT A {m:3d} residual!=head(elems)={sum(r['res_bad'] for r in rs)} "
            f"out_vs_head max_ulp={max(r['head_max_ulp'] for r in rs):.2f} "
            f"within1ulp_min={min(r['head_within1'] for r in rs):.6f} "
            f"out_vs_fp64 max_ulp={max(r['ref_max_ulp'] for r in rs):.2f} "
            f"within1ulp_min={min(r['ref_within1'] for r in rs):.6f} "
            f"eager: residual!=head={sum(r['eager_res_bad'] for r in rs)} "
            f"out_maxabs={max(r['eager_out_maxdiff'] for r in rs):.3g} "
            f"ranks_identical={rs[0]['hash'] == rs[1]['hash']} "
            f"repeat_bad={sum(r['rep_bad'] for r in rs)}/200 rows_invariant={all(r['rows_ok'] for r in rs)}")


def report_bc(b, c):
    if isinstance(b, str):
        log("RESULT B", b)
    else:
        log("RESULT B  us per GDN layer, chain of 30 layers (own weights) in one HIP graph")
        log("RESULT B  M  today(2 GEMM)  pair(1 launch)  one_cfg  | +conv: today  pair")
        for m, r in b.items():
            log(f"RESULT B {m:2d} {r['today_us']:12.2f} {r['pair_us']:14.2f} {r['one_cfg_us']:8.2f}"
                f" | {r['today_conv_us']:12.2f} {r['pair_conv_us']:6.2f}  "
                f"bitwise_mismatch pair={r['bad']} one_cfg={r['one_cfg_bad']}")
    if isinstance(c, str):
        log("RESULT C", c)
    else:
        log("RESULT C  us per attention layer, chain of 10 in one HIP graph")
        for m, r in c.items():
            log(f"RESULT C {m:2d} today(2 kernels)={r['today_us']:.2f} fused={r['fused_us']:.2f} "
                f"maxabs={r['maxdiff']} mismatched_elems={r['neq']}")


def gate(a0, b, c, ga):
    verdict = {}
    a_ok = not isinstance(a0, str)
    if a_ok:
        m = 2
        ar = max(r[m]["ar_us"] for r in ga)
        fu = max(r[m]["fused_us"] for r in ga)
        hd = max(r[m]["head_us"] for r in ga)
        cost_ok = fu <= ar + 1.5
        corr_ok = all(
            r[mm]["res_bad"] == 0 and r[mm]["eager_res_bad"] == 0 and r[mm]["rep_bad"] == 0
            and r[mm]["rows_ok"] and r[mm]["head_within1"] >= 0.999 and r[mm]["head_max_ulp"] <= 2.0
            for r in ga for mm in A_ROWS
        ) and all(ga[0][mm]["hash"] == ga[1][mm]["hash"] for mm in A_ROWS)
        verdict["A"] = (cost_ok and corr_ok, f"M=2 fused-ar={fu - ar:.2f}us (<=1.5) "
                        f"saves {hd - fu:.2f}us/boundary; correctness={corr_ok}")
        save_a1, save_a2 = 39 * (hd - fu), 40 * (hd - fu)
    else:
        verdict["A"] = (False, a0)
        save_a1 = save_a2 = 0.0
    if isinstance(b, str):
        verdict["B"], save_b = (False, b), 0.0
    else:
        bit = all(v == 0 for r in b.values() for v in r["bad"].values())
        save_b = 30 * (b[2]["today_us"] - b[2]["pair_us"])
        verdict["B"] = (bit, f"bitwise={bit} M=2 saves {b[2]['today_us'] - b[2]['pair_us']:.2f}us/layer")
    if isinstance(c, str):
        verdict["C"], save_c = (False, c), 0.0
    else:
        md = max(v for r in c.values() for v in r["maxdiff"].values())
        save_c = 10 * (c[2]["today_us"] - c[2]["fused_us"])
        verdict["C"] = (md <= 0.0625, f"max abs diff {md:.4g} M=2 saves "
                        f"{c[2]['today_us'] - c[2]['fused_us']:.2f}us/layer")
    total = (save_a1 if verdict["A"][0] else 0) + (save_b if verdict["B"][0] else 0) + \
        (save_c if verdict["C"][0] else 0)
    for k, (ok, why) in verdict.items():
        log(f"GATE {k}: {'PASS' if ok else 'FAIL'} {why}")
    log(f"GATE projected saving per bs=2 step: A1(39)={save_a1:.0f}us A2(40)={save_a2:.0f}us "
        f"B(30)={save_b:.0f}us C(10)={save_c:.0f}us; passing parts without A2={total:.0f}us "
        f"(need >=250) -> {'PASS' if total >= 250 else 'FAIL'}")


def main():
    log("devices:", torch.cuda.device_count())
    tc = load_text_config()
    mp.spawn(worker, args=(2, tc), nprocs=2, join=True)


if __name__ == "__main__":
    main()
