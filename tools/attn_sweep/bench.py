"""Plan 14 step 1: sweep Triton decode and extend attention launch geometry on one gfx90a GCD.

Shapes are one TP=2 rank of Qwen3.6-35B-A3B's ten full-attention layers: 8 q heads, 1 kv
head, head_dim 256, bf16 KV, page size 1. Every timing replays a HIP graph that calls the
kernel once per layer on ten separate KV buffers (so L2 does not hold one layer's KV across
calls, as in the server) and reports us per layer call.

Decode: the split cap (--triton-attention-num-kv-splits, HIP default 16) x the grouped stage-1
geometry (BLOCK_N, num_warps, waves_per_eu). The per-sequence split count comes from the same
get_num_kv_splits_triton call the backend makes. Extend: BLOCK_M x BLOCK_N x num_warps x
waves_per_eu for one request of 200/600/1500 new tokens on a 3k prefix.

Every config is checked against the current config (max abs diff) and against an fp32
reference, and run twice to check that its output is bitwise repeatable.
"""

import math
import sys
import traceback

import torch
import triton

from sglang.kernels.ops.attention import decode_attention as da
from sglang.kernels.ops.attention import extend_attention as ea
from sglang.kernels.ops.attention.metadata import get_num_kv_splits_triton

DEV = torch.device("cuda", 0)
HQ, HKV, D = 8, 1, 256
LAYERS = 10
SM_SCALE = 1.0 / math.sqrt(D)
CORES = torch.cuda.get_device_properties(0).multi_processor_count


class Override:
    """Wraps a Triton kernel so that kernel[grid](...) launches with some kwargs replaced."""

    def __init__(self, kernel, overrides):
        self.kernel, self.overrides = kernel, overrides

    def __getitem__(self, grid):
        launch = self.kernel[grid]
        ov = self.overrides

        def run(*args, **kwargs):
            kwargs.update(ov)
            return launch(*args, **kwargs)

        return run


def graph_us(fn, reps=20):
    """us per fn() (fn runs LAYERS kernel calls; result is per call)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(7):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            g.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000.0 / reps / LAYERS)
    times.sort()
    return times[len(times) // 2]


# ----------------------------------------------------------------------------- decode


def decode_setup(bs, ctx, gen):
    tot = bs * ctx
    ks = [torch.randn(tot, HKV, D, device=DEV, generator=gen).to(torch.bfloat16) for _ in range(LAYERS)]
    vs = [torch.randn(tot, HKV, D, device=DEV, generator=gen).to(torch.bfloat16) for _ in range(LAYERS)]
    q = torch.randn(bs, HQ, D, device=DEV, generator=gen).to(torch.bfloat16)
    seq_lens = torch.full((bs,), ctx, dtype=torch.int32, device=DEV)
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=DEV)
    kv_indptr[1:] = torch.cumsum(seq_lens, 0)
    kv_indices = torch.arange(tot, dtype=torch.int32, device=DEV)
    return dict(q=q, ks=ks, vs=vs, seq_lens=seq_lens, kv_indptr=kv_indptr, kv_indices=kv_indices)


def decode_ref(s, layer=0):
    bs, ctx = s["q"].shape[0], s["seq_lens"][0].item()
    k = s["ks"][layer].float().view(bs, ctx, D)
    v = s["vs"][layer].float().view(bs, ctx, D)
    q = s["q"].float()
    att = torch.einsum("bhd,btd->bht", q, k) * SM_SCALE
    return torch.einsum("bht,btd->bhd", att.softmax(-1), v)


def decode_fn(s, cap, out):
    bs = s["q"].shape[0]
    num_kv_splits = torch.empty(bs, dtype=torch.int32, device=DEV)
    get_num_kv_splits_triton[(1,)](
        num_kv_splits, s["seq_lens"], bs, 1, HQ, HKV, cap, CORES, MAX_NUM_SEQ=256
    )
    logits = torch.empty(bs, HQ, cap, D, dtype=torch.float32, device=DEV)
    lse = torch.empty(bs, HQ, cap, dtype=torch.float32, device=DEV)

    def fn():
        for l in range(LAYERS):
            da.decode_attention_fwd(
                s["q"], s["ks"][l], s["vs"][l], out, s["kv_indptr"], s["kv_indices"],
                logits, lse, num_kv_splits, cap, SM_SCALE, 1.0, 1.0,
            )

    return fn, num_kv_splits


DECODE_GEOMS = [  # (BLOCK_N, num_warps, waves_per_eu); first = current HIP config
    (32, 4, 1),
    (32, 4, 2),
    (64, 4, 1),
    (64, 4, 2),
    (16, 4, 1),
    (32, 2, 1),
    (32, 2, 2),
    (64, 8, 1),
]
CAPS = [16, 32, 64, 128]


def decode_sweep():
    print(f"# device {torch.cuda.get_device_name(0)} CUs={CORES} "
          f"triton {triton.__version__} torch {torch.__version__}", flush=True)
    orig_s1 = da._fwd_grouped_kernel_stage1
    gen = torch.Generator(device=DEV).manual_seed(0)
    rows = []
    print("\n## decode: us per layer call (stage1+stage2), 8q/1kv heads, D=256", flush=True)
    print("bs ctx  | " + " ".join(f"cap{c}" for c in CAPS) + " | geom(BLOCK_N,warps,wpe) splits", flush=True)
    for bs in (1, 2, 4):
        for ctx in (2048, 4096, 8192):
            s = decode_setup(bs, ctx, gen)
            ref = decode_ref(s)
            base_out = None
            for geom in DECODE_GEOMS:
                bn, nw, wpe = geom
                da._fwd_grouped_kernel_stage1 = Override(
                    orig_s1, {"BLOCK_N": bn, "num_warps": nw, "waves_per_eu": wpe}
                )
                cells, splits = [], []
                for cap in CAPS:
                    try:
                        out = torch.empty(bs, HQ, D, dtype=torch.bfloat16, device=DEV)
                        fn, nks = decode_fn(s, cap, out)
                        fn()
                        torch.cuda.synchronize()
                        o1 = out.clone()
                        fn()
                        torch.cuda.synchronize()
                        rep = torch.equal(o1, out)
                        # layer LAYERS-1 is what out holds; compare layer 0 by a single call
                        da.decode_attention_fwd(
                            s["q"], s["ks"][0], s["vs"][0], out, s["kv_indptr"], s["kv_indices"],
                            torch.empty(bs, HQ, cap, D, dtype=torch.float32, device=DEV),
                            torch.empty(bs, HQ, cap, dtype=torch.float32, device=DEV),
                            nks, cap, SM_SCALE, 1.0, 1.0,
                        )
                        o0 = out.float()
                        if base_out is None:
                            base_out = o0.clone()
                        d_base = (o0 - base_out).abs().max().item()
                        d_ref = (o0 - ref).abs().max().item()
                        us = graph_us(fn)
                        cells.append(f"{us:6.1f}")
                        splits.append(int(nks[0].item()))
                        rows.append((bs, ctx, geom, cap, us, d_base, d_ref, rep))
                        if not rep or d_base > 2e-2:
                            print(f"  !! bs{bs} ctx{ctx} {geom} cap{cap}: rep={rep} "
                                  f"d_base={d_base:.2e} d_ref={d_ref:.2e}", flush=True)
                    except Exception as e:  # noqa: BLE001
                        cells.append("  fail")
                        print(f"  !! bs{bs} ctx{ctx} {geom} cap{cap}: {type(e).__name__}: "
                              f"{str(e).splitlines()[0][:160]}", flush=True)
                print(f"{bs} {ctx:5d} | " + " ".join(cells) + f" | {geom} {splits}", flush=True)
            del s
            torch.cuda.empty_cache()
    da._fwd_grouped_kernel_stage1 = orig_s1

    # stage-2 alone at the current geometry: the merge loop is MAX_KV_SPLITS long
    print("\n## decode stage2 alone (us per call), current stage-1 geometry", flush=True)
    for bs, ctx in ((2, 4096), (4, 8192)):
        s = decode_setup(bs, ctx, gen)
        cells = []
        for cap in CAPS:
            out = torch.empty(bs, HQ, D, dtype=torch.bfloat16, device=DEV)
            fn, nks = decode_fn(s, cap, out)
            fn()
            logits = torch.randn(bs, HQ, cap, D, dtype=torch.float32, device=DEV)
            lse = torch.randn(bs, HQ, cap, dtype=torch.float32, device=DEV)

            def f2():
                for _ in range(LAYERS):
                    da._decode_softmax_reducev_fwd(
                        logits, lse, s["q"], out, 1.0, s["vs"][0], s["kv_indptr"], nks, cap
                    )

            cells.append(f"cap{cap}={graph_us(f2):5.1f}")
        print(f"bs{bs} ctx{ctx}: " + " ".join(cells), flush=True)
        del s

    print("\n## decode summary: best per shape vs current (geom (32,4,1), cap16)", flush=True)
    worst_err = max((r[5] for r in rows), default=0)
    nonrep = [r for r in rows if not r[7]]
    for bs in (1, 2, 4):
        for ctx in (2048, 4096, 8192):
            cur = [r for r in rows if r[0] == bs and r[1] == ctx and r[2] == DECODE_GEOMS[0] and r[3] == 16]
            sub = [r for r in rows if r[0] == bs and r[1] == ctx and r[5] <= 2e-2 and r[7]]
            if not cur or not sub:
                continue
            best = min(sub, key=lambda r: r[4])
            c64 = [r for r in rows if r[0] == bs and r[1] == ctx and r[2] == DECODE_GEOMS[0] and r[3] == 64]
            c64s = f"{cur[0][4] / c64[0][4]:.2f}x" if c64 else "-"
            print(f"bs{bs} ctx{ctx}: current {cur[0][4]:.1f} us, best {best[4]:.1f} us "
                  f"{best[2]} cap{best[3]} = {cur[0][4] / best[4]:.2f}x; geom0 cap64 {c64s}",
                  flush=True)
    print(f"max |diff| vs current over all configs: {worst_err:.3e}; non-repeatable: {len(nonrep)}",
          flush=True)


# ----------------------------------------------------------------------------- extend


def extend_setup(ext, prefix, gen):
    pool = prefix + 8
    kbs = [torch.randn(pool, HKV, D, device=DEV, generator=gen).to(torch.bfloat16) for _ in range(LAYERS)]
    vbs = [torch.randn(pool, HKV, D, device=DEV, generator=gen).to(torch.bfloat16) for _ in range(LAYERS)]
    q = torch.randn(ext, HQ, D, device=DEV, generator=gen).to(torch.bfloat16)
    ke = torch.randn(ext, HKV, D, device=DEV, generator=gen).to(torch.bfloat16)
    ve = torch.randn(ext, HKV, D, device=DEV, generator=gen).to(torch.bfloat16)
    qo_indptr = torch.tensor([0, ext], dtype=torch.int32, device=DEV)
    kv_indptr = torch.tensor([0, prefix], dtype=torch.int32, device=DEV)
    kv_indices = torch.arange(prefix, dtype=torch.int32, device=DEV)
    return dict(q=q, ke=ke, ve=ve, kbs=kbs, vbs=vbs, qo_indptr=qo_indptr,
                kv_indptr=kv_indptr, kv_indices=kv_indices, ext=ext, prefix=prefix)


def extend_ref(s, layer=0):
    ext, prefix = s["ext"], s["prefix"]
    k = torch.cat([s["kbs"][layer][:prefix], s["ke"]], 0).float()[:, 0]  # [T, D]
    v = torch.cat([s["vbs"][layer][:prefix], s["ve"]], 0).float()[:, 0]
    q = s["q"].float()  # [ext, H, D]
    att = torch.einsum("mhd,td->hmt", q, k) * SM_SCALE
    t = torch.arange(prefix + ext, device=DEV)
    m = torch.arange(ext, device=DEV)
    att = att.masked_fill((t[None, :] > (prefix + m[:, None]))[None], float("-inf"))
    return torch.einsum("hmt,td->mhd", att.softmax(-1), v)


def extend_call(s, out, layer):
    ea.extend_attention_fwd(
        s["q"], s["ke"], s["ve"], out, s["kbs"][layer], s["vbs"][layer], s["qo_indptr"],
        s["kv_indptr"], s["kv_indices"], None, True, None, s["ext"], 1.0, 1.0,
        sm_scale=SM_SCALE,
    )


EXT_GEOMS = [  # (BLOCK_M, BLOCK_N, num_warps, waves_per_eu); first = current gfx90a config
    (64, 64, 4, 1),
    (64, 64, 4, 2),
    (64, 32, 4, 1),
    (32, 64, 4, 1),
    (32, 64, 2, 1),
    (32, 64, 2, 2),
    (32, 32, 2, 1),
    (32, 32, 4, 1),
    (16, 64, 1, 1),
    (16, 64, 2, 1),
    (16, 32, 1, 2),
    (128, 64, 8, 1),
    (128, 32, 8, 1),
]


def extend_sweep():
    orig_bs = ea._get_block_sizes_for_extend_attention
    orig_k = ea._fwd_kernel
    gen = torch.Generator(device=DEV).manual_seed(1)
    rows = []
    print("\n## extend: us per layer call, 1 request, 3072-token prefix, 8q/1kv, D=256", flush=True)
    print("geom(BLOCK_M,BLOCK_N,warps,wpe) | ext200 ext600 ext1500 | max|d| vs current", flush=True)
    setups = {ext: extend_setup(ext, 3072, gen) for ext in (200, 600, 1500)}
    refs = {ext: extend_ref(s) for ext, s in setups.items()}
    base = {}
    for geom in EXT_GEOMS:
        bm, bn, nw, wpe = geom

        def bsz(Lq, Lv, bm=bm, bn=bn, nw=nw):
            dm, dpe, dv, _, _, _ = orig_bs(Lq, Lv)
            return dm, dpe, dv, bm, bn, nw

        ea._get_block_sizes_for_extend_attention = bsz
        ea._fwd_kernel = Override(orig_k, {"waves_per_eu": wpe})
        cells, dmax = [], 0.0
        for ext, s in setups.items():
            try:
                out = torch.empty(ext, HQ, D, dtype=torch.bfloat16, device=DEV)
                extend_call(s, out, 0)
                torch.cuda.synchronize()
                o0 = out.float().clone()
                extend_call(s, out, 0)
                rep = torch.equal(o0, out.float())
                if ext not in base:
                    base[ext] = o0
                d_base = (o0 - base[ext]).abs().max().item()
                d_ref = (o0 - refs[ext]).abs().max().item()
                dmax = max(dmax, d_base)

                def fn(s=s, out=out):
                    for l in range(LAYERS):
                        extend_call(s, out, l)

                us = graph_us(fn, reps=5)
                cells.append(f"{us:7.1f}")
                rows.append((geom, ext, us, d_base, d_ref, rep))
                if not rep or d_base > 2e-2:
                    print(f"  !! {geom} ext{ext}: rep={rep} d_base={d_base:.2e} d_ref={d_ref:.2e}",
                          flush=True)
            except Exception as e:  # noqa: BLE001
                cells.append("   fail")
                print(f"  !! {geom} ext{ext}: {type(e).__name__}: {str(e).splitlines()[0][:160]}",
                      flush=True)
        print(f"{str(geom):18s} | " + " ".join(cells) + f" | {dmax:.2e}", flush=True)
    ea._get_block_sizes_for_extend_attention = orig_bs
    ea._fwd_kernel = orig_k

    print("\n## extend summary: best per length vs current (64,64,4,1)", flush=True)
    for ext in (200, 600, 1500):
        cur = [r for r in rows if r[1] == ext and r[0] == EXT_GEOMS[0]]
        sub = [r for r in rows if r[1] == ext and r[3] <= 2e-2 and r[5]]
        if cur and sub:
            best = min(sub, key=lambda r: r[2])
            print(f"ext{ext}: current {cur[0][2]:.1f} us, best {best[2]:.1f} us {best[0]} "
                  f"= {cur[0][2] / best[2]:.2f}x (d_ref {best[4]:.2e})", flush=True)
    # one geometry for all lengths: geometric-mean speedup
    print("one geometry for all lengths (geomean speedup vs current):", flush=True)
    for geom in EXT_GEOMS:
        sp = []
        for ext in (200, 600, 1500):
            cur = [r for r in rows if r[1] == ext and r[0] == EXT_GEOMS[0]]
            g = [r for r in rows if r[1] == ext and r[0] == geom and r[3] <= 2e-2 and r[5]]
            if cur and g:
                sp.append(cur[0][2] / g[0][2])
        if len(sp) == 3:
            print(f"  {geom}: {math.prod(sp) ** (1 / 3):.2f}x ({' '.join(f'{x:.2f}' for x in sp)})",
                  flush=True)


if __name__ == "__main__":
    torch.cuda.set_device(0)
    for part in (decode_sweep, extend_sweep):
        try:
            part()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            sys.stdout.flush()
