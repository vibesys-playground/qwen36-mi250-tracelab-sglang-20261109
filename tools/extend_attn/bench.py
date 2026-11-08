"""Plan 17 step 1: GQA-packed split-KV extend attention vs the current extend kernel on gfx90a.

Shapes are one TP=2 rank of Qwen3.6-35B-A3B's ten full-attention layers: 8 q heads, 1 kv head,
head_dim 256, bf16 KV, page size 1. A timing replays a HIP graph that calls the op once per
layer on ten separate KV buffers and reports us per layer call (median of 7).

Phase A sweeps the new kernel's geometry at P=2944, N in {184, 617}. Phase B runs the two best
geometries with several split targets and partial dtypes over the grid P in {2944, 6976} x
N in {16, 184, 617, 1024, 1708}, next to the current kernel. Correctness for each phase-B
variant: max |diff| vs an fp32 reference (must be <= 1.5x the current kernel's), 20 eager
repeats bitwise identical, and one sequence bitwise identical alone and in a batch of 2.
Register use and spills come from the compiled kernels' AMDGPU metadata.
"""

import math
import re
import sys
import time
import traceback

import torch
import triton

from sglang.kernels.ops.attention import extend_attention as ea
from sglang.kernels.ops.attention import extend_attention_gqa_split as gs

DEV = torch.device("cuda", 0)
HQ, HKV, D = 8, 1, 256
LAYERS = 10
SM_SCALE = 1.0 / math.sqrt(D)
CORES = torch.cuda.get_device_properties(0).multi_processor_count
PS = (2944, 6976)
NS = (16, 184, 617, 1024, 1708)
T0 = time.time()


def log(*a):
    print(*a, flush=True)


def graph_us(fn, reps=5):
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
    del g
    return times[len(times) // 2]


_META_KEYS = ("vgpr_count", "agpr_count", "vgpr_spill_count", "sgpr_spill_count",
              "private_segment_fixed_size", "group_segment_fixed_size")


def kinfo(k):
    """Register use and spills of a compiled Triton kernel (AMDGPU metadata)."""
    if k is None:
        return "?"
    out = {}
    try:
        asm = k.asm.get("amdgcn", "")
        for key in _META_KEYS:
            m = re.search(r"\." + key + r":\s+(\d+)", asm)
            if m:
                out[key] = int(m.group(1))
    except Exception:  # noqa: BLE001
        pass
    for a in ("n_regs", "n_spills"):
        try:
            out[a] = getattr(k, a)
        except Exception:  # noqa: BLE001
            pass
    return (f"vgpr={out.get('vgpr_count')} agpr={out.get('agpr_count')} "
            f"vspill={out.get('vgpr_spill_count')} sspill={out.get('sgpr_spill_count')} "
            f"scratch={out.get('private_segment_fixed_size')}B lds={out.get('group_segment_fixed_size')}B "
            f"n_spills={out.get('n_spills')}")


class Capture:
    """Wraps a Triton kernel so kernel[grid](...) records the compiled kernel it launched."""

    def __init__(self, kernel):
        self.kernel, self.last = kernel, None

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            self.last = launch(*args, **kwargs)
            return self.last

        return run


# ----------------------------------------------------------------------------- data


def make_case(seqs, gen):
    """seqs: list of (prefix, new). Prefix KV of all seqs sits in one pool per layer."""
    tot_p = sum(p for p, _ in seqs)
    tot_n = sum(n for _, n in seqs)
    pool = tot_p + 8
    kbs = [torch.randn(pool, HKV, D, device=DEV, generator=gen).to(torch.bfloat16) for _ in range(LAYERS)]
    vbs = [torch.randn(pool, HKV, D, device=DEV, generator=gen).to(torch.bfloat16) for _ in range(LAYERS)]
    q = torch.randn(tot_n, HQ, D, device=DEV, generator=gen).to(torch.bfloat16)
    ke = torch.randn(tot_n, HKV, D, device=DEV, generator=gen).to(torch.bfloat16)
    ve = torch.randn(tot_n, HKV, D, device=DEV, generator=gen).to(torch.bfloat16)
    qo, kv, inds, slot = [0], [0], [], 0
    for p, n in seqs:
        inds.append(torch.arange(slot, slot + p, dtype=torch.int32, device=DEV))
        slot += p
        qo.append(qo[-1] + n)
        kv.append(kv[-1] + p)
    return dict(
        seqs=seqs, q=q, ke=ke, ve=ve, kbs=kbs, vbs=vbs,
        qo_indptr=torch.tensor(qo, dtype=torch.int32, device=DEV),
        kv_indptr=torch.tensor(kv, dtype=torch.int32, device=DEV),
        kv_indices=torch.cat(inds) if inds else torch.zeros(0, dtype=torch.int32, device=DEV),
        qo=qo, kv=kv,
    )


def sub_case(c, i):
    """Sequence i of case c as a batch of one, on the same buffers (same bits)."""
    p, n = c["seqs"][i]
    q0, q1, k0 = c["qo"][i], c["qo"][i + 1], c["kv"][i]
    return dict(
        seqs=[(p, n)], q=c["q"][q0:q1], ke=c["ke"][q0:q1], ve=c["ve"][q0:q1],
        kbs=c["kbs"], vbs=c["vbs"],
        qo_indptr=torch.tensor([0, n], dtype=torch.int32, device=DEV),
        kv_indptr=torch.tensor([0, p], dtype=torch.int32, device=DEV),
        kv_indices=c["kv_indices"][k0:k0 + p].clone(),
        qo=[0, n], kv=[0, p],
    )


def ref(c, i=0, layer=0):
    p, n = c["seqs"][i]
    q0, k0 = c["qo"][i], c["kv"][i]
    idx = c["kv_indices"][k0:k0 + p].long()
    k = torch.cat([c["kbs"][layer][idx], c["ke"][q0:q0 + n]], 0).float()[:, 0]
    v = torch.cat([c["vbs"][layer][idx], c["ve"][q0:q0 + n]], 0).float()[:, 0]
    q = c["q"][q0:q0 + n].float()
    att = torch.einsum("mhd,td->hmt", q, k) * SM_SCALE
    t = torch.arange(p + n, device=DEV)
    m = torch.arange(n, device=DEV)
    att = att.masked_fill((t[None, :] > (p + m[:, None]))[None], float("-inf"))
    return torch.einsum("hmt,td->mhd", att.softmax(-1), v)


# ----------------------------------------------------------------------------- ops

CUR_CAP = Capture(ea._fwd_kernel)
ea._fwd_kernel = CUR_CAP


def cur_call(c, out, layer):
    ea.extend_attention_fwd(
        c["q"], c["ke"], c["ve"], out, c["kbs"][layer], c["vbs"][layer], c["qo_indptr"],
        c["kv_indptr"], c["kv_indices"], None, True, None, max(n for _, n in c["seqs"]),
        1.0, 1.0, sm_scale=SM_SCALE,
    )


def new_fn(c, cfg, kernels=None):
    p = gs.plan([s[0] for s in c["seqs"]], [s[1] for s in c["seqs"]], cfg, device=DEV)

    def call(out, layer):
        gs.gqa_split_extend_fwd(
            c["q"], c["ke"], c["ve"], out, c["kbs"][layer], c["vbs"][layer], c["qo_indptr"],
            c["kv_indptr"], c["kv_indices"], SM_SCALE, p, kernels=kernels,
        )

    return call, p


def time_op(call, c):
    out = torch.empty_like(c["q"])

    def fn():
        for l in range(LAYERS):
            call(out, l)

    return graph_us(fn)


def cfg_str(cfg):
    return (f"tok{cfg.bm_tok}(M{cfg.bm_tok * HQ // HKV}) N{cfg.block_n} w{cfg.num_warps} "
            f"wpe{cfg.waves_per_eu} st{cfg.num_stages} sd{cfg.split_d} mf{cfg.mfma} "
            f"tgt{cfg.target_wg} mc{cfg.min_chunk} "
            f"{'bf16' if cfg.part_dtype == torch.bfloat16 else 'fp32'}")


# ----------------------------------------------------------------------------- phase A

GEOMS_A = [  # bm_tok, block_n, warps, wpe, stages, split_d
    (8, 64, 4, 1, 1, 1), (8, 64, 4, 2, 1, 1), (8, 32, 4, 2, 1, 1),
    (16, 64, 4, 1, 1, 1), (16, 64, 4, 1, 1, 2), (16, 32, 4, 1, 1, 1), (16, 32, 4, 2, 1, 1),
    (16, 64, 8, 1, 1, 1), (16, 64, 8, 2, 1, 1), (16, 64, 8, 1, 1, 2), (16, 32, 8, 1, 1, 1),
    (16, 32, 8, 2, 1, 1), (16, 64, 8, 1, 2, 1), (16, 32, 8, 1, 2, 1),
    (32, 32, 8, 1, 1, 2), (32, 64, 8, 1, 1, 2), (32, 32, 8, 1, 1, 1),
]


def phase_a(gen):
    log(f"\n## phase A: geometry sweep at P=2944 (target_wg 220, fp32 partials); us per call")
    cases = {n: make_case([(2944, n)], gen) for n in (184, 617)}
    refs = {n: ref(c) for n, c in cases.items()}
    cur = {}
    for n, c in cases.items():
        out = torch.empty_like(c["q"])
        cur_call(c, out, 0)
        torch.cuda.synchronize()
        cur[n] = ((out.float() - refs[n]).abs().max().item(), time_op(lambda o, l, c=c: cur_call(c, o, l), c))
    log(f"current: N184 {cur[184][1]:.1f} us (d_ref {cur[184][0]:.2e}), "
        f"N617 {cur[617][1]:.1f} us (d_ref {cur[617][0]:.2e}); {kinfo(CUR_CAP.last)}")
    rows = []
    for geom in GEOMS_A:
        bm, bn, nw, wpe, st, sd = geom
        cfg = gs.GqaSplitConfig(bm_tok=bm, block_n=bn, num_warps=nw, waves_per_eu=wpe,
                                num_stages=st, split_d=sd)
        cells, ok, info = {}, True, "?"
        for n, c in cases.items():
            try:
                ks = []
                call, _ = new_fn(c, cfg, ks)
                out = torch.empty_like(c["q"])
                call(out, 0)
                torch.cuda.synchronize()
                d = (out.float() - refs[n]).abs().max().item()
                info = kinfo(ks[0])
                if not d <= max(1.5 * cur[n][0], 1e-3):
                    ok = False
                    log(f"  !! {cfg_str(cfg)} N{n}: d_ref {d:.2e} vs current {cur[n][0]:.2e}")
                cells[n] = time_op(call, c)
            except Exception as e:  # noqa: BLE001
                ok = False
                log(f"  !! {cfg_str(cfg)} N{n}: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
        c184 = f"{cells[184]:7.1f}" if 184 in cells else "   fail"
        c617 = f"{cells[617]:7.1f}" if 617 in cells else "   fail"
        log(f"{str(geom):22s} | N184 {c184} N617 {c617} | {info} [{time.time() - T0:.0f}s]")
        if ok and len(cells) == 2:
            rows.append((cells[184] + cells[617], geom, cells))
    rows.sort(key=lambda r: r[0])
    return [r[1] for r in rows[:2]], cur


def mk(geom, **kw):
    bm, bn, nw, wpe, st, sd = geom
    return gs.GqaSplitConfig(bm_tok=bm, block_n=bn, num_warps=nw, waves_per_eu=wpe,
                             num_stages=st, split_d=sd, **kw)


# ----------------------------------------------------------------------------- phase B


def check_determinism(cfg, p_len, n_len, gen):
    """20 repeats bitwise; alone vs batch of 2 (both orders) bitwise."""
    other = (1000, 300) if n_len != 300 else (1000, 400)
    c2 = make_case([(p_len, n_len), other], gen)
    a = sub_case(c2, 0)
    call, _ = new_fn(a, cfg)
    out = torch.empty_like(a["q"])
    call(out, 0)
    torch.cuda.synchronize()
    first = out.clone()
    rep_ok = True
    for _ in range(19):
        call(out, 0)
        torch.cuda.synchronize()
        rep_ok &= torch.equal(first, out)
    call2, _ = new_fn(c2, cfg)
    out2 = torch.empty_like(c2["q"])
    call2(out2, 0)
    torch.cuda.synchronize()
    batch_ok = torch.equal(first, out2[:n_len])
    # reversed order: the sequence is second in the batch
    c3 = dict(c2)
    c3["seqs"] = [c2["seqs"][1], c2["seqs"][0]]
    nb, pb = other[1], other[0]
    c3["q"] = torch.cat([c2["q"][n_len:], c2["q"][:n_len]])
    c3["ke"] = torch.cat([c2["ke"][n_len:], c2["ke"][:n_len]])
    c3["ve"] = torch.cat([c2["ve"][n_len:], c2["ve"][:n_len]])
    c3["kv_indices"] = torch.cat([c2["kv_indices"][p_len:], c2["kv_indices"][:p_len]])
    c3["qo_indptr"] = torch.tensor([0, nb, nb + n_len], dtype=torch.int32, device=DEV)
    c3["kv_indptr"] = torch.tensor([0, pb, pb + p_len], dtype=torch.int32, device=DEV)
    call3, _ = new_fn(c3, cfg)
    out3 = torch.empty_like(c3["q"])
    call3(out3, 0)
    torch.cuda.synchronize()
    batch_ok &= torch.equal(first, out3[nb:])
    return rep_ok, batch_ok


def phase_b(geoms, gen):
    variants = []
    for g in geoms:
        variants += [mk(g, target_wg=220), mk(g, target_wg=110), mk(g, target_wg=440),
                     mk(g, target_wg=220, part_dtype=torch.bfloat16)]
    log("\n## phase B: us per call over the grid; new variants vs current")
    log("variants:")
    for i, v in enumerate(variants):
        log(f"  v{i}: {cfg_str(v)}")
    hdr = "P     N    | current " + " ".join(f"{'v' + str(i):>7s}" for i in range(len(variants)))
    log(hdr)
    results = {}  # (P, N) -> {name: us}
    fails = []
    for P in PS:
        for N in NS:
            c = make_case([(P, N)], gen)
            r = ref(c)
            row = {}
            out = torch.empty_like(c["q"])
            cur_call(c, out, 0)
            torch.cuda.synchronize()
            d_cur = (out.float() - r).abs().max().item()
            row["cur"] = time_op(lambda o, l, c=c: cur_call(c, o, l), c)
            cells = [f"{row['cur']:7.1f}"]
            for i, v in enumerate(variants):
                try:
                    call, pl = new_fn(c, v)
                    call(out, 0)
                    torch.cuda.synchronize()
                    d = (out.float() - r).abs().max().item()
                    if not d <= max(1.5 * d_cur, 1e-3):
                        fails.append((i, P, N, f"d_ref {d:.2e} vs cur {d_cur:.2e}"))
                    row[i] = time_op(call, c)
                    cells.append(f"{row[i]:7.1f}")
                except Exception as e:  # noqa: BLE001
                    fails.append((i, P, N, f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"))
                    cells.append("   fail")
            results[(P, N)] = row
            log(f"{P:5d} {N:4d} | " + " ".join(cells) + f"   (d_ref cur {d_cur:.2e}) [{time.time() - T0:.0f}s]")
            del c
            torch.cuda.empty_cache()
    return variants, results, fails


def breakdown(cfg, gen):
    """Stage 1 vs merge at P=2944: time the merge alone on filled partials."""
    log(f"\n## stage split for {cfg_str(cfg)}")
    for N in (184, 617, 1708):
        c = make_case([(2944, N)], gen)
        call, pl = new_fn(c, cfg)
        tot = time_op(call, c)
        o_part = torch.randn(pl.part_rows, HQ, D, device=DEV).to(cfg.part_dtype)
        lse_part = torch.randn(pl.part_rows, HQ, device=DEV)

        def merge(out, layer):
            gs._gqa_split_merge[(1, pl.max_ext)](
                out, o_part, lse_part, c["qo_indptr"], c["kv_indptr"], pl.seq_meta,
                out.stride(0), out.stride(1), H=HQ, D=D, BM_TOK=cfg.bm_tok, num_warps=4)

        mu = time_op(merge, c)
        C = int(pl.seq_meta[0, 0].item())
        log(f"N{N}: total {tot:.1f} us, merge {mu:.1f} us, stage1 ~{tot - mu:.1f} us; "
            f"C={C} chunks/tile<= {pl.max_chunks} tiles={pl.max_tiles} "
            f"wg<= {pl.max_tiles * pl.max_chunks} partial {pl.part_rows * HQ * D * (2 if cfg.part_dtype == torch.bfloat16 else 4) / 1e6:.1f} MB")


def main():
    torch.cuda.set_device(0)
    log(f"# device {torch.cuda.get_device_name(0)} {torch.cuda.get_device_properties(0).gcnArchName} "
        f"CUs={CORES} triton {triton.__version__} torch {torch.__version__}")
    gen = torch.Generator(device=DEV).manual_seed(0)
    geoms, _ = phase_a(gen)
    log(f"phase A best geometries: {geoms}")
    if not geoms:
        log("GATE: no passing geometry")
        return
    variants, results, fails = phase_b(geoms, gen)
    for f in fails:
        log(f"  !! v{f[0]} P{f[1]} N{f[2]}: {f[3]}")
    bad = {f[0] for f in fails}

    log("\n## determinism (20 repeats; alone vs batch of 2, both orders)")
    det = {}
    for i, v in enumerate(variants):
        if i in bad:
            continue
        ok = True
        for P, N in ((2944, 16), (2944, 184), (2944, 617), (6976, 1708)):
            try:
                rep, bat = check_determinism(v, P, N, gen)
            except Exception as e:  # noqa: BLE001
                rep, bat = False, False
                log(f"  !! v{i} P{P} N{N}: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
            ok &= rep and bat
            if not (rep and bat):
                log(f"  !! v{i} P{P} N{N}: repeat={rep} batch={bat}")
        det[i] = ok
        log(f"v{i}: {'pass' if ok else 'FAIL'}")

    log("\n## spills of phase-B variants (stage1 / merge)")
    c = make_case([(2944, 617)], gen)
    for i, v in enumerate(variants):
        ks = []
        try:
            call, _ = new_fn(c, v, ks)
            call(torch.empty_like(c["q"]), 0)
            log(f"v{i}: stage1 {kinfo(ks[0])} | merge {kinfo(ks[1])}")
        except Exception as e:  # noqa: BLE001
            log(f"v{i}: {type(e).__name__}")
    cur_call(c, torch.empty_like(c["q"]), 0)
    log(f"current _fwd_kernel: {kinfo(CUR_CAP.last)}")

    good = [i for i in range(len(variants)) if i not in bad and det.get(i)]
    log("\n## summary: best passing variant per shape")
    for (P, N), row in results.items():
        cand = [(row[i], i) for i in good if i in row]
        if cand:
            us, i = min(cand)
            log(f"P{P} N{N}: current {row['cur']:.1f} us, best v{i} {us:.1f} us = {row['cur'] / us:.2f}x")
    # one variant for all shapes (geomean)
    gm = []
    for i in good:
        sp = [row["cur"] / row[i] for row in results.values() if i in row]
        if len(sp) == len(results):
            gm.append((math.prod(sp) ** (1 / len(sp)), i))
    gm.sort(reverse=True)
    for g, i in gm:
        log(f"  v{i}: geomean {g:.2f}x; P2944 N184 {results[(2944, 184)][i]:.1f} N617 "
            f"{results[(2944, 617)][i]:.1f} N1708 {results[(2944, 1708)][i]:.1f}")

    any_det_fail = any(not det.get(i, True) for i in range(len(variants)) if i not in bad)
    b617 = min([results[(2944, 617)][i] for i in good if i in results[(2944, 617)]], default=1e9)
    b184 = min([results[(2944, 184)][i] for i in good if i in results[(2944, 184)]], default=1e9)
    kill = b617 > 400 or b184 > 300 or not good or any_det_fail
    log(f"\nGATE: best N617 {b617:.1f} us (kill > 400, continue <= 350), best N184 {b184:.1f} us "
        f"(kill > 300); determinism failures in any variant: {any_det_fail}; "
        f"=> {'KILL' if kill else 'PASS'}")
    if gm:
        best = variants[gm[0][1]]
        log(f"chosen variant: v{gm[0][1]} {cfg_str(best)}")
        breakdown(best, gen)
        # cold prefill (no prefix) for information: the path is not meant for it
        c = make_case([(0, 2048)], gen)
        cu = time_op(lambda o, l, c=c: cur_call(c, o, l), c)
        call, _ = new_fn(c, best)
        log(f"cold P0 N2048: current {cu:.1f} us, new {time_op(call, c):.1f} us")
    log(f"done in {time.time() - T0:.0f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)
