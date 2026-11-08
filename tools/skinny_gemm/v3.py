"""Plan 18 step 1: HIP skinny GEMV vs today's Triton kernel on one gfx90a GCD (via bench.py).

Variants, each replayed in a HIP graph with its own weight per call:
  e   today: skinny_gemm, _TUNED schedules, fused split-K reduce (control; j83 e_fused)
  s   e + a K-start stagger per pid_n (rotated K walk), same schedules
  s'  e on a padded weight copy (row stride K+64; diagnostic for L2 channel camping)
  h   HIP GEMV (skinny_gemv_gfx90a.hip): best config per (N, K) from a sweep at M=2/16,
      Triton e for shapes it cannot take (K % 512 != 0) or does not win
  r   streaming-read ceiling over the same weight bytes (no math)
Checks: bitwise batch invariance (rows of M=16 == M=1..16 calls) and 100 repeats for e, s, h;
max abs err vs fp32; h vs e difference.
"""

import itertools
import time
import traceback

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm import skinny_gemm_hip as sk
from sglang.kernels.ops.gemm import skinny_gemv_hip as hg
from sglang.kernels.ops.gemm.skinny_gemm_hip import _last_block_reduce

from bench import SHAPES, DEV, graph_time, layer_order, make_weights

MS_ALL = list(range(1, 17))
MS_SHAPE = [1, 2, 8, 16]
NAME_OF = {}
for _name, _n, _k, _c in SHAPES:
    NAME_OF.setdefault((_n, _k), []).append(_name)


def log(*a):
    print(*a, flush=True)


def wbytes(n, k, m):
    return 2 * n * k + 2 * m * k + 2 * m * n


# ------------------------------------------------------------------ (s) Triton + K stagger
@triton.jit
def _stag_kernel(
    x_ptr, w_ptr, y_ptr, ws_ptr, cnt_ptr, M, N, K, stride_xm, stride_wn, stride_ym, STAG,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_split = K // SPLIT_K
    n_it = k_split // BLOCK_K
    rot = (pid_n * STAG) % n_it
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ar = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for it in range(0, n_it):
        kb = pid_k * k_split + ((it + rot) % n_it) * BLOCK_K
        xv = tl.load(x_ptr + offs_m[:, None] * stride_xm + (kb + ar)[None, :], mask=mask_m[:, None], other=0.0)
        wv = tl.load(w_ptr + offs_n[:, None] * stride_wn + (kb + ar)[None, :], mask=mask_n[:, None], other=0.0)
        acc += tl.dot(xv, tl.trans(wv))
    mask = mask_m[:, None] & mask_n[None, :]
    if SPLIT_K == 1:
        tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :], acc.to(y_ptr.dtype.element_ty), mask=mask)
    else:
        ws_ptrs = ws_ptr + pid_k * M * N + offs_m[:, None] * N + offs_n[None, :]
        tl.store(ws_ptrs, acc, mask=mask)
        _last_block_reduce(
            ws_ptr, y_ptr, y_ptr, cnt_ptr, pid_n, offs_m, offs_n, mask, mask_n,
            M, N, stride_ym, False, SPLIT_K,
        )


def stag_gemm(x, w, cfg, stag=1):
    M, K = x.shape
    N = w.shape[0]
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    grid_n = triton.cdiv(N, cfg.block_n)
    ws = y if cfg.split_k == 1 else torch.empty((cfg.split_k, M, N), dtype=torch.float32, device=x.device)
    cnt = sk._ticket_counters(x.device) if cfg.split_k > 1 else ws
    _stag_kernel[(grid_n, cfg.split_k)](
        x, w, y, ws, cnt, M, N, K, x.stride(0), w.stride(0), y.stride(0), stag,
        BLOCK_M=sk.MAX_M, BLOCK_N=cfg.block_n, BLOCK_K=cfg.block_k, SPLIT_K=cfg.split_k,
        num_warps=cfg.num_warps, num_stages=cfg.num_stages,
    )
    return y


# ------------------------------------------------------------------ helpers
def per_shape(op, ws, xs, name, m):
    wl = ws[name]
    x = xs[(m, wl[0].shape[1])]

    def fn():
        for w in wl:
            op(x, w, name)

    return graph_time(fn) / len(wl)


def step(op, ws, xs, m, seq):
    idx = {n: 0 for n, *_ in SHAPES}
    calls = []
    for name in seq:
        calls.append((name, ws[name][idx[name]]))
        idx[name] += 1

    def fn():
        for name, w in calls:
            op(xs[(m, w.shape[1])], w, name)

    return graph_time(fn, iters=10)


def invariant(op, ws, xs, name, k, reps=100):
    w = ws[name][0]
    x16 = xs[(16, k)]
    y16 = op(x16, w, name)
    for m in range(1, 17):
        ym = op(x16[:m].contiguous(), w, name)
        if not torch.equal(ym, y16[:m]):
            return f"batch-invariance FAIL at M={m}"
        # also a strided view of the M=16 input (same rows, no copy)
        if not torch.equal(op(x16[:m], w, name), y16[:m]):
            return f"batch-invariance (view) FAIL at M={m}"
    for _ in range(reps):
        if not torch.equal(op(x16, w, name), y16):
            return "repeatability FAIL"
    return "ok"


def max_err(op, ws, xs, name, k, m):
    w = ws[name][0]
    x = xs[(m, k)]
    ref = x.float() @ w.float().t()
    return (op(x, w, name).float() - ref).abs().max().item()


def read_us(lib, wl):
    n16 = wl[0].numel() * 2 // 16
    out = torch.zeros(4096, dtype=torch.int32, device=DEV)
    best = None
    for blocks_per in (8, 4):
        blocks = max(1, min(4096, triton.cdiv(n16, 256 * blocks_per)))

        def fn():
            for w in wl:
                lib.lib.skg_read(w.data_ptr(), n16, out.data_ptr(), blocks, torch.cuda.current_stream().cuda_stream)

        t = graph_time(fn) / len(wl)
        best = t if best is None else min(best, t)
    return best


def main_v3():
    torch.manual_seed(0)
    props = torch.cuda.get_device_properties(0)
    log("device:", torch.cuda.get_device_name(0), props.gcnArchName, "CUs", props.multi_processor_count)
    log("torch", torch.__version__, "hip", torch.version.hip, "triton", triton.__version__)
    t0 = time.time()
    lib = hg.build()
    log(f"hip build+load {time.time() - t0:.1f}s: {lib.path}, {len(lib.entries)} kernels")

    ws = make_weights()
    xs = {}
    for m in MS_ALL:
        for k in (2048, 256):
            xs[(m, k)] = torch.randn(m, k, device=DEV).to(torch.bfloat16)
    seq = layer_order()
    tuned = {name: sk.default_config(n, k) for name, n, k, _ in SHAPES}
    log("today's schedules:", {k: tuple(v) for k, v in tuned.items()})

    # padded copies for s' (row stride K + 64)
    wpad = {}
    for name, n, k, _ in SHAPES:
        lst = []
        for w in ws[name]:
            buf = torch.empty((n, k + 64), dtype=w.dtype, device=DEV)
            buf[:, :k].copy_(w)
            lst.append(buf[:, :k])
        wpad[name] = lst
    pad_of = {}
    for name in ws:
        for w, wp in zip(ws[name], wpad[name]):
            pad_of[w.data_ptr()] = wp

    ops = {
        "e": lambda x, w, name: sk.skinny_gemm(x, w, None, tuned[name], fused_reduce=True),
        "s": lambda x, w, name: stag_gemm(x, w, tuned[name], 1),
        "s_pad": lambda x, w, name: sk.skinny_gemm(x, pad_of[w.data_ptr()], None, tuned[name], fused_reduce=True),
    }

    # sanity: s and s_pad numerics
    for name, n, k, _ in SHAPES:
        w = ws[name][0]
        for m in (1, 2, 16):
            x = xs[(m, k)]
            ye = ops["e"](x, w, name)
            yp = ops["s_pad"](x, w, name)
            ys = ops["s"](x, w, name)
            if not torch.equal(ye, yp):
                log(f"note: s_pad != e bitwise {name} M={m}")
            d = (ys.float() - ye.float()).abs().max().item()
            if m == 2:
                log(f"s vs e {name}: max|diff| {d:.4g}")

    # ---------------- (h) sweep per (N, K): pick the fastest config at M=2 whose M=16 time is
    # within 3% of e's; shapes without one keep e (decided below from the full tables).
    t0 = time.time()
    e_t = {}
    for name, n, k, _ in SHAPES:
        e_t[name] = {m: per_shape(ops["e"], ws, xs, name, m) for m in (2, 16)}
    h_cfg = {}
    for (n, k), names in NAME_OF.items():
        name = names[0]
        rows = []
        for base in lib.configs(k):
            for stag in (1, 0):
                cfg = base._replace(stag=stag)
                idx = {mb: lib.find(mb, cfg, k) for mb in hg.MBS}
                op = lambda x, w, _n, c=cfg, ix=idx: hg.gemv(lib, x, w, None, c, ix[hg.mb_of(x.shape[0])])
                try:
                    t2 = per_shape(op, ws, xs, name, 2)
                    t16 = per_shape(op, ws, xs, name, 16)
                except Exception as ex:  # noqa: BLE001
                    log("h cfg fail", name, cfg, repr(ex)[:300])
                    continue
                rows.append((t2, t16, cfg, idx))
        if not rows:
            continue
        lim16 = 1.03 * e_t[name][16]
        ok = sorted([r for r in rows if r[1] <= lim16], key=lambda r: r[0])
        by_score = sorted(rows, key=lambda r: r[0] + 0.25 * r[1])
        log(f"[sweep h] {name} N={n} K={k}: {len(rows)} configs; e M2={e_t[name][2]:.2f} M16={e_t[name][16]:.2f}; "
            f"{len(ok)} within M16 <= {lim16:.2f}")
        log("   fastest M2 within the M16 limit:")
        for r in ok[:6]:
            log(f"   M2={r[0]:.2f}us M16={r[1]:.2f}us {tuple(r[2])}")
        log("   best score M2 + 0.25*M16:")
        for r in by_score[:4]:
            log(f"   M2={r[0]:.2f}us M16={r[1]:.2f}us {tuple(r[2])}")
        log("   fastest M16:")
        for r in sorted(rows, key=lambda r: r[1])[:3]:
            log(f"   M2={r[0]:.2f}us M16={r[1]:.2f}us {tuple(r[2])}")
        cands = ok if ok else by_score
        bcfg = cands[0][2]
        for alt in (bcfg._replace(stag=1 - bcfg.stag),):
            ar = [r for r in rows if r[2] == alt]
            if ar:
                log(f"   stagger flipped {tuple(alt)}: M2={ar[0][0]:.2f} M16={ar[0][1]:.2f}")
        for r in cands[:6]:
            op = lambda x, w, _n, c=r[2], ix=r[3]: hg.gemv(lib, x, w, None, c, ix[hg.mb_of(x.shape[0])])
            v = invariant(op, ws, xs, name, k, reps=10)
            if v == "ok":
                h_cfg[(n, k)] = (r[2], r[3])
                break
            log(f"   {tuple(r[2])} rejected: {v}")
    log(f"sweep h took {time.time() - t0:.0f}s")

    def h_pure(x, w, name):
        n, k = w.shape
        c = h_cfg.get((n, k))
        if c is None:
            return ops["e"](x, w, name)
        return hg.gemv(lib, x, w, None, c[0], c[1][hg.mb_of(x.shape[0])])

    ops["h_all"] = h_pure

    # ---------------- per-shape tables
    res = {}
    for key, op in ops.items():
        r = {"shape": {}, "step": {}, "err": {}}
        try:
            for m in MS_SHAPE:
                for name, n, k, _ in SHAPES:
                    r["shape"][(name, m)] = per_shape(op, ws, xs, name, m)
        except Exception:  # noqa: BLE001
            log(f"{key} failed:", traceback.format_exc()[-1500:])
        res[key] = r
    r_us = {}
    for name, n, k, _ in SHAPES:
        r_us[name] = read_us(lib, ws[name])

    # hybrid h: HIP only for shapes where it beats e on score (M2 + 0.25*M16)
    use_hip = {}
    for name, n, k, _ in SHAPES:
        if (n, k) not in h_cfg:
            use_hip[(n, k)] = False
            continue
        use_hip[(n, k)] = (res["h_all"]["shape"][(name, 2)] < res["e"]["shape"][(name, 2)]
                           and res["h_all"]["shape"][(name, 16)] <= 1.03 * res["e"]["shape"][(name, 16)])
    log("h uses HIP for:", {NAME_OF[nk][0]: v for nk, v in use_hip.items()})

    def h_op(x, w, name):
        n, k = w.shape
        if use_hip[(n, k)]:
            return h_pure(x, w, name)
        return ops["e"](x, w, name)

    ops["h"] = h_op
    res["h"] = {"shape": {key: (res["h_all"] if use_hip[NAME_OF_inv(key[0])] else res["e"])["shape"][key]
                          for key in res["e"]["shape"]}, "step": {}, "err": {}}

    # ---------------- correctness
    inv = {}
    for key in ("e", "s", "h"):
        bad = []
        for name, n, k, _ in SHAPES:
            v = invariant(ops[key], ws, xs, name, k, reps=100)
            if v != "ok":
                bad.append(f"{name}: {v}")
        inv[key] = not bad
        log(f"[check] {key}: bitwise batch-invariant (M=1..16, contiguous and view) and 100 repeats: {not bad} {bad}")
    for key in ("e", "s", "s_pad", "h"):
        for m in (1, 2, 16):
            res[key]["err"][m] = max(max_err(ops[key], ws, xs, name, k, m) for name, n, k, _ in SHAPES)
    hd = {}
    for name, n, k, _ in SHAPES:
        w = ws[name][0]
        hd[name] = max((ops["h"](xs[(m, k)], w, name).float() - ops["e"](xs[(m, k)], w, name).float()).abs().max().item()
                       for m in (1, 2, 16))
    log("[check] h vs e max|diff| per shape (M=1,2,16):", {k: round(v, 5) for k, v in hd.items()})
    # bias path of h: error vs the fp32 reference must not exceed e's
    for (n, k), (cfg, idx) in h_cfg.items():
        w = ws[NAME_OF[(n, k)][0]][0]
        b = (torch.randn(n, device=DEV) * 0.1).to(torch.bfloat16)
        for m in (1, 3, 16):
            x = xs[(m, k)]
            ref = x.float() @ w.float().t() + b.float()
            eh = (hg.gemv(lib, x, w, b, cfg, idx[hg.mb_of(m)]).float() - ref).abs().max().item()
            ee = (sk.skinny_gemm(x, w, b, tuned[NAME_OF[(n, k)][0]]).float() - ref).abs().max().item()
            if eh > 1.01 * ee + 1e-6:
                log(f"[check] h bias err above e: N={n} K={k} M={m}: h {eh:.4g} e {ee:.4g}")
    log("[check] h bias path done")

    # ---------------- step graphs
    for key in ("e", "s", "s_pad", "h"):
        for m in MS_ALL:
            res[key]["step"][m] = step(ops[key], ws, xs, m, seq)
        log(f"[{key}] step us: " + ", ".join(f"M{m}={v:.0f}" for m, v in res[key]["step"].items()))
    for key in ("e", "s"):
        rep = [step(ops[key], ws, xs, 2, seq) for _ in range(2)]
        log(f"[{key}] step M=2 repeat timings: {[round(v) for v in rep]}")
    rep = [step(ops["h"], ws, xs, 2, seq) for _ in range(2)]
    log(f"[h] step M=2 repeat timings: {[round(v) for v in rep]}")

    # ---------------- tables
    keys = ["e", "s", "s_pad", "h_all", "h"]
    log("\n## per-shape us/call and GB/s (graph replay of all calls of the shape, own weight each); r = read ceiling")
    for m in MS_SHAPE:
        log(f"\nM={m}")
        log(" | ".join(["shape", "N", "K", "calls"] + [f"{k} us" for k in keys] + ["r us"] + [f"{k} GB/s" for k in keys] + ["r GB/s"]))
        for name, n, k, cnt in SHAPES:
            us = [res[key]["shape"].get((name, m), float("nan")) for key in keys] + [r_us[name]]
            gbs = [wbytes(n, k, m) / u / 1e3 for u in us]
            log(" | ".join([name, str(n), str(k), str(cnt)] + [f"{u:.2f}" for u in us] + [f"{g:.0f}" for g in gbs]))
        log(" | ".join(["sum(count*us)", "", "", "230"] + [
            f"{sum(res[key]['shape'].get((nm, m), float('nan')) * c for nm, _, _, c in SHAPES):.0f}" for key in keys]
            + [f"{sum(r_us[nm] * c for nm, _, _, c in SHAPES):.0f}"]))
    log("\n## 230-call step graph us by M")
    sk_keys = ["e", "s", "s_pad", "h"]
    log(" | ".join(["M"] + sk_keys))
    for m in MS_ALL:
        log(" | ".join([str(m)] + [f"{res[k]['step'].get(m, float('nan')):.0f}" for k in sk_keys]))
    log("\n## max abs err vs fp32 reference")
    for k in ("e", "s", "s_pad", "h"):
        log(k, {m: round(v, 4) for m, v in res[k]["err"].items()})
    log("\nH_TUNED = {")
    for (n, k), (cfg, _) in h_cfg.items():
        log(f"    ({n}, {k}): {tuple(cfg)},  # {NAME_OF[(n, k)]} hip={use_hip[(n, k)]}")
    log("}")

    # ---------------- gate
    e2, e16 = res["e"]["step"][2], res["e"]["step"][16]
    log("\nKILL GATE @M=2 (vs e):")
    for key in ("s", "h"):
        s2, s16 = res[key]["step"][2], res[key]["step"][16]
        q = res[key]["shape"][("gdn_qkvz", 2)]
        ok_inv = inv[key]
        reg16 = s16 / e16 - 1
        passed = ok_inv and (e2 - s2) >= 600 and q <= 25 and reg16 <= 0.05
        log(f"  {key}: step M2 {s2:.0f} vs {e2:.0f} us, saving {(e2 - s2) / 1000:.3f} ms; qkvz {q:.2f} us; "
            f"M16 step {s16:.0f} vs {e16:.0f} ({100 * reg16:+.1f}%); invariant {ok_inv} -> {'PASS' if passed else 'FAIL'}")
    sp = res["s_pad"]["shape"][("gdn_qkvz", 2)]
    log(f"  channel camping diagnostic qkvz M2: e {res['e']['shape'][('gdn_qkvz', 2)]:.2f}, s {res['s']['shape'][('gdn_qkvz', 2)]:.2f}, "
        f"s_pad {sp:.2f}, r {r_us['gdn_qkvz']:.2f} us")
    return 0


def NAME_OF_inv(name):
    for n_, n, k, _ in SHAPES:
        if n_ == name:
            return (n, k)
    raise KeyError(name)


# ====================================================================== plan 18 step 1c: MFMA
def main_q():
    """(q) MFMA GEMV: v_mfma_f32_16x16x16bf16_1k on direct-to-register weights, x padded to 16
    rows; one kernel per shape for every M. Compared with e (today) and r (read ceiling)."""
    torch.manual_seed(0)
    props = torch.cuda.get_device_properties(0)
    log("device:", torch.cuda.get_device_name(0), props.gcnArchName, "CUs", props.multi_processor_count)
    log("torch", torch.__version__, "hip", torch.version.hip, "triton", triton.__version__)
    t0 = time.time()
    lib = hg.build()
    log(f"hip build+load {time.time() - t0:.1f}s: {lib.path}, {len(lib.entries)} scalar + {len(lib.q_entries)} mfma kernels")
    ws = make_weights()
    xs = {}
    for m in MS_ALL:
        for k in (2048, 256):
            xs[(m, k)] = torch.randn(m, k, device=DEV).to(torch.bfloat16)
    seq = layer_order()
    tuned = {name: sk.default_config(n, k) for name, n, k, _ in SHAPES}
    e_op = lambda x, w, name: sk.skinny_gemm(x, w, None, tuned[name], fused_reduce=True)
    ops = {"e": e_op}

    def qop(cfg, idx):
        return lambda x, w, _n: hg.mfma_gemv(lib, x, w, None, cfg, idx)

    # ---------------- C/D layout probe: error vs fp32 must match e's
    def err_of(op, name, k, m):
        w = ws[name][0]
        x = xs[(m, k)]
        ref = x.float() @ w.float().t()
        return (op(x, w, name).float() - ref).abs().max().item()

    dl_ok = None
    for dl in (0, 1):
        good = True
        for name, k in (("gdn_qkvz", 2048), ("shared_down", 256), ("gdn_ba", 2048)):
            cfgs = lib.q_configs(k, dl)
            for cfg in (cfgs[0], cfgs[-1]):
                op = qop(cfg, lib.q_find(cfg, k))
                for m in (1, 3, 16):
                    eq = err_of(op, name, k, m)
                    ee = err_of(e_op, name, k, m)
                    if not eq <= 1.5 * ee + 1e-3:
                        good = False
                        log(f"[probe] dl={dl} {name} {tuple(cfg)} M={m}: err {eq:.4g} vs e {ee:.4g} -> wrong")
        log(f"[probe] C/D layout dl={dl}: {'OK' if good else 'WRONG'}")
        if good and dl_ok is None:
            dl_ok = dl
    if dl_ok is None:
        log("MFMA layout probe failed for both layouts; q not run")
        return 1

    # ---------------- sweep
    t0 = time.time()
    e_t = {name: {m: per_shape(e_op, ws, xs, name, m) for m in (2, 16)} for name, *_ in SHAPES}
    q_cfg = {}
    for (n, k), names in NAME_OF.items():
        name = names[0]
        rows = []
        for cfg in lib.q_configs(k, dl_ok):
            idx = lib.q_find(cfg, k)
            op = qop(cfg, idx)
            try:
                t2 = per_shape(op, ws, xs, name, 2)
                t16 = per_shape(op, ws, xs, name, 16)
            except Exception as ex:  # noqa: BLE001
                log("q cfg fail", name, cfg, repr(ex)[:300])
                continue
            rows.append((t2, t16, cfg, idx))
        lim16 = 1.03 * e_t[name][16]
        ok = sorted([r for r in rows if r[1] <= lim16], key=lambda r: r[0])
        by_score = sorted(rows, key=lambda r: r[0] + 0.25 * r[1])
        log(f"[sweep q] {name} N={n} K={k}: {len(rows)} configs; e M2={e_t[name][2]:.2f} M16={e_t[name][16]:.2f}; "
            f"{len(ok)} within M16 <= {lim16:.2f}")
        for r in ok[:6]:
            log(f"   ok   M2={r[0]:.2f}us M16={r[1]:.2f}us {tuple(r[2])}")
        for r in by_score[:3]:
            log(f"   score M2={r[0]:.2f}us M16={r[1]:.2f}us {tuple(r[2])}")
        cands = ok if ok else by_score
        b = cands[0]
        sc = b[2]._replace(stag=1)
        st = qop(sc, b[3])
        log(f"   stagger on {tuple(sc)}: M2={per_shape(st, ws, xs, name, 2):.2f} M16={per_shape(st, ws, xs, name, 16):.2f}")
        for r in cands[:6]:
            v = invariant(qop(r[2], r[3]), ws, xs, name, k, reps=10)
            if v == "ok":
                q_cfg[(n, k)] = (r[2], r[3])
                break
            log(f"   {tuple(r[2])} rejected: {v}")
    log(f"sweep q took {time.time() - t0:.0f}s")

    def q_all(x, w, name):
        n, k = w.shape
        c = q_cfg.get((n, k))
        if c is None:
            return e_op(x, w, name)
        return hg.mfma_gemv(lib, x, w, None, c[0], c[1])

    ops["q_all"] = q_all
    res = {}
    for key in ("e", "q_all"):
        res[key] = {"shape": {(name, m): per_shape(ops[key], ws, xs, name, m) for m in MS_SHAPE for name, *_ in SHAPES},
                    "step": {}, "err": {}}
    r_us = {name: read_us(lib, ws[name]) for name, *_ in SHAPES}
    use_q = {}
    for name, n, k, _ in SHAPES:
        use_q[(n, k)] = ((n, k) in q_cfg
                         and res["q_all"]["shape"][(name, 2)] < res["e"]["shape"][(name, 2)]
                         and res["q_all"]["shape"][(name, 16)] <= 1.03 * res["e"]["shape"][(name, 16)])
    log("q uses MFMA for:", {NAME_OF[nk][0]: v for nk, v in use_q.items()})

    def q_op(x, w, name):
        return q_all(x, w, name) if use_q[tuple(w.shape)] else e_op(x, w, name)

    ops["q"] = q_op
    res["q"] = {"shape": {key: (res["q_all"] if use_q[NAME_OF_inv(key[0])] else res["e"])["shape"][key]
                          for key in res["e"]["shape"]}, "step": {}, "err": {}}

    # ---------------- correctness
    inv = {}
    for key in ("e", "q_all", "q"):
        bad = []
        for name, n, k, _ in SHAPES:
            v = invariant(ops[key], ws, xs, name, k, reps=100)
            if v != "ok":
                bad.append(f"{name}: {v}")
        inv[key] = not bad
        log(f"[check] {key}: bitwise batch-invariant (M=1..16, contiguous and view) and 100 repeats: {not bad} {bad}")
    for key in ("e", "q_all", "q"):
        for m in (1, 2, 16):
            res[key]["err"][m] = max(max_err(ops[key], ws, xs, name, k, m) for name, n, k, _ in SHAPES)
    qd = {}
    for name, n, k, _ in SHAPES:
        w = ws[name][0]
        qd[name] = max((q_all(xs[(m, k)], w, name).float() - e_op(xs[(m, k)], w, name).float()).abs().max().item()
                       for m in (1, 2, 16))
    log("[check] q_all vs e max|diff| per shape (M=1,2,16):", {k: round(v, 5) for k, v in qd.items()})
    bias_ok = True
    for (n, k), (cfg, idx) in q_cfg.items():
        name = NAME_OF[(n, k)][0]
        w = ws[name][0]
        b = (torch.randn(n, device=DEV) * 0.1).to(torch.bfloat16)
        for m in (1, 3, 16):
            x = xs[(m, k)]
            ref = x.float() @ w.float().t() + b.float()
            eq = (hg.mfma_gemv(lib, x, w, b, cfg, idx).float() - ref).abs().max().item()
            ee = (sk.skinny_gemm(x, w, b, tuned[name]).float() - ref).abs().max().item()
            if eq > 1.5 * ee + 1e-3:
                bias_ok = False
                log(f"[check] q bias err above e: N={n} K={k} M={m}: q {eq:.4g} e {ee:.4g}")
    log(f"[check] q bias path within e's error: {bias_ok}")
    # fresh-input graph replay vs eager (q_all)
    stress_ok = True
    for name, n, k, _ in SHAPES:
        wl = ws[name]
        for m in (2, 16):
            xin = torch.randn(m, k, device=DEV).to(torch.bfloat16)
            outs = []

            def fn():
                outs.clear()
                for w in wl:
                    outs.append(q_all(xin, w, name))

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                fn()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            for it in range(10):
                xin.copy_(torch.randn(m, k, device=DEV))
                g.replay()
                torch.cuda.synchronize()
                for i in (0, len(wl) - 1):
                    if not torch.equal(outs[i], q_all(xin, wl[i], name)):
                        stress_ok = False
            del g
    log(f"[check] q_all graph replay with fresh inputs == eager: {stress_ok}")

    # ---------------- step graphs
    for key in ("e", "q", "q_all"):
        for m in MS_ALL:
            res[key]["step"][m] = step(ops[key], ws, xs, m, seq)
        log(f"[{key}] step us: " + ", ".join(f"M{m}={v:.0f}" for m, v in res[key]["step"].items()))
    for key in ("e", "q"):
        log(f"[{key}] step M=2 repeat timings: {[round(step(ops[key], ws, xs, 2, seq)) for _ in range(2)]}")

    # ---------------- tables
    keys = ["e", "q_all", "q"]
    log("\n## per-shape us/call and GB/s (graph replay, own weight per call); r = read ceiling")
    for m in MS_SHAPE:
        log(f"\nM={m}")
        log(" | ".join(["shape", "N", "K", "calls"] + [f"{k} us" for k in keys] + ["r us"] + [f"{k} GB/s" for k in keys] + ["r GB/s"]))
        for name, n, k, cnt in SHAPES:
            us = [res[key]["shape"][(name, m)] for key in keys] + [r_us[name]]
            log(" | ".join([name, str(n), str(k), str(cnt)] + [f"{u:.2f}" for u in us] + [f"{wbytes(n, k, m) / u / 1e3:.0f}" for u in us]))
        log(" | ".join(["sum(count*us)", "", "", "230"] + [
            f"{sum(res[key]['shape'][(nm, m)] * c for nm, _, _, c in SHAPES):.0f}" for key in keys]
            + [f"{sum(r_us[nm] * c for nm, _, _, c in SHAPES):.0f}"]))
    log("\n## 230-call step graph us by M")
    log(" | ".join(["M"] + keys))
    for m in MS_ALL:
        log(" | ".join([str(m)] + [f"{res[k]['step'][m]:.0f}" for k in keys]))
    log("\n## max abs err vs fp32 reference")
    for k in keys:
        log(k, {m: round(v, 4) for m, v in res[k]["err"].items()})
    log(f"\nQ_TUNED (dl={dl_ok}) = {{")
    for (n, k), (cfg, _) in q_cfg.items():
        log(f"    ({n}, {k}): {tuple(cfg)},  # {NAME_OF[(n, k)]} mfma={use_q[(n, k)]}")
    log("}")

    # ---------------- gate
    e2, e16 = res["e"]["step"][2], res["e"]["step"][16]
    log("\nKILL GATE @M=2 (vs e):")
    for key in ("q", "q_all"):
        s2, s16 = res[key]["step"][2], res[key]["step"][16]
        qk = res[key]["shape"][("gdn_qkvz", 2)]
        reg16 = s16 / e16 - 1
        shape16 = max(res[key]["shape"][(nm, 16)] / res["e"]["shape"][(nm, 16)] - 1 for nm, *_ in SHAPES)
        passed = inv[key] and stress_ok and (e2 - s2) >= 600 and qk <= 25 and reg16 <= 0.05 and shape16 <= 0.05
        log(f"  {key}: step M2 {s2:.0f} vs {e2:.0f} us, saving {(e2 - s2) / 1000:.3f} ms; qkvz {qk:.2f} us; "
            f"M16 step {s16:.0f} vs {e16:.0f} ({100 * reg16:+.1f}%); worst per-shape M16 {100 * shape16:+.1f}%; "
            f"invariant {inv[key]}, replay {stress_ok} -> {'PASS' if passed else 'FAIL'}")
    return 0


# ================================================== plan 18 step 1d: 4x4x4 MFMA (f) + per-shape mix
def main_f():
    """(f) 4x4x4 MFMA GEMV (4 rows x 256 B per load) on every shape; (q) j118's 16x16x16 on the
    M=16-safe shapes; (mix) per shape the fastest at M=2 of e / q / f whose M=16 time is within
    3% of e (one kernel per shape for every M). Compared with e (today) and r (read ceiling)."""
    torch.manual_seed(0)
    props = torch.cuda.get_device_properties(0)
    log("device:", torch.cuda.get_device_name(0), props.gcnArchName, "CUs", props.multi_processor_count)
    log("torch", torch.__version__, "hip", torch.version.hip, "triton", triton.__version__)
    t0 = time.time()
    lib = hg.build()
    log(f"hip build+load {time.time() - t0:.1f}s: {lib.path}, {len(lib.q_entries)} q + {len(lib.f_entries)} f kernels")
    ws = make_weights()
    xs = {}
    for m in MS_ALL:
        for k in (2048, 256):
            xs[(m, k)] = torch.randn(m, k, device=DEV).to(torch.bfloat16)
    seq = layer_order()
    tuned = {name: sk.default_config(n, k) for name, n, k, _ in SHAPES}
    e_op = lambda x, w, name: sk.skinny_gemm(x, w, None, tuned[name], fused_reduce=True)

    def mop(fam, cfg, idx):
        return lambda x, w, _n: hg.mfma_gemv(lib, x, w, None, cfg, idx, fam)

    def err_of(op, name, k, m):
        w = ws[name][0]
        x = xs[(m, k)]
        ref = x.float() @ w.float().t()
        return (op(x, w, name).float() - ref).abs().max().item()

    # ---------------- C/D layout probe per family
    dl_of = {}
    for fam in ("q", "f"):
        for dl in (0, 1):
            good = True
            for name, k in (("gdn_qkvz", 2048), ("shared_down", 256), ("gdn_ba", 2048)):
                cfgs = lib.q_configs(k, dl, fam)
                for cfg in (cfgs[0], cfgs[-1]):
                    op = mop(fam, cfg, lib.q_find(cfg, k, fam))
                    for m in (1, 3, 5, 16):
                        eq, ee = err_of(op, name, k, m), err_of(e_op, name, k, m)
                        if not eq <= 1.5 * ee + 1e-3:
                            good = False
            log(f"[probe] {fam} C/D layout dl={dl}: {'OK' if good else 'WRONG'}")
            if good and fam not in dl_of:
                dl_of[fam] = dl
    if "f" not in dl_of:
        log("[probe] f: no layout gives correct results; f is excluded")

    # ---------------- sweeps
    t0 = time.time()
    e_t = {name: {m: per_shape(e_op, ws, xs, name, m) for m in (2, 16)} for name, *_ in SHAPES}
    best = {}  # (fam, (n, k)) -> (cfg, idx, t2, t16)
    for fam in dl_of:
        for (n, k), names in NAME_OF.items():
            name = names[0]
            rows = []
            for cfg in lib.q_configs(k, dl_of[fam], fam):
                idx = lib.q_find(cfg, k, fam)
                op = mop(fam, cfg, idx)
                try:
                    t2 = per_shape(op, ws, xs, name, 2)
                    t16 = per_shape(op, ws, xs, name, 16)
                except Exception as ex:  # noqa: BLE001
                    log(f"{fam} cfg fail", name, cfg, repr(ex)[:300])
                    continue
                rows.append((t2, t16, cfg, idx))
            lim16 = 1.03 * e_t[name][16]
            ok = sorted([r for r in rows if r[1] <= lim16], key=lambda r: r[0])
            by_score = sorted(rows, key=lambda r: r[0] + 0.25 * r[1])
            log(f"[sweep {fam}] {name} N={n} K={k}: {len(rows)} configs; e M2={e_t[name][2]:.2f} "
                f"M16={e_t[name][16]:.2f}; {len(ok)} within M16 <= {lim16:.2f}")
            for r in ok[:4]:
                log(f"   ok    M2={r[0]:.2f}us M16={r[1]:.2f}us {tuple(r[2])}")
            for r in by_score[:3]:
                log(f"   score M2={r[0]:.2f}us M16={r[1]:.2f}us {tuple(r[2])}")
            for r in (ok if ok else by_score)[:6]:
                v = invariant(mop(fam, r[2], r[3]), ws, xs, name, k, reps=10)
                if v == "ok":
                    best[(fam, (n, k))] = r
                    break
                log(f"   {tuple(r[2])} rejected: {v}")
    log(f"sweeps took {time.time() - t0:.0f}s")

    # per-shape choice: the fastest at M=2 among e and the M16-safe q / f picks
    choice = {}
    for (n, k), names in NAME_OF.items():
        name = names[0]
        pick = ("e", None, None, e_t[name][2])
        for fam in ("q", "f"):
            r = best.get((fam, (n, k)))
            if r and r[1] <= 1.03 * e_t[name][16] and r[0] < pick[3]:
                pick = (fam, r[2], r[3], r[0])
        choice[(n, k)] = pick

    def fam_all(fam):
        def op(x, w, name):
            r = best.get((fam, tuple(w.shape)))
            if r is None:
                return e_op(x, w, name)
            return hg.mfma_gemv(lib, x, w, None, r[2], r[3], fam)
        return op

    def q_safe(x, w, name):
        r = best.get(("q", tuple(w.shape)))
        if r is None or not (r[1] <= 1.03 * e_t[NAME_OF[tuple(w.shape)][0]][16] and r[0] < e_t[NAME_OF[tuple(w.shape)][0]][2]):
            return e_op(x, w, name)
        return hg.mfma_gemv(lib, x, w, None, r[2], r[3], "q")

    def mix(x, w, name):
        fam, cfg, idx, _ = choice[tuple(w.shape)]
        if fam == "e":
            return e_op(x, w, name)
        return hg.mfma_gemv(lib, x, w, None, cfg, idx, fam)

    ops = {"e": e_op, "q": q_safe, "mix": mix}
    if "f" in dl_of:
        ops["f_all"] = fam_all("f")
    keys = list(ops)
    res = {key: {"shape": {(name, m): per_shape(ops[key], ws, xs, name, m) for m in MS_SHAPE for name, *_ in SHAPES},
                 "step": {}, "err": {}} for key in keys}
    r_us = {name: read_us(lib, ws[name]) for name, *_ in SHAPES}
    log("mix choice:", {NAME_OF[nk][0]: (c[0], tuple(c[1]) if c[1] else None) for nk, c in choice.items()})

    # ---------------- correctness
    inv = {}
    for key in keys:
        bad = []
        for name, n, k, _ in SHAPES:
            v = invariant(ops[key], ws, xs, name, k, reps=100)
            if v != "ok":
                bad.append(f"{name}: {v}")
        inv[key] = not bad
        log(f"[check] {key}: bitwise batch-invariant (M=1..16, contiguous and view) and 100 repeats: {not bad} {bad}")
    for key in keys:
        for m in (1, 2, 16):
            res[key]["err"][m] = max(max_err(ops[key], ws, xs, name, k, m) for name, n, k, _ in SHAPES)
    bias_ok = True
    for (fam, (n, k)), r in best.items():
        name = NAME_OF[(n, k)][0]
        w = ws[name][0]
        b = (torch.randn(n, device=DEV) * 0.1).to(torch.bfloat16)
        for m in (1, 3, 16):
            x = xs[(m, k)]
            ref = x.float() @ w.float().t() + b.float()
            eq = (hg.mfma_gemv(lib, x, w, b, r[2], r[3], fam).float() - ref).abs().max().item()
            ee = (sk.skinny_gemm(x, w, b, tuned[name]).float() - ref).abs().max().item()
            if eq > 1.5 * ee + 1e-3:
                bias_ok = False
                log(f"[check] {fam} bias err above e: N={n} K={k} M={m}: {eq:.4g} vs {ee:.4g}")
    log(f"[check] bias path within e's error: {bias_ok}")
    stress = {}
    for key in [k for k in keys if k != "e"]:
        ok_ = True
        for name, n, k, _ in SHAPES:
            wl = ws[name]
            for m in (2, 16):
                xin = torch.randn(m, k, device=DEV).to(torch.bfloat16)
                outs = []

                def fn():
                    outs.clear()
                    for w in wl:
                        outs.append(ops[key](xin, w, name))

                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    fn()
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    fn()
                for it in range(10):
                    xin.copy_(torch.randn(m, k, device=DEV))
                    g.replay()
                    torch.cuda.synchronize()
                    for i in (0, len(wl) - 1):
                        if not torch.equal(outs[i], ops[key](xin, wl[i], name)):
                            ok_ = False
                del g
        stress[key] = ok_
        log(f"[check] {key} graph replay with fresh inputs == eager: {ok_}")

    # ---------------- step graphs
    for key in keys:
        for m in MS_ALL:
            res[key]["step"][m] = step(ops[key], ws, xs, m, seq)
        log(f"[{key}] step us: " + ", ".join(f"M{m}={v:.0f}" for m, v in res[key]["step"].items()))
    for key in keys:
        log(f"[{key}] step M=2 repeat timings: {[round(step(ops[key], ws, xs, 2, seq)) for _ in range(2)]}")

    # ---------------- tables
    log("\n## per-shape us/call and GB/s (graph replay, own weight per call); r = read ceiling")
    for m in MS_SHAPE:
        log(f"\nM={m}")
        log(" | ".join(["shape", "N", "K", "calls"] + [f"{k} us" for k in keys] + ["r us"] + [f"{k} GB/s" for k in keys] + ["r GB/s"]))
        for name, n, k, cnt in SHAPES:
            us = [res[key]["shape"][(name, m)] for key in keys] + [r_us[name]]
            log(" | ".join([name, str(n), str(k), str(cnt)] + [f"{u:.2f}" for u in us] + [f"{wbytes(n, k, m) / u / 1e3:.0f}" for u in us]))
        log(" | ".join(["sum(count*us)", "", "", "230"] + [
            f"{sum(res[key]['shape'][(nm, m)] * c for nm, _, _, c in SHAPES):.0f}" for key in keys]
            + [f"{sum(r_us[nm] * c for nm, _, _, c in SHAPES):.0f}"]))
    log("\n## 230-call step graph us by M")
    log(" | ".join(["M"] + keys))
    for m in MS_ALL:
        log(" | ".join([str(m)] + [f"{res[k]['step'][m]:.0f}" for k in keys]))
    log("\n## max abs err vs fp32 reference")
    for k in keys:
        log(k, {m: round(v, 4) for m, v in res[k]["err"].items()})
    log(f"\nlayouts: {dl_of}")
    log("BEST (fam, (N, K)): M2 us, M16 us, config")
    for (fam, nk), r in sorted(best.items()):
        log(f"    {fam} {nk}: {r[0]:.2f} {r[1]:.2f} {tuple(r[2])}  # {NAME_OF[nk]}")
    log("MIX_TUNED = {")
    for nk, c in choice.items():
        log(f"    {nk}: {(c[0], tuple(c[1]) if c[1] else None)},  # {NAME_OF[nk]}")
    log("}")

    # ---------------- gate
    e2, e16 = res["e"]["step"][2], res["e"]["step"][16]
    log("\nGATE @M=2 (vs e):")
    for key in [k for k in keys if k != "e"]:
        s2, s16 = res[key]["step"][2], res[key]["step"][16]
        qk = res[key]["shape"][("gdn_qkvz", 2)]
        reg16 = s16 / e16 - 1
        shape16 = max(res[key]["shape"][(nm, 16)] / res["e"]["shape"][(nm, 16)] - 1 for nm, *_ in SHAPES)
        passed = inv[key] and stress[key] and (e2 - s2) >= 600 and qk <= 25 and reg16 <= 0.05 and shape16 <= 0.05
        log(f"  {key}: step M2 {s2:.0f} vs {e2:.0f} us, saving {(e2 - s2) / 1000:.3f} ms; qkvz {qk:.2f} us; "
            f"M16 step {s16:.0f} vs {e16:.0f} ({100 * reg16:+.1f}%); worst per-shape M16 {100 * shape16:+.1f}%; "
            f"invariant {inv[key]}, replay {stress[key]} -> {'PASS' if passed else 'FAIL'}")
    return 0


# ============================================ plan 20: 4x4x4 MFMA GEMV with reduce-scatter (f2)
F0_J124 = {  # j124's best skf ("f") configs: (R, P, G, hoist)
    (32, 2048): (1, 8, 1, 0), (256, 2048): (1, 8, 1, 0), (512, 2048): (2, 8, 1, 0),
    (2048, 256): (1, 1, 1, 0), (2048, 2048): (1, 8, 1, 0), (4608, 2048): (2, 8, 1, 1),
    (6144, 2048): (2, 8, 1, 1),
}
QKVZ_N, BA_N = 6144, 32
MOE = ("moe_router", "shared_gate_up", "shared_down")


def layer_order_packed(moe=True):
    """One decode step with qkvz+ba as one packed call ("in_proj"): 200 calls (80 without MoE)."""
    seq = []
    for layer in range(40):
        if layer % 4 == 3:
            seq += ["attn_qkv", "attn_o"]
        else:
            seq += ["in_proj", "gdn_out"]
        if moe:
            seq += list(MOE)
    return seq


def main_g():
    from sglang.kernels.ops.gemm import skinny_gemm_packed_hip as pk

    torch.manual_seed(0)
    props = torch.cuda.get_device_properties(0)
    log("device:", torch.cuda.get_device_name(0), props.gcnArchName, "CUs", props.multi_processor_count)
    log("torch", torch.__version__, "hip", torch.version.hip, "triton", triton.__version__)
    t0 = time.time()
    lib = hg.build(defines=("-DSKG_SWEEP", "-DSKG_NO_SCALAR"))
    log(f"hip build+load {time.time() - t0:.1f}s: {len(lib.f_entries)} f + {len(lib.f2_entries)} f2 kernels")
    ws = make_weights()
    ws["in_proj"] = [torch.cat([a, b_]).contiguous() for a, b_ in zip(ws["gdn_qkvz"], ws["gdn_ba"])]
    names = ["in_proj"] + [nm for nm, *_ in SHAPES if nm not in ("gdn_qkvz", "gdn_ba")]
    allnames = names + ["gdn_qkvz", "gdn_ba"]
    shape_of = {nm: tuple(ws[nm][0].shape) for nm in allnames}
    count = {nm: c for nm, _, _, c in SHAPES}
    count["in_proj"] = 30
    xs = {}
    for m in MS_ALL:
        for k in (2048, 256):
            xs[(m, k)] = torch.randn(m, k, device=DEV).to(torch.bfloat16)

    def head(x, w, name):
        if name == "in_proj":
            return pk.skinny_linear_pair(x, w, QKVZ_N)
        return sk.skinny_linear(x, w, None)

    log("head _GEMV_TUNED:", {k: (v[0], tuple(v[1])) for k, v in sk._GEMV_TUNED.items()})

    def f2op(cfg, idx):
        return lambda x, w, _n: hg.mfma_gemv(lib, x, w, None, cfg, idx, "f2")

    def fop(cfg, idx):
        return lambda x, w, _n: hg.mfma_gemv(lib, x, w, None, cfg, idx, "f")

    t_head = {nm: {m: per_shape(head, ws, xs, nm, m) for m in MS_SHAPE} for nm in allnames}

    # ---------------- f0 = j124's skf configs
    f0 = {}
    for nm in allnames:
        n, k = shape_of[nm]
        c = F0_J124.get((n, k), F0_J124[(6144, 2048)])
        cfg = hg.MfmaConfig(*c, 0)
        f0[nm] = (cfg, lib.q_find(cfg, k, "f"))
    t_f0 = {nm: {m: per_shape(fop(*f0[nm]), ws, xs, nm, m) for m in MS_SHAPE} for nm in allnames}

    # ---------------- f2 sweep per shape (M=2, M=16)
    t0 = time.time()
    sweep = {}
    for nm in allnames:
        n, k = shape_of[nm]
        rows = []
        for cfg in lib.f2_configs(k):
            idx = lib.q_find(cfg, k, "f2")
            op = f2op(cfg, idx)
            try:
                rows.append((per_shape(op, ws, xs, nm, 2), per_shape(op, ws, xs, nm, 16), cfg, idx))
            except Exception as ex:  # noqa: BLE001
                log("f2 cfg fail", nm, cfg, repr(ex)[:200])
        sweep[nm] = rows
    log(f"f2 sweep took {time.time() - t0:.0f}s")

    classes = {
        "f0r": lambda c: c.red == 0 and c.rows <= 2,  # skf's reduce, re-implemented
        "fA": lambda c: c.red == 1 and c.rows <= 2,
        "fB": lambda c: c.red == 0 and c.rows >= 4,
        "fAB": lambda c: c.red == 1 and c.rows >= 4,
    }
    pick = {}  # (class, name) -> (t2, t16, cfg, idx, within_limit)
    for nm in allnames:
        lim = 1.05 * t_head[nm][16]
        log(f"[sweep f2] {nm} {shape_of[nm]}: head M2={t_head[nm][2]:.2f} M16={t_head[nm][16]:.2f} "
            f"(limit {lim:.2f}); f0 M2={t_f0[nm][2]:.2f} M16={t_f0[nm][16]:.2f}")
        for cl, pred in list(classes.items()) + [("any", lambda c: True)]:
            rows = [r for r in sweep[nm] if pred(r[2])]
            if not rows:
                continue
            ok = sorted([r for r in rows if r[1] <= lim], key=lambda r: r[0])
            best = ok[0] if ok else min(rows, key=lambda r: r[0] + 0.25 * r[1])
            pick[(cl, nm)] = (*best, bool(ok))
            fast16 = min(rows, key=lambda r: r[1])
            log(f"   {cl:4s} pick M2={best[0]:.2f} M16={best[1]:.2f} {tuple(best[2])} within={bool(ok)}; "
                f"fastest M16 {fast16[1]:.2f} {tuple(fast16[2])}")

    # ---------------- diagnostics at M=2 / M=16 (timing only)
    log("\n## diagnostics (P=8, G=1): us at M=2 / M=16; base, no cross-lane exchange, one x load")
    for nm in ("in_proj", "gdn_out", "attn_qkv"):
        k = shape_of[nm][1]
        for red in (0, 1):
            for r_ in (2, 4, 8):
                cells = []
                for dg in (0, 1, 2):
                    cfg = hg.F2Config(r_, 8, 1, 0, red, dg)
                    idx = lib.q_find(cfg, k, "f2")
                    if idx < 0:
                        cells.append("n/a")
                        continue
                    op = f2op(cfg, idx)
                    cells.append(f"{per_shape(op, ws, xs, nm, 2):.2f}/{per_shape(op, ws, xs, nm, 16):.2f}")
                log(f"   {nm} red={red} R={r_}: base {cells[0]}  noexch {cells[1]}  onex {cells[2]}")

    # ---------------- mix: per shape the fastest at M=2 of head and f2 with M16 <= 1.05 head
    choice = {}
    for nm in allnames:
        r = pick.get(("any", nm))
        use = r is not None and r[4] and r[0] < t_head[nm][2]
        choice[nm] = (r[2], r[3]) if use else None
    # qkvz and ba alone must give the packed bits: same config as in_proj (rows depend on K, P only)
    if choice["in_proj"] is not None:
        cfg = choice["in_proj"][0]
        for nm in ("gdn_qkvz", "gdn_ba"):
            choice[nm] = (cfg, lib.q_find(cfg, shape_of[nm][1], "f2"))
    else:
        choice["gdn_qkvz"] = choice["gdn_ba"] = None
    log("mix choice:", {nm: (tuple(c[0]) if c else "head") for nm, c in choice.items()})

    def mix(x, w, name):
        c = choice.get(name)
        if c is None:
            return head(x, w, name)
        return hg.mfma_gemv(lib, x, w, None, c[0], c[1], "f2")

    def cls_op(cl):
        def op(x, w, name):
            r = pick.get((cl, name))
            if r is None:
                return head(x, w, name)
            return hg.mfma_gemv(lib, x, w, None, r[2], r[3], "f2")
        return op

    ops = {"head": head, "f0": lambda x, w, nm: hg.mfma_gemv(lib, x, w, None, f0[nm][0], f0[nm][1], "f"),
           "fA": cls_op("fA"), "fB": cls_op("fB"), "fAB": cls_op("fAB"), "mix": mix}
    keys = list(ops)
    res = {key: {(nm, m): per_shape(ops[key], ws, xs, nm, m) for nm in allnames for m in MS_SHAPE} for key in keys}
    r_us = {nm: read_us(lib, ws[nm]) for nm in allnames}

    # ---------------- correctness of the mix (and every f2 config it uses)
    bad = []
    for nm in allnames:
        k = shape_of[nm][1]
        v = invariant(mix, ws, xs, nm, k, reps=100)
        if v != "ok":
            bad.append(f"{nm}: {v}")
    inv_ok = not bad
    log(f"[check] mix: bitwise batch-invariant (M=1..16, contiguous and view) and 100 repeats: {inv_ok} {bad}")
    errs = {nm: {m: (max_err(mix, ws, xs, nm, shape_of[nm][1], m), max_err(head, ws, xs, nm, shape_of[nm][1], m))
                 for m in (1, 2, 16)} for nm in allnames}
    err_ok = all(a <= 1.5 * b + 1e-3 for d in errs.values() for a, b in d.values())
    log(f"[check] mix max err vs fp32 within 1.5x head's: {err_ok}")
    for nm, d in errs.items():
        log(f"   {nm}: " + ", ".join(f"M{m} {a:.4g}/{b:.4g}" for m, (a, b) in d.items()))
    packed_eq = True
    if choice["in_proj"] is not None:
        for m in MS_ALL:
            x = xs[(m, 2048)]
            yp = mix(x, ws["in_proj"][0], "in_proj")
            if not (torch.equal(yp[:, :QKVZ_N], mix(x, ws["gdn_qkvz"][0], "gdn_qkvz"))
                    and torch.equal(yp[:, QKVZ_N:], mix(x, ws["gdn_ba"][0], "gdn_ba"))):
                packed_eq = False
                log(f"[check] packed != unpacked at M={m}")
    log(f"[check] packed in_proj == qkvz and ba alone, bitwise, M=1..16: {packed_eq}")
    bias_ok = True
    for nm in allnames:
        c = choice.get(nm)
        if c is None:
            continue
        n, k = shape_of[nm]
        w = ws[nm][0]
        b = (torch.randn(n, device=DEV) * 0.1).to(torch.bfloat16)
        for m in (1, 3, 16):
            x = xs[(m, k)]
            ref = x.float() @ w.float().t() + b.float()
            eq = (hg.mfma_gemv(lib, x, w, b, c[0], c[1], "f2").float() - ref).abs().max().item()
            ee = (x.float() @ w.float().t()).to(torch.bfloat16).float().add(b.float()).sub(ref).abs().max().item()
            if eq > 1.5 * ee + 2e-2:
                bias_ok = False
                log(f"[check] bias {nm} M={m}: {eq:.4g} (ref rounding {ee:.4g})")
    log(f"[check] f2 bias path: {bias_ok}")
    stress_ok = True
    for nm in allnames:
        k = shape_of[nm][1]
        wl = ws[nm]
        for m in (2, 16):
            xin = torch.randn(m, k, device=DEV).to(torch.bfloat16)
            outs = []

            def fn():
                outs.clear()
                for w in wl:
                    outs.append(mix(xin, w, nm))

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                fn()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            for it in range(10):
                xin.copy_(torch.randn(m, k, device=DEV))
                g.replay()
                torch.cuda.synchronize()
                for i in (0, len(wl) - 1):
                    if not torch.equal(outs[i], mix(xin, wl[i], nm)):
                        stress_ok = False
            del g
    log(f"[check] mix graph replay with fresh inputs == eager: {stress_ok}")

    # ---------------- step graphs (packed in_proj = one call)
    def step_seq(op, m, seq):
        idx = {nm: 0 for nm in names}
        calls = []
        for nm in seq:
            calls.append((nm, ws[nm][idx[nm]]))
            idx[nm] += 1

        def fn():
            for nm, w in calls:
                op(xs[(m, w.shape[1])], w, nm)

        return graph_time(fn, iters=10)

    seq_all, seq_nomoe = layer_order_packed(True), layer_order_packed(False)
    stp = {key: {m: step_seq(ops[key], m, seq_all) for m in MS_ALL} for key in ("head", "f0", "fAB", "mix")}
    stp_nm = {key: {m: step_seq(ops[key], m, seq_nomoe) for m in (1, 2, 4, 8, 16)} for key in ("head", "mix")}
    for key in ("head", "mix"):
        log(f"[{key}] step M=2 repeats: {[round(step_seq(ops[key], 2, seq_all)) for _ in range(2)]}")

    # ---------------- tables
    log("\n## per-shape us/call (graph replay, own weight per call); in_proj = packed qkvz+ba [6176, 2048]")
    for m in MS_SHAPE:
        log(f"\nM={m}")
        log(" | ".join(["shape", "N", "K", "calls"] + keys + ["r"] + [f"{k} GB/s" for k in ("head", "mix")]))
        for nm in allnames:
            n, k = shape_of[nm]
            us = [res[key][(nm, m)] for key in keys] + [r_us[nm]]
            gb = [wbytes(n, k, m) / res[key][(nm, m)] / 1e3 for key in ("head", "mix")]
            log(" | ".join([nm, str(n), str(k), str(count.get(nm, 0))] + [f"{u:.2f}" for u in us] + [f"{g:.0f}" for g in gb]))
        log(" | ".join(["sum(count*us), packed in_proj", "", "", "200"] + [
            f"{sum(res[key][(nm, m)] * count[nm] for nm in names):.0f}" for key in keys]
            + [f"{sum(r_us[nm] * count[nm] for nm in names):.0f}"]))
    log("\n## 200-call step graph us by M (packed in_proj as one call)")
    log(" | ".join(["M", "head", "f0", "fAB", "mix"]))
    for m in MS_ALL:
        log(" | ".join([str(m)] + [f"{stp[k][m]:.0f}" for k in ("head", "f0", "fAB", "mix")]))
    log("\n## non-MoE step graph (80 calls: in_proj, gdn_out, attn_qkv, attn_o) us by M")
    log(" | ".join(["M", "head", "mix"]))
    for m in (1, 2, 4, 8, 16):
        log(" | ".join([str(m)] + [f"{stp_nm[k][m]:.0f}" for k in ("head", "mix")]))
    log("\nMIX_TUNED (f2: R, P, G, hoist, red) = {")
    for nm in allnames:
        c = choice[nm]
        log(f"    {shape_of[nm]}: {tuple(c[0])[:5] if c else None},  # {nm}")
    log("}")

    # ---------------- gate
    h2, h16 = stp["head"][2], stp["head"][16]
    m2, m16 = stp["mix"][2], stp["mix"][16]
    nm_save = stp_nm["head"][2] - stp_nm["mix"][2]
    worst = max((res["mix"][(nm, 16)] / res["head"][(nm, 16)] - 1, nm) for nm in names if choice[nm] is not None) \
        if any(choice[nm] for nm in names) else (0.0, "-")
    passed = (inv_ok and stress_ok and err_ok and packed_eq and (h2 - m2) >= 500
              and worst[0] <= 0.05 and m16 / h16 - 1 <= 0.05)
    log(f"\nGATE: step M2 {m2:.0f} vs head {h2:.0f} us, saving {(h2 - m2) / 1000:.3f} ms (>= 0.5); "
        f"non-MoE saving {nm_save / 1000:.3f} ms (>= 0.4 after plan 13); M16 step {m16:.0f} vs {h16:.0f} "
        f"({100 * (m16 / h16 - 1):+.1f}%); worst chosen shape M16 {100 * worst[0]:+.1f}% ({worst[1]}); "
        f"invariant {inv_ok}, replay {stress_ok}, err {err_ok}, packed bitwise {packed_eq} -> "
        f"{'PASS' if passed else 'FAIL'}")
    return 0
