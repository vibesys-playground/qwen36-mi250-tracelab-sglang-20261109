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
