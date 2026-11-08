"""Plan 11 step 1: skinny GEMM v2 microbench on one gfx90a GCD (run through bench.py).

Options, each replaying one TP=2 decode step's dense GEMMs (every call its own weight) in a
HIP graph:
  a   F.linear (hipBLASLt), for reference
  c   today's kernel: split-K GEMM + _ordered_reduce_kernel, today's _TUNED schedules
  c0  today's GEMM kernel alone (no reduce; wrong output, timing only): splits c by kernel
  e   the fused last-block reduce with today's schedules (must equal c bitwise)
  f   e plus a schedule sweep for the large shapes (6144x2048, 4608x2048, 2048x2048)
  g   a register (non-MFMA) GEMV variant with the fused reduce, large shapes only
Correctness: e == c bitwise on every shape and M = 1..16 (with and without bias), bitwise
batch invariance (rows of M = 16 == M = 1..16 calls) and repeatability for c/e/f/g, and a
fresh-input graph-replay stress of the fused reduce against eager c.
"""

import itertools
import math
import time
import traceback

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.kernels.ops.gemm import skinny_gemm_hip as sk
from sglang.kernels.ops.gemm.skinny_gemm_hip import _last_block_reduce

from bench import SHAPES, DEV, graph_time, layer_order, make_weights

MS_ALL = list(range(1, 17))
MS_SHAPE = [1, 2, 16]
LARGE = [(6144, 2048), (4608, 2048), (2048, 2048)]
NAME_OF = {}
for _name, _n, _k, _c in SHAPES:
    NAME_OF.setdefault((_n, _k), []).append(_name)


def bytes_of(n, k, m):
    return 2 * n * k + 2 * m * k + 2 * m * n


# ------------------------------------------------------------------ register GEMV (g)
@triton.jit
def _reg_gemv_kernel(
    x_ptr, w_ptr, y_ptr, ws_ptr, cnt_ptr, M, N, K, stride_xm, stride_wn, stride_ym,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_split = K // SPLIT_K
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * k_split + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, k_split, BLOCK_K):
        xv = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
        wv = tl.load(w_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(xv[:, None, :] * wv[None, :, :], axis=2)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    mask = mask_m[:, None] & mask_n[None, :]
    if SPLIT_K == 1:
        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :]
        tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask)
    else:
        ws_ptrs = ws_ptr + pid_k * M * N + offs_m[:, None] * N + offs_n[None, :]
        tl.store(ws_ptrs, acc, mask=mask)
        _last_block_reduce(
            ws_ptr, y_ptr, y_ptr, cnt_ptr, pid_n, offs_m, offs_n, mask, mask_n,
            M, N, stride_ym, False, SPLIT_K,
        )


def reg_gemv(x, w, cfg):
    M, K = x.shape
    N = w.shape[0]
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    grid_n = triton.cdiv(N, cfg.block_n)
    ws = y if cfg.split_k == 1 else torch.empty((cfg.split_k, M, N), dtype=torch.float32, device=x.device)
    cnt = sk._ticket_counters(x.device)
    _reg_gemv_kernel[(grid_n, cfg.split_k)](
        x, w, y, ws, cnt, M, N, K, x.stride(0), w.stride(0), y.stride(0),
        BLOCK_M=sk.MAX_M, BLOCK_N=cfg.block_n, BLOCK_K=cfg.block_k, SPLIT_K=cfg.split_k,
        num_warps=cfg.num_warps, num_stages=cfg.num_stages,
    )
    return y


def gemm_only(x, w, cfg):
    """Today's GEMM kernel without its reduce (timing only)."""
    M, K = x.shape
    N = w.shape[0]
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    ws = y if cfg.split_k == 1 else torch.empty((cfg.split_k, M, N), dtype=torch.float32, device=x.device)
    sk._skinny_gemm_kernel[(triton.cdiv(N, cfg.block_n), cfg.split_k)](
        x, w, w, y, ws, ws, M, N, K, x.stride(0), w.stride(0), y.stride(0),
        HAS_BIAS=False, BLOCK_M=sk.MAX_M, BLOCK_N=cfg.block_n, BLOCK_K=cfg.block_k,
        SPLIT_K=cfg.split_k, FUSED_REDUCE=False, num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    return y


# ------------------------------------------------------------------ timing helpers
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


def invariant(op, ws, xs, name, k, reps=3):
    """Bitwise: rows of an M=16 call == M=1..16 calls; and repeated calls agree."""
    w = ws[name][0]
    x16 = xs[(16, k)]
    y16 = op(x16, w, name)
    for m in range(1, 17):
        ym = op(x16[:m].contiguous(), w, name)
        if not torch.equal(ym, y16[:m]):
            return f"batch-invariance FAIL at M={m}"
    for _ in range(reps):
        if not torch.equal(op(x16, w, name), y16):
            return "repeatability FAIL"
    return "ok"


def max_err(op, ws, xs, name, k, m):
    w = ws[name][0]
    x = xs[(m, k)]
    ref = x.float() @ w.float().t()
    return (op(x, w, name).float() - ref).abs().max().item()


def main_v2():
    torch.manual_seed(0)
    props = torch.cuda.get_device_properties(0)
    print("device:", torch.cuda.get_device_name(0), props.gcnArchName, "CUs", props.multi_processor_count, flush=True)
    print("torch", torch.__version__, "hip", torch.version.hip, "triton", triton.__version__, flush=True)
    ws = make_weights()
    xs = {}
    for m in MS_ALL:
        for k in (2048, 256):
            xs[(m, k)] = torch.randn(m, k, device=DEV).to(torch.bfloat16)
    seq = layer_order()
    tuned = {name: sk.default_config(n, k) for name, n, k, _ in SHAPES}
    print("today's schedules:", {k: tuple(v) for k, v in tuned.items()}, flush=True)

    ops = {
        "a_flinear": lambda x, w, name: F.linear(x, w),
        "c_today": lambda x, w, name: sk.skinny_gemm(x, w, None, tuned[name], fused_reduce=False),
        "c0_gemm_only": lambda x, w, name: gemm_only(x, w, tuned[name]),
        "e_fused": lambda x, w, name: sk.skinny_gemm(x, w, None, tuned[name], fused_reduce=True),
    }

    # ---------------- correctness of e: bitwise == c on every shape, M = 1..16, +-bias
    t0 = time.time()
    e_eq = True
    for name, n, k, _ in SHAPES:
        w = ws[name][0]
        bias = (torch.randn(n, device=DEV) * 0.1).to(torch.bfloat16)
        for m in MS_ALL:
            x = xs[(m, k)]
            for b in (None, bias):
                yc = sk.skinny_gemm(x, w, b, tuned[name], fused_reduce=False)
                ye = sk.skinny_gemm(x, w, b, tuned[name], fused_reduce=True)
                if not torch.equal(yc, ye):
                    e_eq = False
                    print(f"e != c: {name} M={m} bias={b is not None} maxdiff="
                          f"{(yc.float() - ye.float()).abs().max().item()}", flush=True)
    print(f"[check] e == c bitwise on all shapes, M=1..16, with/without bias: {e_eq}", flush=True)

    # ---------------- fresh-input graph-replay stress: fused vs eager two-kernel
    stress_ok = True
    for name, n, k, _ in SHAPES:
        cfg = tuned[name]
        wl = ws[name]
        for m in (2, 16):
            xin = torch.empty(m, k, device=DEV, dtype=torch.bfloat16)
            xin.copy_(torch.randn(m, k, device=DEV))
            outs = []

            def fn():
                outs.clear()
                for w in wl:
                    outs.append(sk.skinny_gemm(xin, w, None, cfg, fused_reduce=True))

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                fn()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            for it in range(40):
                xin.copy_(torch.randn(m, k, device=DEV))
                g.replay()
                torch.cuda.synchronize()
                for i in (0, len(wl) // 2, len(wl) - 1):
                    ref = sk.skinny_gemm(xin, wl[i], None, cfg, fused_reduce=False)
                    if not torch.equal(outs[i], ref):
                        stress_ok = False
                        print(f"graph stress FAIL {name} M={m} iter={it} call={i}", flush=True)
                        break
            del g
    leftover = sum(int(b.abs().sum().item()) for b in sk._counters.values())
    print(f"[check] fused graph-replay stress (fresh inputs, 40 replays x shapes x M=2,16): {stress_ok}; "
          f"ticket counters all zero afterwards: {leftover == 0}", flush=True)
    print(f"correctness took {time.time() - t0:.0f}s", flush=True)

    # ---------------- (f) schedule sweep for the large shapes, fused reduce
    t0 = time.time()
    sweep_best = {}
    sweep_rows = {}
    for n, k in LARGE:
        name = NAME_OF[(n, k)][0]
        rows = []
        for bn, bk, spk, nw, ns in itertools.product((16, 32, 64), (128, 256, 512), (1, 2, 4, 8), (4, 8), (2, 3)):
            if k % (spk * bk) != 0:
                continue
            cfg = sk.SkinnyConfig(bn, bk, spk, nw, ns)
            op = lambda x, w, _n, c=cfg: sk.skinny_gemm(x, w, None, c, fused_reduce=True)
            try:
                t2 = per_shape(op, ws, xs, name, 2)
                t16 = per_shape(op, ws, xs, name, 16)
            except Exception as ex:  # noqa: BLE001
                print("cfg fail", name, cfg, repr(ex)[:200], flush=True)
                continue
            rows.append((t2 + 0.25 * t16, t2, t16, cfg))
        rows.sort(key=lambda r: r[0])
        sweep_rows[(n, k)] = rows
        print(f"[sweep f] {name} N={n} K={k}: {len(rows)} configs, top 8 (score = M2 + 0.25*M16)", flush=True)
        for r in rows[:8]:
            print(f"   M2={r[1]:.2f}us M16={r[2]:.2f}us {tuple(r[3])}", flush=True)
        dflt = [r for r in rows if r[3] == sk.SkinnyConfig(*tuple(tuned[name]))]
        if dflt:
            print(f"   today's schedule fused {tuple(dflt[0][3])}: M2={dflt[0][1]:.2f} M16={dflt[0][2]:.2f}", flush=True)
        # Keep the best one that is bitwise batch-invariant and repeatable.
        for r in rows[:10]:
            op = lambda x, w, _n, c=r[3]: sk.skinny_gemm(x, w, None, c, fused_reduce=True)
            verdict = invariant(op, ws, xs, name, k)
            if verdict == "ok":
                sweep_best[(n, k)] = r[3]
                break
            print(f"   {tuple(r[3])} rejected: {verdict}", flush=True)
    print(f"sweep f took {time.time() - t0:.0f}s", flush=True)
    f_cfg = dict(tuned)
    for name, n, k, _ in SHAPES:
        if (n, k) in sweep_best:
            f_cfg[name] = sweep_best[(n, k)]
    print("F_SCHEDULES = {", flush=True)
    for name, n, k, _ in SHAPES:
        print(f"    ({n}, {k}): {tuple(f_cfg[name])},  # {name}", flush=True)
    print("}", flush=True)
    ops["f_fused_sweep"] = lambda x, w, name: sk.skinny_gemm(x, w, None, f_cfg[name], fused_reduce=True)

    def invariant_all(key):
        bad = []
        for name, n, k, _ in SHAPES:
            v = invariant(ops[key], ws, xs, name, k)
            if v != "ok":
                bad.append(f"{name}: {v}")
        print(f"[check] {key} bitwise batch-invariant (M=1..16) and repeatable: {not bad} {bad}", flush=True)
        return not bad

    for key in ("c_today", "e_fused", "f_fused_sweep"):
        invariant_all(key)

    res = {}

    def run_tables(keys_to_run):
        for key in keys_to_run:
            op = ops[key]
            r = {"shape": {}, "step": {}, "err": {}}
            try:
                for m in MS_SHAPE:
                    for name, n, k, _ in SHAPES:
                        r["shape"][(name, m)] = per_shape(op, ws, xs, name, m)
                for m in MS_ALL:
                    r["step"][m] = step(op, ws, xs, m, seq)
                if key != "c0_gemm_only":
                    for m in MS_SHAPE:
                        r["err"][m] = max(max_err(op, ws, xs, name, k, m) for name, n, k, _ in SHAPES)
            except Exception:  # noqa: BLE001
                print(f"{key} failed:", traceback.format_exc()[-1500:], flush=True)
            res[key] = r
            print(f"[{key}] step us: " + ", ".join(f"M{m}={v:.0f}" for m, v in r["step"].items()), flush=True)
        print_tables()

    def print_tables():
        keys = list(res)
        print("\n## per-shape us/call and GB/s (graph replay of all calls of the shape, own weight each)")
        for m in MS_SHAPE:
            print(f"\nM={m}")
            print(" | ".join(["shape", "N", "K", "calls"] + [f"{k} us" for k in keys] + [f"{k} GB/s" for k in keys]))
            for name, n, k, cnt in SHAPES:
                us = [res[key]["shape"].get((name, m), float("nan")) for key in keys]
                gbs = [bytes_of(n, k, m) / u / 1e3 for u in us]
                print(" | ".join([name, str(n), str(k), str(cnt)] + [f"{u:.2f}" for u in us] + [f"{g:.0f}" for g in gbs]))
            print(" | ".join(["sum(count*us)", "", "", "230"] + [
                f"{sum(res[key]['shape'].get((nm, m), float('nan')) * c for nm, _, _, c in SHAPES):.0f}" for key in keys]))
            large = [nm for nm, n, k, _ in SHAPES if (n, k) in LARGE]
            print(" | ".join(["  of which large", "", "", "80"] + [
                f"{sum(res[key]['shape'].get((nm, m), float('nan')) * c for nm, _, _, c in SHAPES if nm in large):.0f}" for key in keys]))
        print("\n## 230-call step graph us by M")
        print(" | ".join(["M"] + keys))
        for m in MS_ALL:
            print(" | ".join([str(m)] + [f"{res[k]['step'].get(m, float('nan')):.0f}" for k in keys]))
        print("\n## max abs err vs fp32 reference")
        for k in keys:
            print(k, {m: round(v, 4) for m, v in res[k]["err"].items()}, flush=True)

    run_tables(list(ops))

    # ---------------- (g) register GEMV, fused reduce, large shapes only
    t0 = time.time()
    g_best = {}
    try:
        for n, k in LARGE:
            name = NAME_OF[(n, k)][0]
            rows = []
            for bn, bk, spk, nw in itertools.product((8, 16, 32), (64, 128), (1, 2, 4, 8), (4, 8)):
                if k % (spk * bk) != 0 or sk.MAX_M * bn * bk > 128 * 64 * nw:
                    continue
                cfg = sk.SkinnyConfig(bn, bk, spk, nw, 2)
                op = lambda x, w, _n, c=cfg: reg_gemv(x, w, c)
                try:
                    t2 = per_shape(op, ws, xs, name, 2)
                    t16 = per_shape(op, ws, xs, name, 16)
                except Exception as ex:  # noqa: BLE001
                    print("g cfg fail", name, cfg, repr(ex)[:200], flush=True)
                    continue
                rows.append((t2 + 0.25 * t16, t2, t16, cfg))
            rows.sort(key=lambda r: r[0])
            print(f"[sweep g] {name} N={n} K={k}: {len(rows)} configs, top 5", flush=True)
            for r in rows[:5]:
                print(f"   M2={r[1]:.2f}us M16={r[2]:.2f}us {tuple(r[3])}", flush=True)
            for r in rows[:5]:
                op = lambda x, w, _n, c=r[3]: reg_gemv(x, w, c)
                verdict = invariant(op, ws, xs, name, k)
                if verdict == "ok":
                    g_best[(n, k)] = r[3]
                    break
                print(f"   {tuple(r[3])} rejected: {verdict}", flush=True)
        g_cfg = {}
        for name, n, k, _ in SHAPES:
            g_cfg[name] = g_best.get((n, k))

        def g_op(x, w, name):
            c = g_cfg[name]
            if c is None:
                return sk.skinny_gemm(x, w, None, f_cfg[name], fused_reduce=True)
            return reg_gemv(x, w, c)

        if g_best:
            ops["g_reg_large"] = g_op
    except Exception:  # noqa: BLE001
        print("g failed:", traceback.format_exc()[-1500:], flush=True)
    print(f"sweep g took {time.time() - t0:.0f}s", flush=True)

    if "g_reg_large" in ops:
        v = invariant_all("g_reg_large")
        run_tables(["g_reg_large"])

    c2 = res["c_today"]["step"].get(2, float("nan"))
    e2 = res["e_fused"]["step"].get(2, float("nan"))
    cands = [(res[k]["step"].get(2, float("inf")), k) for k in ("e_fused", "f_fused_sweep", "g_reg_large") if k in res]
    best = min(cands)
    print(f"\nKILL GATE @M=2: c={c2:.0f}us e={e2:.0f}us best={best[1]} {best[0]:.0f}us "
          f"saving={(c2 - best[0]) / 1000:.3f} ms -> {'PASS' if c2 - best[0] >= 400 else 'KILL'}")
    for n, k in LARGE:
        for name in NAME_OF[(n, k)]:
            ec = res["e_fused"]["shape"].get((name, 2))
            for key in ("f_fused_sweep", "g_reg_large"):
                if key in res:
                    v = res[key]["shape"].get((name, 2))
                    print(f"  large-shape gain @M=2 {name} {key} vs e: {100 * (1 - v / ec):.1f}% ({ec:.2f} -> {v:.2f} us)")
    return 0
