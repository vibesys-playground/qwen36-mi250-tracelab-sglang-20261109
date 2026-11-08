"""Plan 7 step 1 (SKINNY_BENCH=v1; v2 = plan 11; v3 = plan 18 scalar GEMV; the default runs plan 18's MFMA v3.main_q): replay one TP=2 decode step's dense GEMMs on one gfx90a GCD.

Options: (a) F.linear (hipBLASLt), (c) Triton split-K skinny GEMM, (d) sgl tiny_gemm if it
builds, (b) PyTorch TunableOp (run last: it changes F.linear globally). Every call has its own
weight, as in the model, and runs inside a HIP graph. Prints us per shape and per step.
"""

import math
import os
import sys
import time
import traceback

import torch
import torch.nn.functional as F

from sglang.kernels.ops.gemm import skinny_gemm_hip as sk

DEV = torch.device("cuda", 0)
MS = [1, 2, 4, 8, 16]
K = 2048
# (name, N, K, count per step); TP=2 per-rank shapes of Qwen3.6-35B-A3B (plan 7).
SHAPES = [
    ("gdn_qkvz", 6144, 2048, 30),
    ("gdn_ba", 32, 2048, 30),
    ("gdn_out", 2048, 2048, 30),
    ("attn_qkv", 4608, 2048, 10),
    ("attn_o", 2048, 2048, 10),
    ("moe_router", 256, 2048, 40),
    ("shared_gate_up", 512, 2048, 40),
    ("shared_down", 2048, 256, 40),
]


def layer_order():
    """Call sequence of one step: 40 layers, every 4th full attention, each followed by MoE."""
    seq = []
    for layer in range(40):
        if layer % 4 == 3:
            seq += ["attn_qkv", "attn_o"]
        else:
            seq += ["gdn_qkvz", "gdn_ba", "gdn_out"]
        seq += ["moe_router", "shared_gate_up", "shared_down"]
    return seq


def make_weights():
    g = torch.Generator(device=DEV).manual_seed(0)
    ws = {}
    for name, n, k, cnt in SHAPES:
        ws[name] = [
            (torch.randn(n, k, device=DEV, generator=g) * (1.0 / math.sqrt(k))).to(torch.bfloat16)
            for _ in range(cnt)
        ]
    return ws


def graph_time(fn, iters=20):
    """us per replay of a graph that runs fn() once."""
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
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    best = []
    for _ in range(3):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(iters):
            g.replay()
        en.record()
        torch.cuda.synchronize()
        best.append(st.elapsed_time(en) * 1000.0 / iters)
    del g
    return min(best)


def per_shape_us(op, ws, xs, name, m):
    wl = ws[name]
    x = xs[(m, wl[0].shape[1])]

    def fn():
        for w in wl:
            op(x, w, name)

    return graph_time(fn) / len(wl)


def step_us(op, ws, xs, m, seq):
    idx = {n: 0 for n, *_ in SHAPES}
    calls = []
    for name in seq:
        calls.append((name, ws[name][idx[name]]))
        idx[name] += 1

    def fn():
        for name, w in calls:
            op(xs[(m, w.shape[1])], w, name)

    return graph_time(fn, iters=10)


def err_vs_ref(op, ws, xs, m):
    worst = 0.0
    for name, n, k, _ in SHAPES:
        w = ws[name][0]
        x = xs[(m, k)]
        ref = x.float() @ w.float().t()
        y = op(x, w, name).float()
        worst = max(worst, (y - ref).abs().max().item())
    return worst


def fmt_row(cells):
    return " | ".join(str(c) for c in cells)


def main():
    torch.manual_seed(0)
    print("device:", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).gcnArchName,
          "CUs", torch.cuda.get_device_properties(0).multi_processor_count, flush=True)
    print("torch", torch.__version__, "hip", torch.version.hip, flush=True)
    import triton
    print("triton", triton.__version__, flush=True)
    ws = make_weights()
    xs = {}
    for m in MS:
        for k in (2048, 256):
            xs[(m, k)] = torch.randn(m, k, device=DEV).to(torch.bfloat16)
    seq = layer_order()
    print("calls per step:", len(seq), flush=True)

    # ---- (c) Triton config sweep per (N,K), schedule fixed across M (chosen on M=2 + M=16).
    t0 = time.time()
    chosen = {}
    for name, n, k, _ in SHAPES:
        rows = []
        for bn in (16, 32, 64):
            for bk in (128, 256, 512):
                for spk in (1, 2, 4, 8, 16):
                    for nw in (4, 8):
                        cfg = sk.SkinnyConfig(bn, bk, spk, nw, 2)
                        if k % (spk * bk) != 0:
                            continue
                        grid = math.ceil(n / bn) * spk
                        if grid > 2048 or (spk > 1 and grid > 1024 and k // spk < 256):
                            continue
                        try:
                            t2 = per_shape_us(lambda x, w, _n, c=cfg: sk.skinny_gemm(x, w, None, c), ws, xs, name, 2)
                            t16 = per_shape_us(lambda x, w, _n, c=cfg: sk.skinny_gemm(x, w, None, c), ws, xs, name, 16)
                        except Exception as e:  # noqa: BLE001
                            print("cfg fail", name, cfg, repr(e)[:200], flush=True)
                            continue
                        rows.append((t2 + 0.25 * t16, t2, t16, cfg))
        rows.sort(key=lambda r: r[0])
        chosen[name] = rows[0][3]
        print(f"[sweep] {name} N={n} K={k}: top5", flush=True)
        for r in rows[:5]:
            print(f"   M2={r[1]:.2f}us M16={r[2]:.2f}us {r[3]}", flush=True)
        dflt = sk.default_config(n, k)
        dr = [r for r in rows if r[3] == dflt]
        if dr:
            print(f"   default {dflt}: M2={dr[0][1]:.2f} M16={dr[0][2]:.2f}", flush=True)
    print(f"sweep took {time.time() - t0:.0f}s", flush=True)
    print("CHOSEN = {", flush=True)
    for name, n, k, _ in SHAPES:
        print(f"    ({n}, {k}): {tuple(chosen[name])},  # {name}", flush=True)
    print("}", flush=True)

    ops = {}
    ops["a_flinear"] = lambda x, w, name: F.linear(x, w)
    ops["c_triton"] = lambda x, w, name: sk.skinny_gemm(x, w, None, chosen[name])

    # ---- (d) tiny_gemm
    try:
        from sglang.kernels.ops.gemm.tiny_gemm import tiny_gemm_bf16

        y = tiny_gemm_bf16(xs[(2, 2048)], ws["gdn_out"][0])
        torch.cuda.synchronize()
        ops["d_tiny_gemm"] = lambda x, w, name: tiny_gemm_bf16(x, w)
        print("tiny_gemm: available", flush=True)
    except Exception:  # noqa: BLE001
        print("tiny_gemm: unavailable:", traceback.format_exc()[-1500:], flush=True)

    results = {}

    def run_option(key):
        op = ops[key]
        res = {"shape": {}, "step": {}, "err": {}}
        for m in MS:
            try:
                for name, *_ in SHAPES:
                    res["shape"][(name, m)] = per_shape_us(op, ws, xs, name, m)
                res["step"][m] = step_us(op, ws, xs, m, seq)
                res["err"][m] = err_vs_ref(op, ws, xs, m)
            except Exception:  # noqa: BLE001
                print(f"{key} M={m} failed:", traceback.format_exc()[-1500:], flush=True)
        results[key] = res
        print(f"[{key}] step us: " + ", ".join(f"M{m}={v:.0f}" for m, v in res["step"].items()), flush=True)

    for key in list(ops):
        run_option(key)

    # Batch invariance of (c): rows of an M=16 call equal M=1..16 calls bitwise.
    inv_ok = True
    for name, n, k, _ in SHAPES:
        w = ws[name][0]
        x16 = xs[(16, k)]
        y16 = ops["c_triton"](x16, w, name)
        for m in (1, 2, 3, 5, 8, 13):
            ym = ops["c_triton"](x16[:m].contiguous(), w, name)
            if not torch.equal(ym, y16[:m]):
                inv_ok = False
                print("batch-invariance FAIL", name, m, flush=True)
        y_again = ops["c_triton"](x16, w, name)
        if not torch.equal(y_again, y16):
            inv_ok = False
            print("run-to-run FAIL", name, flush=True)
    print("triton batch-invariant and repeatable:", inv_ok, flush=True)

    # ---- (b) TunableOp over hipBLASLt (and rocBLAS) solutions, tuned eagerly then captured.
    try:
        tun = torch.cuda.tunable
        tun.enable(True)
        tun.tuning_enable(True)
        tun.set_max_tuning_duration(30)
        tun.set_max_tuning_iterations(30)
        tun.set_filename("/tmp/tunableop_plan7.csv")
        t0 = time.time()
        for m in MS:
            for name, n, k, _ in SHAPES:
                F.linear(xs[(m, k)], ws[name][0])
        torch.cuda.synchronize()
        print(f"tunableop tuning took {time.time() - t0:.0f}s", flush=True)
        tun.tuning_enable(False)
        ops["b_tunableop"] = lambda x, w, name: F.linear(x, w)
        run_option("b_tunableop")
        print("tunableop results:", flush=True)
        for r in tun.get_results():
            print("  ", r, flush=True)
        try:
            tun.write_file()
            print(open(tun.get_filename()).read(), flush=True)
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        print("tunableop failed:", traceback.format_exc()[-2000:], flush=True)

    # ---- Tables
    keys = [k for k in ("a_flinear", "b_tunableop", "c_triton", "d_tiny_gemm") if k in results]
    print("\n## per-shape us/call (graph replay, cold weights)", flush=True)
    for m in MS:
        print(f"\nM={m}")
        print(fmt_row(["shape", "N", "K", "calls"] + keys))
        for name, n, k, cnt in SHAPES:
            print(fmt_row([name, n, k, cnt] + [f"{results[key]['shape'].get((name, m), float('nan')):.2f}" for key in keys]))
        print(fmt_row(["sum(count*us)", "", "", 230] + [
            f"{sum(results[key]['shape'].get((nm, m), float('nan')) * c for nm, _, _, c in SHAPES):.0f}" for key in keys]))
    print("\n## step graph us (230 calls in layer order) and max abs err vs fp32", flush=True)
    print(fmt_row(["M"] + [f"{k} us" for k in keys] + [f"{k} err" for k in keys]))
    for m in MS:
        print(fmt_row([m] + [f"{results[k]['step'].get(m, float('nan')):.0f}" for k in keys]
                      + [f"{results[k]['err'].get(m, float('nan')):.4f}" for k in keys]))
    a2 = results["a_flinear"]["step"].get(2)
    best = min((results[k]["step"].get(2, float("inf")), k) for k in keys if k != "a_flinear")
    print(f"\nKILL GATE @M=2: a={a2:.0f}us best={best[1]} {best[0]:.0f}us saving={(a2 - best[0]) / 1000:.3f} ms "
          f"-> {'PASS' if a2 - best[0] >= 1000 else 'KILL'}", flush=True)


if __name__ == "__main__":
    which = os.environ.get("SKINNY_BENCH", "v3g")
    if which == "v1":
        sys.exit(main())
    if which == "v2":
        from v2 import main_v2

        sys.exit(main_v2())
    if which == "v3":
        from v3 import main_v3

        sys.exit(main_v3())
    if which == "v3q":
        from v3 import main_q

        sys.exit(main_q())
    if which == "v3f":
        from v3 import main_f

        sys.exit(main_f())
    from v3 import main_g

    sys.exit(main_g())
