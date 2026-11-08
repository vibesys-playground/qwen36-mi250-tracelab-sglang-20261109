"""Tune the fused-MoE Triton kernel (up/down) for Qwen3.6-35B-A3B on gfx90a, BF16.

Runs inside the candidate image on one GCD. Times the whole fused_experts call
(graph-replayed) so the default and the tuned configs are compared on the same op.
Tuning changes only tiling; the check at the end compares tuned vs default outputs.
Prints JSON between ===JSON <name>=== markers so the files can be recovered from job logs.
"""
import argparse, itertools, json, math, os, sys, time, traceback

import torch
import triton

try:  # the tuner never calls aiter; importing sglang's quantization package may touch it on HIP
    import aiter  # noqa: F401
except ImportError:
    import importlib.abc, importlib.machinery, types
    from unittest import mock

    class _AiterFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, name, path, target=None):
            if name == "aiter" or name.startswith("aiter."):
                return importlib.machinery.ModuleSpec(name, self, is_package=True)

        def create_module(self, spec):
            m = mock.MagicMock(name=spec.name)
            m.__path__ = []
            m.__spec__ = spec
            return m

        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, _AiterFinder())


class _Z:
    def __getattr__(self, n):
        return _Z()

    def __bool__(self):
        return False


def _patch_runtime():
    import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as fm
    import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config as fc
    class _Det:
        enable_deterministic_inference = False

    class _Moe:
        enable_fused_moe_sum_all_reduce = False

    class _Exec:
        deterministic = _Det()
        moe = _Moe()

    fm.get_exec = lambda: _Exec()
    fc.get_exec = lambda: _Exec()
    fm.get_parallel = lambda: _Z()
    fm.is_allocation_symmetric = lambda: False
    return fm, fc


DRY = os.environ.get("MOE_TUNE_DRY") == "1"
DEV = "cpu" if DRY else "cuda"
_ALLOWED_OPTS = {"num_warps", "num_stages", "waves_per_eu", "matrix_instr_nonkdim", "kpack",
                 "launch_pdl", "num_ctas", "maxnreg"}
_dry_seen = set()


def _install_dry_patches(fm):
    """CPU-only run: checks imports, argument construction and every kernel argument/constexpr
    type (the things that failed j11/j16). The device launch and sgl_kernel ops are replaced."""
    import types
    import sglang.kernels.ops.moe.fused_moe_triton_kernels as ftk
    import torch.nn.functional as F

    props = types.SimpleNamespace(gcnArchName="gfx90a:sramecc+:xnack-", multi_processor_count=104, name="AMD Instinct MI210")
    torch.cuda.is_available = lambda: True
    torch.cuda.current_device = lambda: 0
    torch.cuda.get_device_properties = lambda i=0: props
    torch.cuda.synchronize = lambda *a, **k: None

    class Ev:
        def __init__(self, **k): pass
        def record(self): self.t = time.perf_counter()
        def elapsed_time(self, o): return (o.t - self.t) * 1000.0
    torch.cuda.Event = Ev

    real = ftk.fused_moe_kernel
    import triton.language as tl

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(*args, **kw):
                names = list(real.arg_names)
                bound = dict(zip(names, args))
                for k, v in kw.items():
                    if k in bound:
                        raise TypeError(f"duplicate argument {k}")
                    if k not in names and k not in _ALLOWED_OPTS:
                        raise TypeError(f"unknown kernel kwarg {k}")
                    bound[k] = v
                for i, n in enumerate(names):
                    if n not in bound and not real.params[i].has_default:
                        raise TypeError(f"missing kernel argument {n}")
                for i, n in enumerate(names):
                    if n in bound and real.params[i].is_constexpr:
                        v = bound[n]
                        if not (v is None or isinstance(v, (bool, int, float, str, tl.dtype))):
                            raise TypeError(f"constexpr {n} has type {type(v).__name__}: {v!r}")
                    elif n in bound:
                        v = bound[n]
                        if not (v is None or isinstance(v, (bool, int, float, torch.Tensor))):
                            raise TypeError(f"argument {n} has type {type(v).__name__}")
                if callable(grid):
                    grid(bound)
                _dry_seen.add(tuple(sorted(k for k in kw)))
            return launch
    ftk.fused_moe_kernel = FakeKernel()
    ftk.is_arch_support_pdl = lambda: False
    fm.moe_sum_reduce_triton = lambda *a, **k: None
    fm.silu_and_mul = lambda x, out: out.copy_(F.silu(x[:, : x.shape[1] // 2]) * x[:, x.shape[1] // 2:])

    def align(topk_ids, bs, E, *a, **k):
        n = topk_ids.numel() + E * (bs - 1)
        n = (n + bs - 1) // bs * bs
        return (torch.zeros(n, dtype=torch.int32), torch.zeros(n // bs, dtype=torch.int32),
                torch.tensor([n], dtype=torch.int32))
    fm.moe_align_block_size = align
    fm.outplace_fused_experts = lambda *a, **k: fm.fused_experts_impl(*a[:7], False, *a[7:], **k)  # the custom-op wrapper only has a CUDA kernel


def bench(fn, iters=20, reps=3):
    fn()  # compile / surface errors eagerly
    torch.cuda.synchronize()
    try:
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            for _ in range(iters):
                fn()
        run = g.replay
        n = iters
    except Exception:
        torch.cuda.synchronize()
        run = fn
        n = 1
    run()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        if n == 1:
            for _ in range(iters):
                run()
        else:
            run()
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) * 1000.0 / iters)
    return best


def p2(x):
    return 1 << (max(1, x) - 1).bit_length()


def cands(BM, out_dim, K, M):
    bns = [b for b in (32, 64, 128, 256) if b <= max(32, out_dim)]
    bks = [b for b in (32, 64, 128, 256) if b <= max(32, p2(K))]
    nws = [1, 2, 4, 8] if BM == 16 else [2, 4, 8]
    res = []
    for bn, bk, nw in itertools.product(bns, bks, nws):
        if (BM * bk + bk * bn) * 2 * 2 > 65536:
            continue
        res.append(dict(BLOCK_SIZE_M=BM, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=1,
                        num_warps=nw, num_stages=2, waves_per_eu=0))
    return res


def refine(c, M):
    out = []
    for ns, wpe, g in itertools.product((2, 3), (0, 2), (1, 8) if M >= 64 else (1,)):
        d = dict(c, num_stages=ns, waves_per_eu=wpe, GROUP_SIZE_M=g)
        if (d["BLOCK_SIZE_M"] * d["BLOCK_SIZE_K"] + d["BLOCK_SIZE_K"] * d["BLOCK_SIZE_N"]) * 2 * ns > 65536:
            continue
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tps", default="8,4,2")
    ap.add_argument("--ms", default="1,2,4,8,12,16,24,32,48,64,96,128,256,512,1024,2048,4096")
    ap.add_argument("--E", type=int, default=256)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--moe-inter", type=int, default=512)
    ap.add_argument("--outdir", default="")
    a = ap.parse_args()

    fm, fc = _patch_runtime()
    if DRY:
        _install_dry_patches(fm)
        fc._is_hip = True
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
    from sglang.srt.utils import get_device_name

    props = torch.cuda.get_device_properties(0)
    print("DEVICE", repr(get_device_name()), getattr(props, "gcnArchName", "?"),
          "CUs", props.multi_processor_count, "triton", triton.__version__, "torch", torch.__version__, flush=True)
    print("CONFIGFILE", fc.get_config_file_name(a.E, a.moe_inter // 8, None, [0, 0], False, False), flush=True)
    torch.manual_seed(0)
    dev = DEV
    real_try = fc.try_get_optimal_moe_config
    cur = {}

    def patched(w1s, w2s, top_k, dtype, M, **kw):
        if "up" in cur:
            return dict(cur["up"]), (dict(cur["down"]), None)
        return real_try(w1s, w2s, top_k, dtype, M, **kw)
    fm.try_get_optimal_moe_config = patched
    rcfg = MoeRunnerConfig(inplace=False)

    summary = []
    _errs = {}
    stage_hint = ["up"]
    for tp in [int(x) for x in a.tps.split(",")]:
        N = a.moe_inter // tp
        K = a.hidden
        E = a.E
        w1 = (torch.randn(E, 2 * N, K, device=dev) * 0.02).to(torch.bfloat16)
        w2 = (torch.randn(E, K, N, device=dev) * 0.02).to(torch.bfloat16)
        up_out, dn_out = {}, {}
        for M in [int(x) for x in a.ms.split(",")]:
            x = torch.randn(M, K, device=dev).to(torch.bfloat16)
            ids = torch.rand(M, E, device=dev).topk(a.topk, dim=1).indices.to(torch.int32)
            wts = torch.softmax(torch.randn(M, a.topk, device=dev), dim=1).float()
            tko = (wts, ids, None)

            def run():
                return fm.fused_experts(x, w1, w2, tko, rcfg)

            def timed(up, down):
                cur["up"], cur["down"] = up, down
                try:
                    return bench(run)
                except Exception as e:
                    torch.cuda.synchronize()
                    msg = type(e).__name__ + ": " + str(e).splitlines()[0][:160] if str(e) else type(e).__name__
                    _errs[msg] = _errs.get(msg, 0) + 1
                    if _errs[msg] == 1 and len(_errs) <= 12:
                        print("CANDIDATE-ERROR", msg, "cfg", cur.get("up") if stage_hint[0] == "up" else cur.get("down"), flush=True)
                    return float("inf")
                finally:
                    cur.clear()

            # default (the lookup finds no file for this device on a clean tree, or the committed one
            # if present; force the generic default so the comparison is against upstream behaviour)
            dcfg = fc.get_default_config(M, E, N, K, a.topk, None, False)
            t_def = timed(dict(dcfg), dict(dcfg))
            cur["up"], cur["down"] = dict(dcfg), dict(dcfg)
            ref = run().float().clone()
            cur.clear()

            BMs = [16] if M <= 16 else [16, 32] if M <= 64 else [16, 32, 64] if M <= 256 else [32, 64, 128]
            best_total = None
            t0 = time.time()
            for BM in BMs:
                base = dict(BLOCK_SIZE_M=BM, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64, GROUP_SIZE_M=1,
                            num_warps=4, num_stages=2, waves_per_eu=0)
                down = dict(base)
                # up search (down fixed)
                for stage, (out_dim, kdim) in (("up", (2 * N, K)), ("down", (K, N))):
                    stage_hint[0] = stage
                    pool = cands(BM, out_dim, kdim, M)
                    res = []
                    for c in pool:
                        t = timed(c, down) if stage == "up" else timed(up, c)
                        res.append((t, c))
                    res.sort(key=lambda r: r[0])
                    top = [c for t, c in res[:4] if t < float("inf")]
                    res2 = list(res[:4])
                    for c in top:
                        for d in refine(c, M):
                            t = timed(d, down) if stage == "up" else timed(up, d)
                            res2.append((t, d))
                    res2.sort(key=lambda r: r[0])
                    if not res2 or res2[0][0] == float("inf"):
                        break
                    if stage == "up":
                        up = res2[0][1]
                        down = dict(up, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64, GROUP_SIZE_M=1, num_warps=4, num_stages=2, waves_per_eu=0)
                    else:
                        down = res2[0][1]
                        tot = res2[0][0]
                        if best_total is None or tot < best_total[0]:
                            best_total = (tot, dict(up), dict(down))
            if best_total is None:
                print(f"TP{tp} M={M}: no valid config", flush=True)
                continue
            t_best, up, down = best_total
            # verify and re-time default/tuned back to back
            cur["up"], cur["down"] = dict(up), dict(down)
            out = run().float()
            cur.clear()
            err = (out - ref).abs().max().item()
            scale = ref.abs().max().item() + 1e-6
            t_t = timed(up, down)
            t_d2 = timed(dict(dcfg), dict(dcfg))
            up_out[M], dn_out[M] = up, down
            row = dict(tp=tp, M=M, default_us=round(t_d2, 1), tuned_us=round(t_t, 1),
                       speedup=round(t_d2 / t_t, 2), rel_err=round(err / scale, 5),
                       up=up, down=down, search_s=round(time.time() - t0))
            summary.append(row)
            print("ROW", json.dumps(row), flush=True)
        del w1, w2
        for suffix, cfgs in (("", up_out), ("_down", dn_out)):
            name = fc.get_config_file_name(E, N, None, [0, 0], False, suffix == "_down")
            body = {str(m): {k: v for k, v in c.items()} for m, c in sorted(cfgs.items())}
            txt = json.dumps(body, indent=4) + "\n"
            print(f"===JSON {name}===\n{txt}===END===", flush=True)
            if a.outdir:
                os.makedirs(a.outdir, exist_ok=True)
                open(os.path.join(a.outdir, name), "w").write(txt)
    print("ERROR-COUNTS", json.dumps(_errs))
    print("SUMMARY")
    for r in summary:
        print(f"TP{r['tp']} M={r['M']:5d} default {r['default_us']:9.1f}us tuned {r['tuned_us']:9.1f}us x{r['speedup']:.2f} relerr {r['rel_err']}")


if __name__ == "__main__":
    main()
