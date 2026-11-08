"""Microbenchmark and correctness check: MoE decode routing on gfx90a (plan 9).

Old path: AOT topk_softmax (topkGatingSoftmax) + moe_align_block_size + count_and_sort.
New path: Triton moe_fused_gate (softmax/top-k/renorm) + moe_align_small_numel.
Both are driven through the real wrappers (fused_topk / select_experts and
moe_align_block_size) by toggling the module gates the change adds, so the
dispatch itself is what gets tested. Shapes are the model's: E=256, top-8,
renormalize, hidden 2048, BF16 router logits. Not on any serving path.
"""
import os
import sys
import time
import traceback

import torch

E, TOPK, HIDDEN = 256, 8, 2048
LAYERS = 40
RESULTS = {}


def log(*a):
    print(*a, flush=True)


def section(name):
    def deco(fn):
        def run(*a, **k):
            log(f"\n===== {name} =====")
            t = time.time()
            try:
                return fn(*a, **k)
            except Exception:
                traceback.print_exc()
                RESULTS[name] = "ERROR"
            finally:
                log(f"[{name}: {time.time() - t:.1f}s]")

        return run

    return deco


from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

import triton

import importlib

# The triton_utils package re-exports the function under the module's own name,
# so `import ... as` would bind the function; fetch the module object itself.
mab = importlib.import_module(
    "sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size"
)
assert hasattr(mab, "_use_small_numel_align"), mab
import sglang.srt.layers.moe.topk as topk_mod
from sglang.srt.layers.moe.topk import TopKConfig, fused_topk, select_experts

dev = torch.device("cuda")


class Path:
    """Context manager selecting the old or new routing path."""

    def __init__(self, new):
        self.new = new

    def __enter__(self):
        self.saved = (topk_mod._use_hip_small_m_triton_router, mab._use_small_numel_align)
        topk_mod._use_hip_small_m_triton_router = self.new
        mab._use_small_numel_align = self.new

    def __exit__(self, *a):
        topk_mod._use_hip_small_m_triton_router, mab._use_small_numel_align = self.saved


def route(logits, block_size=16):
    h = torch.empty((logits.shape[0], HIDDEN), dtype=torch.bfloat16, device=dev)
    w, ids = fused_topk(h, logits, TOPK, True)
    s, e, n = mab.moe_align_block_size(ids, block_size, E)
    return w, ids, s, e, n


@section("environment")
def env():
    p = torch.cuda.get_device_properties(0)
    log("torch", torch.__version__, "hip", torch.version.hip, "triton", triton.__version__)
    log("device", p.name, getattr(p, "gcnArchName", "?"), "warp_size", getattr(p, "warp_size", "?"))
    try:
        log("triton target", triton.runtime.driver.active.get_current_target())
    except Exception as ex:
        log("triton target: n/a", ex)
    log("gates: hip_small_m_router", topk_mod._use_hip_small_m_triton_router,
        "small_numel_align", mab._use_small_numel_align, "aiter", topk_mod._use_aiter)


@section("wave64 compile")
def wave64():
    from sglang.kernels.ops.moe.moe_align_small_numel import _moe_align_small_numel_kernel
    from sglang.kernels.ops.moe.moe_fused_gate import _router_triton_kernel

    with Path(True):
        route(torch.randn(4, E, device=dev).to(torch.bfloat16))
    torch.cuda.synchronize()
    ok = True
    for name, fn in (("router", _router_triton_kernel), ("align_small", _moe_align_small_numel_kernel)):
        found = False
        caches = getattr(fn, "device_caches", None) or {}
        for _, entry in caches.items():
            cache = entry[0] if isinstance(entry, tuple) else entry
            for k in cache.values():
                found = True
                md = k.metadata
                ws = getattr(md, "warp_size", None)
                asm = k.asm.get("amdgcn", "") if hasattr(k, "asm") else ""
                w64 = ".wavefront_size: 64" in asm or "wavefront_size64" in asm or ws == 64
                log(f"{name}: target={getattr(md, 'target', '?')} warp_size={ws} num_warps={md.num_warps} "
                    f"wave64_in_asm={w64} vgpr={getattr(k, 'n_regs', '?')} spills={getattr(k, 'n_spills', '?')}")
                ok = ok and w64
        if not found:
            log(f"{name}: no compiled kernel found in cache")
            ok = False
    RESULTS["wave64"] = ok


def tie_logits(rows, g):
    """Logits with many exact ties among the top experts."""
    out = []
    # small integer grid: heavy ties everywhere
    out.append(torch.randint(-3, 4, (rows, E), generator=g, device=dev).float())
    # top-heavy duplicates: 12 experts share the max value, rest random lower
    x = torch.randn(rows, E, generator=g, device=dev) - 5.0
    pos = torch.rand(rows, E, generator=g, device=dev).argsort(dim=1)[:, :12]
    x.scatter_(1, pos, 2.0)
    out.append(x)
    # all-equal rows
    out.append(torch.zeros(rows, E, device=dev))
    # pairs of ties straddling the k-th boundary
    x = torch.randn(rows, E, generator=g, device=dev) * 3
    v, i = x.topk(9, dim=1)
    x.scatter_(1, i[:, 7:9], v[:, 7:8].expand(-1, 2))
    out.append(x)
    return torch.cat(out).to(torch.bfloat16)


@section("routing equality (a)")
def routing_equality():
    from sglang.kernels.ops.moe import topk_softmax
    from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

    g = torch.Generator(device=dev).manual_seed(0)
    sets = {
        "randn*3": (torch.randn(131072, E, generator=g, device=dev) * 3).to(torch.bfloat16),
        "randn*1": torch.randn(65536, E, generator=g, device=dev).to(torch.bfloat16),
        "randn*8": (torch.randn(32768, E, generator=g, device=dev) * 8).to(torch.bfloat16),
        "ties": tie_logits(16384, g),
    }
    ok = True
    for name, x in sets.items():
        M = x.shape[0]
        w0 = torch.empty(M, TOPK, dtype=torch.float32, device=dev)
        i0 = torch.empty(M, TOPK, dtype=torch.int32, device=dev)
        topk_softmax(w0, i0, x, True)
        w1, i1 = moe_fused_gate(x, None, TOPK, scoring_func="softmax", renormalize=True)
        ids_eq = torch.equal(i0, i1)
        ids_set_eq = torch.equal(i0.sort(1).values, i1.sort(1).values)
        dw = (w0 - w1).abs().max().item()
        nbits = (w0 != w1).float().mean().item()
        # fp64 reference on the selected ids
        ref = torch.softmax(x.double(), dim=1).gather(1, i0.long())
        ref = ref / ref.sum(1, keepdim=True)
        dref0 = (w0.double() - ref).abs().max().item()
        dref1 = (w1.double() - ref).abs().max().item()
        log(f"{name:8s} rows={M:6d} ids_identical={ids_eq} id_sets_equal={ids_set_eq} "
            f"max|dw|={dw:.3e} frac_w_differ={nbits:.4f} |old-fp64|={dref0:.3e} |new-fp64|={dref1:.3e}")
        if not ids_eq:
            bad = (i0 != i1).any(1).nonzero()[:3, 0]
            for r in bad.tolist():
                log("  row", r, "old", i0[r].tolist(), "new", i1[r].tolist())
        ok = ok and ids_eq and dw <= 1e-6
    # dispatch: fused_topk at M<=8 must take the new kernel and match the direct call
    x = sets["randn*3"][:4096]
    for M in (1, 2, 4, 8, 9):
        with Path(True):
            outs = [fused_topk(torch.empty(M, HIDDEN, device=dev), x[r:r + M], TOPK, True)
                    for r in range(0, 4096 - M + 1, M)]
        w = torch.cat([o[0] for o in outs])
        i = torch.cat([o[1] for o in outs])
        n = w.shape[0]
        wd, idd = moe_fused_gate(x[:n], None, TOPK, scoring_func="softmax", renormalize=True)
        w0 = torch.empty(n, TOPK, dtype=torch.float32, device=dev)
        i0 = torch.empty(n, TOPK, dtype=torch.int32, device=dev)
        topk_softmax(w0, i0, x[:n], True)
        log(f"fused_topk M={M}: equals_triton={torch.equal(w, wd) and torch.equal(i, idd)} "
            f"equals_aot={torch.equal(w, w0) and torch.equal(i, i0)}  (M=9 must stay on AOT)")
    RESULTS["routing_equal"] = ok


def align_canon(ids, s, e, n, block):
    """Per-expert sets of pair indices plus per-block expert ids, order-free within a block run."""
    total = int(n[0].item())
    numel = ids.numel()
    s = s[:total].tolist()
    e = e[: total // block].tolist()
    segs = {}
    for b, ex in enumerate(e):
        segs.setdefault(ex, []).extend(t for t in s[b * block:(b + 1) * block] if t < numel)
    return total, e, {k: sorted(v) for k, v in segs.items()}


@section("align equality")
def align_equality():
    g = torch.Generator(device=dev).manual_seed(1)
    ok = True
    n_cases = 0
    for M in (1, 2, 3, 4, 5, 6, 7, 8):
        for block in (16, 32, 64):
            for trial in range(40):
                if trial % 4 == 0:  # heavy expert collisions
                    ids = torch.randint(0, 6, (M, TOPK), generator=g, device=dev, dtype=torch.int32)
                elif trial % 4 == 1:  # padded rows routed to expert 0 (HIP pad masking)
                    ids = torch.rand(M, E, generator=g, device=dev).argsort(1)[:, :TOPK].int()
                    ids[M // 2:] = 0
                else:
                    ids = torch.rand(M, E, generator=g, device=dev).argsort(1)[:, :TOPK].int()
                with Path(False):
                    a = align_canon(ids, *mab.moe_align_block_size(ids, block, E), block)
                with Path(True):
                    b = align_canon(ids, *mab.moe_align_block_size(ids, block, E), block)
                n_cases += 1
                if a != b:
                    ok = False
                    log("MISMATCH M", M, "block", block, "ids", ids.tolist())
                    log(" old", a)
                    log(" new", b)
                    break
    log(f"align cases={n_cases} identical_semantics={ok}")
    RESULTS["align_equal"] = ok


def moe_weights():
    inter = 512
    try:
        import glob
        import json

        cfgs = glob.glob("/vibesys/mounts/hf/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/*/config.json")
        if cfgs:
            c = json.load(open(cfgs[0]))
            c = c.get("text_config", c)
            inter = c.get("moe_intermediate_size", inter)
    except Exception:
        pass
    n = inter // 2  # TP=2 shard
    g = torch.Generator(device=dev).manual_seed(2)
    w1 = (torch.randn(E, 2 * n, HIDDEN, generator=g, device=dev) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(E, HIDDEN, n, generator=g, device=dev) * 0.02).to(torch.bfloat16)
    log(f"moe_intermediate_size={inter}, TP=2 shard N={n}")
    return w1, w2


@section("fused_moe bit-identity (b) and batch invariance (c)")
def fused_moe_checks():
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe

    w1, w2 = moe_weights()
    g = torch.Generator(device=dev).manual_seed(3)
    cfg = TopKConfig(top_k=TOPK, renormalize=True)
    same_route_ok = True
    full_ok = True
    worst_full = 0.0
    for M in (1, 2, 4, 8):
        for trial in range(25):
            x = (torch.randn(M, HIDDEN, generator=g, device=dev)).to(torch.bfloat16)
            logits = (torch.randn(M, E, generator=g, device=dev) * 3).to(torch.bfloat16)
            if trial % 5 == 0:
                logits = tie_logits(M, g)[torch.randperm(4 * M, device=dev)[:M]]
            with Path(False):
                t_old = select_experts(hidden_states=x, router_logits=logits, topk_config=cfg)
                y_old = fused_moe(x.clone(), w1, w2, t_old)
            with Path(True):
                # (b) same router output, new align -> must be bit-identical
                y_new_align = fused_moe(x.clone(), w1, w2, t_old)
                t_new = select_experts(hidden_states=x, router_logits=logits, topk_config=cfg)
                y_new = fused_moe(x.clone(), w1, w2, t_new)
            same_route_ok &= torch.equal(y_old, y_new_align)
            ids_eq = torch.equal(t_old.topk_ids, t_new.topk_ids)
            eq = torch.equal(y_old, y_new)
            full_ok &= eq and ids_eq
            worst_full = max(worst_full, (y_old.float() - y_new.float()).abs().max().item())
    log(f"(b) old router + new align vs old chain: bit-identical={same_route_ok}")
    log(f"(b) full new chain vs old chain: bit-identical={full_ok} max|dy|={worst_full:.3e}")
    RESULTS["fused_moe_same_route_bitident"] = same_route_ok
    RESULTS["fused_moe_full_bitident"] = full_ok

    # (c) batch invariance: row r at M=1 equals row r inside M=8
    for new in (False, True):
        r_ok, y_ok = True, True
        for trial in range(25):
            x = (torch.randn(8, HIDDEN, generator=g, device=dev)).to(torch.bfloat16)
            logits = (torch.randn(8, E, generator=g, device=dev) * 3).to(torch.bfloat16)
            with Path(new):
                t8 = select_experts(hidden_states=x, router_logits=logits, topk_config=cfg)
                r8 = route(logits)
                y8 = fused_moe(x.clone(), w1, w2, t8)
                for r in range(8):
                    t1 = select_experts(hidden_states=x[r:r + 1], router_logits=logits[r:r + 1], topk_config=cfg)
                    r_ok &= torch.equal(t1.topk_ids[0], t8.topk_ids[r]) and torch.equal(t1.topk_weights[0], t8.topk_weights[r])
                    y1 = fused_moe(x[r:r + 1].clone(), w1, w2, t1)
                    y_ok &= torch.equal(y1[0], y8[r])
        tag = "new" if new else "old"
        log(f"(c) {tag} path: routing row(M=1)==row(M=8): {r_ok}; fused_moe row(M=1)==row(M=8): {y_ok}")
        RESULTS[f"batch_invariant_routing_{tag}"] = r_ok
        RESULTS[f"batch_invariant_moe_{tag}"] = y_ok


def graph_time(fn, reps=30, inner=1):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(7):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            graph.replay()
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b) * 1e3 / reps / inner)
    samples.sort()
    return samples[len(samples) // 2]


@section("timing in HIP graph")
def timing():
    from sglang.kernels.ops.moe import topk_softmax

    g = torch.Generator(device=dev).manual_seed(4)
    rows = {}
    for M in (1, 2, 4, 8):
        logits = [(torch.randn(M, E, generator=g, device=dev) * 3).to(torch.bfloat16) for _ in range(LAYERS)]
        h = torch.empty(M, HIDDEN, dtype=torch.bfloat16, device=dev)
        ids_in = [torch.rand(M, E, generator=g, device=dev).argsort(1)[:, :TOPK].int() for _ in range(LAYERS)]

        def chain():
            for l in range(LAYERS):
                w, ids = fused_topk(h, logits[l], TOPK, True)
                mab.moe_align_block_size(ids, 16, E)

        def router_only():
            for l in range(LAYERS):
                fused_topk(h, logits[l], TOPK, True)

        def align_only():
            for l in range(LAYERS):
                mab.moe_align_block_size(ids_in[l], 16, E)

        res = {}
        for new in (False, True):
            with Path(new):
                tag = "new" if new else "old"
                res[f"chain_{tag}"] = graph_time(chain, inner=LAYERS)
                res[f"router_{tag}"] = graph_time(router_only, inner=LAYERS)
                res[f"align_{tag}"] = graph_time(align_only, inner=LAYERS)
        rows[M] = res
        log(f"M={M}: per layer us  chain old {res['chain_old']:.2f} -> new {res['chain_new']:.2f} "
            f"(saved {res['chain_old'] - res['chain_new']:.2f}, x40 = {(res['chain_old'] - res['chain_new']) * 40 / 1000:.3f} ms/step) | "
            f"router {res['router_old']:.2f} -> {res['router_new']:.2f} | align {res['align_old']:.2f} -> {res['align_new']:.2f}")
    RESULTS["timing"] = rows
    worst = max(rows[M]["chain_new"] for M in (1, 2, 4))
    RESULTS["kill_new_gt_15us_at_M_le_4"] = worst > 15.0
    log(f"worst new chain at M<=4: {worst:.2f} us/layer (kill if > 15)")


@section("repeatability")
def repeatability():
    g = torch.Generator(device=dev).manual_seed(5)
    ok = True
    with Path(True):
        for M in (1, 3, 8):
            logits = (torch.randn(M, E, generator=g, device=dev) * 3).to(torch.bfloat16)
            first = route(logits)
            for _ in range(50):
                again = route(logits)
                t = int(first[4][0])
                ok &= all(torch.equal(a, b) for a, b in zip(first[:2], again[:2]))
                ok &= torch.equal(first[2][:t], again[2][:t]) and torch.equal(first[3][: t // 16], again[3][: t // 16])
    log(f"new path deterministic over 50 repeats: {ok}")
    RESULTS["repeatable"] = ok


if __name__ == "__main__":
    torch.cuda.set_device(0)
    with torch.inference_mode():
        env()
        wave64()
        routing_equality()
        align_equality()
        repeatability()
    timing()
    with torch.inference_mode():
        fused_moe_checks()
    log("\n===== SUMMARY =====")
    for k, v in RESULTS.items():
        log(f"{k}: {v}")
