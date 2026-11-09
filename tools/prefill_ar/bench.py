"""Plan 31 step 1: the TP=2 prefill layer boundary (custom all-reduce + Gemma add+RMSNorm) on gfx90a.

Per-rank shapes: M rows x 2048 bf16, M in MS. Every all-reduce input is written by a
(M x 2048) @ (2048 x 2048) GEMM just before the call (dirty L2), in HIP graphs (CHAIN layers per
graph, REPLAYS replays), per-kernel times from the torch profiler on rank 0.
  (a) head: the AOT custom all-reduce (block_limit 16) + Triton _gemma_fused_add_rmsnorm;
      a2: the same all-reduce with no GEMM in front (clean L2) to split the fences from streaming.
  (b) the same 1-stage kernel at block_limit BLOCKS (JIT copy of the checkout's header, kMaxBlocks
      128); output bitwise against (a).
  (c) link probes on the pair: peer read, peer push (non-temporal) and hipMemcpyAsync, one
      direction and both at once, 2.5 MB (rotating over 64 MB) and 64 MB.
  (d) prototype push all-reduce + add + Gemma norm in one kernel (p2p_prefill.hip); residual
      bitwise against (a), normed within 1 ulp, ranks bitwise identical, CORR iterations.
Runs pair (0,1) alone, then the 4 pairs concurrently. Prints RESULT / TABLE / GATE lines.
Not on any serving path.
"""
import json
import os
import sys
import tempfile
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HID = 2048
EPS = 1e-6
MS = (184, 617, 1024, 1708, 2816)
BLOCKS = (32, 64, 104, 128)
CHAIN = 8
REPLAYS = 25  # 200 calls per variant and size
CORR = 200
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(REPO, "sglang/python/sglang/kernels/aot/csrc/allreduce")
BUILD = os.path.join(tempfile.gettempdir(), "prefill_ar_build")


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------------------------------
# builds (parent process, once)
# ------------------------------------------------------------------------------------------
VAR_BINDING = r'''
#include "custom_all_reduce.hip"
#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_custom_ar", &init_custom_ar);
  m.def("all_reduce_reg", &all_reduce_reg);
  m.def("all_reduce_unreg", &all_reduce_unreg);
  m.def("dispose", &dispose);
  m.def("meta_size", &meta_size);
  m.def("register_buffer", &register_buffer);
  m.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
  m.def("register_graph_buffers", &register_graph_buffers);
  m.def("allocate_meta_buffer", &allocate_meta_buffer);
  m.def("get_meta_buffer_ipc_handle", &get_meta_buffer_ipc_handle);
  m.def("set_block_limit", [](int n) { sglang::g_block_override = n; });
}
'''
VAR_FUNCS = ["init_custom_ar", "all_reduce_reg", "all_reduce_unreg", "dispose", "meta_size",
             "register_buffer", "get_graph_buffer_ipc_meta", "register_graph_buffers",
             "allocate_meta_buffer", "get_meta_buffer_ipc_handle"]


def _write(path, text):
    if os.path.exists(path) and open(path).read() == text:
        return
    with open(path, "w") as f:
        f.write(text)


def build_variant():
    from torch.utils.cpp_extension import load

    d = os.path.join(BUILD, "var")
    os.makedirs(d, exist_ok=True)
    cuh = open(os.path.join(SRC, "custom_all_reduce_hip.cuh")).read()
    a = "constexpr int kMaxBlocks = 64;"
    b = "  int blocks = ::min(block_limit, (size + threads - 1) / threads);"
    assert cuh.count(a) == 1 and cuh.count(b) == 1, "header text changed; update bench.py"
    cuh = cuh.replace(a, "constexpr int kMaxBlocks = 128;\ninline int g_block_override = 0;")
    cuh = cuh.replace(b, "  int blocks = ::min(g_block_override > 0 ? g_block_override : block_limit,"
                         " (size + threads - 1) / threads);")
    _write(os.path.join(d, "custom_all_reduce.hip"), open(os.path.join(SRC, "custom_all_reduce.hip")).read())
    _write(os.path.join(d, "custom_all_reduce_hip.cuh"), cuh)
    _write(os.path.join(d, "binding.hip"), VAR_BINDING)
    os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx90a")
    return load(name="prefill_ar_var_ext", sources=[os.path.join(d, "binding.hip")],
                extra_include_paths=[d], build_directory=d,
                extra_cflags=["-O3", "-DUSE_ROCM"],
                extra_cuda_cflags=["-O3", "-DUSE_ROCM", "-DNDEBUG", "-DENABLE_BF16",
                                   "--offload-arch=gfx90a"])


def build_p2p():
    from torch.utils.cpp_extension import load

    d = os.path.join(BUILD, "p2p")
    os.makedirs(d, exist_ok=True)
    _write(os.path.join(d, "p2p_prefill.hip"), open(os.path.join(HERE, "p2p_prefill.hip")).read())
    os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx90a")
    return load(name="p2p_prefill_ext", sources=[os.path.join(d, "p2p_prefill.hip")],
                build_directory=d, extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "--offload-arch=gfx90a"])


# ------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------
def bits_hash(t):
    v = t.contiguous().view(torch.int16).to(torch.int64).flatten()
    w = (torch.arange(v.numel(), device=v.device, dtype=torch.int64) % 65521) + 1
    return int((v * w).sum().item())


def ulp_err(out, ref):
    ref = ref.double()
    a = ref.abs().clamp_min(2.0 ** -126)
    ulp = torch.exp2(torch.floor(torch.log2(a)) - 7)
    e = (out.double() - ref).abs() / ulp
    return e.max().item(), (e <= 1.0).double().mean().item()


class Ctx:
    pass


class OpsSwitch:
    """Points sglang's custom_all_reduce_ops at the AOT library or the JIT variant."""

    def __init__(self, ops, var):
        self.ops = ops
        self.var = var
        self.orig = {f: getattr(ops, f) for f in VAR_FUNCS}

    def head(self):
        for f, v in self.orig.items():
            setattr(self.ops, f, v)

    def variant(self, n=0):
        for f in VAR_FUNCS:
            setattr(self.ops, f, getattr(self.var, f))
        self.var.set_block_limit(n)


def capture(fn, ca=None, warm=2):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warm):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    if ca is not None:
        with ca.capture():
            with torch.cuda.graph(g, stream=s):
                res = fn()
    else:
        with torch.cuda.graph(g, stream=s):
            res = fn()
    torch.cuda.synchronize()
    return g, res


def time_and_profile(c, g):
    """Wall us per layer (all ranks), and rank 0's per-kernel table {name: avg us}."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    dist.barrier(group=c.pg)
    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        g.replay()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / (REPLAYS * CHAIN) * 1e6
    dist.barrier(group=c.pg)
    if c.local != 0:
        for _ in range(REPLAYS):
            g.replay()
        torch.cuda.synchronize()
        dist.barrier(group=c.pg)
        return wall, None
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(REPLAYS):
            g.replay()
        torch.cuda.synchronize()
    dist.barrier(group=c.pg)
    agg = {}
    for e in p.events():
        dt = getattr(e, "device_type", None)
        if dt is None or "CUDA" not in str(dt):
            continue
        dur = e.device_time if hasattr(e, "device_time") else e.cuda_time
        n, t = agg.get(e.name, (0, 0.0))
        agg[e.name] = (n + 1, t + dur)
    return wall, {k: (n, t / n) for k, (n, t) in agg.items()}


def kavg(tab, key):
    if tab is None:
        return None
    hits = [(n, t) for k, (n, t) in tab.items() if key in k]
    if not hits:
        return None
    n = sum(h[0] for h in hits)
    return round(sum(h[0] * h[1] for h in hits) / n, 2)


# ------------------------------------------------------------------------------------------
# setup
# ------------------------------------------------------------------------------------------
def setup(rank, world):
    c = Ctx()
    c.rank = rank
    c.local = rank % 2
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    from sglang.srt.distributed.device_communicators import custom_all_reduce as car
    import sglang.srt.distributed.device_communicators.custom_all_reduce_ops as ops
    from sglang.kernels.ops.layernorm.minimax_m3_rmsnorm import gemma_fused_add_rmsnorm

    pgs = [dist.new_group([2 * i, 2 * i + 1], backend="gloo") for i in range(world // 2)]
    c.pg = pgs[rank // 2]
    c.gemma = gemma_fused_add_rmsnorm
    c.ca = car.CustomAllreduce(group=c.pg, device=rank)
    c.var = c.p2p = None
    c.notes = []
    try:
        c.var = build_variant()
    except Exception as e:  # noqa: BLE001
        c.notes.append(f"variant build failed: {e!r}"[:500])
    try:
        c.p2p = build_p2p()
    except Exception as e:  # noqa: BLE001
        c.notes.append(f"p2p build failed: {e!r}"[:500])
    c.sw = OpsSwitch(ops, c.var) if c.var is not None else None
    c.ca_var = None
    if c.sw is not None:
        c.sw.variant(0)
        c.ca_var = car.CustomAllreduce(group=c.pg, device=rank)
        c.sw.head()
    c.cu = torch.cuda.get_device_properties(rank).multi_processor_count
    if c.p2p is not None:
        rows = max(MS)
        c.recv = c.p2p.alloc(2 * rows * HID * 2, True)
        c.ctl = c.p2p.alloc(4096 * 4, True)
        c.fs_rows = rows
        mine = (c.p2p.ipc_handle(c.recv), c.p2p.ipc_handle(c.ctl))
        allh = [None, None]
        dist.all_gather_object(allh, mine, group=c.pg)
        peer = allh[1 - c.local]
        c.peer_recv = c.p2p.ipc_open(peer[0])
        c.peer_ctl = c.p2p.ipc_open(peer[1])
    log(f"rank {rank}: {torch.cuda.get_device_name()} cu={c.cu} ca.disabled={c.ca.disabled} "
        f"var={'ok' if c.var else 'none'} p2p={'ok' if c.p2p else 'none'} notes={c.notes}")
    return c


def weights(c):
    g = torch.Generator(device="cuda")
    g.manual_seed(1000 + c.local)
    W = (torch.randn(HID, HID, generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    g.manual_seed(77)  # replicated
    nw = (torch.randn(HID, generator=g, device="cuda") * 0.3).to(torch.bfloat16)
    return W, nw


# ------------------------------------------------------------------------------------------
# (a), (b), (d) per M
# ------------------------------------------------------------------------------------------
def boundary_part(c, ms=MS, corr=CORR):
    W, nw = weights(c)
    res = {}
    for M in ms:
        r = {}
        A = torch.randn(M, HID, device="cuda", dtype=torch.bfloat16)
        g = torch.Generator(device="cuda")
        g.manual_seed(5)
        resid = torch.randn(M, HID, generator=g, device="cuda").to(torch.bfloat16)

        def chain(ca):
            def fn():
                rr = resid
                out = None
                for _ in range(CHAIN):
                    y = torch.mm(A, W)
                    ar = ca.custom_all_reduce(y)
                    out, rr = c.gemma(ar, rr, nw, EPS, round_sum=True)
                return out, rr
            return fn

        def chain_clean(ca):
            def fn():
                y = torch.mm(A, W)
                outs = [ca.custom_all_reduce(y) for _ in range(CHAIN)]
                return outs[-1]
            return fn

        def one(ca):
            def fn():
                y = torch.mm(A, W)
                ar = ca.custom_all_reduce(y)
                out, rr = c.gemma(ar, resid, nw, EPS, round_sum=True)
                return ar, out, rr
            return fn

        def fused_one(grid, fence):
            def fn():
                y = torch.mm(A, W)
                out = torch.empty_like(y)
                rr = torch.empty_like(y)
                c.p2p.push_ar_norm(y, resid, nw, out, rr, c.peer_recv, c.recv, c.peer_ctl, c.ctl,
                                   c.fs_rows, c.local, EPS, grid, fence)
                return out, rr
            return fn

        def fused_chain(grid, fence):
            def fn():
                rr = resid
                out = None
                for _ in range(CHAIN):
                    y = torch.mm(A, W)
                    out = torch.empty_like(y)
                    nr = torch.empty_like(y)
                    c.p2p.push_ar_norm(y, rr, nw, out, nr, c.peer_recv, c.recv, c.peer_ctl, c.ctl,
                                       c.fs_rows, c.local, EPS, grid, fence)
                    rr = nr
                return out, rr
            return fn

        timed = {}  # name -> graph
        single = {}
        # head (AOT library)
        if c.sw is not None:
            c.sw.head()
        timed["a"] = capture(chain(c.ca), c.ca)[0]
        timed["a_clean"] = capture(chain_clean(c.ca), c.ca)[0]
        single["a"] = capture(one(c.ca), c.ca)
        if c.ca_var is not None:
            for n in BLOCKS:
                c.sw.variant(n)
                timed[f"b{n}"] = capture(chain(c.ca_var), c.ca_var)[0]
                single[f"b{n}"] = capture(one(c.ca_var), c.ca_var)
                if n in (64, 104):
                    timed[f"b{n}_clean"] = capture(chain_clean(c.ca_var), c.ca_var)[0]
            c.sw.head()
        dgrids = (c.cu, 2 * c.cu)
        if c.p2p is not None:
            for gr in dgrids:
                for f in (0, 1):
                    timed[f"d{gr}f{f}"] = capture(fused_chain(gr, f))[0]
                    single[f"d{gr}f{f}"] = capture(fused_one(gr, f))
        # timing
        for name, gph in timed.items():
            wall, tab = time_and_profile(c, gph)
            ent = {"wall": round(wall, 2)}
            if tab is not None:
                ent["ar"] = kavg(tab, "cross_device_reduce_1stage")
                ent["norm"] = kavg(tab, "gemma_fused_add_rmsnorm")
                ent["fused"] = kavg(tab, "push_ar_norm_kernel")
                ent["gemm"] = kavg(tab, "Cijk")
                if name == "a" and M == 617:
                    ent["top"] = sorted(((k[:60], n, round(t, 1)) for k, (n, t) in tab.items()),
                                        key=lambda x: -x[1] * x[2])[:6]
            r[name] = ent
        # correctness
        mism = {k: 0 for k in single if k != "a"}
        ulpmax = {k: 0.0 for k in single if k.startswith("d")}
        hashes = []
        gA = torch.Generator(device="cuda")
        for i in range(corr):
            gA.manual_seed(10_000 + 2 * i + c.local)
            A.copy_(torch.randn(M, HID, generator=gA, device="cuda").to(torch.bfloat16))
            single["a"][0].replay()
            ar_h, out_h, r_h = (t.clone() for t in single["a"][1])
            hs = [bits_hash(ar_h), bits_hash(r_h), bits_hash(out_h)]
            for k, (gph, outs) in single.items():
                if k == "a":
                    continue
                gph.replay()
                if k.startswith("b"):
                    ok = torch.equal(outs[0].view(torch.int16), ar_h.view(torch.int16)) and \
                        torch.equal(outs[1].view(torch.int16), out_h.view(torch.int16))
                    hs.append(bits_hash(outs[0]))
                else:
                    ok = torch.equal(outs[1].view(torch.int16), r_h.view(torch.int16))
                    u, _ = ulp_err(outs[0], out_h)
                    ulpmax[k] = max(ulpmax[k], u)
                    ok = ok and u <= 1.0
                    hs.append(bits_hash(outs[0]))
                    hs.append(bits_hash(outs[1]))
                mism[k] += 0 if ok else 1
            hashes.append(hs)
        allh = [None, None]
        dist.all_gather_object(allh, hashes, group=c.pg)
        r["corr"] = {"iters": corr, "mismatch_vs_head": mism, "ulp_max": ulpmax,
                     "ranks_identical": allh[0] == allh[1],
                     "p2p_err": c.p2p.read_err(c.ctl) if c.p2p is not None else None}
        res[M] = r
        if c.rank == 0:
            log(f"RESULT M={M} " + json.dumps(r))
        del timed, single
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        dist.barrier(group=c.pg)
    return res


# ------------------------------------------------------------------------------------------
# (c) link probes
# ------------------------------------------------------------------------------------------
def link_part(c):
    if c.p2p is None:
        return "no p2p ext"
    big = 64 << 20
    src = c.p2p.alloc(big, False)
    dst = c.p2p.alloc(big, False)
    c.p2p.fill_random(src, big, c.local)
    allh = [None, None]
    dist.all_gather_object(allh, (c.p2p.ipc_handle(src), c.p2p.ipc_handle(dst)), group=c.pg)
    psrc = c.p2p.ipc_open(allh[1 - c.local][0])
    pdst = c.p2p.ipc_open(allh[1 - c.local][1])
    out = {}
    win = 2621440  # 2.5 MB
    grids = (16, 32, 64, c.cu, 2 * c.cu, 4 * c.cu)
    for size, reps in ((win, 100), (big, 10)):
        for mode in ("read", "push", "memcpy"):
            for gr in (grids if mode != "memcpy" else (0,)):
                for both in (False, True):
                    active = both or c.local == 0
                    torch.cuda.synchronize()
                    dist.barrier(group=c.pg)
                    ms = None
                    if active:
                        e0 = torch.cuda.Event(enable_timing=True)
                        e1 = torch.cuda.Event(enable_timing=True)
                        e0.record()
                        for i in range(reps):
                            off = (i % 25) * win if size == win else 0
                            if mode == "read":
                                c.p2p.probe(0, dst + off, psrc + off, size, gr)
                            elif mode == "push":
                                c.p2p.probe(1, pdst + off, src + off, size, gr)
                            else:
                                c.p2p.probe(2, dst + off, psrc + off, size, 0)
                        e1.record()
                        torch.cuda.synchronize()
                        ms = e0.elapsed_time(e1)
                    dist.barrier(group=c.pg)
                    if active and c.local == 0:
                        gbs = size * reps / (ms * 1e-3) / 1e9
                        out[f"{size >> 10}KB:{mode}:g{gr}:{'bi' if both else 'uni'}"] = (
                            round(gbs, 1), round(ms * 1e3 / reps, 1))
    if c.rank == 0:
        log("RESULT link " + json.dumps(out))
    return out


# ------------------------------------------------------------------------------------------
def worker(rank, world, port, tag):
    import faulthandler

    faulthandler.dump_traceback_later(2400, exit=True)
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}")
    c = setup(rank, world)
    out = {"notes": c.notes, "cu": c.cu}
    parts = [("link", link_part), ("boundary", boundary_part)]
    for name, fn in parts:
        try:
            out[name] = fn(c)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            out[name] = f"ERR {type(e).__name__}: {e}"
        dist.barrier()
        log(f"[{tag}] rank {rank} done {name}")
    allr = [None] * world
    dist.all_gather_object(allr, out)
    if rank == 0:
        try:
            report(tag, allr)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
    dist.barrier()
    faulthandler.cancel_dump_traceback_later()
    sys.stdout.flush()
    os._exit(0)


def fit(points):
    """least squares t = a + bytes / B; points [(bytes, us)] -> (a us, B GB/s)."""
    n = len(points)
    if n < 2:
        return None, None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    k = sxy / sxx
    a = my - k * mx
    return round(a, 1), round(1e-3 / k, 1) if k > 0 else None  # bytes/us -> GB/s


def report(tag, allr):
    r0 = allr[0]
    log(f"RAW[{tag}] " + json.dumps(r0, default=str)[:20000])
    for i in range(0, len(allr), 2):
        if i:
            b = allr[i].get("boundary")
            if isinstance(b, dict):
                log(f"PAIR{i // 2}[{tag}] " + json.dumps({M: {k: v.get("ar") if isinstance(v, dict) and "ar" in v else None
                                                               for k, v in e.items() if k != "corr"}
                                                           for M, e in b.items()}, default=str))
    b = r0.get("boundary")
    lk = r0.get("link")
    if not isinstance(b, dict):
        log(f"GATE[{tag}] no boundary results: {b}")
        return
    names = list(next(iter(b.values())).keys())
    names = [n for n in names if n != "corr"]
    log(f"TABLE[{tag}] per-kernel us (rank 0). boundary = AR + norm, or the fused kernel")
    log("TABLE M | " + " | ".join(names))
    bnd = {}
    for M, e in b.items():
        cells = []
        for n in names:
            v = e.get(n, {})
            if n.startswith("d"):
                bt = v.get("fused")
            elif n.endswith("_clean"):
                bt = v.get("ar")
            else:
                bt = (v.get("ar") or 0) + (v.get("norm") or 0) if v.get("ar") else None
            bnd[(int(M), n)] = bt
            cells.append(f"ar {v.get('ar')} nrm {v.get('norm')} fz {v.get('fused')} bnd {bt} wall {v.get('wall')}")
        log(f"TABLE {M} | " + " | ".join(cells))
        log(f"CORR {M} {json.dumps(e.get('corr'))}")
    # fits
    for n in names:
        key = "fused" if n.startswith("d") else "ar"
        pts = [(int(M) * HID * 2, b[M][n].get(key)) for M in b if b[M].get(n, {}).get(key)]
        a, B = fit(pts)
        log(f"FIT[{tag}] {n} ({key}): fixed {a} us + bytes / {B} GB/s")
    a_head, B_head = fit([(int(M) * HID * 2, b[M]["a"]["ar"]) for M in b if b[M]["a"].get("ar")])
    # link
    blink = None
    if isinstance(lk, dict):
        bi = {k: v[0] for k, v in lk.items() if k.endswith(":bi") and k.startswith("65536KB")}
        if bi:
            kbest = max(bi, key=bi.get)
            blink = bi[kbest]
            log(f"LINK[{tag}] best 64MB both-directions per-rank rate {blink} GB/s ({kbest})")
        for k, v in sorted(lk.items()):
            log(f"LINK {k} {v[0]} GB/s {v[1]} us")
    M = 617
    if M in b or str(M) in b:
        key = M if M in b else str(M)
        head_b = bnd.get((M, "a"))
        cands = {n: bnd[(M, n)] for n in names if (n.startswith("b") and not n.endswith("_clean") or n.startswith("d"))
                 and bnd.get((M, n))}
        best = min(cands, key=cands.get) if cands else None
        bestb = min((n for n in cands if n.startswith("b")), key=cands.get, default=None)
        bestd = min((n for n in cands if n.startswith("d")), key=cands.get, default=None)
        save = head_b - cands[best] if best and head_b else None
        log(f"GATE[{tag}] head boundary @617 {head_b} us; best {best} {cands.get(best)} us; save {save} us "
            f"-> {'PASS' if save is not None and save >= 20 else 'KILL'} (need >= 20)")
        har = b[key]["a"].get("ar")
        log(f"GATE[{tag}] isolated head AR @617 {har} us -> {'PASS' if har and har > 30 else 'KILL'} (kill if <= 30)")
        log(f"GATE[{tag}] B_link {blink} vs 1.3 x head streaming {B_head} GB/s (fixed {a_head} us) -> "
            f"{'PASS' if blink and B_head and blink >= 1.3 * B_head else 'KILL'}")
        if bestb and bestd:
            log(f"GATE[{tag}] fused {bestd} {cands[bestd]} vs widened {bestb} {cands[bestb]}: "
                f"fused saves {round(cands[bestb] - cands[bestd], 2)} us/boundary (2b needs >= 8)")


def main():
    log("devices:", torch.cuda.device_count())
    os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx90a")
    t = time.time()
    for name, fn in (("variant", build_variant), ("p2p", build_p2p)):
        try:
            fn()
            log(f"build {name} ok")
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            log(f"build {name} FAILED")
    log(f"builds {time.time() - t:.0f}s")
    modes = os.environ.get("PREFILL_AR_MODES", "pair01,pairs4").split(",")
    if "pair01" in modes:
        mp.spawn(worker, args=(2, 29631, "pair01"), nprocs=2, join=True)
    if "pairs4" in modes and torch.cuda.device_count() >= 8:
        mp.spawn(worker, args=(8, 29632, "pairs4"), nprocs=8, join=True)


if __name__ == "__main__":
    main()
