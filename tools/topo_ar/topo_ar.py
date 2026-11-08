"""Topology and small all-reduce latency on the node's GCDs (plan 4, step 1).

Prints rocm-smi topology, the HIP device -> PCI bus / card mapping, then measures RCCL
all-reduce latency through torch.distributed for BF16 messages of 8-64 KB (plus two larger
sizes for bandwidth), eager and inside a HIP graph, on several GCD groups. Not on any serving
path; it reads nothing of the model.
"""
import os
import subprocess
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SIZES_KB = [8, 16, 32, 64, 2304, 16384]
# Group sets measured one after another. Each entry is a list of disjoint groups run
# concurrently (all ranks in all groups run the same loop at once).
GROUP_SETS = [
    ("pair01", [[0, 1]]),
    ("pair23", [[2, 3]]),
    ("pair45", [[4, 5]]),
    ("pair67", [[6, 7]]),
    ("pair02", [[0, 2]]),
    ("pair04", [[0, 4]]),
    ("pair06", [[0, 6]]),
    ("pair13", [[1, 3]]),
    ("pair15", [[1, 5]]),
    ("pair17", [[1, 7]]),
    ("4pairs_concurrent", [[0, 1], [2, 3], [4, 5], [6, 7]]),
    ("quad0123", [[0, 1, 2, 3]]),
    ("quad4567", [[4, 5, 6, 7]]),
    ("2quads_concurrent", [[0, 1, 2, 3], [4, 5, 6, 7]]),
    ("all8", [list(range(8))]),
]
EAGER_ITERS = 300
GRAPH_OPS = 100
GRAPH_REPLAYS = 20


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        print(out.stdout + out.stderr, flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"failed: {e}", flush=True)


def topology():
    for c in [
        "rocm-smi --showtopo",
        "rocm-smi --showtopotype",
        "rocm-smi --showtopoweight",
        "rocm-smi --showtopohops",
        "rocm-smi --showbus",
        "rocm-smi --showproductname",
        "rocm-smi --showuniqueid",
        "rocm-smi --shownodesbw",
        "amd-smi list",
        "amd-smi topology",
        "env | grep -E 'HIP_VISIBLE|ROCR_VISIBLE|CUDA_VISIBLE|NCCL|RCCL|HSA' || true",
        "cat /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null | grep -E 'location_id|domain|unique_id|simd_count' | head -64",
        "for n in /sys/class/kfd/kfd/topology/nodes/*; do echo $n; cat $n/io_links/*/properties 2>/dev/null | grep -E 'node_to|weight|type|max_bandwidth' | paste -sd' '; done",
    ]:
        sh(c)
    print("HIP device mapping (torch):", flush=True)
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        bus = getattr(p, "pci_bus_id", None)
        dev = getattr(p, "pci_device_id", None)
        dom = getattr(p, "pci_domain_id", None)
        uuid = getattr(p, "uuid", None)
        print(f"  hip:{i} name={p.name} arch={getattr(p, 'gcnArchName', '?')} "
              f"pci={dom}:{bus}:{dev} uuid={uuid} mem={p.total_memory >> 30}GiB", flush=True)
    print("torch", torch.__version__, "hip", torch.version.hip, "rccl", torch.cuda.nccl.version(),
          flush=True)


def bench_group(rank, pg, size_kb, mode):
    n = size_kb * 1024 // 2
    x = torch.ones(n, dtype=torch.bfloat16, device="cuda")
    for _ in range(10):
        dist.all_reduce(x, group=pg)
    torch.cuda.synchronize()
    if mode == "eager":
        iters = EAGER_ITERS if size_kb <= 64 else 50
        dist.barrier(group=pg)
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x, group=pg)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e6
    ops = GRAPH_OPS if size_kb <= 64 else 20
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        dist.all_reduce(x, group=pg)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    # thread_local: the ProcessGroupNCCL watchdog thread queries events while this thread
    # captures; in the default global mode that query invalidates the capture.
    with torch.cuda.graph(g, capture_error_mode="thread_local"):
        for _ in range(ops):
            dist.all_reduce(x, group=pg)
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    dist.barrier(group=pg)
    t0 = time.perf_counter()
    for _ in range(GRAPH_REPLAYS):
        g.replay()
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / (GRAPH_REPLAYS * ops) * 1e6
    del g
    return us


def worker(rank, world):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world,
                            init_method="tcp://127.0.0.1:29591",
                            device_id=torch.device("cuda", rank))
    if rank == 0:
        topology()
    dist.barrier()
    # new_group must be called by every rank, for every group, in the same order.
    all_pgs = [[dist.new_group(g) for g in groups] for _, groups in GROUP_SETS]
    # All eager numbers first, so a failed graph capture cannot lose them.
    for mode in ("eager", "graph"):
        for (name, groups), pgs in zip(GROUP_SETS, all_pgs):
            mine = [(g, pg) for g, pg in zip(groups, pgs) if rank in g]
            for kb in SIZES_KB:
                dist.barrier()
                val = None
                if mine:
                    try:
                        val = bench_group(rank, mine[0][1], kb, mode)
                    except Exception as e:  # noqa: BLE001
                        val = f"ERR {type(e).__name__}: {e}"
                gathered = [None] * world
                dist.all_gather_object(gathered, val)
                if rank == 0:
                    per_group = []
                    for g in groups:
                        vals = [gathered[r] for r in g]
                        if any(isinstance(v, str) for v in vals):
                            per_group.append(str([v for v in vals if isinstance(v, str)][0]))
                        else:
                            per_group.append(f"{max(vals):.1f}")
                    line = f"RESULT {name:20s} {mode:5s} {kb:6d}KB us/op(max rank) per group: {' '.join(per_group)}"
                    if kb >= 1024 and all(not isinstance(gathered[r], str) for r in groups[0]):
                        t = max(gathered[r] for r in groups[0]) * 1e-6
                        k = len(groups[0])
                        busbw = kb * 1024 * 2 * (k - 1) / k / t / 1e9
                        line += f"  busbw(g0)={busbw:.1f}GB/s"
                    print(line, flush=True)
    dist.barrier()
    dist.destroy_process_group()


PYNCCL_SETS = [
    ("pair01", [[0, 1]]),
    ("pair02", [[0, 2]]),
    ("pair04", [[0, 4]]),
    ("4pairs_concurrent", [[0, 1], [2, 3], [4, 5], [6, 7]]),
    ("quad0123", [[0, 1, 2, 3]]),
    ("2quads_concurrent", [[0, 1, 2, 3], [4, 5, 6, 7]]),
    ("all8", [list(range(8))]),
]
PYNCCL_SIZES_KB = [8, 16, 32, 64, 2304]


def pynccl_worker(rank, world):
    """Graph-mode all-reduce through SGLang's PyNcclCommunicator, the path SGLang's HIP-graph
    decode uses when custom all-reduce is off. torch.distributed is gloo-only here, so no
    ProcessGroupNCCL watchdog exists to poll events during capture."""
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world,
                            init_method="tcp://127.0.0.1:29592")
    from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator

    all_pgs = [[dist.new_group(g, backend="gloo") for g in groups] for _, groups in PYNCCL_SETS]
    for (name, groups), pgs in zip(PYNCCL_SETS, all_pgs):
        mine = [pg for g, pg in zip(groups, pgs) if rank in g]
        comm = PyNcclCommunicator(mine[0], device=rank) if mine else None
        for kb in PYNCCL_SIZES_KB:
            dist.barrier()
            val = None
            if comm is not None:
                try:
                    n = kb * 1024 // 2
                    x = torch.ones(n, dtype=torch.bfloat16, device="cuda")
                    ops = GRAPH_OPS if kb <= 64 else 20
                    s = torch.cuda.Stream()
                    with comm.change_state(enable=True):
                        with torch.cuda.stream(s):
                            for _ in range(10):
                                comm.all_reduce(x)
                        s.synchronize()
                        eager_iters = 300 if kb <= 64 else 50
                        dist.barrier(group=mine[0])
                        t0 = time.perf_counter()
                        with torch.cuda.stream(s):
                            for _ in range(eager_iters):
                                comm.all_reduce(x)
                        s.synchronize()
                        eager_us = (time.perf_counter() - t0) / eager_iters * 1e6
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g, stream=s):
                            for _ in range(ops):
                                comm.all_reduce(x)
                    torch.cuda.synchronize()
                    g.replay()
                    torch.cuda.synchronize()
                    dist.barrier(group=mine[0])
                    t0 = time.perf_counter()
                    for _ in range(GRAPH_REPLAYS):
                        g.replay()
                    torch.cuda.synchronize()
                    graph_us = (time.perf_counter() - t0) / (GRAPH_REPLAYS * ops) * 1e6
                    del g
                    val = (eager_us, graph_us)
                except Exception as e:  # noqa: BLE001
                    val = f"ERR {type(e).__name__}: {e}"
            gathered = [None] * world
            dist.all_gather_object(gathered, val)
            if rank == 0:
                cols = []
                for g in groups:
                    vals = [gathered[r] for r in g]
                    bad = [v for v in vals if isinstance(v, str)]
                    if bad:
                        cols.append(bad[0])
                    else:
                        cols.append(f"eager={max(v[0] for v in vals):.1f} "
                                    f"graph={max(v[1] for v in vals):.1f}")
                print(f"RESULT pynccl {name:20s} {kb:6d}KB us/op(max rank) per group: "
                      + " | ".join(cols), flush=True)
    dist.barrier()
    dist.destroy_process_group()


# ---------------------------------------------------------------------------------------------
# Plan 6 step 1: SGLang's HIP CustomAllreduce on same-card GCD pairs (the TP=2 groups the
# 4xTP=2 layout uses), against PyNccl (RCCL) on the same groups, plus correctness.
# ---------------------------------------------------------------------------------------------
CA_SETS = [
    ("pair01", [[0, 1]]),
    ("4pairs_concurrent", [[0, 1], [2, 3], [4, 5], [6, 7]]),
]
CA_SIZES_KB = [8, 16, 32, 64, 2304]
HIDDEN = 2048
CORR_ITERS = 500


def kernel_probe():
    """Which load/store branch of custom_all_reduce_hip.cuh the gfx90a code object holds."""
    import glob
    import tempfile

    import sgl_kernel

    d = os.path.dirname(sgl_kernel.__file__)
    sos = glob.glob(os.path.join(d, "**", "*.so"), recursive=True)
    llvm = "/opt/rocm/llvm/bin"
    tmp = tempfile.mkdtemp()
    for so in sos:
        fb = os.path.join(tmp, "fb.bin")
        co = os.path.join(tmp, "k.co")
        r = subprocess.run(f"objcopy --dump-section .hip_fatbin={fb} {so} {tmp}/x.so",
                           shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            continue
        lst = subprocess.run(f"{llvm}/clang-offload-bundler --list --type=o --input={fb}",
                             shell=True, capture_output=True, text=True).stdout.split()
        tgt = [t for t in lst if "gfx90a" in t]
        print("kernel_probe:", so, "targets:", lst, flush=True)
        if not tgt:
            continue
        subprocess.run(f"{llvm}/clang-offload-bundler --unbundle --type=o --input={fb} "
                       f"--targets={tgt[0]} --output={co}", shell=True)
        dis = subprocess.run(f"{llvm}/llvm-objdump -d --mcpu=gfx90a {co}", shell=True,
                             capture_output=True, text=True).stdout
        # Print the body of the 2-rank bf16 1-stage kernel's memory instructions.
        cur, shown = None, 0
        for line in dis.splitlines():
            if line.endswith(">:"):
                cur = line
                if "cross_device_reduce_1stage" in line and "Li2E" in line and "bfloat" in line \
                        and shown == 0:
                    print("kernel_probe symbol:", line, flush=True)
                    shown = 1
                elif shown == 1:
                    shown = 2
                continue
            if shown == 1 and any(k in line for k in ("global_", "flat_", "buffer_wbl2",
                                                      "buffer_invl2", "buffer_wbinvl1",
                                                      "s_waitcnt vmcnt(0)")):
                print("  ", line.strip()[:140], flush=True)
        if shown == 0:
            names = sorted({l for l in dis.splitlines() if "cross_device_reduce_1stage" in l
                            and l.endswith(">:")})
            print("kernel_probe: 1stage symbols:", names[:8], flush=True)


def _seeded(rank, it, kb, n):
    g = torch.Generator(device="cuda")
    g.manual_seed(1_000_003 * rank + 7919 * it + kb)
    return torch.randn(n, generator=g, device="cuda", dtype=torch.float32).to(torch.bfloat16)


def _graph_time(fn_step, x0, pg, ops, comm_ctx):
    """Capture `ops` chained calls of fn_step(x) -> x in one HIP graph, replay, us/op."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with comm_ctx():
        with torch.cuda.graph(g, stream=s):
            x = x0
            for _ in range(ops):
                x = fn_step(x)
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    dist.barrier(group=pg)
    t0 = time.perf_counter()
    for _ in range(GRAPH_REPLAYS):
        g.replay()
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / (GRAPH_REPLAYS * ops) * 1e6
    del g
    return us


def ca_worker(rank, world):
    import contextlib
    import faulthandler
    import hashlib

    faulthandler.dump_traceback_later(1000, exit=True)
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world,
                            init_method="tcp://127.0.0.1:29593")
    from sglang.srt.distributed.device_communicators import custom_all_reduce as car
    from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
        gather_physical_device_ids,
        is_full_nvlink,
    )
    from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator

    if rank == 0:
        print("CA class (dispatch):", car.dispatch_custom_allreduce(None, None),
              "amd_deterministic:", car._use_amd_deterministic_impl(), flush=True)
        try:
            kernel_probe()
        except Exception as e:  # noqa: BLE001
            print("kernel_probe failed:", e, flush=True)
    all_pgs = [[dist.new_group(g, backend="gloo") for g in groups] for _, groups in CA_SETS]
    for (name, groups), pgs in zip(CA_SETS, all_pgs):
        mine = [(g, pg) for g, pg in zip(groups, pgs) if rank in g]
        dist.barrier()
        g_ranks, pg = mine[0] if mine else (None, None)
        ca = nccl = None
        if pg is not None:
            fn = is_full_nvlink(gather_physical_device_ids(pg, torch.device("cuda", rank)), 2)
            ca = car.CustomAllreduce(group=pg, device=rank)
            nccl = PyNcclCommunicator(pg, device=rank)
            probe = torch.empty(4096, dtype=torch.bfloat16, device="cuda")
            print(f"CAINIT {name} rank{rank} group={g_ranks} is_full_nvlink={fn} "
                  f"disabled={ca.disabled} full_nvlink={getattr(ca, 'full_nvlink', None)} "
                  f"max_size={getattr(ca, 'max_size', None)} "
                  f"should_custom_ar(8KB)={ca.should_custom_ar(probe)} "
                  f"should_custom_ar(2.3MB)={ca.should_custom_ar(torch.empty(1179648, dtype=torch.bfloat16, device='cuda'))}",
                  flush=True)
        for kb in CA_SIZES_KB:
            dist.barrier()
            val = None
            if ca is not None and not ca.disabled:
                try:
                    val = ca_bench_one(rank, g_ranks, pg, ca, nccl, kb)
                except Exception as e:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    val = f"ERR {type(e).__name__}: {e}"
            gathered = [None] * world
            dist.all_gather_object(gathered, val)
            if rank == 0:
                for gi, g in enumerate(groups):
                    vals = [gathered[r] for r in g]
                    bad = [v for v in vals if isinstance(v, str) or v is None]
                    if bad:
                        print(f"RESULT ca {name:18s} g{gi} {kb:5d}KB {bad[0]}", flush=True)
                        continue
                    t = {k: max(v[k] for v in vals) for k in vals[0] if k.endswith("_us")}
                    c = vals[0]
                    same = all(v["hashes"] == vals[0]["hashes"] for v in vals)
                    print(f"RESULT ca {name:18s} g{gi} {kb:5d}KB "
                          f"ca_eager={t['ca_eager_us']:.1f} ca_graph={t['ca_graph_us']:.1f} "
                          f"nccl_graph={t['nccl_graph_us']:.1f} | in-context(after GEMM): "
                          f"gemm={t['gemm_us']:.1f} +ca={t['ca_ctx_us'] - t['gemm_us']:.1f} "
                          f"+nccl={t['nccl_ctx_us'] - t['gemm_us']:.1f} | "
                          f"correct graph: bad={sum(v['g_bad'] for v in vals)} "
                          f"maxerr_ulp={max(v['g_maxulp'] for v in vals):.2f} "
                          f"exact={min(v['g_exact'] for v in vals):.5f} "
                          f"eager: bad={sum(v['e_bad'] for v in vals)} "
                          f"maxerr_ulp={max(v['e_maxulp'] for v in vals):.2f} "
                          f"| ranks_bitwise_identical={same} iters={CORR_ITERS}",
                          flush=True)
    dist.barrier()
    faulthandler.cancel_dump_traceback_later()
    dist.destroy_process_group()


def _ulp_err(out, ref32):
    """Error of out in units of one BF16 ulp of the correctly rounded reference."""
    refb = ref32.to(torch.bfloat16).float()
    a = refb.abs().clamp_min(2.0 ** -126)
    ulp = torch.exp2(torch.floor(torch.log2(a)) - 7)
    e = (out.float() - ref32).abs() / ulp
    exact = (out.float() == refb).float().mean().item()
    return e.max().item(), int((e > 1.0).sum().item()), exact


def ca_bench_one(rank, g_ranks, pg, ca, nccl, kb):
    import contextlib

    n = kb * 1024 // 2
    rows = n // HIDDEN
    res = {}
    big = kb > 64
    ops = 20 if big else GRAPH_OPS
    # --- eager (unregistered path: copy into the IPC buffer, then the kernel) ---
    x = torch.ones(n, dtype=torch.bfloat16, device="cuda")
    for _ in range(10):
        ca.custom_all_reduce(x)
    torch.cuda.synchronize()
    iters = 50 if big else 300
    dist.barrier(group=pg)
    t0 = time.perf_counter()
    for _ in range(iters):
        ca.custom_all_reduce(x)
    torch.cuda.synchronize()
    res["ca_eager_us"] = (time.perf_counter() - t0) / iters * 1e6

    def ca_ctx():
        return ca.capture()

    def nccl_ctx():
        return nccl.change_state(enable=True)

    # --- graph, back-to-back all-reduces on one buffer ---
    res["ca_graph_us"] = _graph_time(lambda v: ca.custom_all_reduce(v), x, pg, ops, ca_ctx)

    def nccl_step(v):
        nccl.all_reduce(v)
        return v
    res["nccl_graph_us"] = _graph_time(nccl_step, x, pg, ops, nccl_ctx)

    # --- graph, in context: x <- AR(x @ W), like a row-parallel projection feeding the
    # next layer; the AR's input is written by the kernel just before it.
    torch.manual_seed(rank)
    W = (torch.randn(HIDDEN, HIDDEN, device="cuda") * 0.02).to(torch.bfloat16)
    xm = torch.randn(rows, HIDDEN, device="cuda").to(torch.bfloat16)
    res["gemm_us"] = _graph_time(lambda v: v @ W, xm, pg, ops, contextlib.nullcontext)
    res["ca_ctx_us"] = _graph_time(lambda v: ca.custom_all_reduce(v @ W), xm, pg, ops, ca_ctx)

    def nccl_ctx_step(v):
        h = v @ W
        nccl.all_reduce(h)
        return h
    res["nccl_ctx_us"] = _graph_time(nccl_ctx_step, xm, pg, ops, nccl_ctx)

    # --- correctness, graph mode: static input refreshed every iteration with fresh random
    # values; inside the graph a kernel writes h = 2*inp, then the AR reads h on both ranks.
    inp = torch.zeros(n, dtype=torch.bfloat16, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with ca.capture():
        with torch.cuda.graph(g, stream=s):
            out_g = ca.custom_all_reduce(inp * 2)
    torch.cuda.synchronize()
    wpos = (torch.arange(n, device="cuda", dtype=torch.int64) % 65521) + 1
    hashes = []
    g_bad = e_bad = 0
    g_max = e_max = 0.0
    g_exact = 1.0
    for it in range(CORR_ITERS):
        mine_in = _seeded(rank, it, kb, n)
        ref = torch.zeros(n, dtype=torch.float32, device="cuda")
        for r in g_ranks:  # fp32 sum of every rank's input, regenerated locally
            ref += _seeded(r, it, kb, n).float()
        inp.copy_(mine_in)
        dist.barrier(group=pg)  # all ranks have new input before any rank reduces
        g.replay()
        torch.cuda.synchronize()
        m, b, ex = _ulp_err(out_g, ref * 2)
        g_max, g_bad, g_exact = max(g_max, m), g_bad + b, min(g_exact, ex)
        hashes.append(int((out_g.view(torch.int16).to(torch.int64) * wpos).sum().item()))
        # eager on the same inputs (the server's prefill/extend path)
        out_e = ca.custom_all_reduce(mine_in)
        torch.cuda.synchronize()
        m, b, _ = _ulp_err(out_e, ref)
        e_max, e_bad = max(e_max, m), e_bad + b
        hashes.append(int((out_e.view(torch.int16).to(torch.int64) * wpos).sum().item()))
    del g
    res.update(g_bad=g_bad, g_maxulp=g_max, g_exact=g_exact, e_bad=e_bad, e_maxulp=e_max,
               hashes=hashes)
    return res


def main():
    world = torch.cuda.device_count()
    if os.environ.get("TOPO_AR_MODE") == "ca":
        print("devices:", world, "mode: ca", flush=True)
        mp.spawn(ca_worker, args=(world,), nprocs=world, join=True)
        return
    if os.environ.get("TOPO_AR_MODE") == "pynccl":
        print("devices:", world, "mode: pynccl", flush=True)
        mp.spawn(pynccl_worker, args=(world,), nprocs=world, join=True)
        return
    print("devices:", world, flush=True)
    if world < 8:
        topology()
        sys.exit("need 8 GCDs")
    mp.spawn(worker, args=(world,), nprocs=world, join=True)


if __name__ == "__main__":
    main()
