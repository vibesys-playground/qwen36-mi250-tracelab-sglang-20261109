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


def main():
    world = torch.cuda.device_count()
    print("devices:", world, flush=True)
    if world < 8:
        topology()
        sys.exit("need 8 GCDs")
    mp.spawn(worker, args=(world,), nprocs=world, join=True)


if __name__ == "__main__":
    main()
