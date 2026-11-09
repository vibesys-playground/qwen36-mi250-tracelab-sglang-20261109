"""Plan 32 stage 2 gate 0 (one GCD): does warming the first S MB of a decode GEMV's weight in L2
save GEMV time, and do skf2's nontemporal loads hit L2?

Every iteration in a captured graph is [flush L2 (32 MB plain read), pre, X]. pre reads S bytes
of either the target (the bytes X reads first) or a same-size decoy, so pre costs the same and
T(target) - T(decoy) is X's saving from a warm start. Gates: qkvz >= 1.5 us, gdn_out >= 1.0 us."""
import ctypes, os, statistics, subprocess, sys, tempfile

import torch

from sglang.kernels.ops.gemm import skinny_gemm_hip as sk

HERE = os.path.dirname(os.path.abspath(__file__))
so = os.path.join(tempfile.mkdtemp(), "l2probe.so")
subprocess.run(["hipcc", "-O3", "--offload-arch=gfx90a", "-shared", "-fPIC", "-o", so,
                os.path.join(HERE, "l2probe.hip")], check=True)
lib = ctypes.CDLL(so)
lib.touch.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
dev = "cuda"
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).gcnArchName,
      "L2", getattr(torch.cuda.get_device_properties(0), "L2_cache_size", "?"))
sink = torch.zeros(4096, dtype=torch.int32, device=dev)
flushbuf = torch.ones(32 << 20, dtype=torch.uint8, device=dev)
decoy = torch.ones(64 << 20, dtype=torch.uint8, device=dev)
MB = 1 << 20


def touch(t, nbytes, nt=0, blocks=220):
    s = torch.cuda.current_stream().cuda_stream
    assert lib.touch(ctypes.c_void_p(t.data_ptr()), nbytes, nt, blocks, ctypes.c_void_p(sink.data_ptr()), ctypes.c_void_p(s)) == 0


def chain_us(body, iters=100, reps=15):
    """us per iteration of body() (flush included), graph-replayed; median over reps."""
    for _ in range(2):
        body()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            body()
    ts = []
    for _ in range(reps + 2):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / iters)
    return statistics.median(ts[2:])


def flush():
    touch(flushbuf, flushbuf.numel())


res = {}
# (A) a plain read of S bytes, cold vs pre-warmed by a plain or NT touch; and the same for NT reads
print("\n## (A) read kernel over S MB: us of [flush, pre, read]; saving = decoy - target")
tgt = torch.ones(16 << 20, dtype=torch.uint8, device=dev)
for S in (2, 4, 6):
    for rd_nt in (0, 1):
        row = {}
        for pre_nt in (0, 1):
            for which, src in (("target", tgt), ("decoy", decoy)):
                row[(pre_nt, which)] = chain_us(lambda: (flush(), touch(src, S * MB, pre_nt), touch(tgt, S * MB, rd_nt)))
        base = chain_us(lambda: (flush(), touch(decoy, S * MB, 0)))
        print(f"S={S}MB read {'NT' if rd_nt else 'plain'}: read cold ~{row[(0,'decoy')]-base:.2f} us | "
              f"pre plain: saving {row[(0,'decoy')]-row[(0,'target')]:.2f} us | pre NT: saving {row[(1,'decoy')]-row[(1,'target')]:.2f} us", flush=True)

# (B) skf2 GEMVs at M=2 with the first S MB of their weight warmed by a plain touch
print("\n## (B) skf2 GEMV at M=2: us of [flush, pre(S MB), GEMV]; saving = decoy - target")
torch.manual_seed(0)
shapes = {"qkvz+ba [6176,2048]": (6176, 2048), "gdn_out [2048,2048]": (2048, 2048), "attn_qkv [4608,2048]": (4608, 2048)}
for name, (n, k) in shapes.items():
    w = (torch.randn(n, k, device=dev) * 0.02).to(torch.bfloat16)
    x = torch.randn(2, k, device=dev).to(torch.bfloat16)
    assert sk.gemv_active(n, k)
    y0 = sk.skinny_linear(x, w, None).clone()
    base = chain_us(lambda: (flush(), touch(decoy, 4 * MB, 0)))
    full = chain_us(lambda: (flush(), touch(decoy, 4 * MB, 0), sk.skinny_linear(x, w, None)))
    print(f"{name}: GEMV after flush {full-base:.2f} us; weight {n*k*2/MB:.1f} MB", flush=True)
    for S in (1, 2, 4, 6):
        for pre_nt in (0, 1):
            t = chain_us(lambda: (flush(), touch(w, S * MB, pre_nt), sk.skinny_linear(x, w, None)))
            d = chain_us(lambda: (flush(), touch(decoy, S * MB, pre_nt), sk.skinny_linear(x, w, None)))
            res[(name, S, pre_nt)] = d - t
            print(f"  S={S}MB pre {'NT' if pre_nt else 'plain'}: target {t:.2f} decoy {d:.2f} -> saving {d-t:.2f} us", flush=True)
    # whole weight warm where it fits (gdn_out 8.4 MB does not; upper bound with 7 MB)
    assert torch.equal(sk.skinny_linear(x, w, None), y0)

best = lambda nm: max(res[(nm, S, 0)] for S in (1, 2, 4, 6))
q, g = best("qkvz+ba [6176,2048]"), best("gdn_out [2048,2048]")
print(f"\nGATE0 qkvz best saving {q:.2f} us (need >= 1.5), gdn_out best {g:.2f} us (need >= 1.0): {'PASS' if q >= 1.5 and g >= 1.0 else 'KILL'}")
