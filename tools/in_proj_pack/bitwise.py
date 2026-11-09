"""Plan 32 stage 1: the packed qkvz+ba in_proj (as finalize_fused_in_proj builds it) against the
two separate skinny projections the head runs, bitwise, M=1..16, eager and graph-replayed, and
the fused decode unpack+Conv1D fed the packed (strided) views against contiguous copies.
Per-rank TP=2 shapes of Qwen3.6-35B-A3B: qkvz [6144, 2048], ba [32, 2048]."""
import sys

import torch

from sglang.kernels.ops.gemm import skinny_gemm_hip as sk

torch.manual_seed(0)
dev = "cuda"
K, NQKVZ, NBA = 2048, 6144, 32
wq = (torch.randn(NQKVZ, K, device=dev) * 0.02).to(torch.bfloat16)
wb = (torch.randn(NBA, K, device=dev) * 0.02).to(torch.bfloat16)
fused = torch.cat([wq, wb], 0).contiguous()
wq_alias, wb_alias = fused[:NQKVZ], fused[NQKVZ:]
print("gemv_active packed", sk.gemv_active(*fused.shape), "qkvz", sk.gemv_active(NQKVZ, K), "ba", sk.gemv_active(NBA, K))

ok = True
for m in range(1, 17):
    x = torch.randn(m, K, device=dev).to(torch.bfloat16)
    assert sk.applicable(x, fused, None) and sk.applicable(x, wq, None) and sk.applicable(x, wb, None)
    sep_q, sep_b = sk.skinny_linear(x, wq, None), sk.skinny_linear(x, wb, None)
    out = sk.skinny_linear(x, fused, None)
    pq, pb = out[:, :NQKVZ], out[:, NQKVZ:]
    # prefill-sized calls still run the separate projections on the aliased rows
    aq, ab = sk.skinny_linear(x, wq_alias, None), sk.skinny_linear(x, wb_alias, None)
    eq = torch.equal(pq, sep_q) and torch.equal(pb, sep_b) and torch.equal(aq, sep_q) and torch.equal(ab, sep_b)
    ok &= eq
    print(f"M={m:2d} packed==separate {torch.equal(pq, sep_q)}/{torch.equal(pb, sep_b)} alias==separate {torch.equal(aq, sep_q)}/{torch.equal(ab, sep_b)}"
          f" maxdiff {(pq.float()-sep_q.float()).abs().max().item():.3g}/{(pb.float()-sep_b.float()).abs().max().item():.3g}")

# graph replay at M=2 and M=16
for m in (2, 16):
    xs = torch.randn(m, K, device=dev).to(torch.bfloat16)
    for _ in range(3):
        sk.skinny_linear(xs, fused, None); sk.skinny_linear(xs, wq, None); sk.skinny_linear(xs, wb, None)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        go = sk.skinny_linear(xs, fused, None)
    for trial in range(5):
        xs.copy_(torch.randn(m, K, device=dev).to(torch.bfloat16))
        g.replay(); torch.cuda.synchronize()
        e = torch.equal(go[:, :NQKVZ], sk.skinny_linear(xs, wq, None)) and torch.equal(go[:, NQKVZ:], sk.skinny_linear(xs, wb, None))
        ok &= e
    print(f"graph M={m}: packed replay == separate eager over 5 inputs: {e}")

# fused decode unpack + Conv1D on strided views of the packed output vs contiguous copies
try:
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import fused_qkvzba_causal_conv1d_update_contiguous as fz
    qkv_dim, v_dim, nvh, hvd, W = 4096, 2048, 16, 128, 4
    conv_w = (torch.randn(qkv_dim, W, device=dev) * 0.1).to(torch.bfloat16)
    conv_b = (torch.randn(qkv_dim, device=dev) * 0.1).to(torch.bfloat16)
    for m in (1, 2, 8, 16):
        x = torch.randn(m, K, device=dev).to(torch.bfloat16)
        out = sk.skinny_linear(x, fused, None)
        pq, pb = out[:, :NQKVZ], out[:, NQKVZ:]
        state0 = torch.randn(40, qkv_dim, W - 1, device=dev).to(torch.bfloat16)
        idx = torch.randperm(40, device=dev)[:m].to(torch.int32)
        s1, s2 = state0.clone(), state0.clone()
        r1 = fz(pq, pb, s1, conv_w, conv_b, idx, qkv_dim=qkv_dim, v_dim=v_dim, num_v_heads=nvh, head_v_dim=hvd, activation="silu")
        r2 = fz(pq.contiguous(), pb.contiguous(), s2, conv_w, conv_b, idx, qkv_dim=qkv_dim, v_dim=v_dim, num_v_heads=nvh, head_v_dim=hvd, activation="silu")
        e = all(torch.equal(a, b) for a, b in zip(r1, r2)) and torch.equal(s1, s2)
        ok &= e
        print(f"fused unpack+conv M={m}: strided views == contiguous (outputs + state): {e}")
except Exception as ex:  # report, do not hide
    ok = False
    print("fused unpack+conv check error:", repr(ex))

print(f"BITWISE {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
