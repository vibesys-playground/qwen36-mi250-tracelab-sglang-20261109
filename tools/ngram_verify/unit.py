"""Plan 27 stage 1 unit check (one GPU, random weights): the decode-shaped verify kernels
against T successive decode steps with the serving decode kernels.

* GDN: dsv_gdn_conv + dsv_gdn_recurrence vs T x (fused_qkvzba_causal_conv1d_update_contiguous
  + fused_recurrent_gated_delta_rule_packed_decode). Outputs, z/a/b, the per-position conv
  windows and SSM states must be bitwise equal; the persistent states must stay untouched.
* Attention: decode_attention_fwd over the bs*T rows with dsv_num_kv_splits vs T decode
  batches (column t: every request at length seq_len + t + 1, get_num_kv_splits_triton).
* Drafter: prompt lookup on a toy context.

Run from / in the candidate image: python3 <checkout>/tools/ngram_verify/unit.py
"""
import sys
import time
import types

import numpy as np
import torch

from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from sglang.kernels.ops.attention.metadata import get_num_kv_splits_triton
from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkvzba_causal_conv1d_update_contiguous,
)
from sglang.srt.speculative.decode_verify import (
    PromptLookupDrafter,
    dsv_gdn_conv,
    dsv_gdn_recurrence,
    dsv_num_kv_splits,
)

dev = "cuda"
FAIL = []


def check(name, a, b):
    eq = torch.equal(a, b)
    md = (a.float() - b.float()).abs().max().item() if a.shape == b.shape else float("nan")
    print(f"  {'OK  ' if eq else 'FAIL'} {name}: bitwise={eq} maxdiff={md:.3e}", flush=True)
    if not eq:
        FAIL.append(name)


def gdn_case(bs, T, bias, slots=12, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    nk, hk, nv, hv = 8, 128, 16, 128  # Qwen3.6 GDN per rank at TP=2
    qd = kd = nk * hk
    vd = nv * hv
    qkv = qd + kd + vd
    rows = bs * T
    r = lambda *s, dt=torch.bfloat16, sc=1.0: (torch.randn(*s, generator=g, device=dev) * sc).to(dt)
    qkvz = r(rows, qkv + vd)
    ba = r(rows, 2 * nv)
    conv_state = r(slots, qkv, 4)  # physical width 4 > K-1, as the pool may be
    conv_w = r(qkv, 4, sc=0.5)
    conv_b = r(qkv, sc=0.1) if bias else None
    ssm = r(slots, nv, hv, hk, dt=torch.float32, sc=0.1)
    A_log = r(nv, dt=torch.float32, sc=0.5)
    dt_bias = r(nv, dt=torch.float32, sc=0.5)
    idx = torch.tensor([3, 7, 1, 9, 4, 2, 8, 5][:bs], dtype=torch.int32, device=dev)
    inter_idx = torch.arange(bs, dtype=torch.int32, device=dev)
    scale = hk**-0.5

    # reference: T decode steps, column t = rows n*T+t
    cs, ss = conv_state.clone(), ssm.clone()
    ref_o = torch.empty(rows, 1, nv, hv, dtype=torch.bfloat16, device=dev)
    ref_mixed = torch.empty(rows, qkv, dtype=torch.bfloat16, device=dev)
    ref_z = torch.empty(rows, nv, hv, dtype=torch.bfloat16, device=dev)
    ref_a = torch.empty(rows, nv, dtype=torch.bfloat16, device=dev)
    ref_b = torch.empty(rows, nv, dtype=torch.bfloat16, device=dev)
    ref_win, ref_state = [], []
    for t in range(T):
        sel = torch.arange(bs, device=dev) * T + t
        m, z, b, a = fused_qkvzba_causal_conv1d_update_contiguous(
            qkvz[sel].contiguous(), ba[sel].contiguous(), cs, conv_w, conv_b, idx,
            qkv_dim=qkv, v_dim=vd, num_v_heads=nv, head_v_dim=hv, activation="silu",
        )
        o = torch.empty(bs, 1, nv, hv, dtype=torch.bfloat16, device=dev)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=m, a=a, b=b, A_log=A_log, dt_bias=dt_bias, scale=scale,
            initial_state=ss, out=o, ssm_state_indices=idx, use_qk_l2norm_in_kernel=True,
        )
        ref_o[sel], ref_mixed[sel], ref_z[sel], ref_a[sel], ref_b[sel] = o, m, z, a, b
        ref_win.append(cs[idx.long(), :, :3].clone())
        ref_state.append(ss[idx.long()].clone())

    inter_win = torch.zeros(bs + 1, T, qkv, 3, dtype=torch.bfloat16, device=dev)
    inter_ssm = torch.zeros(bs + 1, T, nv, hv, hk, dtype=torch.float32, device=dev)
    cs2, ss2 = conv_state.clone(), ssm.clone()
    m2, z2, b2, a2 = dsv_gdn_conv(
        qkvz, ba, cs2, conv_w, conv_b, idx, inter_win, inter_idx, T=T, qkv_dim=qkv,
        v_dim=vd, num_v_heads=nv, head_v_dim=hv, activation="silu",
    )
    o2 = dsv_gdn_recurrence(
        m2, a2, b2, A_log=A_log, dt_bias=dt_bias, scale=scale, ssm_states=ss2,
        cache_indices=idx, inter_ssm=inter_ssm, inter_indices=inter_idx, T=T,
        num_v_heads=nv, head_v_dim=hv,
    )
    tag = f"gdn bs={bs} T={T} bias={bias}"
    check(f"{tag} conv out", m2, ref_mixed)
    check(f"{tag} z", z2, ref_z)
    check(f"{tag} a", a2, ref_a)
    check(f"{tag} b", b2, ref_b)
    check(f"{tag} core out", o2.transpose(0, 1), ref_o)
    check(f"{tag} conv windows", inter_win[:bs].permute(1, 0, 2, 3), torch.stack(ref_win))
    check(f"{tag} ssm states", inter_ssm[:bs].permute(1, 0, 2, 3, 4), torch.stack(ref_state))
    check(f"{tag} persistent conv untouched", cs2, conv_state)
    check(f"{tag} persistent ssm untouched", ss2, ssm)


def attn_case(seq_lens, T, max_kv_splits, seed=1):
    g = torch.Generator(device=dev).manual_seed(seed)
    H, KVH, D = 8, 1, 256
    bs = len(seq_lens)
    pool = sum(seq_lens) + bs * T + 64
    kbuf = torch.randn(pool, KVH, D, generator=g, device=dev).to(torch.bfloat16)
    vbuf = torch.randn(pool, KVH, D, generator=g, device=dev).to(torch.bfloat16)
    perm = torch.randperm(pool, generator=g, device=dev)
    req_ids, off = [], 0
    for L in seq_lens:
        req_ids.append(perm[off : off + L + T])
        off += L + T
    q = torch.randn(bs * T, H, D, generator=g, device=dev).to(torch.bfloat16)
    be = types.SimpleNamespace(
        static_kv_splits=False, device_core_count=torch.cuda.get_device_properties(0).multi_processor_count,
        enable_deterministic=False, split_tile_size=None, num_head=H, num_kv_head=KVH,
        max_kv_splits=max_kv_splits,
    )
    sm = D**-0.5

    def run(rows_q, rows_lens, rows_req, nks):
        n = len(rows_lens)
        indptr = torch.zeros(n + 1, dtype=torch.int32, device=dev)
        indptr[1:] = torch.cumsum(torch.tensor(rows_lens, device=dev), 0)
        indices = torch.cat([req_ids[r][:l] for r, l in zip(rows_req, rows_lens)]).to(torch.int64)
        o = torch.empty_like(rows_q)
        logits = torch.empty(n, H, max_kv_splits, D, dtype=torch.float32, device=dev)
        lse = torch.empty(n, H, max_kv_splits, dtype=torch.float32, device=dev)
        decode_attention_fwd(rows_q, kbuf, vbuf, o, indptr, indices, logits, lse, nks,
                             max_kv_splits, sm, 1.0, 1.0)
        return o

    ref = torch.empty_like(q)
    for t in range(T):
        sel = [n * T + t for n in range(bs)]
        lens = torch.tensor([L + t + 1 for L in seq_lens], dtype=torch.int64, device=dev)
        nks = torch.empty(bs, dtype=torch.int32, device=dev)
        get_num_kv_splits_triton[(1,)](nks, lens, bs, 1, H, KVH, max_kv_splits,
                                       be.device_core_count, MAX_NUM_SEQ=256)
        ref[sel] = run(q[sel].contiguous(), lens.tolist(), list(range(bs)), nks)
    nks2 = torch.empty(bs * T, dtype=torch.int32, device=dev)
    dsv_num_kv_splits(be, nks2, torch.tensor(seq_lens, dtype=torch.int64, device=dev), bs, T)
    rl = [L + t + 1 for L in seq_lens for t in range(T)]
    rr = [n for n in range(bs) for t in range(T)]
    out = run(q, rl, rr, nks2)
    check(f"attn lens={seq_lens} T={T} splits={max_kv_splits} (splits {nks2.tolist()})", out, ref)


def drafter_case():
    d = PromptLookupDrafter(3)
    req = types.SimpleNamespace(origin_input_ids_unpadded=[5, 6, 7, 8, 9, 5, 6, 7], output_ids=[])
    r = d.draft(req, [])
    ok1 = r.tolist() == [7, 8, 9, 5]
    req.output_ids = [8]
    r2 = d.draft(req, [9])  # context ...5 6 7 8 9 -> last "8 9" -> continues 5 6 7
    ok2 = r2.tolist() == [9, 5, 6, 7]
    req3 = types.SimpleNamespace(origin_input_ids_unpadded=[1, 2, 3], output_ids=[])
    r3 = d.draft(req3, [])
    ok3 = r3.tolist() == [3, 3, 3, 3]
    req4 = types.SimpleNamespace(origin_input_ids_unpadded=[4, 1, 2, 1, 2], output_ids=[])
    r4 = d.draft(req4, [])  # "1 2" at 1 -> cont [1, 2] period 2 -> 1 2 1
    ok4 = r4.tolist() == [2, 1, 2, 1]
    for name, ok, got in (("simple", ok1, r), ("pending", ok2, r2), ("nomatch", ok3, r3), ("period", ok4, r4)):
        print(f"  {'OK  ' if ok else 'FAIL'} drafter {name}: {got.tolist()}", flush=True)
        if not ok:
            FAIL.append("drafter " + name)


def main():
    print(torch.cuda.get_device_properties(0).gcnArchName if torch.version.hip else "cuda", flush=True)
    drafter_case()
    for bs, T, bias in ((1, 4, False), (2, 4, False), (2, 3, True), (3, 4, False), (8, 4, False)):
        gdn_case(bs, T, bias)
    for lens, T, sp in (([1000], 4, 8), ([1000, 3500], 4, 8), ([37, 20000], 3, 16), ([5000, 5003, 4990, 64], 4, 8)):
        attn_case(lens, T, sp)
    # speed: one GDN layer, bs=2, T=4 vs one decode step at bs=2
    print("FAILED:" if FAIL else "ALL OK", FAIL, flush=True)
    return 1 if FAIL else 0


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    print(f"done in {time.time() - t0:.1f}s")
    sys.exit(rc)
