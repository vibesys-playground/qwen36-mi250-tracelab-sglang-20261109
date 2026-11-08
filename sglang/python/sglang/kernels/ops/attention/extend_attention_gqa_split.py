"""GQA-packed, split-KV extend attention (gfx90a, head_dim 256).

The generic extend kernel (`extend_attention._fwd_kernel`) launches one workgroup per
(sequence, q head, 64-token tile) and walks the whole prefix serially. With 8 q heads per
KV head, each prefix K/V tile is loaded eight times, and at a few hundred new tokens the grid
does not fill a 110-CU GCD. This path:

1. packs the G q heads that share one KV head into the M dimension of the tile
   (BLOCK_M = BM_TOK tokens x G heads), so each K/V tile is loaded once for all G heads;
2. splits the keys of each token tile into chunks of C keys (prefix chunks first, then the
   causal chunks of the new tokens), one workgroup per chunk, each writing a partial
   (O, LSE);
3. merges the partials of each token in chunk order with an fp32 log-sum-exp merge.

Determinism: the chunk size C of a sequence depends only on that sequence's own
(prefix_len, extend_len) and the launch config, never on its batch-mates; the merge runs in a
fixed order. The same request gives the same bits alone or in any batch.

Supported: causal, no custom mask, no sliding window, no sinks, no logit cap, no score_mod,
page size 1, k/v scale 1, Lq == Lv.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

import torch
import triton
import triton.language as tl


@triton.jit
def _gqa_split_stage1(
    Q,
    K_Ext,
    V_Ext,
    K_Buf,
    V_Buf,
    qo_indptr,
    kv_indptr,
    kv_indices,
    seq_meta,  # int32 [bs, 2]: chunk size C, offset of the sequence's partial rows
    O_part,  # [rows, H, D] fp32 or bf16
    LSE_part,  # [rows, H] fp32
    sm_scale,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    HKV: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    BM_TOK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_D: tl.constexpr,
):
    BLOCK_M: tl.constexpr = BM_TOK * G
    H: tl.constexpr = HKV * G
    DH: tl.constexpr = D // SPLIT_D

    seq = tl.program_id(0)
    tile = tl.program_id(1) // HKV
    kvh = tl.program_id(1) % HKV
    c = tl.program_id(2)

    q_start = tl.load(qo_indptr + seq)
    n_ext = tl.load(qo_indptr + seq + 1) - q_start
    t0 = tile * BM_TOK
    if t0 >= n_ext:
        return
    kv_start = tl.load(kv_indptr + seq)
    n_pre = tl.load(kv_indptr + seq + 1) - kv_start
    C = tl.load(seq_meta + seq * 2)
    part_off = tl.load(seq_meta + seq * 2 + 1)

    n_pc = (n_pre + C - 1) // C
    t_end = tl.minimum(n_ext, t0 + BM_TOK)
    n_ec = (t_end + C - 1) // C
    if c >= n_pc + n_ec:
        return

    offs_r = tl.arange(0, BLOCK_M)
    tok = t0 + offs_r // G
    head = kvh * G + offs_r % G
    mask_m = tok < n_ext
    offs_dh = tl.arange(0, DH)
    offs_n = tl.arange(0, BLOCK_N)

    q_row = (q_start + tok).to(tl.int64) * stride_qbs + head * stride_qh
    q0 = tl.load(Q + q_row[:, None] + offs_dh[None, :], mask=mask_m[:, None], other=0.0)
    if SPLIT_D == 2:
        q1 = tl.load(
            Q + q_row[:, None] + DH + offs_dh[None, :], mask=mask_m[:, None], other=0.0
        )

    offs_dv = tl.arange(0, D)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    if c < n_pc:
        lo = c * C
        hi = tl.minimum(lo + C, n_pre)
        for start_n in range(lo, hi, BLOCK_N):
            n = start_n + offs_n
            mask_n = n < hi
            loc = tl.load(kv_indices + kv_start + n, mask=mask_n, other=0).to(tl.int64)
            k_base = K_Buf + loc[None, :] * stride_buf_kbs + kvh * stride_buf_kh
            k0 = tl.load(k_base + offs_dh[:, None], mask=mask_n[None, :], other=0.0)
            qk = tl.dot(q0, k0)
            if SPLIT_D == 2:
                k1 = tl.load(k_base + DH + offs_dh[:, None], mask=mask_n[None, :], other=0.0)
                qk += tl.dot(q1, k1)
            qk = qk * sm_scale
            qk = tl.where(mask_n[None, :], qk, float("-inf"))
            row_max = tl.max(qk, 1)
            n_e_max = tl.maximum(tl.where(row_max == float("-inf"), -1e20, row_max), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)
            v = tl.load(
                V_Buf
                + loc[:, None] * stride_buf_vbs
                + kvh * stride_buf_vh
                + offs_dv[None, :],
                mask=mask_n[:, None],
                other=0.0,
            )
            acc = acc * re_scale[:, None] + tl.dot(p.to(v.dtype), v)
            e_max = n_e_max
    else:
        lo = (c - n_pc) * C
        hi = tl.minimum(lo + C, t_end)
        for start_n in range(lo, hi, BLOCK_N):
            n = start_n + offs_n
            mask_n = n < hi
            k_base = (
                K_Ext + (q_start + n[None, :]).to(tl.int64) * stride_kbs + kvh * stride_kh
            )
            k0 = tl.load(k_base + offs_dh[:, None], mask=mask_n[None, :], other=0.0)
            qk = tl.dot(q0, k0)
            if SPLIT_D == 2:
                k1 = tl.load(k_base + DH + offs_dh[:, None], mask=mask_n[None, :], other=0.0)
                qk += tl.dot(q1, k1)
            qk = qk * sm_scale
            keep = mask_n[None, :] & (n[None, :] <= tok[:, None])
            qk = tl.where(keep, qk, float("-inf"))
            row_max = tl.max(qk, 1)
            n_e_max = tl.maximum(tl.where(row_max == float("-inf"), -1e20, row_max), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)
            v = tl.load(
                V_Ext
                + (q_start + n[:, None]).to(tl.int64) * stride_vbs
                + kvh * stride_vh
                + offs_dv[None, :],
                mask=mask_n[:, None],
                other=0.0,
            )
            acc = acc * re_scale[:, None] + tl.dot(p.to(v.dtype), v)
            e_max = n_e_max

    no_kv = deno == 0.0
    lse = tl.where(no_kv, float("-inf"), tl.log(deno) + e_max)
    deno_safe = tl.where(no_kv, 1.0, deno)
    prow = (part_off + c * n_ext + tok).to(tl.int64) * H + head
    tl.store(LSE_part + prow, lse, mask=mask_m)
    tl.store(
        O_part + prow[:, None] * D + offs_dv[None, :],
        (acc / deno_safe[:, None]).to(O_part.dtype.element_ty),
        mask=mask_m[:, None],
    )


@triton.jit
def _gqa_split_merge(
    O,
    O_part,
    LSE_part,
    qo_indptr,
    kv_indptr,
    seq_meta,
    stride_obs,
    stride_oh,
    H: tl.constexpr,
    D: tl.constexpr,
    BM_TOK: tl.constexpr,
):
    seq = tl.program_id(0)
    t = tl.program_id(1)
    q_start = tl.load(qo_indptr + seq)
    n_ext = tl.load(qo_indptr + seq + 1) - q_start
    if t >= n_ext:
        return
    n_pre = tl.load(kv_indptr + seq + 1) - tl.load(kv_indptr + seq)
    C = tl.load(seq_meta + seq * 2)
    part_off = tl.load(seq_meta + seq * 2 + 1)
    t_end = tl.minimum(n_ext, (t // BM_TOK) * BM_TOK + BM_TOK)
    n_chunks = (n_pre + C - 1) // C + (t_end + C - 1) // C

    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    m = tl.zeros([H], dtype=tl.float32) - 1e30
    den = tl.zeros([H], dtype=tl.float32)
    acc = tl.zeros([H, D], dtype=tl.float32)
    for c in range(0, n_chunks):
        prow = (part_off + c * n_ext + t).to(tl.int64) * H + offs_h
        lse = tl.load(LSE_part + prow)
        o = tl.load(O_part + prow[:, None] * D + offs_d[None, :]).to(tl.float32)
        n_m = tl.maximum(m, lse)
        a = tl.exp(m - n_m)
        w = tl.exp(lse - n_m)
        den = den * a + w
        acc = acc * a[:, None] + w[:, None] * o
        m = n_m
    out = acc / den[:, None]
    tl.store(
        O + (q_start + t).to(tl.int64) * stride_obs + offs_h[:, None] * stride_oh + offs_d[None, :],
        out.to(O.dtype.element_ty),
    )


class GqaSplitConfig(NamedTuple):
    # Defaults measured on MI250 (gfx90a), 8 q heads per KV head, head_dim 256
    # (tools/extend_attn/bench.py): 16-token tiles at 4 warps use AGPRs and are slower.
    bm_tok: int = 8  # tokens per tile; BLOCK_M = bm_tok * G
    block_n: int = 64
    num_warps: int = 4
    waves_per_eu: int = 2
    num_stages: int = 1
    split_d: int = 1
    target_wg: int = 220  # workgroups a sequence aims for (about 2 per CU on 110 CUs)
    min_chunk: int = 256  # smallest chunk of keys per workgroup
    part_dtype: torch.dtype = torch.float32
    mfma: int = 16


DEFAULT_CONFIG = GqaSplitConfig()


def chunk_size(n_pre: int, n_ext: int, cfg: GqaSplitConfig) -> int:
    """Keys per workgroup for one sequence. Depends only on its own lengths and cfg."""
    tiles = triton.cdiv(n_ext, cfg.bm_tok)
    per_tile = max(1, triton.cdiv(cfg.target_wg, tiles))
    keys = n_pre + (n_ext + 1) // 2  # mean keys per tile
    c = max(cfg.min_chunk, triton.cdiv(keys, per_tile))
    return triton.cdiv(c, cfg.block_n) * cfg.block_n


class GqaSplitPlan(NamedTuple):
    seq_meta: torch.Tensor  # int32 [bs, 2] on device
    part_rows: int
    max_tiles: int
    max_chunks: int
    max_ext: int
    cfg: GqaSplitConfig


def plan(
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
    cfg: GqaSplitConfig = DEFAULT_CONFIG,
    device=None,
) -> GqaSplitPlan:
    meta, off, max_chunks = [], 0, 1
    for p, n in zip(prefix_lens, extend_lens):
        p, n = int(p), int(n)
        c = chunk_size(p, n, cfg)
        s = triton.cdiv(p, c) + triton.cdiv(n, c)
        meta += [c, off]
        off += s * n
        max_chunks = max(max_chunks, s)
    max_ext = max([int(n) for n in extend_lens] + [1])
    seq_meta = torch.tensor(meta, dtype=torch.int32).view(-1, 2).to(device, non_blocking=True)
    return GqaSplitPlan(
        seq_meta=seq_meta,
        part_rows=max(off, 1),
        max_tiles=triton.cdiv(max_ext, cfg.bm_tok),
        max_chunks=max_chunks,
        max_ext=max_ext,
        cfg=cfg,
    )


def _hip_kwargs(cfg: GqaSplitConfig) -> dict:
    if torch.version.hip is None:
        return {}
    return {"waves_per_eu": cfg.waves_per_eu, "matrix_instr_nonkdim": cfg.mfma}


def gqa_split_extend_fwd(
    q: torch.Tensor,  # [T, H, D]
    k: torch.Tensor,  # [T, HKV, D]
    v: torch.Tensor,
    o: torch.Tensor,  # [T, H, D]
    k_buffer: torch.Tensor,  # [slots, HKV, D]
    v_buffer: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
    p: GqaSplitPlan,
    kernels: Optional[list] = None,
):
    cfg = p.cfg
    T, H, D = q.shape
    HKV = k.shape[1]
    G = H // HKV
    bs = qo_indptr.shape[0] - 1
    o_part = torch.empty((p.part_rows, H, D), dtype=cfg.part_dtype, device=q.device)
    lse_part = torch.empty((p.part_rows, H), dtype=torch.float32, device=q.device)
    k1 = _gqa_split_stage1[(bs, p.max_tiles * HKV, p.max_chunks)](
        q,
        k,
        v,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        p.seq_meta,
        o_part,
        lse_part,
        sm_scale,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        HKV=HKV,
        G=G,
        D=D,
        BM_TOK=cfg.bm_tok,
        BLOCK_N=cfg.block_n,
        SPLIT_D=cfg.split_d,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
        **_hip_kwargs(cfg),
    )
    k2 = _gqa_split_merge[(bs, p.max_ext)](
        o,
        o_part,
        lse_part,
        qo_indptr,
        kv_indptr,
        p.seq_meta,
        o.stride(0),
        o.stride(1),
        H=H,
        D=D,
        BM_TOK=cfg.bm_tok,
        num_warps=4,
    )
    if kernels is not None:
        kernels[:] = [k1, k2]
    return o


def can_handle(
    *,
    q_head_num: int,
    kv_head_num: int,
    head_dim: int,
    v_head_dim: int,
    is_causal: bool,
    custom_mask,
    sliding_window_size: int,
    sinks,
    logit_cap: float,
    score_mod,
    xai_temperature_len,
    page_size: int,
    k_scale: float,
    v_scale: float,
    kv_dtype: torch.dtype,
) -> bool:
    if kv_head_num <= 0 or q_head_num % kv_head_num:
        return False
    G = q_head_num // kv_head_num
    return (
        head_dim == 256
        and v_head_dim == 256
        and G in (4, 8, 16)
        and is_causal
        and custom_mask is None
        and (sliding_window_size is None or sliding_window_size <= 0)
        and sinks is None
        and not logit_cap
        and score_mod is None
        and (xai_temperature_len is None or xai_temperature_len <= 0)
        and page_size == 1
        and k_scale == 1.0
        and v_scale == 1.0
        and kv_dtype in (torch.bfloat16, torch.float16)
    )
