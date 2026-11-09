"""Prompt-lookup drafts and a decode-shaped target verify for chain n-gram speculation.

Two independent switches, both off unless set (``serving/serve.sh`` sets them):

``SGLANG_NGRAM_PROMPT_LOOKUP=1``
    The NGRAM worker drafts with :class:`PromptLookupDrafter` instead of the global n-gram
    trie. A request's drafts are a function of that request's own prompt and output token
    ids only: the most recent earlier occurrence of its last n tokens (n = 4 down to 1) in
    its own context, and the k tokens that followed it. Nothing is shared between requests
    and nothing outside the request is read.

``SGLANG_NGRAM_DECODE_VERIFY=1``
    A chain verify batch (bs requests x T = k + 1 tokens) runs through the model as a
    decode batch of M = bs * T rows: row (n, t) is request n's token at position
    seq_len[n] + t and attends to the first seq_len[n] + t + 1 KV entries of its request,
    exactly as a decode step at that position would. Every layer therefore takes its decode
    path for M rows (skinny GEMVs, the decode MoE dispatch, decode attention, the decode
    norm/all-reduce boundaries), except the Gated DeltaNet layers, which run a T-step
    recurrence per request (:func:`dsv_gdn_conv`, :func:`dsv_gdn_recurrence`). Those keep
    the persistent conv/SSM state untouched and write the state after every position into
    the speculative scratch (``intermediate_conv_window`` / ``intermediate_ssm``), so the
    stock ``commit_mamba_states_after_verify`` commits the state at the accepted length,
    and rollback is exact. The GDN kernels are the decode kernels with a loop over the T
    positions: per position they do the same operations in the same order, so a row's
    result is bitwise the one a decode step would give.

The batch keeps ``ForwardMode.TARGET_VERIFY`` everywhere outside the layer stack
(scheduler, graph runner, logits, sampling, KV/state commit); only the model's layer stack
sees ``DECODE`` (:func:`decode_shaped_verify`).
"""

from __future__ import annotations

import contextlib
import logging
from typing import Optional

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.op import exp
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.utils import get_bool_env_var, get_int_env_var

logger = logging.getLogger(__name__)


def prompt_lookup_enabled() -> bool:
    return get_bool_env_var("SGLANG_NGRAM_PROMPT_LOOKUP", "false")


def decode_verify_enabled() -> bool:
    return get_bool_env_var("SGLANG_NGRAM_DECODE_VERIFY", "false")


def dsv_group(forward_batch) -> int:
    """T (tokens per request) while the layer stack runs a decode-shaped verify, else 0."""
    return getattr(forward_batch, "_dsv_group", 0) if forward_batch is not None else 0


def is_dsv_batch(forward_batch) -> bool:
    """A chain TARGET_VERIFY batch that runs decode-shaped (see the module docstring)."""
    if forward_batch is None or not forward_batch.forward_mode.is_target_verify():
        return False
    spec_info = forward_batch.spec_info
    return (
        _DSV_ON
        and spec_info is not None
        and getattr(spec_info, "draft_token_num", 0) > 1
        and getattr(forward_batch, "ragged_verify_layout", None) is None
    )


_DSV_ON = decode_verify_enabled()


@contextlib.contextmanager
def decode_shaped_verify(forward_batch):
    """Present a chain verify batch to the layer stack as a decode batch of bs*T rows."""
    if not is_dsv_batch(forward_batch):
        yield
        return
    orig_mode = forward_batch.forward_mode
    forward_batch.forward_mode = ForwardMode.DECODE
    forward_batch._dsv_group = forward_batch.spec_info.draft_token_num
    try:
        yield
    finally:
        forward_batch.forward_mode = orig_mode
        forward_batch._dsv_group = 0


# ----------------------------------------------------------------------------------------
# Drafts: per-request prompt lookup
# ----------------------------------------------------------------------------------------


def _as_np(ids) -> np.ndarray:
    if hasattr(ids, "typecode"):  # array.array
        return np.frombuffer(ids, dtype=np.dtype(ids.typecode)) if len(ids) else np.zeros(0, np.int64)
    return np.asarray(ids, dtype=np.int64)


class _ReqContext:
    """The tail of one request's own token ids (prompt, then outputs), kept on the request."""

    __slots__ = ("buf", "n", "nout")

    def __init__(self, prompt, window: int):
        tail = _as_np(prompt)[-window:]
        self.buf = np.empty(2 * window + 64, dtype=np.int64)
        self.buf[: tail.shape[0]] = tail
        self.n = int(tail.shape[0])
        self.nout = 0

    def append(self, ids: np.ndarray, window: int, reserve: int) -> None:
        m = ids.shape[0]
        if self.n + m + reserve > self.buf.shape[0]:
            keep = min(self.n, window)
            self.buf[:keep] = self.buf[self.n - keep : self.n]
            self.n = keep
            if self.n + m + reserve > self.buf.shape[0]:
                grown = np.empty(self.n + m + reserve + window, dtype=np.int64)
                grown[: self.n] = self.buf[: self.n]
                self.buf = grown
        self.buf[self.n : self.n + m] = ids
        self.n += m


class PromptLookupDrafter:
    """Chain drafts of k tokens from a request's own context (prompt lookup decoding).

    For the context c[0..L) (prompt ids, then output ids, then the accepted ids not yet
    added to the request), find the most recent i < L - n with c[i:i+n] == c[L-n:L],
    trying n = max_ngram down to min_ngram, and draft c[i+n : i+n+k]. When the match is
    so recent that fewer than k tokens follow it, the continuation repeats with its
    period L - (i + n), which is what the text did since the match. Without a match the
    drafts repeat the last token (they are verified like any other drafts). Only the last
    ``window`` ids are searched.

    The draft for a request depends only on that request's own token ids, so it is the
    same on every TP rank and on every run.
    """

    def __init__(
        self,
        k: int,
        *,
        max_ngram: int = 4,
        min_ngram: int = 1,
        window: int = 8192,
        corrupt_p: float = 0.0,
        vocab_size: int = 0,
    ):
        self.k = k
        self.max_ngram = max_ngram
        self.min_ngram = min_ngram
        self.window = window
        # DEBUG ONLY (SGLANG_SPEC_DEBUG_CORRUPT_DRAFT, never set when serving): replace
        # this fraction of draft tokens with random ids, to test that outputs do not
        # depend on the drafts. Fixed seed, so every TP rank draws the same ids.
        self.corrupt_p = corrupt_p
        self.vocab_size = vocab_size
        self._rng = np.random.default_rng(1234)
        self.mask_one = np.tril(np.ones((k + 1, k + 1), dtype=bool))

    def _context(self, req, pending) -> tuple[np.ndarray, int]:
        st = getattr(req, "_p27_lookup_ctx", None)
        out = req.output_ids
        if st is None or len(out) < st.nout:
            st = _ReqContext(req.origin_input_ids_unpadded, self.window)
            req._p27_lookup_ctx = st
        if len(out) > st.nout:
            st.append(_as_np(out)[st.nout :], self.window, len(pending) + 1)
            st.nout = len(out)
        p = len(pending)
        if st.n + p > st.buf.shape[0]:
            st.append(np.zeros(0, np.int64), self.window, p + 1)
        if p:
            # Accepted last step, not yet in output_ids: placed after the committed
            # tail without being committed (the next call overwrites them).
            st.buf[st.n : st.n + p] = pending
        return st.buf, st.n + p

    def draft(self, req, pending) -> np.ndarray:
        """[last token, d_1 .. d_k] for one request."""
        buf, L = self._context(req, pending)
        k = self.k
        res = np.empty(k + 1, dtype=np.int64)
        last = buf[L - 1] if L > 0 else 0
        res[0] = last
        start = max(0, L - self.window)
        found = -1
        for n in range(min(self.max_ngram, L - 1), self.min_ngram - 1, -1):
            end = L - n  # candidate starts i in [start, end): i + n < L, and i != L - n
            if end <= start:
                continue
            pat = buf[L - n : L]
            m = buf[start:end] == pat[0]
            for j in range(1, n):
                m &= buf[start + j : end + j] == pat[j]
            hit = np.flatnonzero(m)
            if hit.shape[0]:
                found = start + int(hit[-1]) + n
                break
        if found < 0:
            res[1:] = last
        else:
            cont = buf[found:L]
            period = cont.shape[0]
            if period >= k:
                res[1:] = cont[:k]
            else:
                for j in range(k):
                    res[1 + j] = cont[j % period]
        if self.corrupt_p > 0:
            hit = self._rng.random(k) < self.corrupt_p
            if hit.any():
                res[1:][hit] = self._rng.integers(0, self.vocab_size, int(hit.sum()))
        return res

    def batch(self, reqs, pending_lists) -> tuple[np.ndarray, np.ndarray]:
        """Flat drafts [bs * (k + 1)] and the chain tree masks [bs * (k + 1) ** 2]."""
        bs = len(reqs)
        drafts = np.empty(bs * (self.k + 1), dtype=np.int64)
        for i, (req, pend) in enumerate(zip(reqs, pending_lists)):
            drafts[i * (self.k + 1) : (i + 1) * (self.k + 1)] = self.draft(req, pend)
        mask = np.tile(self.mask_one.reshape(-1), bs)
        return drafts, mask


def make_drafter(k: int, vocab_size: int) -> PromptLookupDrafter:
    corrupt = float(get_int_env_var("SGLANG_SPEC_DEBUG_CORRUPT_DRAFT_PCT", 0)) / 100.0
    if corrupt > 0:
        logger.warning(
            "SGLANG_SPEC_DEBUG_CORRUPT_DRAFT_PCT: corrupting %.0f%% of drafts (debug only)",
            100 * corrupt,
        )
    return PromptLookupDrafter(
        k,
        max_ngram=get_int_env_var("SGLANG_NGRAM_LOOKUP_MAX_N", 4),
        min_ngram=get_int_env_var("SGLANG_NGRAM_LOOKUP_MIN_N", 1),
        window=get_int_env_var("SGLANG_NGRAM_LOOKUP_WINDOW", 8192),
        corrupt_p=corrupt,
        vocab_size=vocab_size,
    )


# ----------------------------------------------------------------------------------------
# Attention: per-row decode split counts
# ----------------------------------------------------------------------------------------


@triton.jit
def _dsv_num_kv_splits_kernel(
    num_kv_splits_ptr,
    seq_lens_ptr,
    num_seq,
    T,
    num_head,
    num_kv_head,
    max_kv_splits,
    device_core_count,
    MAX_NUM_SEQ: tl.constexpr,
):
    """get_num_kv_splits_triton (num_group=1) for the decode batch of column t.

    Column t holds every request's row at length seq_len + t + 1. Each column gets the
    split counts a decode step of those rows would get, written to rows n * T + t.
    """
    t = tl.program_id(0)
    offs_seq = tl.arange(0, MAX_NUM_SEQ)
    mask_seq = offs_seq < num_seq

    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=0) + (t + 1)
    seq_lens = tl.where(mask_seq, seq_lens, 0)
    max_seq_len = tl.max(seq_lens)
    seq_lens = tl.where(mask_seq, seq_lens, max_seq_len)
    min_seq_len = tl.min(seq_lens)
    if max_seq_len * 8 < min_seq_len * 10:
        min_seq_len = max_seq_len
    max_kv_splits_1 = tl.minimum(tl.cdiv(max_seq_len, min_seq_len), max_kv_splits)
    kv_chunk_size_1 = tl.cdiv(max_seq_len, max_kv_splits_1)

    ext_seq_len = tl.cast(max_seq_len, tl.float32) / 64.0
    ext_device_core_count = tl.cast(
        device_core_count * tl.maximum(tl.log2(ext_seq_len), 1.0), tl.int32
    )
    block_h, num_kv_group = 16, num_head // num_kv_head
    if num_kv_group == 1:
        token_grid = num_seq * num_head
    else:
        block_h = tl.minimum(block_h, num_kv_group)
        token_grid = num_seq * tl.cdiv(num_head, block_h)
    max_kv_splits_2 = tl.minimum(
        tl.cdiv(ext_device_core_count, token_grid), max_kv_splits
    )
    kv_chunk_size_2 = tl.cdiv(max_seq_len, max_kv_splits_2)

    num_kv_splits = tl.maximum(
        tl.cdiv(seq_lens, kv_chunk_size_1), tl.cdiv(seq_lens, kv_chunk_size_2)
    )
    tl.store(num_kv_splits_ptr + offs_seq * T + t, num_kv_splits, mask=mask_seq)


def dsv_rows(seq_lens: torch.Tensor, req_pool_indices: torch.Tensor, bs: int, T: int):
    """Per-row KV lengths (seq_len + t + 1) and request rows of the M = bs*T decode rows."""
    steps = torch.arange(1, T + 1, dtype=seq_lens.dtype, device=seq_lens.device)
    row_lens = (seq_lens[:bs, None] + steps[None, :]).reshape(-1)
    row_reqs = req_pool_indices[:bs].repeat_interleave(T)
    return row_lens, row_reqs


def dsv_num_kv_splits(
    backend, num_kv_splits: torch.Tensor, seq_lens: torch.Tensor, bs: int, T: int
) -> None:
    """Fill num_kv_splits[: bs*T] as TritonAttnBackend.get_num_kv_splits would for each
    column's decode batch (see _dsv_num_kv_splits_kernel)."""
    if (
        backend.static_kv_splits or backend.device_core_count <= 0
    ) and not backend.enable_deterministic:
        num_kv_splits[: bs * T].fill_(backend.max_kv_splits)
        return
    assert not (backend.split_tile_size is not None and backend.enable_deterministic), (
        "decode-shaped verify: deterministic split tiles unsupported"
    )
    _dsv_num_kv_splits_kernel[(T,)](
        num_kv_splits,
        seq_lens,
        bs,
        T,
        backend.num_head,
        backend.num_kv_head,
        backend.max_kv_splits,
        backend.device_core_count,
        MAX_NUM_SEQ=max(256, triton.next_power_of_2(bs)),
    )


# ----------------------------------------------------------------------------------------
# Gated DeltaNet: T-step conv and recurrence with per-position states
# ----------------------------------------------------------------------------------------


@triton.jit
def _dsv_gdn_conv_kernel(
    mixed_qkv,
    z,
    b,
    a,
    mixed_qkvz,
    mixed_ba,
    conv_state,
    conv_weight,
    conv_bias,
    conv_state_indices,
    inter_win,
    inter_indices,
    stride_qkvz_batch,
    stride_qkvz_dim,
    stride_ba_batch,
    stride_ba_dim,
    stride_state_batch,
    stride_state_dim,
    stride_state_pos,
    stride_weight_dim,
    stride_weight_width,
    stride_win_slot,
    stride_win_step,
    stride_win_dim,
    stride_win_pos,
    T: tl.constexpr,
    QKV_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    NUM_STATE_SLOTS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """_fused_qkvzba_causal_conv1d_update_contiguous_kernel (width 4) over the T rows of
    one request. Per row: the same loads, products and sums in the same order as the
    decode kernel, with the window carried in registers; the window after row t goes to
    the scratch step t. The persistent conv state is not written."""
    req = tl.program_id(0)
    dim_idx = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    qkv_mask = dim_idx < QKV_DIM

    state_slot = tl.load(conv_state_indices + req).to(tl.int64)
    valid_slot = (
        (state_slot != PAD_SLOT_ID) & (state_slot >= 0) & (state_slot < NUM_STATE_SLOTS)
    )
    state_base = (
        conv_state + state_slot * stride_state_batch + dim_idx * stride_state_dim
    )
    w0 = tl.load(state_base, mask=qkv_mask & valid_slot, other=0.0)
    w1 = tl.load(state_base + stride_state_pos, mask=qkv_mask & valid_slot, other=0.0)
    w2 = tl.load(
        state_base + 2 * stride_state_pos, mask=qkv_mask & valid_slot, other=0.0
    )
    k0 = tl.load(conv_weight + dim_idx * stride_weight_dim, mask=qkv_mask, other=0.0)
    k1 = tl.load(
        conv_weight + dim_idx * stride_weight_dim + stride_weight_width,
        mask=qkv_mask,
        other=0.0,
    )
    k2 = tl.load(
        conv_weight + dim_idx * stride_weight_dim + 2 * stride_weight_width,
        mask=qkv_mask,
        other=0.0,
    )
    k3 = tl.load(
        conv_weight + dim_idx * stride_weight_dim + 3 * stride_weight_width,
        mask=qkv_mask,
        other=0.0,
    )
    if HAS_BIAS:
        bias = tl.load(conv_bias + dim_idx, mask=qkv_mask, other=0.0).to(tl.float32)
    win_row = tl.load(inter_indices + req).to(tl.int64)
    win_base = inter_win + win_row * stride_win_slot + dim_idx * stride_win_dim

    z_mask = dim_idx < V_DIM
    gate_mask = dim_idx < NUM_V_HEADS
    for t in tl.static_range(T):
        row = req * T + t
        x = tl.load(
            mixed_qkvz + row * stride_qkvz_batch + dim_idx * stride_qkvz_dim,
            mask=qkv_mask,
            other=0.0,
        )
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        if HAS_BIAS:
            acc += bias
        acc += w0 * k0
        acc += w1 * k1
        acc += w2 * k2
        acc += x * k3
        if SILU_ACTIVATION:
            conv_out = acc / (1.0 + tl.exp(-acc))
        else:
            conv_out = acc
        conv_out = tl.where(valid_slot, conv_out, x)
        tl.store(mixed_qkv + row * QKV_DIM + dim_idx, conv_out, mask=qkv_mask)

        w0 = w1
        w1 = w2
        w2 = x
        step_base = win_base + t * stride_win_step
        tl.store(step_base, w0, mask=qkv_mask)
        tl.store(step_base + stride_win_pos, w1, mask=qkv_mask)
        tl.store(step_base + 2 * stride_win_pos, w2, mask=qkv_mask)

        z_value = tl.load(
            mixed_qkvz + row * stride_qkvz_batch + (QKV_DIM + dim_idx) * stride_qkvz_dim,
            mask=z_mask,
            other=0.0,
        )
        tl.store(z + row * V_DIM + dim_idx, z_value, mask=z_mask)
        b_value = tl.load(
            mixed_ba + row * stride_ba_batch + dim_idx * stride_ba_dim,
            mask=gate_mask,
            other=0.0,
        )
        a_value = tl.load(
            mixed_ba + row * stride_ba_batch + (NUM_V_HEADS + dim_idx) * stride_ba_dim,
            mask=gate_mask,
            other=0.0,
        )
        tl.store(b + row * NUM_V_HEADS + dim_idx, b_value, mask=gate_mask)
        tl.store(a + row * NUM_V_HEADS + dim_idx, a_value, mask=gate_mask)


def dsv_gdn_conv(
    mixed_qkvz: torch.Tensor,
    mixed_ba: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: Optional[torch.Tensor],
    conv_state_indices: torch.Tensor,
    inter_win: torch.Tensor,
    inter_indices: torch.Tensor,
    *,
    T: int,
    qkv_dim: int,
    v_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    activation: Optional[str],
    pad_slot_id: int = -1,
):
    """(mixed_qkv, z, b, a) for M = bs*T rows; conv windows after each row go to
    inter_win[inter_indices[n], t] (a [slots, T, dim, 3] view, any strides)."""
    rows = mixed_qkvz.shape[0]
    assert rows % T == 0
    bs = rows // T
    assert conv_weight.shape[1] == 4, "decode-shaped verify: Conv1D width 4 only"
    assert inter_win.ndim == 4 and inter_win.shape[1] >= T and inter_win.shape[-1] == 3
    assert conv_state.shape[1] == qkv_dim and mixed_ba.shape == (rows, 2 * num_v_heads)
    mixed_qkv = torch.empty((rows, qkv_dim), dtype=mixed_qkvz.dtype, device=mixed_qkvz.device)
    z = torch.empty((rows, num_v_heads, head_v_dim), dtype=mixed_qkvz.dtype, device=mixed_qkvz.device)
    b = torch.empty((rows, num_v_heads), dtype=mixed_ba.dtype, device=mixed_ba.device)
    a = torch.empty_like(b)
    if rows == 0:
        return mixed_qkv, z, b, a
    block_size = 256
    grid = (bs, triton.cdiv(qkv_dim, block_size))
    _dsv_gdn_conv_kernel[grid](
        mixed_qkv,
        z,
        b,
        a,
        mixed_qkvz,
        mixed_ba,
        conv_state,
        conv_weight,
        conv_bias,
        conv_state_indices,
        inter_win,
        inter_indices,
        mixed_qkvz.stride(0),
        mixed_qkvz.stride(1),
        mixed_ba.stride(0),
        mixed_ba.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        conv_weight.stride(0),
        conv_weight.stride(1),
        inter_win.stride(0),
        inter_win.stride(1),
        inter_win.stride(2),
        inter_win.stride(3),
        T=T,
        QKV_DIM=qkv_dim,
        V_DIM=v_dim,
        NUM_V_HEADS=num_v_heads,
        NUM_STATE_SLOTS=conv_state.shape[0],
        HAS_BIAS=conv_bias is not None,
        SILU_ACTIVATION=activation in ("silu", "swish"),
        PAD_SLOT_ID=pad_slot_id,
        BLOCK_SIZE=block_size,
        num_warps=8,
        num_stages=2,
    )
    return mixed_qkv, z, b, a


@triton.jit
def _dsv_gdn_recurrence_kernel(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    ssm_state_indices,
    inter_indices,
    scale,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_inter_slot,
    stride_inter_step,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
):
    """fused_recurrent_gated_delta_rule_packed_decode_kernel over the T rows of one
    request. Per row the operations and their order are the decode kernel's; the state
    is carried through the state dtype between rows (as a decode step stores and reloads
    it), written to the scratch step t after row t, and never to the persistent pool."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    state_idx = tl.load(ssm_state_indices + i_n).to(tl.int64)
    if state_idx < 0:
        zero = tl.zeros([BV], dtype=tl.float32).to(o.dtype.element_ty)
        for t in tl.static_range(T):
            tl.store(o + ((i_n * T + t) * HV + i_hv) * V + o_v, zero, mask=mask_v)
        return

    p_h0 = h0 + state_idx * stride_init_state_token
    p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    inter_row = tl.load(inter_indices + i_n).to(tl.int64)
    p_inter = (
        ht
        + inter_row * stride_inter_slot
        + i_hv * V * K
        + o_v[:, None] * K
        + o_k[None, :]
    )
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)

    for t in tl.static_range(T):
        row = i_n * T + t
        p_o = o + (row * HV + i_hv) * V + o_v
        p_mixed = mixed_qkv + row * stride_mixed_qkv_tok
        q_off = i_h * K + o_k
        k_off = (H * K) + i_h * K + o_k
        v_off = (2 * H * K) + i_hv * V + o_v
        b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_mixed + v_off, mask=mask_v, other=0).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale

        a_val = tl.load(a + row * stride_a_tok + i_hv).to(tl.float32)
        b_val = tl.load(b + row * stride_b_tok + i_hv).to(tl.float32)
        x = a_val + dt_bias_val
        softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
        g_val = -tl.exp(A_log_val) * softplus_x
        beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)

        b_h *= exp(g_val)
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= beta_val
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        b_h = b_h.to(h0.dtype.element_ty).to(tl.float32)
        tl.store(
            p_inter + t * stride_inter_step,
            b_h.to(ht.dtype.element_ty),
            mask=mask_h,
        )


def dsv_gdn_recurrence(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    inter_ssm: torch.Tensor,
    inter_indices: torch.Tensor,
    T: int,
    num_v_heads: int,
    head_v_dim: int,
) -> torch.Tensor:
    """Core attention output [1, M, HV, V] for M = bs*T rows (the packed decode layout);
    the state after row t of request n goes to inter_ssm[inter_indices[n], t]."""
    rows = mixed_qkv.shape[0]
    bs = rows // T
    out = mixed_qkv.new_empty(rows, 1, num_v_heads, head_v_dim)
    if rows == 0:
        return out.transpose(0, 1)
    HV, V, K = ssm_states.shape[-3:]
    assert HV == num_v_heads and V == head_v_dim
    assert ssm_states.stride(-1) == 1 and ssm_states.stride(-2) == K
    assert ssm_states.stride(-3) == V * K
    assert inter_ssm.ndim == 5 and inter_ssm.shape[1] >= T
    assert tuple(inter_ssm.shape[-3:]) == (HV, V, K)
    assert inter_ssm.stride(-1) == 1 and inter_ssm.stride(-2) == K
    assert inter_ssm.stride(-3) == V * K
    assert a.is_contiguous() and b.is_contiguous() and mixed_qkv.stride(-1) == 1
    qkv_dim = mixed_qkv.shape[1]
    qk_dim = qkv_dim - HV * V
    H = qk_dim // 2 // K
    BK = triton.next_power_of_2(K)
    assert BK == K
    # The decode wrapper's tile choice: BV=8 is bitwise equal to BV=32 (same one-warp K
    # reductions); a request's T rows run in one program, so the grid is bs * HV wide.
    BV = min(triton.next_power_of_2(V), 8 if bs <= 8 else 32)
    grid = (triton.cdiv(V, BV), bs * HV)
    _dsv_gdn_recurrence_kernel[grid](
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        out,
        ssm_states,
        inter_ssm,
        cache_indices,
        inter_indices,
        scale,
        stride_mixed_qkv_tok=mixed_qkv.stride(0),
        stride_a_tok=a.stride(0),
        stride_b_tok=b.stride(0),
        stride_init_state_token=ssm_states.stride(0),
        stride_inter_slot=inter_ssm.stride(0),
        stride_inter_step=inter_ssm.stride(1),
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=True,
        num_warps=1,
        num_stages=3,
    )
    return out.transpose(0, 1)


# ----------------------------------------------------------------------------------------
# DEBUG ONLY (SGLANG_DEBUG_STEP_TIMING=<path>, never set when serving): synchronized forward
# time of decode/verify steps, measured while the file <path> exists.
# ----------------------------------------------------------------------------------------

import os as _os

_STEP_TIMING_FLAG = _os.environ.get("SGLANG_DEBUG_STEP_TIMING", "")
_step_acc: dict = {}


@contextlib.contextmanager
def debug_step_timer(forward_batch):
    """Logs the mean synchronized model forward time per (mode, requests) every 50
    forwards. The syncs remove CPU/GPU overlap; for comparing step costs only."""
    if (
        not _STEP_TIMING_FLAG
        or not (
            forward_batch.forward_mode.is_decode()
            or forward_batch.forward_mode.is_target_verify()
        )
        or not _os.path.exists(_STEP_TIMING_FLAG)
    ):
        yield
        return
    import time

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    key = (forward_batch.forward_mode.name, int(forward_batch.batch_size))
    n, tot = _step_acc.get(key, (0, 0.0))
    n, tot = n + 1, tot + dt
    if n >= 50:
        logger.warning(
            "STEP_TIMING mode=%s bs=%d n=%d mean_ms=%.3f", key[0], key[1], n, 1000 * tot / n
        )
        n, tot = 0, 0.0
    _step_acc[key] = (n, tot)
