"""Fused Gated DeltaNet single-token decode bodies (Qwen3.5 contiguous layout).

Two decode-only kernels that replace the tail of today's four-launch body
(unpack, Conv1D update, packed recurrence, gated RMSNorm):

* ``gdn_packed_decode_norm`` (level 2): the packed recurrence plus the gated
  RMSNorm. One program owns one (sequence, v-head) and the whole V x K state
  tile, so it can normalize over V in-program and multiply by silu(z).
* ``gdn_decode_fused_body`` (level 3): unpack + indexed Conv1D update +
  recurrence + gated RMSNorm in one launch. One program owns one
  (sequence, k-head) and that k-head's HV/H v-heads, so it alone reads and
  writes the Conv1D state of its q, k and v channels: no cross-program
  ordering is needed.

Both keep today's math and storage: the Conv1D expression and BF16 output
rounding of ``_fused_qkvzba_causal_conv1d_update_contiguous_kernel``, the FP32
recurrence of ``fused_recurrent_gated_delta_rule_packed_decode_kernel`` with
its output rounded to the activation dtype, and the gated RMSNorm of
``_layer_norm_fwd_1pass_kernel`` (norm before gate). Every output row depends
only on its own inputs and state slot, and the reduction order inside a
program is fixed by the shape, so results are batch invariant and
repeatable. A negative state index writes zero output and leaves every state
untouched.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.op import exp


@triton.jit
def _gdn_gate_beta(a, b, A_log, dt_bias, i_n, i_hv, stride_a_tok, stride_b_tok,
                   SOFTPLUS_THRESHOLD: tl.constexpr):
    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)
    return g_val, beta_val


@triton.jit
def _gdn_head_update_norm(
    h_base,  # state of this (slot, v-head): [V, K], K contiguous
    b_q,  # [BK] fp32, normalized and scaled
    b_k,  # [BK] fp32, normalized
    b_v,  # [BV] fp32
    g_val,
    beta_val,
    z_ptr,  # this (row, v-head)'s z, V contiguous
    norm_w,
    o_ptr,  # this (row, v-head)'s output, V contiguous
    eps,
    o_k,
    o_v,
    mask_k,
    mask_v,
    K: tl.constexpr,
    V: tl.constexpr,
    ACT_SIGMOID: tl.constexpr,
):
    mask_h = mask_v[:, None] & mask_k[None, :]
    p_h = h_base + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32)
    b_h *= exp(g_val)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= beta_val
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)
    tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mask_h)

    # Gated RMSNorm over V (norm before gate), from the activation-dtype
    # output exactly as the separate norm launch reads it.
    x = b_o.to(o_ptr.dtype.element_ty).to(tl.float32)
    x = tl.where(mask_v, x, 0.0)
    var = tl.sum(x * x, 0) / V
    rstd = tl.rsqrt(var + eps)
    w = tl.load(norm_w + o_v, mask=mask_v, other=0.0).to(tl.float32)
    y = x * rstd * w
    z = tl.load(z_ptr + o_v, mask=mask_v, other=0.0).to(tl.float32)
    if ACT_SIGMOID:
        y *= tl.sigmoid(z)
    else:
        y *= z * tl.sigmoid(z)
    tl.store(o_ptr + o_v, y.to(o_ptr.dtype.element_ty), mask=mask_v)


@triton.jit
def _gdn_packed_decode_norm_kernel(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    z,
    norm_w,
    o,
    h,
    ssm_state_indices,
    scale,
    eps,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_z_tok: tl.constexpr,
    stride_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    ACT_SIGMOID: tl.constexpr,
):
    i_nh = tl.program_id(0)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
    p_o = o + (i_n * HV + i_hv) * V
    if state_idx < 0:
        zero = tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty)
        tl.store(p_o + o_v, zero, mask=mask_v)
        return

    p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
    b_q = tl.load(p_mixed + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(p_mixed + H * K + i_h * K + o_k, mask=mask_k, other=0).to(
        tl.float32
    )
    b_v = tl.load(p_mixed + 2 * H * K + i_hv * V + o_v, mask=mask_v, other=0).to(
        tl.float32
    )
    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    g_val, beta_val = _gdn_gate_beta(
        a, b, A_log, dt_bias, i_n, i_hv, stride_a_tok, stride_b_tok,
        SOFTPLUS_THRESHOLD,
    )
    _gdn_head_update_norm(
        h + state_idx * stride_state_token + i_hv * V * K,
        b_q,
        b_k,
        b_v,
        g_val,
        beta_val,
        z + i_n * stride_z_tok + i_hv * V,
        norm_w,
        p_o,
        eps,
        o_k,
        o_v,
        mask_k,
        mask_v,
        K,
        V,
        ACT_SIGMOID,
    )


def _check_norm_activation(activation: str) -> bool:
    if activation in ("swish", "silu"):
        return False
    if activation == "sigmoid":
        return True
    raise ValueError(f"Unsupported gated-norm activation {activation!r}")


def gdn_packed_decode_norm(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    z: torch.Tensor,
    *,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    norm_activation: str,
    scale: float,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    num_warps: int = 8,
) -> torch.Tensor:
    """Packed recurrence + gated RMSNorm. Returns ``[B, HV, V]``.

    ``mixed_qkv`` is ``[B, q|k|v]`` after Conv1D; ``z`` is ``[B, HV, V]`` (or
    any ``[B, HV*V]`` view with a unit last stride).
    """
    B, qkv_dim = mixed_qkv.shape
    HV, V, K = ssm_states.shape[-3:]
    H = (qkv_dim - HV * V) // (2 * K)
    assert 2 * H * K + HV * V == qkv_dim and HV % H == 0
    assert mixed_qkv.stride(-1) == 1 and z.stride(-1) == 1
    assert ssm_states.is_contiguous() or ssm_states[0].is_contiguous()
    z2 = z.reshape(B, HV * V)
    assert z2.stride(-1) == 1
    out = mixed_qkv.new_empty(B, HV, V)
    if B == 0:
        return out
    _gdn_packed_decode_norm_kernel[(B * HV,)](
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        z2,
        norm_weight,
        out,
        ssm_states,
        cache_indices,
        scale,
        norm_eps,
        stride_mixed_qkv_tok=mixed_qkv.stride(0),
        stride_a_tok=a.stride(0),
        stride_b_tok=b.stride(0),
        stride_z_tok=z2.stride(0),
        stride_state_token=ssm_states.stride(0),
        stride_indices_seq=cache_indices.stride(0),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=triton.next_power_of_2(K),
        BV=triton.next_power_of_2(V),
        SOFTPLUS_THRESHOLD=20.0,
        ACT_SIGMOID=_check_norm_activation(norm_activation),
        num_warps=num_warps,
        num_stages=1,
    )
    return out


@triton.jit
def _gdn_conv_channels(
    qkvz_row,
    state_slot_base,
    conv_weight,
    conv_bias,
    ch,
    upd_mask,
    stride_state_dim: tl.constexpr,
    stride_state_pos: tl.constexpr,
    stride_weight_dim: tl.constexpr,
    stride_weight_width: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
):
    """Conv1D decode update of channels ``ch`` of one valid row; same
    expression and operand types as the fused unpack+Conv1D kernel.
    Returns the activation-dtype output widened to fp32."""
    x = tl.load(qkvz_row + ch)
    state_base = state_slot_base + ch * stride_state_dim
    acc = tl.zeros(ch.shape, dtype=tl.float32)
    if HAS_BIAS:
        acc += tl.load(conv_bias + ch).to(tl.float32)
    for pos in tl.static_range(KERNEL_WIDTH - 1):
        state_value = tl.load(state_base + pos * stride_state_pos, mask=upd_mask,
                              other=0.0)
        weight_value = tl.load(
            conv_weight + ch * stride_weight_dim + pos * stride_weight_width
        )
        acc += state_value * weight_value
    last_weight = tl.load(
        conv_weight + ch * stride_weight_dim + (KERNEL_WIDTH - 1) * stride_weight_width
    )
    acc += x * last_weight
    if SILU_ACTIVATION:
        conv_out = acc / (1.0 + tl.exp(-acc))
    else:
        conv_out = acc
    for pos in tl.static_range(KERNEL_WIDTH - 2):
        next_value = tl.load(state_base + (pos + 1) * stride_state_pos,
                             mask=upd_mask, other=0.0)
        tl.store(state_base + pos * stride_state_pos, next_value, mask=upd_mask)
    tl.store(state_base + (KERNEL_WIDTH - 2) * stride_state_pos, x, mask=upd_mask)
    return conv_out.to(x.dtype).to(tl.float32)


@triton.jit
def _gdn_decode_fused_body_kernel(
    mixed_qkvz,
    mixed_ba,
    conv_state,
    conv_weight,
    conv_bias,
    A_log,
    dt_bias,
    norm_w,
    o,
    h,
    state_indices,
    scale,
    eps,
    stride_qkvz_tok: tl.constexpr,
    stride_ba_tok: tl.constexpr,
    stride_conv_slot: tl.constexpr,
    stride_conv_dim: tl.constexpr,
    stride_conv_pos: tl.constexpr,
    stride_weight_dim: tl.constexpr,
    stride_weight_width: tl.constexpr,
    stride_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    ACT_SIGMOID: tl.constexpr,
):
    G: tl.constexpr = HV // H
    QKV_DIM: tl.constexpr = 2 * H * K + HV * V
    i_nh = tl.program_id(0)
    i_n, i_h = i_nh // H, i_nh % H

    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V

    state_idx = tl.load(state_indices + i_n * stride_indices_seq).to(tl.int64)
    p_o = o + (i_n * HV + i_h * G) * V
    if state_idx < 0:
        zero = tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty)
        for j in tl.static_range(G):
            tl.store(p_o + j * V + o_v, zero, mask=mask_v)
        return

    qkvz_row = mixed_qkvz + i_n * stride_qkvz_tok
    conv_base = conv_state + state_idx * stride_conv_slot
    b_q = _gdn_conv_channels(
        qkvz_row, conv_base, conv_weight, conv_bias, i_h * K + o_k, mask_k,
        stride_conv_dim, stride_conv_pos, stride_weight_dim, stride_weight_width,
        KERNEL_WIDTH, HAS_BIAS, SILU_ACTIVATION,
    )
    b_k = _gdn_conv_channels(
        qkvz_row, conv_base, conv_weight, conv_bias, H * K + i_h * K + o_k, mask_k,
        stride_conv_dim, stride_conv_pos, stride_weight_dim, stride_weight_width,
        KERNEL_WIDTH, HAS_BIAS, SILU_ACTIVATION,
    )
    b_q = tl.where(mask_k, b_q, 0.0)
    b_k = tl.where(mask_k, b_k, 0.0)
    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    for j in tl.static_range(G):
        i_hv = i_h * G + j
        b_v = _gdn_conv_channels(
            qkvz_row, conv_base, conv_weight, conv_bias,
            2 * H * K + i_hv * V + o_v, mask_v,
            stride_conv_dim, stride_conv_pos, stride_weight_dim,
            stride_weight_width, KERNEL_WIDTH, HAS_BIAS, SILU_ACTIVATION,
        )
        g_val, beta_val = _gdn_gate_beta(
            mixed_ba + HV, mixed_ba, A_log, dt_bias, i_n, i_hv, stride_ba_tok,
            stride_ba_tok, SOFTPLUS_THRESHOLD,
        )
        _gdn_head_update_norm(
            h + state_idx * stride_state_token + i_hv * V * K,
            b_q,
            b_k,
            b_v,
            g_val,
            beta_val,
            qkvz_row + QKV_DIM + i_hv * V,
            norm_w,
            p_o + j * V,
            eps,
            o_k,
            o_v,
            mask_k,
            mask_v,
            K,
            V,
            ACT_SIGMOID,
        )


def gdn_decode_fused_body(
    mixed_qkvz: torch.Tensor,
    mixed_ba: torch.Tensor,
    *,
    conv_states: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias,
    conv_activation,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    norm_activation: str,
    scale: float,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    num_warps: int = 8,
) -> torch.Tensor:
    """Whole decode body from the raw projections. Returns ``[B, HV, V]``.

    ``mixed_qkvz`` is ``[B, q|k|v|z]`` and ``mixed_ba`` is ``[B, b|a]``, both
    with a unit last stride. The Conv1D state is updated in place.
    """
    B = mixed_qkvz.shape[0]
    HV, V, K = ssm_states.shape[-3:]
    qkv_dim = conv_weight.shape[0]
    H = (qkv_dim - HV * V) // (2 * K)
    assert 2 * H * K + HV * V == qkv_dim and HV % H == 0
    assert mixed_qkvz.shape[1] == qkv_dim + HV * V
    assert mixed_ba.shape[1] == 2 * HV
    assert mixed_qkvz.stride(-1) == 1 and mixed_ba.stride(-1) == 1
    assert conv_states.shape[1] == qkv_dim
    assert conv_states.shape[2] >= conv_weight.shape[1] - 1
    assert 2 <= conv_weight.shape[1] <= 4
    assert conv_activation in (None, "silu", "swish")
    out = mixed_qkvz.new_empty(B, HV, V)
    if B == 0:
        return out
    _gdn_decode_fused_body_kernel[(B * H,)](
        mixed_qkvz,
        mixed_ba,
        conv_states,
        conv_weight,
        conv_bias,
        A_log,
        dt_bias,
        norm_weight,
        out,
        ssm_states,
        cache_indices,
        scale,
        norm_eps,
        stride_qkvz_tok=mixed_qkvz.stride(0),
        stride_ba_tok=mixed_ba.stride(0),
        stride_conv_slot=conv_states.stride(0),
        stride_conv_dim=conv_states.stride(1),
        stride_conv_pos=conv_states.stride(2),
        stride_weight_dim=conv_weight.stride(0),
        stride_weight_width=conv_weight.stride(1),
        stride_state_token=ssm_states.stride(0),
        stride_indices_seq=cache_indices.stride(0),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=triton.next_power_of_2(K),
        BV=triton.next_power_of_2(V),
        KERNEL_WIDTH=conv_weight.shape[1],
        HAS_BIAS=conv_bias is not None,
        SILU_ACTIVATION=conv_activation in ("silu", "swish"),
        SOFTPLUS_THRESHOLD=20.0,
        ACT_SIGMOID=_check_norm_activation(norm_activation),
        num_warps=num_warps,
        num_stages=1,
    )
    return out
