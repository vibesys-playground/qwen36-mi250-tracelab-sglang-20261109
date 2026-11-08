"""Skinny BF16 GEMM ``y[m, n] = x[m, k] @ w[n, k].T (+ bias)`` for a few rows on gfx90a.

At M <= 16 hipBLASLt on gfx90a (MI210/MI250) tiles only N and streams the whole K loop in
one workgroup, so a decode-step linear costs ~20 us whatever its weight size. This kernel
gives every workgroup a [BLOCK_N, K / SPLIT_K] slab of the weight, read once for all
(up to 16) rows of x through MFMA ``tl.dot``, so the grid covers the GCD's CUs.

Numerics: BF16 inputs, FP32 accumulation, BF16 output. With SPLIT_K > 1 each split writes
FP32 partials and a second kernel sums them in a fixed order (split 0, 1, ...); there are
no atomics. The schedule (tile sizes, split count) is a function of (N, K) only and x is
always padded to one 16-row tile, so a row's result does not depend on how many other
rows share the call.
"""

from __future__ import annotations

import functools
from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl

# Rows of x per call: one MFMA tile. The same tile serves every M in 1..16.
MAX_M = 16


class SkinnyConfig(NamedTuple):
    block_n: int
    block_k: int
    split_k: int
    num_warps: int = 4
    num_stages: int = 2


@triton.jit
def _skinny_gemm_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    ws_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_ym,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_split = K // SPLIT_K
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * k_split + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, k_split, BLOCK_K):
        xv = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
        wv = tl.load(w_ptrs, mask=mask_n[:, None], other=0.0)
        acc += tl.dot(xv, tl.trans(wv))
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    mask = mask_m[:, None] & mask_n[None, :]
    if SPLIT_K == 1:
        if HAS_BIAS:
            acc += tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)[None, :]
        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :]
        tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask)
    else:
        ws_ptrs = ws_ptr + pid_k * M * N + offs_m[:, None] * N + offs_n[None, :]
        tl.store(ws_ptrs, acc, mask=mask)


@triton.jit
def _ordered_reduce_kernel(
    ws_ptr,
    b_ptr,
    y_ptr,
    M,
    N,
    stride_ym,
    HAS_BIAS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M * N
    acc = tl.load(ws_ptr + offs, mask=mask, other=0.0)
    for s in tl.static_range(1, SPLIT_K):
        acc += tl.load(ws_ptr + s * M * N + offs, mask=mask, other=0.0)
    m = offs // N
    n = offs % N
    if HAS_BIAS:
        acc += tl.load(b_ptr + n, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + m * stride_ym + n, acc.to(y_ptr.dtype.element_ty), mask=mask)


def skinny_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor],
    cfg: SkinnyConfig,
) -> torch.Tensor:
    """y = x @ w.T + bias for 2-D ``x`` [M <= 16, K] and contiguous ``w`` [N, K]."""
    M, K = x.shape
    N = w.shape[0]
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    grid_n = triton.cdiv(N, cfg.block_n)
    has_bias = bias is not None
    if cfg.split_k == 1:
        ws = y
    else:
        ws = torch.empty((cfg.split_k, M, N), dtype=torch.float32, device=x.device)
    _skinny_gemm_kernel[(grid_n, cfg.split_k)](
        x,
        w,
        bias if has_bias else w,
        y,
        ws,
        M,
        N,
        K,
        x.stride(0),
        w.stride(0),
        y.stride(0),
        HAS_BIAS=has_bias,
        BLOCK_M=MAX_M,
        BLOCK_N=cfg.block_n,
        BLOCK_K=cfg.block_k,
        SPLIT_K=cfg.split_k,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    if cfg.split_k > 1:
        block = 256
        _ordered_reduce_kernel[(triton.cdiv(M * N, block),)](
            ws,
            bias if has_bias else ws,
            y,
            M,
            N,
            y.stride(0),
            HAS_BIAS=has_bias,
            SPLIT_K=cfg.split_k,
            BLOCK=block,
            num_warps=4,
        )
    return y


def default_config(n: int, k: int) -> SkinnyConfig:
    """The fixed schedule for an (N, K) weight; a function of the shape only, never of M."""
    # Aim for about two workgroups per CU of a 104/110-CU gfx90a GCD.
    block_k = 256
    block_n = 16 if n <= 8192 else 64
    grid_n = triton.cdiv(n, block_n)
    split_k = 1
    while grid_n * split_k < 192 and k // (split_k * 2) >= block_k and split_k < 16:
        split_k *= 2
    return SkinnyConfig(block_n, block_k, split_k)


def supports(n: int, k: int, cfg: SkinnyConfig) -> bool:
    return k % (cfg.split_k * cfg.block_k) == 0


# Larger weights (an lm_head) are bandwidth-bound in hipBLASLt already.
_MAX_N = 16384


@functools.lru_cache(maxsize=None)
def _config_for(n: int, k: int) -> Optional[SkinnyConfig]:
    cfg = default_config(n, k)
    return cfg if supports(n, k, cfg) else None


def applicable(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> bool:
    k = x.shape[-1]
    return (
        x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and (bias is None or bias.dtype == torch.bfloat16)
        and weight.dim() == 2
        and weight.shape[1] == k
        and weight.is_contiguous()
        and 0 < x.numel() <= MAX_M * k
        and weight.shape[0] <= _MAX_N
        and not weight.requires_grad
        and _config_for(weight.shape[0], k) is not None
    )


def skinny_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    n, k = weight.shape
    x2 = x.reshape(-1, k)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    y = skinny_gemm(x2, weight, bias, _config_for(n, k))
    return y.view(*x.shape[:-1], n)
