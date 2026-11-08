"""Two skinny BF16 GEMMs that share their input, in one launch, each with its own schedule.

``y = x @ [w1; w2].T`` for a packed weight ``[w1; w2]`` (rows N1 + N2, contiguous) and a few rows
of ``x``. Programs [0, grid1) run w1's tiles and the rest run w2's, each part with the (N, K)
schedule ``skinny_gemm_hip`` picks for it alone and the same tile code, so every output column
has the bits the two separate ``skinny_linear`` calls give (tools/ar_norm/bench.py checks it).
A single schedule for the packed N would change the split-K order of the narrow part.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm import skinny_gemm_hip as _sk


@triton.jit
def _skinny_tile(
    x_ptr,
    w_ptr,
    y_ptr,
    ws_ptr,
    cnt_ptr,
    pid_n,
    pid_k,
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_ym,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    # The body of skinny_gemm_hip._skinny_gemm_kernel (no bias, fused split-K reduce).
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
        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :]
        tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask)
    else:
        ws_ptrs = ws_ptr + pid_k * M * N + offs_m[:, None] * N + offs_n[None, :]
        tl.store(ws_ptrs, acc, mask=mask)
        _sk._last_block_reduce(
            ws_ptr, y_ptr, y_ptr, cnt_ptr, pid_n, offs_m, offs_n, mask, mask_n,
            M, N, stride_ym, False, SPLIT_K,
        )


@triton.jit
def _skinny_gemm_pair_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    ws1_ptr,
    ws2_ptr,
    cnt_ptr,
    M,
    N1,
    N2,
    K,
    stride_xm,
    stride_wn,
    stride_ym,
    BLOCK_M: tl.constexpr,
    BN1: tl.constexpr,
    BK1: tl.constexpr,
    SK1: tl.constexpr,
    BN2: tl.constexpr,
    BK2: tl.constexpr,
    SK2: tl.constexpr,
    GRID_N1: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < GRID_N1 * SK1:
        _skinny_tile(
            x_ptr, w_ptr, y_ptr, ws1_ptr, cnt_ptr, pid % GRID_N1, pid // GRID_N1,
            M, N1, K, stride_xm, stride_wn, stride_ym,
            BLOCK_M, BN1, BK1, SK1,
        )
    else:
        p = pid - GRID_N1 * SK1
        grid_n2 = tl.cdiv(N2, BN2)
        _skinny_tile(
            x_ptr, w_ptr + N1 * stride_wn, y_ptr + N1, ws2_ptr, cnt_ptr + GRID_N1,
            p % grid_n2, p // grid_n2,
            M, N2, K, stride_xm, stride_wn, stride_ym,
            BLOCK_M, BN2, BK2, SK2,
        )


def _schedules(n1: int, n2: int, k: int):
    c1, c2 = _sk._config_for(n1, k), _sk._config_for(n2, k)
    if c1 is None or c2 is None:
        return None
    # One launch has one num_warps / num_stages; the tiles keep their code only if both agree.
    if (c1.num_warps, c1.num_stages) != (c2.num_warps, c2.num_stages):
        return None
    return c1, c2


def applicable(x: torch.Tensor, w_packed: torch.Tensor, n1: int) -> bool:
    k = x.shape[-1]
    n2 = w_packed.shape[0] - n1
    return (
        _sk._fused_reduce_ok
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and w_packed.dtype == torch.bfloat16
        and w_packed.is_contiguous()
        and w_packed.shape[1] == k
        and 0 < x.shape[0] <= _sk.MAX_M
        and 0 < n1 < w_packed.shape[0]
        and _schedules(n1, n2, k) is not None
        # Both parts must be on the Triton kernel: this copies its tile, not the HIP GEMV's.
        and (n1, k) not in _sk._GEMV_TUNED
        and (n2, k) not in _sk._GEMV_TUNED
    )


def skinny_linear_pair(
    x: torch.Tensor, w_packed: torch.Tensor, n1: int
) -> Optional[torch.Tensor]:
    """x [M, K] @ w_packed[N1 + N2, K].T as the two schedules of skinny_linear on w_packed[:n1]
    and w_packed[n1:]; returns [M, N1 + N2], or None when the fused split-K counters are not
    available yet (first call inside a graph capture)."""
    M, K = x.shape
    n2 = w_packed.shape[0] - n1
    c1, c2 = _schedules(n1, n2, K)
    if x.stride(-1) != 1:
        x = x.contiguous()
    grid_n1 = triton.cdiv(n1, c1.block_n)
    grid_n2 = triton.cdiv(n2, c2.block_n)
    cnt = None
    if c1.split_k > 1 or c2.split_k > 1:
        if grid_n1 + grid_n2 > _sk._COUNTER_LEN:
            return None
        cnt = _sk._ticket_counters(x.device)
        if cnt is None:
            return None
    y = torch.empty((M, n1 + n2), dtype=x.dtype, device=x.device)
    ws1 = (
        torch.empty((c1.split_k, M, n1), dtype=torch.float32, device=x.device)
        if c1.split_k > 1
        else y
    )
    ws2 = (
        torch.empty((c2.split_k, M, n2), dtype=torch.float32, device=x.device)
        if c2.split_k > 1
        else y
    )
    _skinny_gemm_pair_kernel[(grid_n1 * c1.split_k + grid_n2 * c2.split_k,)](
        x,
        w_packed,
        y,
        ws1,
        ws2,
        cnt if cnt is not None else ws1,
        M,
        n1,
        n2,
        K,
        x.stride(0),
        w_packed.stride(0),
        y.stride(0),
        BLOCK_M=_sk.MAX_M,
        BN1=c1.block_n,
        BK1=c1.block_k,
        SK1=c1.split_k,
        BN2=c2.block_n,
        BK2=c2.block_k,
        SK2=c2.split_k,
        GRID_N1=grid_n1,
        num_warps=c1.num_warps,
        num_stages=c1.num_stages,
    )
    return y
