"""Skinny BF16 GEMM ``y[m, n] = x[m, k] @ w[n, k].T (+ bias)`` for a few rows on gfx90a.

At M <= 16 hipBLASLt on gfx90a (MI210/MI250) tiles only N and streams the whole K loop in
one workgroup, so a decode-step linear costs ~20 us whatever its weight size. This kernel
gives every workgroup a [BLOCK_N, K / SPLIT_K] slab of the weight, read once for all
(up to 16) rows of x through MFMA ``tl.dot``, so the grid covers the GCD's CUs.

Numerics: BF16 inputs, FP32 accumulation, BF16 output. With SPLIT_K > 1 each split writes
FP32 partials, and the last block of each N tile to finish (elected by an int32 ticket
counter) sums them in a fixed order (split 0, 1, ...) inside the same kernel; no float
atomics, and the sum order does not depend on which block arrives last. The schedule (tile sizes, split count) is a function of (N, K) only and x is
always padded to one 16-row tile, so a row's result does not depend on how many other
rows share the call.

For the shapes in ``_GEMV_TUNED`` a hand-written HIP MFMA GEMV (skinny_gemv_gfx90a.hip, built
with hipcc at first use) replaces this kernel; it loads the weight straight into registers,
pads x to 16 MFMA rows the same way and keeps a fixed K order, so it is batch-invariant too.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm import skinny_gemv_hip as _gemv
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

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
    cnt_ptr,
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
    FUSED_REDUCE: tl.constexpr,
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
        if FUSED_REDUCE:
            _last_block_reduce(
                ws_ptr, b_ptr, y_ptr, cnt_ptr, pid_n, offs_m, offs_n, mask, mask_n,
                M, N, stride_ym, HAS_BIAS, SPLIT_K,
            )


@triton.jit
def _wait_vm_stores(dep):
    tl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)", "=v,v", [dep], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def _last_block_reduce(
    ws_ptr, b_ptr, y_ptr, cnt_ptr, pid_n, offs_m, offs_n, mask, mask_n,
    M, N, stride_ym, HAS_BIAS: tl.constexpr, SPLIT_K: tl.constexpr,
):
    """Sum the SPLIT_K partials of this [16, BLOCK_N] tile in split order 0, 1, ...

    Every (pid_n, pid_k) block takes a ticket from ``cnt[pid_n]`` after its partial is
    stored; the block that draws the last ticket does the sum. The ticket only elects
    that block: the sum always runs over splits 0..SPLIT_K-1 in index order, exactly as
    ``_ordered_reduce_kernel`` does, so the result is bitwise the same as the two-kernel
    path whichever block arrives last.

    Memory ordering (gfx90a): every wave waits ``vmcnt(0)`` so its partial stores have
    reached L2 (L2 is the device coherence point; L1 is write-through), the barrier
    orders all waves before the ticket, the ticket is an agent-scope acq_rel atomic
    (release before it, ``buffer_wbinvl1_vol`` after it), and the partials are re-read
    with ``.cg`` so they come from L2, never from a stale line in this CU's L1.
    """
    # Each wave waits until its own partial stores are acknowledged by L2 (Triton's
    # barrier waits on LDS only), then the barrier orders all waves before the ticket.
    _wait_vm_stores(pid_n)
    tl.debug_barrier()
    ticket = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
    if ticket == SPLIT_K - 1:
        base = ws_ptr + offs_m[:, None] * N + offs_n[None, :]
        tot = tl.load(base, mask=mask, other=0.0, cache_modifier=".cg")
        for s in tl.static_range(1, SPLIT_K):
            tot += tl.load(base + s * M * N, mask=mask, other=0.0, cache_modifier=".cg")
        if HAS_BIAS:
            tot += tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)[None, :]
        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :]
        tl.store(y_ptrs, tot.to(y_ptr.dtype.element_ty), mask=mask)
        # Every block of this call has taken its ticket: re-arm the counter for the next
        # call on this stream (and for the next graph replay) without a memset.
        tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="relaxed", scope="gpu")


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


# Arrival-ticket counters of the fused split-K reduce: one int32 per N tile, zero between
# calls (the last block of each tile re-arms its own), one buffer per device. Calls that
# share it must not overlap, which holds while a device runs its skinny GEMMs on one stream
# at a time: on HIP the Qwen3.5/3.6 models use an alternate stream only when
# SGLANG_ALT_STREAM is set, and then the two-kernel reduce is used. The buffer is
# allocated on the first eager call, never inside graph capture (which falls back).
_COUNTER_LEN = 4096
_counters: dict[int, torch.Tensor] = {}
_fused_reduce_ok = os.environ.get("SGLANG_ALT_STREAM", "").lower() not in ("1", "true", "yes")
_warned_capture_fallback = False


def _ticket_counters(device: torch.device) -> Optional[torch.Tensor]:
    global _warned_capture_fallback
    buf = _counters.get(device.index)
    if buf is None:
        if torch.cuda.is_current_stream_capturing():
            if not _warned_capture_fallback:
                _warned_capture_fallback = True
                logger.warning(
                    "skinny GEMM: first split-K call on device %s is inside graph "
                    "capture; it uses the two-kernel reduce",
                    device,
                )
            return None
        buf = torch.zeros(_COUNTER_LEN, dtype=torch.int32, device=device)
        _counters[device.index] = buf
    return buf


def skinny_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor],
    cfg: SkinnyConfig,
    fused_reduce: bool = True,
) -> torch.Tensor:
    """y = x @ w.T + bias for 2-D ``x`` [M <= 16, K] and contiguous ``w`` [N, K].

    With split_k > 1 the splits are summed in split order inside the GEMM by the last
    block of each N tile (``fused_reduce``), or by a second kernel; both give the same bits.
    """
    M, K = x.shape
    N = w.shape[0]
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    grid_n = triton.cdiv(N, cfg.block_n)
    has_bias = bias is not None
    cnt = None
    if cfg.split_k == 1:
        ws = y
    else:
        ws = torch.empty((cfg.split_k, M, N), dtype=torch.float32, device=x.device)
        if fused_reduce and _fused_reduce_ok and grid_n <= _COUNTER_LEN:
            cnt = _ticket_counters(x.device)
    fused = cnt is not None
    _skinny_gemm_kernel[(grid_n, cfg.split_k)](
        x,
        w,
        bias if has_bias else w,
        y,
        ws,
        cnt if fused else ws,
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
        FUSED_REDUCE=fused,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    if cfg.split_k > 1 and not fused:
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


# Schedules measured on an MI210 (gfx90a, 104 CUs) with HIP-graph replay at M = 2 and 16
# over block_n 16-64, block_k 128-512, split_k 1-16, 4 or 8 warps (tools/skinny_gemm).
_TUNED: dict[tuple[int, int], SkinnyConfig] = {
    (6144, 2048): SkinnyConfig(64, 128, 2),
    (4608, 2048): SkinnyConfig(64, 128, 2),
    (2048, 2048): SkinnyConfig(64, 128, 16),
    (512, 2048): SkinnyConfig(64, 128, 16),
    (256, 2048): SkinnyConfig(16, 128, 16),
    (32, 2048): SkinnyConfig(16, 128, 16),
    (2048, 256): SkinnyConfig(64, 128, 1),
}


def default_config(n: int, k: int) -> SkinnyConfig:
    """The fixed schedule for an (N, K) weight; a function of the shape only, never of M."""
    if (n, k) in _TUNED:
        return _TUNED[(n, k)]
    # Untuned shapes: split K until the grid is about two workgroups per CU.
    block_k = 128
    block_n = 16 if n <= 256 else 64
    grid_n = triton.cdiv(n, block_n)
    split_k = 1
    while grid_n * split_k < 192 and k % (split_k * 2 * block_k) == 0 and split_k < 16:
        split_k *= 2
    return SkinnyConfig(block_n, block_k, split_k)


def supports(n: int, k: int, cfg: SkinnyConfig) -> bool:
    return k % (cfg.split_k * cfg.block_k) == 0


# Larger weights (an lm_head) are bandwidth-bound in hipBLASLt already.
_MAX_N = 16384


@functools.lru_cache(maxsize=None)
def _config_for(n: int, k: int) -> Optional[SkinnyConfig]:
    cfg = default_config(n, k)
    if not supports(n, k, cfg):
        return None
    logger.info("skinny GEMM: N=%d K=%d uses %s for M<=%d", n, k, cfg, MAX_M)
    return cfg


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


# HIP MFMA GEMV (skinny_gemv_gfx90a.hip) for the (N, K) where it beats this Triton kernel at
# M = 2 with M = 16 within 3%; one kernel for every M of a shape. Measured on an MI250 GCD
# (tools/skinny_gemm v3.main_q, j118), us at M = 2 / 16, Triton -> GEMV:
_GEMV_TUNED: dict[tuple[int, int], tuple[str, _gemv.MfmaConfig]] = {
    (2048, 2048): ("q", _gemv.MfmaConfig(1, 4, 1, 1)),  # 15.8/18.3 -> 13.5/18.0
    (4608, 2048): ("q", _gemv.MfmaConfig(1, 4, 1, 1)),  # 30.6/32.0 -> 22.2/29.4
    (512, 2048): ("q", _gemv.MfmaConfig(1, 16, 1, 1)),  # 8.3/8.8 -> 6.8/9.0
    (2048, 256): ("q", _gemv.MfmaConfig(1, 4, 1, 0)),  # 5.9/5.9 -> 3.2/3.7
}

_gemv_lib: Optional[_gemv.GemvLib] = None
_gemv_idx: dict[tuple[int, int], int] = {}
_gemv_decided = False


def _gemv_ready() -> bool:
    """Builds (hipcc, cached on disk) and loads the GEMV library once per process, on the first
    skinny call; any failure keeps the process on the Triton kernel, so the kernel used for a
    weight never changes during a run."""
    global _gemv_lib, _gemv_decided
    if _gemv_decided:
        return _gemv_lib is not None
    _gemv_decided = True
    if not _GEMV_TUNED or envs.SGLANG_ROCM_DISABLE_SKINNY_GEMV.get():
        return False
    if torch.cuda.is_current_stream_capturing():
        logger.warning("skinny GEMV: first use is inside graph capture; using the Triton kernel")
        return False
    try:
        entries = [(fam, cfg, k) for (n, k), (fam, cfg) in _GEMV_TUNED.items()]
        lib = _gemv.build(defines=_gemv.table_defines(entries))
        idx = {}
        dev = torch.cuda.current_device()
        for (n, k), (fam, cfg) in _GEMV_TUNED.items():
            i = lib.q_find(cfg, k, fam)
            if i < 0:
                raise RuntimeError(f"no kernel for N={n} K={k} {fam} {cfg}")
            idx[(n, k)] = i
            # Load the code object on this device before any graph capture.
            xw = torch.zeros((1, k), dtype=torch.bfloat16, device=dev)
            _gemv.mfma_gemv(lib, xw, xw, None, cfg, i, fam)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        logger.warning("skinny GEMV: HIP build or load failed, using the Triton kernel: %s", e)
        return False
    _gemv_idx.update(idx)
    _gemv_lib = lib
    logger.info("skinny GEMV: HIP MFMA kernels for %s", sorted(_GEMV_TUNED))
    return True


def skinny_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    n, k = weight.shape
    x2 = x.reshape(-1, k)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    tuned = _GEMV_TUNED.get((n, k))
    if tuned is not None and weight.data_ptr() % 16 == 0 and _gemv_ready():
        if x2.data_ptr() % 16 != 0 or x2.stride(0) % 8 != 0:
            x2 = x2.clone(memory_format=torch.contiguous_format)
        fam, cfg = tuned
        y = _gemv.mfma_gemv(_gemv_lib, x2, weight, bias, cfg, _gemv_idx[(n, k)], fam)
    else:
        y = skinny_gemm(x2, weight, bias, _config_for(n, k))
    return y.view(*x.shape[:-1], n)
