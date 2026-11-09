"""One-launch decode MoE block on gfx90a (sgl-kernel AOT op ``hip_decode_moe_block``).

For a BF16 Qwen3.5/3.6-MoE block (hidden 2048, 256 experts of intermediate 256 per rank,
top-8 softmax routing with renormalize, one shared expert of intermediate 256 per rank with
a sigmoid gate) and a few decode rows, one persistent launch computes the router linear, the
softmax top-8 routing (lowest expert id wins ties), the routed experts, the shared expert
and ``routed + sigmoid(gate) * shared``. The TP all-reduce stays with the caller. The model
sends M <= DISPATCH_MAX_M rows, in launches of at most CHUNK_M rows.

The kernel needs a workspace whose barrier counters start at zero and reset themselves; it
is allocated once per device outside graph capture. The workspace is per device, not per
stream: launches that share it must be ordered, which holds because the model runs its
decode forward (eager or graph replay) on one stream per device. If the first call comes
during capture the caller keeps the per-op path. ``SGLANG_HIP_DECODE_MEGAKERNEL=0`` turns
it off. A barrier spin that times out sets an error word in the workspace; ``check_error``
reads it, logs it and resets the counters (decode_moe_block calls it on eager calls).
"""

from __future__ import annotations

import functools
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

HIDDEN, NUM_EXPERTS, INTER, TOPK, MAX_M = 2048, 256, 256, 8, 8
# Measured with tools/megakernel/moe_block_bench.py on gfx90a (us per block, graph replay,
# j119): M=1 40 vs 84 per-op, M=2 64 vs 107; M=4 as two M=2 launches 127 vs 147. Larger
# batches keep the per-op path.
DISPATCH_MAX_M = 4
CHUNK_M = 2
# Kernel variant bits (see hip_decode_moe_block.hip): 1 DPP top-8, 2 pipelined w13/w2 loops,
# 4 DPP wave sums, 8 phase stamps (debug). Built for every M <= CHUNK_M.
F_TOPK, F_PIPE, F_DSUM, F_STAMP = 1, 2, 4, 8
DEFAULT_VARIANT = 0
# Workgroups per CU in the grid, capped by what is co-resident for the variant and M.
DEFAULT_WG_PER_CU = 1
_WS_BYTES = 131072
_SYNC_ERR, _SYNC_WORDS = 128, (0, 128, 192)  # int32 indices: error word; count, err, router
_ERR_CHECK_EVERY = 256
_workspaces: dict[int, torch.Tensor] = {}
_grids: dict[tuple, int] = {}
_calls = 0
_logged = False


def env_enabled() -> bool:
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_HIP_DECODE_MEGAKERNEL.get())


@functools.lru_cache(maxsize=None)
def platform_ok() -> bool:
    if not torch.version.hip or not torch.cuda.is_available():
        return False
    try:
        arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
        if not arch.startswith("gfx90a"):
            return False
        import sgl_kernel  # noqa: F401  (loads common_ops)

        torch.ops.sgl_kernel.hip_decode_moe_block  # noqa: B018  (absent in older builds)
        return True
    except Exception:  # noqa: BLE001
        return False


def weights_ok(w_router, w_sgu, w_sgate, w_sdown, w13, w2) -> bool:
    shapes = (
        (w_router, (NUM_EXPERTS, HIDDEN)),
        (w_sgu, (2 * INTER, HIDDEN)),
        (w_sdown, (HIDDEN, INTER)),
        (w13, (NUM_EXPERTS, 2 * INTER, HIDDEN)),
        (w2, (NUM_EXPERTS, HIDDEN, INTER)),
    )
    for w, shape in shapes:
        if tuple(w.shape) != shape or w.dtype != torch.bfloat16 or not w.is_contiguous():
            return False
    return w_sgate.numel() == HIDDEN and w_sgate.dtype == torch.bfloat16 and w_sgate.is_contiguous()


def _dev_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def max_grid(m: int, variant: int, device_index: int, wg_per_cu: int = DEFAULT_WG_PER_CU) -> int:
    """Grid for M rows and a variant: wg_per_cu workgroups per CU, capped by co-residency
    (cached per device; the grid barrier needs every workgroup resident)."""
    key = (device_index, m, variant, wg_per_cu)
    g = _grids.get(key)
    if g is None:
        cu = torch.cuda.get_device_properties(device_index).multi_processor_count
        with torch.cuda.device(device_index):
            co = torch.ops.sgl_kernel.hip_decode_moe_block_max_grid(m, variant)
        if co < cu:
            raise RuntimeError(f"hip_decode_moe_block: M={m} variant={variant} fits {co} workgroups < {cu} CUs")
        g = min(co, wg_per_cu * cu)
        _grids[key] = g
    return g


def _prepare(device: torch.device) -> Optional[torch.Tensor]:
    """Workspace for ``device``; None if it cannot be set up here (capture)."""
    idx = _dev_index(device)
    ws = _workspaces.get(idx)
    if ws is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        for m in range(1, CHUNK_M + 1):  # co-residency of the default grid, checked up front
            max_grid(m, DEFAULT_VARIANT, idx)
        ws = torch.zeros(_WS_BYTES, dtype=torch.uint8, device=device)
        _workspaces[idx] = ws
    return ws


def check_error(device: torch.device) -> bool:
    """True if a launch on ``device`` timed out in a spin since the last check. Logs it and
    resets the counters so later launches start clean. Synchronizes; not for graph capture."""
    ws = _workspaces.get(_dev_index(device))
    if ws is None:
        return False
    sync = ws[:1024].view(torch.int32)
    if int(sync[_SYNC_ERR].item()) == 0:
        return False
    logger.error(
        "hip_decode_moe_block: a grid barrier or router wait timed out on %s (grid not "
        "co-resident or a launch was cut short); outputs since the last check are not "
        "trustworthy. Counters reset.",
        device,
    )
    for i in _SYNC_WORDS:
        sync[i] = 0
    return True


def decode_moe_block(
    x: torch.Tensor,
    w_router: torch.Tensor,
    w_sgu: torch.Tensor,
    w_sgate: torch.Tensor,
    w_sdown: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    grid: Optional[int] = None,
    variant: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """MoE block output for ``x`` [M <= 8, 2048] (before the TP all-reduce), or None if the
    kernel cannot run here (the caller then takes the per-op path). Rows go in launches of
    CHUNK_M; each row's result is the same for any chunking, M or grid."""
    global _logged, _calls
    ws = _prepare(x.device)
    if ws is None:
        return None
    idx = _dev_index(x.device)
    v = DEFAULT_VARIANT if variant is None else variant
    if out is None:
        out = torch.empty_like(x)
    for r0 in range(0, x.shape[0], CHUNK_M):
        m = min(CHUNK_M, x.shape[0] - r0)
        g = grid or max_grid(m, v, idx)
        torch.ops.sgl_kernel.hip_decode_moe_block(
            x[r0 : r0 + CHUNK_M],
            w_router,
            w_sgu,
            w_sgate,
            w_sdown,
            w13,
            w2,
            out[r0 : r0 + CHUNK_M],
            ws,
            g,
            v,
        )
    if not torch.cuda.is_current_stream_capturing():
        _calls += 1
        if _calls % _ERR_CHECK_EVERY == 0:
            check_error(x.device)
    if not _logged:
        _logged = True
        logger.info(
            "HIP decode MoE megakernel: on (decode M<=%d, %d rows per launch, variant=%d, grid=%d); "
            "SGLANG_HIP_DECODE_MEGAKERNEL=0 disables",
            DISPATCH_MAX_M,
            CHUNK_M,
            v,
            grid or max_grid(CHUNK_M, v, idx),
        )
    return out


def workspace_views(device: torch.device) -> dict:
    """Debug views of the workspace: router logits, routing ids/weights of the last call."""
    ws = _workspaces[_dev_index(device)]
    r = ws[1024 : 1024 + MAX_M * 768 * 2].view(torch.bfloat16).view(MAX_M, 768)
    ids_off = 16384 + MAX_M * TOPK * INTER * 2
    ids = ws[ids_off : ids_off + MAX_M * TOPK * 4].view(torch.int32).view(MAX_M, TOPK)
    wts = ws[ids_off + MAX_M * TOPK * 4 : ids_off + 2 * MAX_M * TOPK * 4].view(torch.float32).view(MAX_M, TOPK)
    sync = ws[:1024].view(torch.int32)
    stamps = ws[65536:131072].view(torch.int64).view(256, 4, 8)
    return {"logits": r[:, :NUM_EXPERTS], "ids": ids, "wts": wts, "sync": sync, "stamps": stamps}
