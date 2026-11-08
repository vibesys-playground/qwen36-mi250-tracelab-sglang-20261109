"""One-launch decode MoE block on gfx90a (sgl-kernel AOT op ``hip_decode_moe_block``).

For a BF16 Qwen3.5/3.6-MoE block (hidden 2048, 256 experts of intermediate 256 per rank,
top-8 softmax routing with renormalize, one shared expert of intermediate 256 per rank with
a sigmoid gate) and M <= 8 rows, one persistent launch computes the router linear, the
softmax top-8 routing (lowest expert id wins ties), the routed experts, the shared expert
and ``routed + sigmoid(gate) * shared``. The TP all-reduce stays with the caller.

The kernel needs a 64 KiB workspace whose barrier counters start at zero and reset
themselves; it is allocated once per device outside graph capture. If the first call comes
during capture the caller keeps the per-op path. ``SGLANG_HIP_DECODE_MEGAKERNEL=0`` turns it
off.
"""

from __future__ import annotations

import functools
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

HIDDEN, NUM_EXPERTS, INTER, TOPK, MAX_M = 2048, 256, 256, 8, 8
# Measured with tools/megakernel/moe_block_bench.py on gfx90a (us per block, graph replay):
# M=1 40 vs 84 per-op, M=2 64 vs 107; M=4 as two M=2 launches 127 vs 147; M=8 254 vs 225.
# So the model sends M <= 4 here, in launches of at most CHUNK_M rows (rows are independent).
DISPATCH_MAX_M = 4
CHUNK_M = 2
_WS_BYTES = 65536
_workspaces: dict[int, torch.Tensor] = {}
_grids: dict[int, int] = {}
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


def _prepare(device: torch.device) -> Optional[tuple[torch.Tensor, int]]:
    """Workspace and grid for ``device``; None if they cannot be set up here (capture)."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    ws = _workspaces.get(idx)
    if ws is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        cu = torch.cuda.get_device_properties(idx).multi_processor_count
        # One workgroup per CU must be co-resident for every M.
        for m in range(1, MAX_M + 1):
            g = torch.ops.sgl_kernel.hip_decode_moe_block_max_grid(m)
            if g < cu:
                raise RuntimeError(f"hip_decode_moe_block: M={m} fits {g} workgroups < {cu} CUs")
        ws = torch.zeros(_WS_BYTES, dtype=torch.uint8, device=device)
        _workspaces[idx] = ws
        _grids[idx] = cu
    return ws, _grids[idx]


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
) -> Optional[torch.Tensor]:
    """MoE block output for ``x`` [M <= 8, 2048] (before the TP all-reduce), or None if the
    kernel cannot run here (the caller then takes the per-op path). Rows go in launches of
    CHUNK_M; each row's result is the same for any chunking, M or grid."""
    global _logged
    prep = _prepare(x.device)
    if prep is None:
        return None
    ws, g = prep
    if out is None:
        out = torch.empty_like(x)
    for r0 in range(0, x.shape[0], CHUNK_M):
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
            grid or g,
        )
    if not _logged:
        _logged = True
        logger.info(
            "HIP decode MoE megakernel: on (decode M<=%d, %d rows per launch, grid=%d); "
            "SGLANG_HIP_DECODE_MEGAKERNEL=0 disables",
            DISPATCH_MAX_M,
            CHUNK_M,
            grid or g,
        )
    return out


def workspace_views(device: torch.device) -> dict:
    """Debug views of the workspace: router logits, routing ids/weights of the last call."""
    ws = _workspaces[device.index if device.index is not None else torch.cuda.current_device()]
    r = ws[1024 : 1024 + MAX_M * 768 * 2].view(torch.bfloat16).view(MAX_M, 768)
    ids_off = 16384 + MAX_M * TOPK * INTER * 2
    ids = ws[ids_off : ids_off + MAX_M * TOPK * 4].view(torch.int32).view(MAX_M, TOPK)
    wts = ws[ids_off + MAX_M * TOPK * 4 : ids_off + 2 * MAX_M * TOPK * 4].view(torch.float32).view(MAX_M, TOPK)
    sync = ws[:1024].view(torch.int32)
    return {"logits": r[:, :NUM_EXPERTS], "ids": ids, "wts": wts, "sync": sync}
