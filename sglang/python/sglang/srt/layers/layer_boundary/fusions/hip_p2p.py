"""gfx90a TP=2 decode boundaries over P2P (kernels/ops/gemm/skinny_p2p_hip.py).

* A: the attention output projection (GDN out_proj / attention o_proj, 2048x2048) runs as the
  skf2 GEMV whose last workgroup does the all-reduce, the residual add and the post-attention
  Gemma RMSNorm. The layer asks ``defers_out_proj`` before its attention runs; when it says
  yes the attention returns the projection's input, the layer marks it with ``mark_deferred``
  and this provider's FFN-input fusion, which then never declines, runs the fused kernel.
* B: every other decode boundary (the MoE output's sum + residual add + input norm, and the
  attention output's when A is off) as one P2P all-reduce + add + norm kernel.

Rounding points are the head's (custom all-reduce, then the Triton Gemma add+norm with
round_sum): the residual is bitwise the head's, the normed output differs only in the order
of the sum of squares. Every decision depends only on the batch and the shapes, which both
ranks share, so both ranks always launch the same kernels.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Optional

import torch

from sglang.srt.distributed.parallel_state import GroupCoordinator, get_tp_group
from sglang.srt.layers.layer_boundary import (
    FfnInputFusion,
    SumGroup,
    TokenAxis,
    UnreducedOutput,
    get_attn_tp_context,
)
from sglang.srt.layers.layernorm import GemmaRMSNorm, hip_gemma_norm_is_triton
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)

# Logits columns per rank the gather channel holds (Qwen3.5/3.6: 248320 / 2 = 124160).
MAX_GATHER_COLS = 131072

_state = None
_decided = False


def _is_gfx90a() -> bool:
    return (
        is_hip()
        and torch.cuda.is_available()
        and torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.startswith(
            "gfx90a"
        )
    )


def part_enabled(flag) -> bool:
    """Whether a part's switch and the master switch are on, on gfx90a at TP=2."""
    from sglang.srt.environ import envs

    parallel = get_parallel()
    return bool(
        envs.SGLANG_HIP_DECODE_BOUNDARY_FUSION.get()
        and flag.get()
        and parallel.tp_size == 2
        and parallel.attn_tp_size == 2
        and _is_gfx90a()
    )


def hip_p2p():
    """The process's HipP2PBoundary, built on first use outside graph capture, or None.
    Both ranks reach the first call in the same forward and agree on the outcome."""
    global _state, _decided
    if _decided:
        return _state
    if torch.cuda.is_current_stream_capturing():
        return None
    _decided = True
    import torch.distributed as dist

    from sglang.kernels.ops.gemm import skinny_gemm_hip
    from sglang.kernels.ops.gemm.skinny_p2p_hip import HipP2PBoundary

    group = get_tp_group()

    def all_gather_object(obj):
        out = [None] * group.world_size
        dist.all_gather_object(out, obj, group=group.cpu_group)
        return out

    ok = skinny_gemm_hip._gemv_ready()
    if not all(all_gather_object(ok)):
        logger.warning("HIP P2P boundaries off: the HIP GEMV library is not loaded on every rank")
        return None
    state, err = None, None
    try:
        state = HipP2PBoundary(
            skinny_gemm_hip._gemv_lib,
            group.rank_in_group,
            all_gather_object,
            gather_cols=MAX_GATHER_COLS,
        )
    except Exception as e:  # noqa: BLE001
        err = repr(e)
    errs = all_gather_object(err)
    if any(e is not None for e in errs):
        logger.warning("HIP P2P boundaries off: %s", errs)
        return None
    _state = state
    logger.info("HIP P2P decode boundaries ready (rank %d)", group.rank_in_group)
    return _state


def _batch_eligible(forward_batch: ForwardBatch, rows: int) -> bool:
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

    return bool(
        forward_batch.forward_mode.is_decode()
        and 0 < rows <= 16
        and hip_gemma_norm_is_triton()
        and not forward_batch.can_run_tbo
        and not is_batch_invariant_mode_enabled()
        and not get_attn_tp_context().input_scattered
        and not get_exec().comm.enable_quant_communications
    )


def _norm_eligible(norm, dtype) -> bool:
    return isinstance(norm, GemmaRMSNorm) and norm.weight.dtype == dtype


# The attention output a layer left for the FFN-input fusion to project: (tensor, weight).
_pending: Optional[tuple] = None


def mark_deferred(x: torch.Tensor, weight: torch.Tensor) -> None:
    global _pending
    if _pending is not None:
        raise RuntimeError("an unconsumed deferred out_proj")
    _pending = (x, weight)


def check_consumed() -> None:
    global _pending
    if _pending is not None:
        _pending = None
        raise RuntimeError("the deferred out_proj was not run by the FFN-input fusion")


def _take_pending(x) -> Optional[torch.Tensor]:
    global _pending
    if _pending is None:
        return None
    t, w = _pending
    if t is not x and not (
        isinstance(x, torch.Tensor) and t.data_ptr() == x.data_ptr() and t.shape == x.shape
    ):
        return None
    _pending = None
    return w


class HipP2PFusion:
    """One layer's P2P boundary kernels, given to both of its stages."""

    def __init__(self, *, out_proj_norm: bool, ar_norm: bool) -> None:
        self.out_proj_norm = out_proj_norm
        self.ar_norm = ar_norm

    def attn_input_fusions(self, plan) -> tuple:
        return (partial(self._attn_input, plan),) if self.ar_norm else ()

    def ffn_input_fusions(self, plan) -> tuple:
        if TokenAxis.ATTN_TP in plan.incoming_residual_rows.sharded:
            return ()
        return (
            FfnInputFusion(completes=SumGroup.ATTN_TP, run=partial(self._ffn_input, plan)),
        )

    def can_defer_all_reduce(self, plan, forward_batch: ForwardBatch) -> bool:
        return False

    def can_defer_finalize(self, plan, forward_batch: ForwardBatch) -> bool:
        return False

    def defers_out_proj(
        self, forward_batch: ForwardBatch, x: torch.Tensor, norm, weight: torch.Tensor
    ) -> bool:
        """Asked before the attention runs: x is its input (rows), weight the projection's."""
        if not (
            self.out_proj_norm
            and isinstance(x, torch.Tensor)
            and x.dtype == torch.bfloat16
            and _batch_eligible(forward_batch, x.shape[0])
            and _norm_eligible(norm, torch.bfloat16)
            and tuple(weight.shape) == (2048, 2048)
            and weight.dtype == torch.bfloat16
            and weight.is_contiguous()
            and weight.data_ptr() % 16 == 0
        ):
            return False
        from sglang.kernels.ops.gemm import skinny_gemm_hip

        return skinny_gemm_hip.gemv_active(2048, 2048) and hip_p2p() is not None

    def _ffn_input(self, plan, hidden_states, residual, forward_batch):
        norm = plan.norm
        weight = _take_pending(hidden_states)
        if weight is not None:
            x = hidden_states.reshape(-1, hidden_states.shape[-1])
            return hip_p2p().out_proj_add_norm(
                x, weight, residual, norm.weight.data, norm.variance_epsilon
            )
        if not self.ar_norm:
            return None
        return self._run(forward_batch, get_tp_group(), hidden_states, residual, norm)

    def _attn_input(self, plan, owed, residual, forward_batch, post_residual_addition):
        if (
            not isinstance(owed, UnreducedOutput)
            or not isinstance(owed.group, GroupCoordinator)
            or owed.complete_sum is not None
            or owed.reduce_to_dp_local is not None
            or residual is None
            or owed.group.world_size != 2
        ):
            return None
        if post_residual_addition is not None:
            if not self._eligible(forward_batch, owed.group, owed.partial, residual, plan.norm):
                return None
            residual = residual + post_residual_addition
        return self._run(forward_batch, owed.group, owed.partial, residual, plan.norm)

    def _run(self, forward_batch, group, partial_sum, residual, norm):
        if not self._eligible(forward_batch, group, partial_sum, residual, norm):
            return None
        p2p = hip_p2p()
        if p2p is None or not p2p.ar_norm_fits(partial_sum):
            return None
        return p2p.ar_add_norm(partial_sum, residual, norm.weight.data, norm.variance_epsilon)

    @staticmethod
    def _eligible(forward_batch, group, partial_sum, residual, norm) -> bool:
        return bool(
            isinstance(partial_sum, torch.Tensor)
            and partial_sum.dim() == 2
            and partial_sum.dtype == torch.bfloat16
            and residual is not None
            and residual.dtype == partial_sum.dtype
            and residual.shape == partial_sum.shape
            and residual.is_contiguous()
            and _norm_eligible(norm, partial_sum.dtype)
            and _batch_eligible(forward_batch, partial_sum.shape[0])
            and group.world_size == 2
            # The head's all-reduce for this tensor is the custom AR (same rank-order sum).
            and group._resolve_outplace_all_reduce_method(partial_sum) == "ca"
        )


def hip_p2p_fusion(is_nextn: bool) -> Optional[HipP2PFusion]:
    """The provider a gfx90a TP=2 decoder layer uses, or None when its parts are off."""
    from sglang.srt.environ import envs

    if is_nextn:
        return None
    a = part_enabled(envs.SGLANG_HIP_P2P_OUT_PROJ_NORM)
    b = part_enabled(envs.SGLANG_HIP_P2P_AR_NORM)
    if not (a or b):
        return None
    return HipP2PFusion(out_proj_norm=a, ar_norm=b)
