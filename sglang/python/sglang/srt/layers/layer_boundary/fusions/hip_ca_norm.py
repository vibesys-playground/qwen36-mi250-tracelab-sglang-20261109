"""TP all-reduce + residual add + GemmaRMSNorm in one HIP custom all-reduce launch (gfx90a).

At a decode layer input the head runs the HIP custom all-reduce and then the Triton Gemma
add+norm (``round_sum=True``). CustomAllreduce.custom_all_reduce_add_rmsnorm does both in
one kernel with the same rounding points: the residual it writes is bitwise the head's, the
normed output differs only in the order of the sum of squares (tools/ar_norm/bench.py).
It never defers the MoE finalize or the all-reduce, so producers behave as without it.
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import torch

from sglang.srt.distributed.parallel_state import GroupCoordinator, get_attn_tp_group
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
from sglang.srt.utils import get_bool_env_var


class HipCustomARNormFusion:
    """One layer's fused AR + add + norm candidates, given to both of its stages.

    ``ffn_input``: also take the attention output's sum at the post-attention norm; off when
    another kernel owns that norm (SGLANG_HIP_DECODE_MEGAKERNEL, plan 13)."""

    def __init__(self, *, ffn_input: bool) -> None:
        self.ffn_input = ffn_input

    def attn_input_fusions(self, plan) -> tuple:
        return (partial(self._all_reduce_add_norm, plan),)

    def ffn_input_fusions(self, plan) -> tuple:
        parallel = get_parallel()
        if (
            not self.ffn_input
            or parallel.attn_tp_size != parallel.tp_size
            or TokenAxis.ATTN_TP in plan.incoming_residual_rows.sharded
        ):
            return ()
        return (
            FfnInputFusion(
                completes=SumGroup.ATTN_TP,
                run=partial(self._ffn_input_add_norm, plan),
            ),
        )

    def can_defer_all_reduce(self, plan, forward_batch: ForwardBatch) -> bool:
        return False

    def can_defer_finalize(self, plan, forward_batch: ForwardBatch) -> bool:
        return False

    def _all_reduce_add_norm(
        self, plan, owed, residual, forward_batch, post_residual_addition
    ):
        if (
            not isinstance(owed, UnreducedOutput)
            or not isinstance(owed.group, GroupCoordinator)
            or owed.complete_sum is not None
            or owed.reduce_to_dp_local is not None
            or residual is None
        ):
            return None
        if post_residual_addition is not None:
            if not self._eligible(plan, forward_batch, owed.group, owed.partial, residual):
                return None
            residual = residual + post_residual_addition
        return self._run(plan, forward_batch, owed.group, owed.partial, residual)

    def _ffn_input_add_norm(self, plan, hidden_states, residual, forward_batch):
        return self._run(
            plan, forward_batch, get_attn_tp_group(), hidden_states, residual
        )

    def _run(
        self,
        plan,
        forward_batch: ForwardBatch,
        group: GroupCoordinator,
        partial_sum: torch.Tensor,
        residual: Optional[torch.Tensor],
    ):
        if not self._eligible(plan, forward_batch, group, partial_sum, residual):
            return None
        norm = plan.norm
        return group.ca_comm.custom_all_reduce_add_rmsnorm(
            partial_sum,
            residual,
            norm.weight.data,
            norm.variance_epsilon,
            weight_shift=1.0,
        )

    @staticmethod
    def _eligible(
        plan,
        forward_batch: ForwardBatch,
        group: GroupCoordinator,
        partial_sum: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> bool:
        from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

        mode = forward_batch.forward_mode
        return bool(
            (mode.is_decode() or mode.is_target_verify())
            and residual is not None
            and residual.dtype == partial_sum.dtype
            and residual.shape == partial_sum.shape
            and isinstance(plan.norm, GemmaRMSNorm)
            and plan.norm.weight.dtype == partial_sum.dtype
            # The kernel reproduces the Triton Gemma add+norm's rounding points only.
            and hip_gemma_norm_is_triton()
            and group.ca_comm is not None
            # The head's all-reduce for this tensor is the same custom AR kernel.
            and group._resolve_outplace_all_reduce_method(partial_sum) == "ca"
            and not forward_batch.can_run_tbo
            and not is_batch_invariant_mode_enabled()
            and not get_attn_tp_context().input_scattered
            and not get_exec().comm.enable_quant_communications
            and TokenAxis.ATTN_TP not in plan.fused_input_rows(forward_batch).sharded
        )


def hip_ca_norm_fusion() -> Optional[HipCustomARNormFusion]:
    """The provider a gfx90a decoder layer uses, or None when it is switched off."""
    from sglang.srt.environ import envs

    if not envs.SGLANG_HIP_AR_NORM_FUSION.get() or get_parallel().tp_size < 2:
        return None
    return HipCustomARNormFusion(
        ffn_input=not get_bool_env_var("SGLANG_HIP_DECODE_MEGAKERNEL", "false")
    )
