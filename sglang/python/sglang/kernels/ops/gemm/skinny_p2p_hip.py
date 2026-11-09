"""TP=2 decode-boundary kernels over P2P on gfx90a (skinny_gemv_gfx90a.hip, "P2P boundary").

Three channels, each with its own receive buffer (uncached, IPC-shared with the peer GCD) and
control block (epoch, arrival counter, error flag):

* ``out_proj_add_norm``: the skf2 out_proj GEMV (N = K = 2048) whose last workgroup runs the
  all-reduce, the residual add and the Gemma RMSNorm (one launch instead of three).
* ``ar_add_norm``: all-reduce + residual add + Gemma RMSNorm of a partial sum (one launch
  instead of two).
* ``gather``: the column all-gather of the lm_head logits (instead of RCCL's).

Rounding points are those of the custom all-reduce followed by the Triton Gemma add+norm with
``round_sum=True``; the rank sum is always rank 0 + rank 1 in fp32, so both ranks get the same
bits. Every call on a channel must be made on both ranks, in the same order, with the same M
(as for any collective). Launches go on torch's current stream and can be graph-captured.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Callable, List, Optional, Tuple

import torch

from sglang.kernels.ops.gemm import skinny_gemv_hip as _gemv

logger = logging.getLogger(__name__)

HIDDEN = 2048
MAX_ROWS = 16
# Words of a channel's receive buffer: 2 parities x MAX_ROWS rows x cols / 2 LL words.
_CTL_STRIDE = 256  # int32 per control block (epoch at 0, counter at 64, error at 128)
_CH_A, _CH_B, _CH_G = 0, 1, 2


def _words(cols: int) -> int:
    return 2 * MAX_ROWS * cols // 2


class HipP2PBoundary:
    """One rank's side of the two-rank P2P channels. Built eagerly (not under graph capture)
    on both ranks at the same point; ``all_gather_object`` exchanges the IPC handles."""

    def __init__(
        self,
        lib: _gemv.GemvLib,
        rank: int,
        all_gather_object: Callable[[object], List[object]],
        gather_cols: int = 0,
    ) -> None:
        if rank not in (0, 1):
            raise ValueError(f"rank {rank}")
        self.lib = lib.lib
        self.rank = rank
        self.gather_cols = gather_cols
        dev = torch.cuda.current_device()
        self.device = torch.device("cuda", dev)
        self._off = [0, _words(HIDDEN), 2 * _words(HIDDEN)]
        total_words = 2 * _words(HIDDEN) + (_words(gather_cols) if gather_cols else 0)
        nbytes = total_words * 8
        base = self.lib.skp_alloc_uncached(nbytes)
        if not base:
            raise RuntimeError("hipExtMallocWithFlags(uncached) failed")
        self._mine = base
        h = ctypes.create_string_buffer(64)
        if self.lib.skp_ipc_handle(base, h) != 0:
            raise RuntimeError("hipIpcGetMemHandle failed")
        handles = all_gather_object(bytes(h.raw))
        peer = self.lib.skp_ipc_open(ctypes.c_char_p(handles[1 - rank]))
        if not peer:
            raise RuntimeError("hipIpcOpenMemHandle failed")
        self._peer = peer
        self.ctl = torch.zeros(3 * _CTL_STRIDE, dtype=torch.int32, device=self.device)
        self.yloc = torch.empty((MAX_ROWS, HIDDEN), dtype=torch.bfloat16, device=self.device)
        self.cus = torch.cuda.get_device_properties(dev).multi_processor_count
        torch.cuda.synchronize()

    # -- helpers --------------------------------------------------------------------------
    def _args(self, ch: int, cols: int) -> _gemv.P2pArgs:
        a = _gemv.P2pArgs()
        a.peer = self._peer + self._off[ch] * 8
        a.mine = self._mine + self._off[ch] * 8
        a.ctl = self.ctl.data_ptr() + ch * _CTL_STRIDE * 4
        a.rank = self.rank
        a.n = cols
        a.rows = MAX_ROWS
        return a

    def errors(self) -> List[int]:
        """Spin-timeout flags of the three channels (nonzero: a peer never arrived)."""
        return [int(self.ctl[c * _CTL_STRIDE + 128].item()) for c in range(3)]

    @staticmethod
    def _norm_args(a, residual, weight, eps, out, res_out):
        a.residual = residual.data_ptr() if residual is not None else None
        a.weight = weight.data_ptr()
        a.out = out.data_ptr()
        a.res_out = res_out.data_ptr()
        a.eps = float(eps)

    # -- A: out_proj GEMV + all-reduce + add + norm ----------------------------------------
    @staticmethod
    def out_proj_fits(x: torch.Tensor, w: torch.Tensor) -> bool:
        return (
            x.dim() == 2
            and 0 < x.shape[0] <= MAX_ROWS
            and x.shape[1] == HIDDEN
            and tuple(w.shape) == (HIDDEN, HIDDEN)
            and x.dtype == torch.bfloat16
            and w.dtype == torch.bfloat16
            and w.is_contiguous()
            and x.stride(1) == 1
            and w.data_ptr() % 16 == 0
        )

    def out_proj_add_norm(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        residual: Optional[torch.Tensor],
        weight: torch.Tensor,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(normed, residual + allreduce(x @ w.T)) for x [M <= 16, 2048], w [2048, 2048]."""
        if x.data_ptr() % 16 != 0 or x.stride(0) % 8 != 0:
            x = x.clone(memory_format=torch.contiguous_format)
        m = x.shape[0]
        out = torch.empty((m, HIDDEN), dtype=x.dtype, device=x.device)
        res_out = torch.empty_like(out)
        a = self._args(_CH_A, HIDDEN)
        a.yloc = self.yloc.data_ptr()
        self._norm_args(a, residual, weight, eps, out, res_out)
        rc = self.lib.skf2ar_launch(
            x.data_ptr(), w.data_ptr(), m, x.stride(0), ctypes.byref(a),
            torch.cuda.current_stream().cuda_stream,
        )
        if rc != 0:
            raise RuntimeError(f"skf2ar launch failed: {rc}")
        return out, res_out

    # -- B: all-reduce + add + norm ----------------------------------------------------------
    @staticmethod
    def ar_norm_fits(x: torch.Tensor) -> bool:
        return (
            x.dim() == 2
            and 0 < x.shape[0] <= MAX_ROWS
            and x.shape[1] == HIDDEN
            and x.dtype == torch.bfloat16
            and x.stride(1) == 1
            and x.stride(0) % 4 == 0
            and x.data_ptr() % 8 == 0
        )

    def ar_add_norm(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor],
        weight: torch.Tensor,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m = x.shape[0]
        out = torch.empty((m, HIDDEN), dtype=x.dtype, device=x.device)
        res_out = torch.empty_like(out)
        a = self._args(_CH_B, HIDDEN)
        a.inp = x.data_ptr()
        a.in_stride = x.stride(0)
        self._norm_args(a, residual, weight, eps, out, res_out)
        rc = self.lib.skp_ar_norm_launch(
            m, ctypes.byref(a), torch.cuda.current_stream().cuda_stream
        )
        if rc != 0:
            raise RuntimeError(f"p2p_ar_norm launch failed: {rc}")
        return out, res_out

    # -- C: logits gather ----------------------------------------------------------------------
    def gather_fits(self, x: torch.Tensor) -> bool:
        return (
            self.gather_cols > 0
            and x.dim() == 2
            and 0 < x.shape[0] <= MAX_ROWS
            and x.shape[1] <= self.gather_cols
            and x.shape[1] % 4 == 0
            and x.dtype == torch.bfloat16
            and x.stride(1) == 1
            and x.stride(0) % 4 == 0
            and x.data_ptr() % 8 == 0
        )

    def gather(self, x: torch.Tensor) -> torch.Tensor:
        """[M, S] per rank -> [M, 2S]: rank 0's columns, then rank 1's."""
        m, s = x.shape
        out = torch.empty((m, 2 * s), dtype=x.dtype, device=x.device)
        a = self._args(_CH_G, s)
        a.inp = x.data_ptr()
        a.in_stride = x.stride(0)
        a.out = out.data_ptr()
        blocks = max(1, min(self.cus, (m * s // 4 + 255) // 256))
        rc = self.lib.skp_gather_launch(
            m, blocks, ctypes.byref(a), torch.cuda.current_stream().cuda_stream
        )
        if rc != 0:
            raise RuntimeError(f"p2p_gather launch failed: {rc}")
        return out
