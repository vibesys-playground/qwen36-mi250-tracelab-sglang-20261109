"""Deferred decode launch for the overlap scheduler (SGLANG_DEFER_DECODE_LAUNCH).

The overlap loop launches decode step k+1 right after step k, then blocks on k. A prefill
that arrives while k runs is ingested only after k ends and then queues on the GPU behind
k+1. With this on, the receiving TP rank holds the launch of a pure decode step k+1 until
shortly before k is estimated to end, polling the request sockets meanwhile; a request
that arrives in that window is ingested first, so its prefill goes ahead of k+1.

Only the launch order changes. The other TP ranks do not wait here: they block in the
request broadcast of ``ingest_requests`` until the receiving rank reaches it.

Estimates on the host clock, per decode batch size, medians of the last 16 steps:
  step: GPU time of a decode step = done(k) - max(done(k-1), launch_start(k))
  launch: host time from leaving the wait to the end of run_batch of the decode step
The wait ends at start(k) + step - launch - margin, when step k completes, or when a
request arrives, whichever is first.
"""

from __future__ import annotations

import time
from collections import deque
from statistics import median
from typing import Deque, Dict, Optional, Tuple

import zmq

_WINDOW = 16
_MIN_SAMPLES = 4


class DeferDecodeLaunch:
    def __init__(self, *, sockets, margin_s: float):
        self._poller = zmq.Poller()
        for s in sockets:
            self._poller.register(s, zmq.POLLIN)
        self._margin_s = margin_s
        self._step_s: Dict[int, Deque[float]] = {}
        self._launch_s: Dict[int, Deque[float]] = {}
        # (decode batch size or 0 for any other batch, launch start), oldest first
        self._inflight: Deque[Tuple[int, float]] = deque()
        self._last_done: Optional[float] = None
        self._wait_exit: Optional[float] = None
        self.num_waits = 0
        self.num_cut_by_request = 0

    @staticmethod
    def create(*, sockets, margin_us: int) -> Optional["DeferDecodeLaunch"]:
        sockets = [s for s in sockets if s is not None]
        if not sockets or not all(isinstance(s, zmq.Socket) for s in sockets):
            return None  # not the receiving rank, or a Rust ring / scripted proxy
        return DeferDecodeLaunch(sockets=sockets, margin_s=margin_us * 1e-6)

    @staticmethod
    def _push(d: Dict[int, Deque[float]], bs: int, x: float) -> None:
        q = d.get(bs)
        if q is None:
            q = d[bs] = deque(maxlen=_WINDOW)
        q.append(x)

    def on_launch(self, *, decode_bs: int, t_start: float, t_end: float) -> None:
        """A batch was launched by run_batch between t_start and t_end."""
        if decode_bs and self._wait_exit is not None:
            self._push(self._launch_s, decode_bs, t_end - self._wait_exit)
        self._wait_exit = None
        self._inflight.append((decode_bs, t_start))

    def on_done(self, t_done: float) -> None:
        """The oldest in-flight batch was observed complete at t_done."""
        if not self._inflight:
            return
        decode_bs, t_launch = self._inflight.popleft()
        if decode_bs:
            start = t_launch if self._last_done is None else max(self._last_done, t_launch)
            if t_done > start:
                self._push(self._step_s, decode_bs, t_done - start)
        self._last_done = t_done

    def wait(self, *, done_evt) -> None:
        """Hold the launch of the next decode step; return early on a request."""
        try:
            until = self._deadline()
            if until is None:
                return
            self.num_waits += 1
            while True:
                rem = until - time.perf_counter()
                if rem <= 0:
                    return
                if done_evt is not None and done_evt.query():
                    return
                tmo_ms = 1 if rem > 1.5e-3 else 0
                if self._poller.poll(tmo_ms):
                    self.num_cut_by_request += 1
                    return
                if tmo_ms == 0:
                    time.sleep(min(rem, 2e-4))
        finally:
            self._wait_exit = time.perf_counter()

    def _deadline(self) -> Optional[float]:
        if len(self._inflight) != 1:
            return None
        bs, t_launch = self._inflight[0]
        steps = self._step_s.get(bs)
        launches = self._launch_s.get(bs)
        if (
            not bs
            or steps is None
            or launches is None
            or len(steps) < _MIN_SAMPLES
            or len(launches) < _MIN_SAMPLES
        ):
            return None
        start = t_launch if self._last_done is None else max(self._last_done, t_launch)
        return start + median(steps) - median(launches) - self._margin_s
