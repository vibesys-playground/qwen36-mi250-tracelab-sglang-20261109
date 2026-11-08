"""Runtime build and ctypes binding of the gfx90a HIP skinny GEMV (skinny_gemv_gfx90a.hip).

The .hip source is compiled once per (source, flags, arch) with hipcc into a cache directory
(``SGLANG_SKINNY_GEMV_CACHE``, default ``$XDG_CACHE_HOME`` or ``/tmp``) under a file lock, so
the ranks of one server share one build. Launches go on torch's current stream and can be
captured in a HIP graph.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
import time
from typing import NamedTuple, Optional

import torch

logger = logging.getLogger(__name__)

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skinny_gemv_gfx90a.hip")
MBS = (1, 2, 4, 8, 16)


class GemvConfig(NamedTuple):
    rows: int  # R: rows of N per wave
    kparts: int  # P: waves splitting K for one row group
    groups: int  # G: row groups per workgroup
    nt: int = 1  # nontemporal weight loads
    lpr: int = 64  # lanes per row: 8, 16 or 64
    stag: int = 1  # K-start rotation per workgroup (0 = off)

    def nseg(self, k: int) -> int:
        return k // (self.lpr * 8)

    def takes(self, k: int, nseg: int) -> bool:
        return k == nseg * self.lpr * 8


# K per segment: "q" = 16x16x16 MFMA (16 rows x 64 B per load), "f" = 4x4x4 MFMA (4 rows x 256 B).
SEG_K = {"q": 32, "f": 128}


class MfmaConfig(NamedTuple):
    rows: int  # R: 16-row tiles of w per wave
    kparts: int  # P: waves splitting K for one tile group
    groups: int  # G: tile groups per workgroup
    hoist: int  # 1: issue every load before the first MFMA
    dl: int = 0  # MFMA C/D lane layout (0 or 1; the bench probes which one is right)
    stag: int = 0  # K-start rotation per workgroup (0 = off)



def _cache_dir() -> str:
    d = os.environ.get("SGLANG_SKINNY_GEMV_CACHE")
    if not d:
        base = os.environ.get("XDG_CACHE_HOME") or tempfile.gettempdir()
        d = os.path.join(base, "sglang_skinny_gemv")
    os.makedirs(d, exist_ok=True)
    return d


def _hipcc() -> Optional[str]:
    h = shutil.which("hipcc")
    if h is None and os.path.exists("/opt/rocm/bin/hipcc"):
        h = "/opt/rocm/bin/hipcc"
    return h


class GemvLib:
    def __init__(self, path: str):
        lib = ctypes.CDLL(path)
        vp, ci = ctypes.c_void_p, ctypes.c_int
        lib.skg_count.restype = ci
        lib.skg_entry.argtypes = [ci, ctypes.POINTER(ci)]
        lib.skg_find.argtypes = [ci] * 7
        lib.skg_launch.argtypes = [ci, vp, vp, vp, vp, ci, ci, ci, ci, ci, vp]
        lib.skg_read.argtypes = [vp, ctypes.c_longlong, vp, ci, vp]
        lib.skq_count.restype = ci
        lib.skq_entry.argtypes = [ci, ctypes.POINTER(ci)]
        lib.skq_find.argtypes = [ci] * 6
        lib.skq_launch.argtypes = [ci, vp, vp, vp, vp, ci, ci, ci, ci, ci, vp]
        lib.skf_count.restype = ci
        lib.skf_entry.argtypes = [ci, ctypes.POINTER(ci)]
        lib.skf_find.argtypes = [ci] * 6
        lib.skf_launch.argtypes = [ci, vp, vp, vp, vp, ci, ci, ci, ci, ci, vp]
        self.lib = lib
        self.path = path
        self.entries = []
        buf = (ci * 7)()
        for i in range(lib.skg_count()):
            lib.skg_entry(i, buf)
            self.entries.append(tuple(buf))
        self.q_entries = []
        qbuf = (ci * 6)()
        for i in range(lib.skq_count()):
            lib.skq_entry(i, qbuf)
            self.q_entries.append(tuple(qbuf))
        self.f_entries = []
        for i in range(lib.skf_count()):
            lib.skf_entry(i, qbuf)
            self.f_entries.append(tuple(qbuf))

    def find(self, mb: int, cfg: GemvConfig, k: int) -> int:
        return self.lib.skg_find(mb, cfg.rows, cfg.kparts, cfg.groups, cfg.nt, cfg.nseg(k), cfg.lpr)

    def configs(self, k: int) -> list[GemvConfig]:
        """Configs built for weights with K columns."""
        return [
            GemvConfig(r, p, g, nt, lpr)
            for mb, r, p, g, nt, ns, lpr in self.entries
            if mb == 1 and ns * lpr * 8 == k
        ]


    def q_find(self, cfg: MfmaConfig, k: int, fam: str = "q") -> int:
        find = self.lib.skq_find if fam == "q" else self.lib.skf_find
        return find(cfg.rows, cfg.kparts, cfg.groups, cfg.hoist, k // SEG_K[fam], cfg.dl)

    def q_configs(self, k: int, dl: int, fam: str = "q") -> list[MfmaConfig]:
        entries = self.q_entries if fam == "q" else self.f_entries
        return [
            MfmaConfig(r, p, g, hb, d)
            for r, p, g, hb, ns, d in entries
            if ns * SEG_K[fam] == k and d == dl
        ]


def build(defines: tuple[str, ...] = ("-DSKG_SWEEP",)) -> GemvLib:
    hipcc = _hipcc()
    if hipcc is None:
        raise RuntimeError("hipcc not found")
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0]
    flags = ["-O3", f"--offload-arch={arch}", "-ffp-contract=off", "-fPIC", "-shared", "-std=c++17",
             "-Wno-unused-result", *defines]
    h = hashlib.sha256(open(_SRC, "rb").read())
    h.update(" ".join(flags).encode())
    d = _cache_dir()
    so = os.path.join(d, f"skinny_gemv_{h.hexdigest()[:16]}.so")
    if not os.path.exists(so):
        with open(os.path.join(d, "build.lock"), "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                if not os.path.exists(so):
                    tmp = f"{so}.{os.getpid()}.tmp"
                    t0 = time.time()
                    r = subprocess.run([hipcc, *flags, "-o", tmp, _SRC], capture_output=True, text=True)
                    if r.returncode != 0:
                        raise RuntimeError(f"hipcc failed ({r.returncode}): {r.stderr[-3000:]}")
                    os.replace(tmp, so)
                    logger.info("skinny GEMV: built %s in %.1fs", so, time.time() - t0)
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)
    return GemvLib(so)


def mb_of(m: int) -> int:
    for mb in MBS:
        if m <= mb:
            return mb
    raise ValueError(m)


def gemv(lib: GemvLib, x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor],
         cfg: GemvConfig, idx: Optional[int] = None) -> torch.Tensor:
    """y = x @ w.T + bias with x [M <= 16, K] (unit inner stride, 16-byte aligned rows) and
    contiguous w [N, K], K = nseg * lpr * 8 for a built config."""
    M, K = x.shape
    N = w.shape[0]
    if idx is None:
        idx = lib.find(mb_of(M), cfg, K)
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    rc = lib.lib.skg_launch(
        idx, x.data_ptr(), w.data_ptr(), bias.data_ptr() if bias is not None else None,
        y.data_ptr(), M, N, x.stride(0), N, cfg.stag,
        torch.cuda.current_stream().cuda_stream,
    )
    if rc != 0:
        raise RuntimeError(f"skinny GEMV launch failed: hip error {rc} (idx {idx})")
    return y


def mfma_gemv(lib: GemvLib, x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor],
              cfg: MfmaConfig, idx: int, fam: str = "q") -> torch.Tensor:
    """y = x @ w.T + bias through the MFMA kernel ``idx`` (``lib.q_find(cfg, K)``); x [M <= 16, K]
    with unit inner stride and 16-byte aligned rows, w contiguous [N, K]."""
    M, K = x.shape
    N = w.shape[0]
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    launch = lib.lib.skq_launch if fam == "q" else lib.lib.skf_launch
    rc = launch(
        idx, x.data_ptr(), w.data_ptr(), bias.data_ptr() if bias is not None else None,
        y.data_ptr(), M, N, x.stride(0), N, cfg.stag,
        torch.cuda.current_stream().cuda_stream,
    )
    if rc != 0:
        raise RuntimeError(f"skinny MFMA GEMV launch failed: hip error {rc} (idx {idx})")
    return y
