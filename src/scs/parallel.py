"""Deterministic CPU multiprocessing helpers."""

from __future__ import annotations

import os


def resolve_workers(workers: int | None) -> int:
    """Resolve a worker count where ``None`` or ``0`` means all available CPU cores."""
    if isinstance(workers, bool):
        raise ValueError("workers must be a non-negative integer, not a boolean")
    if workers is None or int(workers) == 0:
        return max(1, os.cpu_count() or 1)
    value = int(workers)
    if value < 0:
        raise ValueError(f"workers must be >= 0, got {value}")
    return value


def configure_worker_threads(*, configure_torch: bool = False) -> None:
    """Pin BLAS/OpenMP (and optionally PyTorch) intra-op threads to 1 inside worker processes."""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    if not configure_torch:
        return
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass
