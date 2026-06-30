"""Device selection helper for ihcinfer models."""

from __future__ import annotations

from typing import List

import torch


def resolve_device(gpu_ids: List[int] | None) -> torch.device:
    """Return a torch device from the first valid GPU id, else MPS/CPU."""
    if gpu_ids:
        for idx in gpu_ids:
            if idx >= 0 and torch.cuda.is_available() and idx < torch.cuda.device_count():
                return torch.device(f"cuda:{idx}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
