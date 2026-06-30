"""Blank-patch detection and zero scoring helpers."""

from __future__ import annotations

import numpy as np
from PIL import Image


def is_blank_patch(img: Image.Image, threshold: float = 0.95) -> bool:
    """Return True if most pixels are white or black."""
    arr = np.asarray(img.convert("RGB"))
    white = np.all(arr >= 245, axis=-1)
    black = np.all(arr <= 10, axis=-1)
    blank_ratio = np.count_nonzero(white | black) / arr.shape[0] / arr.shape[1]
    return bool(blank_ratio >= threshold)


def blank_scoring() -> dict:
    """Zero cell-count scoring for blank patches."""
    return {
        "num_total": 0,
        "num_pos": 0,
        "num_neg": 0,
        "percent_pos": 0.0,
    }
