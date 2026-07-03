"""WSI-level output helpers: CSV and thumbnail reading."""

from __future__ import annotations

import os
from typing import List

import pandas as pd
from PIL import Image


def write_patch_csv(records: List[dict], csv_path: str) -> None:
    """Write patch records to CSV."""
    columns = [
        "patch_id",
        "x",
        "y",
        "width",
        "height",
        "num_total",
        "num_pos",
        "num_neg",
        "percent_pos",
    ]
    df = pd.DataFrame(records, columns=columns)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df.to_csv(csv_path, index=False)


def read_slide_thumbnail(slide_path: str, target_size: tuple[int, int]) -> Image.Image:
    """Return an RGB thumbnail of *slide_path* resized exactly to *target_size*.

    The function tries the following strategies in order:

    1. Use the reader's ``read_thumbnail`` method if available
       (``OpenSlideReader`` uses OpenSlide's native pyramid;
       ``PILReader`` resizes the in-memory image).
    2. Fall back to opening the slide directly with ``openslide``.

    Raises:
        Exception: If no strategy can produce a thumbnail.
    """
    from ..readers import create_reader

    target_w, target_h = target_size

    reader = create_reader(slide_path)
    try:
        if hasattr(reader, "read_thumbnail"):
            thumb = reader.read_thumbnail(target_size)
        else:
            import openslide

            slide = openslide.OpenSlide(slide_path)
            try:
                thumb = slide.get_thumbnail((target_w, target_h)).convert("RGB")
            finally:
                slide.close()
    finally:
        reader.close()

    if thumb.size != target_size:
        thumb = thumb.resize(target_size, Image.LANCZOS)
    return thumb.convert("RGB")
