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


def read_slide_thumbnail(
    slide_path: str,
    target_size: tuple[int, int],
    bounds: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    """Return an RGB thumbnail of *slide_path* resized exactly to *target_size*.

    The function tries the following strategies in order:

    1. Use the reader's ``read_thumbnail`` method if available
       (``OpenSlideReader`` uses OpenSlide's native pyramid;
       ``PILReader`` resizes the in-memory image).
    2. Fall back to opening the slide directly with ``openslide``.

    If *bounds* is provided (or reported by the reader, as for MIRAX .mrxs),
    the thumbnail is cropped to that region before resizing, so the useful
    tissue fills the output instead of being a small island in a black frame.
    To preserve visual quality, the initial thumbnail is requested at a larger
    size so that the cropped tissue region still has approximately
    *target_size* resolution.

    Raises:
        Exception: If no strategy can produce a thumbnail.
    """
    from ..readers import create_reader

    target_w, target_h = target_size

    reader = create_reader(slide_path)
    try:
        full_w, full_h = reader.width, reader.height
        if hasattr(reader, "read_thumbnail"):
            thumb = reader.read_thumbnail(target_size)
        else:
            import openslide

            slide = openslide.OpenSlide(slide_path)
            try:
                thumb = slide.get_thumbnail((target_w, target_h)).convert("RGB")
            finally:
                slide.close()

        if bounds is None:
            bounds = getattr(reader, "bounds", None)
    finally:
        reader.close()

    if bounds is not None and full_w > 0 and full_h > 0:
        bx, by, bw, bh = bounds
        if bw > 0 and bh > 0:
            # For slides whose level-0 canvas is much larger than the scanned
            # tissue (e.g. MIRAX), request a larger thumbnail so that the
            # tissue region inside the bounds retains full target resolution.
            scale = max(full_w / bw, full_h / bh, 1.0)
            request_w = max(1, int(round(target_w * scale)))
            request_h = max(1, int(round(target_h * scale)))

            reader = create_reader(slide_path)
            try:
                thumb = reader.read_thumbnail((request_w, request_h))
            finally:
                reader.close()

            thumb_w, thumb_h = thumb.size
            x1 = max(0, min(thumb_w, int(round(bx / full_w * thumb_w))))
            y1 = max(0, min(thumb_h, int(round(by / full_h * thumb_h))))
            x2 = max(0, min(thumb_w, int(round((bx + bw) / full_w * thumb_w))))
            y2 = max(0, min(thumb_h, int(round((by + bh) / full_h * thumb_h))))
            if x2 > x1 and y2 > y1:
                thumb = thumb.crop((x1, y1, x2, y2))

    if thumb.size != target_size:
        thumb = thumb.resize(target_size, Image.LANCZOS)
    return thumb.convert("RGB")
