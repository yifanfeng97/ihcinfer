"""High-level patch output construction and saving.

This module encapsulates the standard DeepLIIF patch output pipeline:

- final segmentation image (G5)
- original patch with red/blue cell contours (overlay)
- ``cells.json`` with per-cell centroid/boundary/positive/size
- ``scoring.json`` with total/positive/negative counts
- optional inferred marker modality (G4)

The :class:`PatchOutput` dataclass, the lower-level ``build_patch_output`` /
``save_patch_output`` helpers, and the convenience :func:`save_patch` function
are library-level primitives; the example script only handles CLI argument
parsing and directory naming.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from PIL import Image

from ..scoring import compute_scoring_from_cells, draw_contours, extract_cells


@dataclass
class PatchOutput:
    """User-facing outputs for a single DeepLIIF patch."""

    name: str
    original: Image.Image
    images: Dict[str, Image.Image]
    seg_key: str
    marker_key: str | None
    scoring: Dict[str, int | float]
    cells: List[Dict[str, object]]
    overlay: Image.Image


def build_patch_output(
    name: str,
    original: Image.Image,
    images: Dict[str, Image.Image],
    *,
    seg_key: str = "segmentation",
    marker_key: str | None = "marker",
    overlay_thickness: int = 2,
    resolution: str = "40x",
) -> PatchOutput:
    """Build the standard set of outputs for one patch.

    Args:
        name: Identifier used for logging / directory naming.
        original: The input RGB patch.
        images: Dictionary of model outputs (must contain *seg_key*).
        seg_key: Key for the final segmentation image (default ``segmentation``).
        marker_key: Key for the inferred marker image (default ``marker``).
        overlay_thickness: Thickness of the contour lines drawn on the overlay.
        resolution: Magnification string used for cell-size thresholds.

    Returns:
        A :class:`PatchOutput` instance ready to be saved.
    """
    seg = images.get(seg_key)
    if seg is None:
        raise ValueError(f"Segmentation image '{seg_key}' not found in model outputs")

    marker = images.get(marker_key) if marker_key else None
    cells = extract_cells(seg, marker, resolution=resolution)
    overlay = draw_contours(original, cells, thickness=overlay_thickness)
    scoring = compute_scoring_from_cells(cells)

    return PatchOutput(
        name=name,
        original=original,
        images=images,
        seg_key=seg_key,
        marker_key=marker_key,
        scoring=scoring,
        cells=cells,
        overlay=overlay,
    )


def _save_image(img: Image.Image, path: Path, image_format: str) -> None:
    """Save a PIL image, applying JPEG quality 95 when applicable."""
    kwargs = {"quality": 95} if image_format.lower() in ("jpg", "jpeg") else {}
    img.save(path, **kwargs)


def save_patch_output(
    output: PatchOutput,
    output_dir: str | Path,
    *,
    image_format: str = "jpg",
    save_marker: bool = False,
) -> None:
    """Save a :class:`PatchOutput` to disk.

    Files written:
        - ``segmentation.{image_format}``
        - ``overlay.{image_format}``
        - ``cells.json``
        - ``scoring.json``
        - ``marker.{image_format}`` (only if *save_marker* is True and a marker
          image is present)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seg = output.images.get(output.seg_key)
    if seg is not None:
        _save_image(seg, output_dir / f"segmentation.{image_format}", image_format)

    _save_image(output.overlay, output_dir / f"overlay.{image_format}", image_format)

    if save_marker and output.marker_key:
        marker = output.images.get(output.marker_key)
        if marker is not None:
            _save_image(marker, output_dir / f"marker.{image_format}", image_format)

    with open(output_dir / "cells.json", "w", encoding="utf-8") as f:
        json.dump({"cells": output.cells}, f, indent=2)

    with open(output_dir / "scoring.json", "w", encoding="utf-8") as f:
        json.dump(output.scoring, f, indent=2)


def save_patch(
    name: str,
    original: Image.Image,
    images: Dict[str, Image.Image],
    output_dir: str | Path,
    *,
    seg_key: str = "segmentation",
    marker_key: str | None = "marker",
    overlay_thickness: int = 2,
    resolution: str = "40x",
    image_format: str = "jpg",
    save_marker: bool = False,
) -> PatchOutput:
    """Build and save the standard patch outputs in one call.

    This is a convenience wrapper around :func:`build_patch_output` and
    :func:`save_patch_output`.
    """
    output = build_patch_output(
        name=name,
        original=original,
        images=images,
        seg_key=seg_key,
        marker_key=marker_key,
        overlay_thickness=overlay_thickness,
        resolution=resolution,
    )
    save_patch_output(
        output,
        output_dir,
        image_format=image_format,
        save_marker=save_marker,
    )
    return output
