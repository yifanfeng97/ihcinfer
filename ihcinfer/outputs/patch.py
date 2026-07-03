"""High-level patch output construction and saving.

This module encapsulates the standard IHC patch output pipeline:

- final segmentation image
- original patch with red/blue cell contours (overlay)
- ``cells.json`` with per-cell centroid/boundary/positive/size
- ``scoring.json`` with total/positive/negative counts
- optional inferred marker modality
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
    """User-facing outputs for a single patch."""

    name: str
    original: Image.Image
    segmentation: Image.Image
    marker: Image.Image | None
    scoring: Dict[str, int | float]
    cells: List[Dict[str, object]]
    overlay: Image.Image
    extra_images: Dict[str, Image.Image] | None = None


def build_patch_output(
    name: str,
    original: Image.Image,
    segmentation: Image.Image,
    marker: Image.Image | None = None,
    *,
    extra_images: Dict[str, Image.Image] | None = None,
    overlay_thickness: int = 2,
    resolution: str = "40x",
) -> PatchOutput:
    """Build the standard set of outputs for one patch.

    Args:
        name: Identifier used for logging / directory naming.
        original: The input RGB patch.
        segmentation: Final RGB segmentation image.
        marker: Optional inferred marker modality.
        extra_images: Optional dict of additional images to save.
        overlay_thickness: Thickness of the contour lines drawn on the overlay.
        resolution: Magnification string used for cell-size thresholds.

    Returns:
        A :class:`PatchOutput` instance ready to be saved.
    """
    cells = extract_cells(segmentation, marker, resolution=resolution)
    overlay = draw_contours(original, cells, thickness=overlay_thickness)
    scoring = compute_scoring_from_cells(cells)

    return PatchOutput(
        name=name,
        original=original,
        segmentation=segmentation,
        marker=marker,
        scoring=scoring,
        cells=cells,
        overlay=overlay,
        extra_images=extra_images,
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
        - any keys in ``extra_images`` as ``{key}.{image_format}``
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_image(
        output.segmentation, output_dir / f"segmentation.{image_format}", image_format
    )
    _save_image(output.overlay, output_dir / f"overlay.{image_format}", image_format)

    if save_marker and output.marker is not None:
        _save_image(output.marker, output_dir / f"marker.{image_format}", image_format)

    if output.extra_images:
        for key, img in output.extra_images.items():
            _save_image(img, output_dir / f"{key}.{image_format}", image_format)

    with open(output_dir / "cells.json", "w", encoding="utf-8") as f:
        json.dump({"cells": output.cells}, f, indent=2)

    with open(output_dir / "scoring.json", "w", encoding="utf-8") as f:
        json.dump(output.scoring, f, indent=2)


def save_patch(
    name: str,
    original: Image.Image,
    segmentation: Image.Image,
    output_dir: str | Path,
    *,
    marker: Image.Image | None = None,
    extra_images: Dict[str, Image.Image] | None = None,
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
        segmentation=segmentation,
        marker=marker,
        extra_images=extra_images,
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
