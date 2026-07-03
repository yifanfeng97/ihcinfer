"""Tests for ihcinfer.outputs.patch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ihcinfer.outputs import PatchOutput, build_patch_output, save_patch_output


def _red_segmentation(size: int = 64) -> Image.Image:
    """Create a segmentation with one positive red square in the center."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    margin = size // 4
    arr[margin:-margin, margin:-margin] = [255, 0, 0]
    return Image.fromarray(arr)


def _make_output(
    original: Image.Image | None = None,
    seg: Image.Image | None = None,
    marker: Image.Image | None = None,
) -> PatchOutput:
    original = original or Image.new("RGB", (64, 64), (200, 200, 200))
    seg = seg or _red_segmentation()
    return build_patch_output(
        name="test_patch",
        original=original,
        segmentation=seg,
        marker=marker,
    )


def test_build_patch_output_returns_cells_and_overlay():
    output = _make_output()
    assert output.name == "test_patch"
    assert len(output.cells) >= 1
    assert all("centroid" in c for c in output.cells)
    assert all("boundary" in c for c in output.cells)
    assert all("positive" in c for c in output.cells)
    assert all("size" in c for c in output.cells)
    assert output.overlay.size == (64, 64)
    assert output.scoring["num_total"] >= 1


def test_save_patch_output_writes_jpg_by_default(tmp_path: Path):
    output = _make_output()
    save_patch_output(output, tmp_path)

    assert (tmp_path / "segmentation.jpg").exists()
    assert (tmp_path / "overlay.jpg").exists()
    assert (tmp_path / "cells.json").exists()
    assert (tmp_path / "scoring.json").exists()
    assert not (tmp_path / "marker.jpg").exists()

    cells = json.loads((tmp_path / "cells.json").read_text())
    assert "cells" in cells
    scoring = json.loads((tmp_path / "scoring.json").read_text())
    assert set(scoring.keys()) >= {"num_total", "num_pos", "num_neg", "percent_pos"}


def test_save_patch_output_can_write_png(tmp_path: Path):
    output = _make_output()
    save_patch_output(output, tmp_path, image_format="png")
    assert (tmp_path / "segmentation.png").exists()
    assert (tmp_path / "overlay.png").exists()


def test_save_patch_output_marker_when_requested(tmp_path: Path):
    marker = Image.new("RGB", (64, 64), (128, 128, 128))
    output = _make_output(marker=marker)
    save_patch_output(output, tmp_path, save_marker=True)
    assert (tmp_path / "marker.jpg").exists()


def test_cells_json_does_not_include_mean_marker():
    """The reference schema only contains centroid/boundary/positive/size."""
    output = _make_output()
    for cell in output.cells:
        assert "mean_marker" not in cell
