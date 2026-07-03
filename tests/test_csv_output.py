"""Tests for CSV and heatmap output utilities."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ihcinfer import SlideInference
from ihcinfer.outputs import (
    blend_heatmap_overlay,
    build_heatmap,
    read_slide_thumbnail,
    write_patch_csv,
)
from ihcinfer.prep import TissueMask

MODEL_DIR = "/home/fengyifan/disk/code/DeepLIIF/model-server/DeepLIIF_Latest_Model"
PATCH = "tests/data/patches/22_2.png"


@pytest.fixture
def sample_records():
    return [
        {
            "patch_id": "p0",
            "x": 0,
            "y": 0,
            "width": 512,
            "height": 512,
            "num_total": 100,
            "num_pos": 30,
            "num_neg": 70,
            "percent_pos": 30.0,
        },
        {
            "patch_id": "p1",
            "x": 512,
            "y": 0,
            "width": 512,
            "height": 512,
            "num_total": 50,
            "num_pos": 5,
            "num_neg": 45,
            "percent_pos": 10.0,
        },
    ]


def test_write_patch_csv(sample_records, tmp_path):
    csv_path = tmp_path / "out.csv"
    write_patch_csv(sample_records, str(csv_path))
    df = pd.read_csv(csv_path)
    assert list(df.columns) == [
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
    assert len(df) == 2
    assert df.iloc[0]["num_total"] == 100


def test_build_heatmap(sample_records, tmp_path):
    hm_path = tmp_path / "hm.png"
    build_heatmap(sample_records, 1024, 512, str(hm_path), mode="percent_pos")
    assert hm_path.exists()
    img = Image.open(hm_path)
    # default upscale is 2x
    assert img.size == (4, 2)


def test_build_heatmap_no_upscale(sample_records, tmp_path):
    hm_path = tmp_path / "hm_no_upscale.png"
    build_heatmap(
        sample_records, 1024, 512, str(hm_path), mode="percent_pos", upscale=1
    )
    assert hm_path.exists()
    img = Image.open(hm_path)
    assert img.size == (2, 1)


def test_build_heatmap_max_size(sample_records, tmp_path):
    hm_path = tmp_path / "hm_max.png"
    build_heatmap(
        sample_records,
        1024,
        512,
        str(hm_path),
        mode="percent_pos",
        upscale=None,
        max_size=512,
    )
    assert hm_path.exists()
    img = Image.open(hm_path)
    # 2x1 grid scaled to fit max_size=512 on long edge -> 512x256
    assert img.size == (512, 256)


def test_build_heatmap_full_resolution(sample_records, tmp_path):
    hm_path = tmp_path / "hm_full.png"
    build_heatmap(
        sample_records, 1024, 512, str(hm_path), mode="percent_pos", tile_size=1
    )
    assert hm_path.exists()
    img = Image.open(hm_path)
    # default upscale is 2x
    assert img.size == (2048, 1024)


def test_blend_heatmap_overlay():
    thumb = Image.new("RGB", (100, 80), (200, 150, 100))
    heat = Image.new("RGB", (100, 80), (50, 100, 200))
    overlay = blend_heatmap_overlay(thumb, heat, alpha=0.5)
    assert overlay.size == (100, 80)
    arr = np.asarray(overlay)
    # uniform blend: (200+50)/2, (150+100)/2, (100+200)/2
    expected = np.array([125, 125, 150], dtype=np.uint8)
    assert np.allclose(arr[0, 0], expected, atol=1)


def test_blend_heatmap_overlay_with_mask():
    thumb = Image.new("RGB", (100, 80), (200, 150, 100))
    heat = Image.new("RGB", (100, 80), (50, 100, 200))
    mask = np.zeros((40, 50), dtype=np.uint8)
    mask[10:30, 10:40] = 1
    overlay = blend_heatmap_overlay(thumb, heat, alpha=0.5, tissue_mask=mask)
    assert overlay.size == (100, 80)
    arr = np.asarray(overlay)
    # A corner outside the mask should remain exactly the thumbnail color.
    assert np.array_equal(arr[0, 0], [200, 150, 100])


def test_read_slide_thumbnail_pil(tmp_path):
    img_path = tmp_path / "slide.jpg"
    Image.new("RGB", (512, 384), (128, 64, 32)).save(img_path, quality=95)
    thumb = read_slide_thumbnail(str(img_path), (200, 150))
    assert thumb.size == (200, 150)
    assert thumb.mode == "RGB"


def test_heatmap_thumbnail_overlay_same_size(sample_records, tmp_path):
    hm_path = tmp_path / "hm.png"
    build_heatmap(
        sample_records,
        1024,
        512,
        str(hm_path),
        mode="percent_pos",
        upscale=None,
        max_size=512,
    )
    hm = Image.open(hm_path)
    thumb = Image.new("RGB", hm.size, (200, 150, 100))
    overlay = blend_heatmap_overlay(thumb, hm, alpha=0.4)
    assert overlay.size == hm.size


def test_build_heatmap_interpolated_grid_factor(tmp_path):
    """Interpolated mode with grid_factor produces a fine heatmap grid."""
    records = [
        {"x": 0, "y": 0, "width": 512, "height": 512, "percent_pos": 10.0},
        {"x": 512, "y": 0, "width": 512, "height": 512, "percent_pos": 20.0},
        {"x": 0, "y": 512, "width": 512, "height": 512, "percent_pos": 30.0},
        {"x": 512, "y": 512, "width": 512, "height": 512, "percent_pos": 40.0},
    ]
    hm_path = tmp_path / "hm_interp.png"
    build_heatmap(
        records,
        1024,
        1024,
        str(hm_path),
        mode="percent_pos",
        grid_factor=4,
        upscale=1,
    )
    hm = Image.open(hm_path)
    # 1024/512*4 = 8 cells per axis
    assert hm.size == (8, 8)


def test_blend_heatmap_overlay_with_tissue_mask_object():
    thumb = Image.new("RGB", (100, 80), (200, 150, 100))
    heat = Image.new("RGB", (100, 80), (50, 100, 200))
    # A 50x40 mask; only the right half is tissue.
    mask = np.zeros((40, 50), dtype=np.uint8)
    mask[:, 25:] = 1
    tissue_mask = TissueMask(mask, scale=(2.0, 2.0), contours=[], holes=[])
    overlay = blend_heatmap_overlay(thumb, heat, alpha=0.5, tissue_mask=tissue_mask)
    arr = np.asarray(overlay)
    # Left side outside tissue should stay thumbnail; right side blended.
    assert np.array_equal(arr[20, 10], [200, 150, 100])
    expected_blend = np.array([125, 125, 150], dtype=np.uint8)
    assert np.allclose(arr[20, 75], expected_blend, atol=1)
