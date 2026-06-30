"""Slow integration tests for whole-slide inference."""

from pathlib import Path

import openslide
import pytest
from PIL import Image

from ihcinfer import SlideInference

MODEL_DIR = "/home/fengyifan/disk/code/DeepLIIF/model-server/DeepLIIF_Latest_Model"
SVS = "tests/data/slides/98140-6 CD3.svs"

# A 4096x4096 level-0 region known to contain tissue, selected from a full WSI run.
CROP_X = 16384
CROP_Y = 16384
CROP_SIZE = 4096


@pytest.fixture
def small_slide_path(tmp_path: Path) -> Path:
    """Crop a tissue-rich region from the test SVS into a small JPEG."""
    slide = openslide.OpenSlide(SVS)
    region = slide.read_region((CROP_X, CROP_Y), 0, (CROP_SIZE, CROP_SIZE))
    slide.close()

    path = tmp_path / "small_slide.jpg"
    region.convert("RGB").save(path, quality=95)
    return path


@pytest.mark.slow
def test_run_on_wsi_cropped_slide(small_slide_path: Path, tmp_path: Path):
    inf = SlideInference(model_dir=MODEL_DIR, batch_size=4)
    result = inf.run_on_wsi(
        str(small_slide_path),
        str(tmp_path / "out"),
        tissue_min_ratio=0.01,
    )
    assert result.csv_path.exists()
    assert result.heatmap_path.exists()
    assert len(result.records) > 0
    assert len(result.region_sample_paths) <= 2 * 2
    assert len(result.patch_sample_dirs) <= 4
    if result.thumbnail_path:
        assert result.thumbnail_path.exists()
        assert result.overlay_path.exists()
        hm = Image.open(result.heatmap_path)
        th = Image.open(result.thumbnail_path)
        ov = Image.open(result.overlay_path)
        assert hm.size == th.size == ov.size


@pytest.mark.slow
def test_run_on_wsi_arbitrary_chunk_size(small_slide_path: Path, tmp_path: Path):
    """chunk_size no longer needs to be a multiple of region_size."""
    inf = SlideInference(model_dir=MODEL_DIR, batch_size=4)
    result = inf.run_on_wsi(
        str(small_slide_path),
        str(tmp_path / "out"),
        tissue_min_ratio=0.01,
        chunk_size=10000,
        num_region_samples=2,
        num_patch_samples=3,
    )
    assert result.csv_path.exists()
    assert result.heatmap_path.exists()
    assert len(result.records) > 0
    assert len(result.region_sample_paths) <= 2 * 2
    assert len(result.patch_sample_dirs) <= 3
    if result.thumbnail_path:
        assert result.thumbnail_path.exists()
        assert result.overlay_path.exists()
        hm = Image.open(result.heatmap_path)
        th = Image.open(result.thumbnail_path)
        ov = Image.open(result.overlay_path)
        assert hm.size == th.size == ov.size


@pytest.mark.slow
def test_run_on_wsi_skip_thumbnail(small_slide_path: Path, tmp_path: Path):
    inf = SlideInference(model_dir=MODEL_DIR, batch_size=4)
    result = inf.run_on_wsi(
        str(small_slide_path),
        str(tmp_path / "out"),
        tissue_min_ratio=0.01,
        skip_thumbnail=True,
    )
    assert result.csv_path.exists()
    assert result.heatmap_path.exists()
    assert result.thumbnail_path is None
    assert result.overlay_path is None
    assert not (tmp_path / "out" / "he_thumbnail.jpg").exists()
    assert not (tmp_path / "out" / "overlay.jpg").exists()
