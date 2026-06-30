"""Tests for tissue segmentation."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ihcinfer.prep import TissueMask, TissueSegmenter

SVS = "tests/data/slides/98140-6 CD3.svs"
PATCH = "tests/data/patches/22_2.png"


@pytest.mark.skipif(not Path(SVS).exists(), reason="test slide not available")
def test_segmenter_on_svs():
    seg = TissueSegmenter(seg_level="auto").segment(SVS)
    assert seg.mask.sum() > 0
    assert len(seg.contours) > 0


def test_segmenter_on_patch_image():
    # the patch should be all tissue, so mask is mostly foreground
    img = Image.open(PATCH).convert("RGB")
    seg = TissueSegmenter(seg_level=0).segment(img)
    ratio = seg.mask.sum() / seg.mask.size
    assert 0.1 < ratio < 1.0


def test_tissue_mask_contains_patch():
    img = Image.open(PATCH).convert("RGB")
    seg = TissueSegmenter(seg_level=0).segment(img)
    assert seg.contains_patch(0, 0, img.width, img.height, min_ratio=0.05) is True
    # bottom-right corner is background in this sample patch
    assert seg.contains_patch(500, 500, 10, 10, min_ratio=1.0) is False


def test_tissue_mask_contains_center():
    img = Image.open(PATCH).convert("RGB")
    mask = TissueSegmenter(seg_level=0).segment(img)
    # Result should be a boolean for any level-0 coordinate.
    result = mask.contains_center(img.width // 2, img.height // 2)
    assert result in (True, False)


def test_tissue_mask_contains_center_respects_scale():
    """contains_center maps level-0 coords back to the downsampled mask."""
    mask_arr = np.zeros((10, 10), dtype=np.uint8)
    mask_arr[5, 5] = 1
    tissue_mask = TissueMask(mask_arr, scale=(10.0, 10.0), contours=[], holes=[])
    assert tissue_mask.contains_center(55, 55) is True
    assert tissue_mask.contains_center(45, 45) is False
    assert tissue_mask.contains_center(200, 200) is False


def test_tissue_mask_bbox_on_patch():
    img = Image.open(PATCH).convert("RGB")
    mask = TissueSegmenter(seg_level=0).segment(img)
    x_min, y_min, x_max, y_max = mask.bbox()
    assert 0 <= x_min <= x_max <= img.width
    assert 0 <= y_min <= y_max <= img.height
    assert x_max > x_min
    assert y_max > y_min


@pytest.mark.skipif(not Path(SVS).exists(), reason="test slide not available")
def test_tissue_segmenter_clam_vs_ihc():
    """IHC mode should retain at least as much foreground as CLAM on IHC slides."""
    clam_mask = TissueSegmenter(seg_level="auto", mode="clam").segment(SVS)
    ihc_mask = TissueSegmenter(seg_level="auto", mode="ihc").segment(SVS)

    assert clam_mask.mask.sum() > 0
    assert ihc_mask.mask.sum() > 0
    assert ihc_mask.mask.sum() >= clam_mask.mask.sum()


@pytest.mark.skipif(not Path(SVS).exists(), reason="test slide not available")
def test_ihc_mask_has_bbox():
    mask = TissueSegmenter(seg_level="auto", mode="ihc").segment(SVS)
    x_min, y_min, x_max, y_max = mask.bbox()
    assert x_max > x_min
    assert y_max > y_min


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        TissueSegmenter(mode="unknown")
