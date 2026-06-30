"""Tests for the Tiler generator and stitcher."""

import numpy as np
import pytest
from PIL import Image

from ihcinfer.prep import Tiler


@pytest.fixture
def rgb_image():
    arr = np.zeros((1024, 1024, 3), dtype=np.uint8)
    arr[:, :, 0] = 128
    arr[:, :, 1] = 64
    arr[:, :, 2] = 200
    return Image.fromarray(arr)


def test_tiler_yields_expected_number_of_tiles(rgb_image):
    tiler = Tiler(rgb_image, tile_size=512, overlap_size=0)
    tiles = list(tiler)
    assert len(tiles) == 4
    for tile in tiles:
        assert tile.img.size == (512, 512)


def test_tiler_coordinates_no_overlap(rgb_image):
    tiler = Tiler(rgb_image, tile_size=512, overlap_size=0)
    tiles = list(tiler)
    positions = [(tile.abs_cx, tile.abs_cy) for tile in tiles]
    assert sorted(positions) == [(0, 0), (0, 512), (512, 0), (512, 512)]


def test_tiler_stitch_identity_no_overlap(rgb_image):
    tiler = Tiler(rgb_image, tile_size=512, overlap_size=0)
    for tile in tiler:
        tiler.stitch(tile, {"out": tile.img.copy()})
    out = tiler.results()["out"]
    np.testing.assert_array_equal(np.array(out), np.array(rgb_image))


def test_tiler_stitch_with_overlap_ignores_border(rgb_image):
    tiler = Tiler(rgb_image, tile_size=512, overlap_size=32)
    for tile in tiler:
        tiler.stitch(tile, {"out": tile.img.copy()})
    out = np.array(tiler.results()["out"])
    ref = np.array(rgb_image)
    # central regions should match; border of size `overlap` is background
    np.testing.assert_array_equal(out[32:-32, 32:-32], ref[32:-32, 32:-32])


def test_tiler_small_image_with_overlap_starts_at_zero():
    """When total <= tile_size, the only tile must start at (0, 0)."""
    img = Image.new("RGB", (256, 256), (10, 20, 30))
    tiler = Tiler(img, tile_size=512, overlap_size=32)
    tiles = list(tiler)
    assert len(tiles) == 1
    assert tiles[0].abs_cx == 0
    assert tiles[0].abs_cy == 0
    assert tiles[0].x == 0
    assert tiles[0].y == 0


def test_tiler_small_image_with_overlap_stitch_identity():
    """A single overlapping tile on a small image stitches without shifting content."""
    img = Image.new("RGB", (256, 256), (40, 50, 60))
    tiler = Tiler(img, tile_size=512, overlap_size=32)
    for tile in tiler:
        tiler.stitch(tile, {"out": tile.img.copy()})
    out = np.array(tiler.results()["out"])
    ref = np.array(img)
    # The interior (excluding the artificial overlap border) matches the input.
    np.testing.assert_array_equal(out[32:-32, 32:-32], ref[32:-32, 32:-32])
