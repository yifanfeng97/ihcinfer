"""Tests for the WSI reader abstraction layer."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from ihcinfer.readers import KFBReader, OpenSlideReader, SlideReader, create_reader
from ihcinfer.readers.pil_reader import PILReader


# ---------------------------------------------------------------------------
# OpenSlideReader tests
# ---------------------------------------------------------------------------


class TestOpenSlideReader:
    """Tests for :class:`OpenSlideReader`."""

    def test_init_opens_slide(self, tmp_path):
        """Opening a valid image via OpenSlideReader works."""
        # Use a real SVS file if available, otherwise skip
        sample = "/home/fengyifan/disk/code/kfbslide/tests/sample.svs"
        if not __import__("os").path.exists(sample):
            pytest.skip("No sample.svs available")
        reader = OpenSlideReader(sample)
        assert reader.width > 0
        assert reader.height > 0
        reader.close()

    def test_context_manager(self, tmp_path):
        """``OpenSlideReader`` works as a context manager."""
        sample = "/home/fengyifan/disk/code/kfbslide/tests/sample.svs"
        if not __import__("os").path.exists(sample):
            pytest.skip("No sample.svs available")
        with OpenSlideReader(sample) as reader:
            assert reader.width > 0

    def test_read_region(self, tmp_path):
        """``read`` returns a correctly-shaped RGB uint8 array."""
        sample = "/home/fengyifan/disk/code/kfbslide/tests/sample.svs"
        if not __import__("os").path.exists(sample):
            pytest.skip("No sample.svs available")
        reader = OpenSlideReader(sample)
        region = reader.read((0, 0, 512, 512))
        assert region.shape == (512, 512, 3)
        assert region.dtype == np.uint8
        reader.close()

    def test_close_idempotent(self, tmp_path):
        """Calling ``close`` multiple times should not raise."""
        sample = "/home/fengyifan/disk/code/kfbslide/tests/sample.svs"
        if not __import__("os").path.exists(sample):
            pytest.skip("No sample.svs available")
        reader = OpenSlideReader(sample)
        reader.close()
        reader.close()  # idempotent


# ---------------------------------------------------------------------------
# KFBReader tests
# ---------------------------------------------------------------------------


class TestKFBReader:
    """Tests for :class:`KFBReader`."""

    def test_kfb_reader_import(self):
        """``KFBReader`` class exists regardless of kfbslide install status."""
        assert KFBReader is not None

    def test_kfb_reader_without_library(self, tmp_path):
        """Creating a ``KFBReader`` without kfbslide raises ``ImportError``."""
        if "kfbslide" in sys.modules:
            pytest.skip("kfbslide is installed")
        fake_kfb = tmp_path / "fake.kfb"
        fake_kfb.write_text("not real")
        with pytest.raises((ImportError, Exception)):
            create_reader(str(fake_kfb), backend="kfb")

    def test_kfb_reader_with_library(self, tmp_path):
        """When kfbslide is present, ``KFBReader`` can be instantiated."""
        pytest.importorskip("kfbslide")
        # We cannot easily test with a real KFB file, but we can verify
        # the constructor path by mocking kfbslide.
        fake_kfb = tmp_path / "fake.kfb"
        fake_kfb.write_text("not real")

        mock_slide = MagicMock()
        mock_slide.dimensions = (1000, 800)
        mock_kfb = MagicMock()
        mock_kfb.open.return_value = mock_slide

        with patch.dict(sys.modules, {"kfbslide": mock_kfb}):
            # Force re-import so the lazy import picks up the mock
            import importlib
            from ihcinfer.readers import kfb_reader as kfb_mod

            importlib.reload(kfb_mod)
            reader = kfb_mod.KFBReader(str(fake_kfb))
            assert reader.width == 1000
            assert reader.height == 800
            reader.close()


# ---------------------------------------------------------------------------
# create_reader factory tests
# ---------------------------------------------------------------------------


class TestCreateReader:
    """Tests for :func:`create_reader`."""

    def test_create_reader_openslide_backend(self):
        """``backend='openslide'`` returns an ``OpenSlideReader``."""
        sample = "/home/fengyifan/disk/code/kfbslide/tests/sample.svs"
        if not __import__("os").path.exists(sample):
            pytest.skip("No sample.svs available")
        reader = create_reader(sample, backend="openslide")
        assert isinstance(reader, OpenSlideReader)
        reader.close()

    def test_create_reader_auto_with_svs(self):
        """``backend='auto'`` picks OpenSlide for ``.svs`` files."""
        sample = "/home/fengyifan/disk/code/kfbslide/tests/sample.svs"
        if not __import__("os").path.exists(sample):
            pytest.skip("No sample.svs available")
        reader = create_reader(sample, backend="auto")
        assert isinstance(reader, OpenSlideReader)
        reader.close()

    def test_create_reader_auto_kfb_extension_uses_kfb(self, tmp_path, monkeypatch):
        """When OpenSlide fails and path ends with ``.kfb``, auto falls back to KFB."""
        fake_kfb = tmp_path / "test.kfb"
        fake_kfb.write_text("not real")

        # Mock OpenSlideReader to always raise
        def mock_osr(path):
            raise RuntimeError("OpenSlide cannot open this")

        # Mock KFBReader to return a simple mock
        class MockKFB:
            def __init__(self, path):
                self._w = 100
                self._h = 200

            @property
            def width(self):
                return self._w

            @property
            def height(self):
                return self._h

            def read(self, xywh):
                return np.zeros((xywh[3], xywh[2], 3), dtype=np.uint8)

            def close(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        monkeypatch.setattr(
            "ihcinfer.readers.openslide_reader.OpenSlideReader", mock_osr
        )
        monkeypatch.setattr(
            "ihcinfer.readers.kfb_reader.KFBReader", MockKFB
        )

        reader = create_reader(str(fake_kfb), backend="auto")
        assert reader.width == 100
        assert reader.height == 200
        reader.close()

    def test_create_reader_auto_raises_original_when_not_kfb(self, tmp_path, monkeypatch):
        """When OpenSlide fails and path is not ``.kfb``, auto raises the original error."""
        fake_svs = tmp_path / "test.svs"
        fake_svs.write_text("not real")

        original_err = RuntimeError("OpenSlide cannot open this")

        def mock_osr(path):
            raise original_err

        monkeypatch.setattr(
            "ihcinfer.readers.openslide_reader.OpenSlideReader", mock_osr
        )

        with pytest.raises(RuntimeError, match="OpenSlide cannot open this"):
            create_reader(str(fake_svs), backend="auto")

    def test_create_reader_unknown_backend(self):
        """An unknown backend string raises ``ValueError``."""
        with pytest.raises(ValueError, match="Unknown backend"):
            create_reader("/some/path", backend="not_a_backend")

    def test_create_reader_kfb_backend_without_library(self, tmp_path):
        """``backend='kfb'`` without kfbslide installed raises."""
        if "kfbslide" in sys.modules:
            pytest.skip("kfbslide is installed")
        fake_kfb = tmp_path / "fake.kfb"
        fake_kfb.write_text("not real")
        with pytest.raises((ImportError, Exception)):
            create_reader(str(fake_kfb), backend="kfb")


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocol:
    """Ensure concrete readers satisfy the ``SlideReader`` protocol."""

    def test_openslide_is_slide_reader(self):
        """``OpenSlideReader`` satisfies ``SlideReader``."""
        sample = "/home/fengyifan/disk/code/kfbslide/tests/sample.svs"
        if not __import__("os").path.exists(sample):
            pytest.skip("No sample.svs available")
        reader = OpenSlideReader(sample)
        assert isinstance(reader, SlideReader)
        reader.close()


class TestPILReader:
    """Tests for :class:`PILReader`."""

    def test_read_thumbnail_resizes_to_target(self, tmp_path):
        """``read_thumbnail`` returns an image of exactly the requested size."""
        img_path = tmp_path / "slide.jpg"
        Image.new("RGB", (512, 384), (128, 64, 32)).save(img_path, quality=95)
        reader = PILReader(str(img_path))
        thumb = reader.read_thumbnail((200, 150))
        assert thumb.size == (200, 150)
        assert thumb.mode == "RGB"
        reader.close()

    def test_read_region_and_close(self, tmp_path):
        """``read`` returns the requested region and ``close`` is idempotent."""
        img_path = tmp_path / "slide.png"
        Image.new("RGB", (256, 256), (255, 0, 0)).save(img_path)
        reader = PILReader(str(img_path))
        region = reader.read((0, 0, 128, 128))
        assert region.shape == (128, 128, 3)
        reader.close()
        reader.close()
