"""OpenSlide-backed WSI reader."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


class OpenSlideReader:
    """Read whole-slide images via ``openslide.OpenSlide``.

    This class conforms to the :class:`~ihcinfer.readers.base.SlideReader`
    protocol.
    """

    def __init__(self, path: str) -> None:
        """Open *path* with OpenSlide.

        Args:
            path: Filesystem path to the slide (e.g. ``.svs``, ``.tiff``).
        """
        import openslide

        self._slide = openslide.OpenSlide(path)

    @property
    def width(self) -> int:
        """Width of the slide in pixels at level 0."""
        return self._slide.dimensions[0]

    @property
    def height(self) -> int:
        """Height of the slide in pixels at level 0."""
        return self._slide.dimensions[1]

    def read(self, xywh: Tuple[int, int, int, int]) -> np.ndarray:
        """Read a region from the slide.

        Args:
            xywh: ``(x, y, w, h)`` in level-0 coordinates.

        Returns:
            RGB ``uint8`` array of shape ``(h, w, 3)``.
        """
        x, y, w, h = xywh
        pil_img: Image.Image = self._slide.read_region((x, y), 0, (w, h))
        rgb = pil_img.convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)

    def read_thumbnail(self, target_size: Tuple[int, int]) -> Image.Image:
        """Return a thumbnail via OpenSlide's native pyramid, resized to *target_size*."""
        thumb: Image.Image = self._slide.get_thumbnail(target_size)
        if thumb.size != target_size:
            thumb = thumb.resize(target_size, Image.LANCZOS)
        return thumb.convert("RGB")

    def close(self) -> None:
        """Close the underlying OpenSlide object."""
        if hasattr(self, "_slide"):
            self._slide.close()
            delattr(self, "_slide")

    def __enter__(self) -> "OpenSlideReader":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and close the reader."""
        self.close()
