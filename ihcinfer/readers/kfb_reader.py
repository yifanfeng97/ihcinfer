"""KFB (Kuang-fu-biao) WSI reader using ``kfbslide``."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

# Lazy import so the module loads even when kfbslide is not installed.
_kfbslide = None  # type: ignore


def _get_kfbslide():
    """Return the ``kfbslide`` module, importing it on first use."""
    global _kfbslide
    if _kfbslide is None:
        import kfbslide as _kfbslide_mod  # type: ignore

        _kfbslide = _kfbslide_mod
    return _kfbslide


class KFBReader:
    """Read KFB whole-slide images via ``kfbslide``.

    This class conforms to the :class:`~ihcinfer.readers.base.SlideReader`
    protocol.
    """

    def __init__(self, path: str) -> None:
        """Open *path* with kfbslide.

        Args:
            path: Filesystem path to the ``.kfb`` file.

        Raises:
            ImportError: If ``kfbslide`` is not installed.
            Exception: If kfbslide cannot open the file.
        """
        kfb = _get_kfbslide()
        if kfb is None:
            raise ImportError(
                "kfbslide is required for KFB support but is not installed. "
                "Install it with:  pip install kfbslide"
            )
        self._slide = kfb.open(path)

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

    def close(self) -> None:
        """Close the underlying kfbslide object."""
        self._slide.close()

    def __enter__(self) -> "KFBReader":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and close the reader."""
        self.close()
