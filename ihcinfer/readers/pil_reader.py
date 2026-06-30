"""PIL-backed reader for standard image formats (PNG, JPEG, TIFF)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


class PILReader:
    """Read regular images via Pillow.

    Conforms to the :class:`~ihcinfer.readers.base.SlideReader` protocol.
    """

    def __init__(self, path: str) -> None:
        self._img = Image.open(path).convert("RGB")
        self._path = path

    @property
    def width(self) -> int:
        return self._img.width

    @property
    def height(self) -> int:
        return self._img.height

    def read(self, xywh: Tuple[int, int, int, int]) -> np.ndarray:
        """Read a region from the image, padding with white if needed."""
        x, y, w, h = xywh
        x1 = max(0, min(self.width, x))
        y1 = max(0, min(self.height, y))
        x2 = max(0, min(self.width, x + w))
        y2 = max(0, min(self.height, y + h))
        crop = self._img.crop((x1, y1, x2, y2))
        if crop.size != (w, h):
            padded = Image.new("RGB", (w, h), (255, 255, 255))
            padded.paste(crop, (x1 - x, y1 - y))
            crop = padded
        return np.asarray(crop, dtype=np.uint8)

    def read_thumbnail(self, target_size: Tuple[int, int]) -> Image.Image:
        """Return the image resized to *target_size*."""
        return self._img.resize(target_size, Image.LANCZOS)

    def close(self) -> None:
        self._img.close()

    def __enter__(self) -> "PILReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
