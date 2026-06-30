"""Tiling helper for image patches."""

from __future__ import annotations

import math
from typing import List, NamedTuple, Tuple

import numpy as np
from PIL import Image


Tile = NamedTuple(
    "Tile",
    [
        ("x", int),
        ("y", int),
        ("abs_cx", int),
        ("abs_cy", int),
        ("cw", int),
        ("ch", int),
        ("img", Image.Image),
    ],
)


class Tiler:
    """Tile an image into overlapping patches and stitch results back."""

    def __init__(
        self,
        image_or_size: Image.Image | np.ndarray | Tuple[int, int],
        tile_size: int,
        overlap_size: int = 0,
        pad_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if overlap_size < 0 or 2 * overlap_size >= tile_size:
            raise ValueError("overlap_size must be in [0, tile_size/2)")

        if isinstance(image_or_size, Image.Image):
            self.image = image_or_size.convert("RGB")
            self.width, self.height = self.image.size
        elif isinstance(image_or_size, np.ndarray):
            self.image = Image.fromarray(image_or_size)
            self.width, self.height = self.image.size
        else:
            self.image = None
            self.width, self.height = image_or_size

        self.tile_size = tile_size
        self.overlap_size = overlap_size
        self.stride = tile_size - 2 * overlap_size
        self.pad_color = pad_color
        self._canvas: dict[str, Image.Image] = {}
        self._tiles = list(self._generate_tiles())

    def _axis_starts(self, total: int) -> List[int]:
        """Central-region start coordinates along one axis."""
        if total <= self.tile_size:
            # A single tile covers the whole axis; its central region starts at 0.
            return [0]
        length = total - 2 * self.overlap_size
        n = math.ceil(length / self.stride)
        starts = [self.overlap_size + i * self.stride for i in range(n - 1)]
        remainder = length - (n - 1) * self.stride
        starts.append(total - self.overlap_size - remainder)
        return starts

    def _generate_tiles(self):
        xs = self._axis_starts(self.width)
        ys = self._axis_starts(self.height)
        blank = Image.new("RGB", (self.tile_size, self.tile_size), self.pad_color)

        for cy in ys:
            for cx in xs:
                x = max(0, cx - self.overlap_size)
                y = max(0, cy - self.overlap_size)
                x2 = min(x + self.tile_size, self.width)
                y2 = min(y + self.tile_size, self.height)
                cw = min(self.stride, self.width - cx)
                ch = min(self.stride, self.height - cy)

                patch = blank.copy()
                if self.image is not None:
                    patch.paste(self.image.crop((x, y, x2, y2)), (0, 0))

                yield Tile(
                    x=x,
                    y=y,
                    abs_cx=cx,
                    abs_cy=cy,
                    cw=cw,
                    ch=ch,
                    img=patch,
                )

    def __iter__(self):
        return iter(self._tiles)

    def __len__(self) -> int:
        return len(self._tiles)

    def stitch(self, tile: Tile, result_images: dict[str, Image.Image]) -> None:
        """Paste the central region of each result image into the canvas."""
        left = self.overlap_size
        top = self.overlap_size
        right = left + tile.cw
        bottom = top + tile.ch

        for key, img in result_images.items():
            if key not in self._canvas:
                self._canvas[key] = Image.new(
                    "RGB", (self.width, self.height), self.pad_color
                )
            crop = img.crop((left, top, right, bottom))
            self._canvas[key].paste(crop, (tile.abs_cx, tile.abs_cy))

    def results(self) -> dict[str, Image.Image]:
        return self._canvas.copy()
