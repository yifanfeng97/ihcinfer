"""Base WSI reader protocol and factory."""

from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable

import numpy as np


@runtime_checkable
class SlideReader(Protocol):
    """Protocol for whole-slide image readers.

    Implementations must provide ``width`` and ``height`` properties, a
    ``read`` method that returns an RGB ``uint8`` NumPy array, and
    context-manager support.
    """

    @property
    def width(self) -> int:
        """Width of the slide in pixels at level 0."""
        ...

    @property
    def height(self) -> int:
        """Height of the slide in pixels at level 0."""
        ...

    def read(self, xywh: Tuple[int, int, int, int]) -> np.ndarray:
        """Read a region from the slide.

        Args:
            xywh: A ``(x, y, w, h)`` tuple in level-0 coordinates.

        Returns:
            An RGB ``uint8`` array of shape ``(h, w, 3)``.
        """
        ...

    def close(self) -> None:
        """Release underlying resources."""
        ...

    # Optional method that may be implemented for efficient thumbnail access.
    # def read_thumbnail(self, target_size: Tuple[int, int]) -> Image.Image: ...

    def __enter__(self) -> SlideReader:
        """Enter context manager."""
        ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and close the reader."""
        ...


_PIL_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def create_reader(path: str, backend: str = "auto") -> SlideReader:
    """Create a :class:`SlideReader` for *path*.

    Args:
        path: Filesystem path to the whole-slide image.
        backend: One of ``"auto"``, ``"openslide"``, ``"kfb"``, or ``"pil"``.

    Returns:
        A concrete :class:`SlideReader` instance.

    Raises:
        ValueError: If *backend* is not recognised.
        Exception: If the selected backend cannot open the slide.
    """
    if backend == "auto":
        from .openslide_reader import OpenSlideReader

        try:
            return OpenSlideReader(path)
        except Exception as exc:
            if path.lower().endswith(".kfb"):
                from .kfb_reader import KFBReader

                return KFBReader(path)
            if any(path.lower().endswith(ext) for ext in _PIL_EXTS):
                from .pil_reader import PILReader

                return PILReader(path)
            raise exc

    if backend == "openslide":
        from .openslide_reader import OpenSlideReader

        return OpenSlideReader(path)

    if backend == "kfb":
        from .kfb_reader import KFBReader

        return KFBReader(path)

    if backend == "pil":
        from .pil_reader import PILReader

        return PILReader(path)

    raise ValueError(f"Unknown backend: {backend!r}. Choose from 'auto', 'openslide', 'kfb', 'pil'.")
