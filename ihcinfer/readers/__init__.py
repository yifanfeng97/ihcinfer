"""WSI reader implementations for ihcinfer."""

from .base import SlideReader, create_reader
from .kfb_reader import KFBReader
from .openslide_reader import OpenSlideReader
from .pil_reader import PILReader

__all__ = [
    "SlideReader",
    "create_reader",
    "OpenSlideReader",
    "KFBReader",
    "PILReader",
]
