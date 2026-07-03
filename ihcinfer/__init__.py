"""Fast patch-based inference library for DeepLIIF on whole-slide images."""

from .inference import IHCAnalyzer
from .prep.tissue import segment_tissue

__all__ = ["IHCAnalyzer", "segment_tissue"]
