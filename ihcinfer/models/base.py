"""Abstract base class for IHC inference models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
from PIL import Image


class InferenceModel(ABC):
    """Abstract interface for a model that produces IHC segmentation outputs.

    Implementations are responsible for loading their own checkpoints and
    translating inputs/outputs into the common representation used by the
    rest of the pipeline.
    """

    @property
    @abstractmethod
    def seg_key(self) -> str:
        """Key used for the final segmentation image in result dicts."""
        ...

    @property
    @abstractmethod
    def marker_key(self) -> str | None:
        """Key used for the inferred marker image, or ``None`` if unavailable."""
        ...

    @abstractmethod
    def forward(self, images: List[Image.Image]) -> List[Dict[str, Image.Image]]:
        """Run the full model and return a dict of output images per input.

        Each returned dict must contain at least ``self.seg_key``.
        """
        ...

    @abstractmethod
    def forward_arrays(self, images: List[Image.Image]) -> List[np.ndarray]:
        """Scoring-only fast path returning uint8 RGB segmentation arrays.

        This should produce the same semantic segmentation as ``forward`` but
        skip expensive PIL/marker creation when only cell counts are needed.
        """
        ...
