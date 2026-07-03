"""Abstract base class for IHC inference models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from PIL import Image


@dataclass
class ModelOutput:
    """Typed output produced by an :class:`InferenceModel`.

    Attributes:
        segmentation: Final RGB segmentation image.
        marker: Optional inferred marker modality.
        modalities: Optional dict of additional model-specific modalities.
    """

    segmentation: Image.Image
    marker: Image.Image | None = None
    modalities: Dict[str, Image.Image] | None = None


class InferenceModel(ABC):
    """Abstract interface for a model that produces IHC segmentation outputs.

    Implementations are responsible for loading their own checkpoints and
    translating inputs/outputs into the common representation used by the
    rest of the pipeline.
    """

    @abstractmethod
    def forward(self, images: List[Image.Image]) -> List[ModelOutput]:
        """Run the full model and return typed outputs, one per input image."""
        ...

    @abstractmethod
    def forward_arrays(self, images: List[Image.Image]) -> List[np.ndarray]:
        """Scoring-only fast path returning uint8 RGB segmentation arrays.

        This should produce the same semantic segmentation as :meth:`forward`
        but skip expensive PIL/marker creation when only cell counts are needed.
        """
        ...
