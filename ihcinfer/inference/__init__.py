"""High-level inference entry points."""

from .patch import PatchInference, run_batches_adaptive
from .region import RegionInference
from .wsi import SlideInference, WSIResult

__all__ = [
    "SlideInference",
    "PatchInference",
    "RegionInference",
    "WSIResult",
    "run_batches_adaptive",
]
