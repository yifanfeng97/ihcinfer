"""High-level inference entry points."""

from .patch import PatchInference, _run_batches_adaptive
from .region import RegionInference
from .wsi import SlideInference, WSIResult

__all__ = [
    "SlideInference",
    "PatchInference",
    "RegionInference",
    "WSIResult",
    "_run_batches_adaptive",
]
