"""Patch-level IHC inference and scoring."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple, TypeVar

import numpy as np
import torch
from PIL import Image


T = TypeVar("T")
R = TypeVar("R")

from ..models import DeepLIIFModel, InferenceModel, ModelOutput
from ..models.device import resolve_device
from ..prep import blank_scoring, is_blank_patch
from ..scoring import compute_scoring


# Output keys used in the public images dictionary.
SEG_KEY = "segmentation"
MARKER_KEY = "marker"


def _resolution_for_tile(tile_size: int) -> str:
    if tile_size > 384:
        return "40x"
    if tile_size > 192:
        return "20x"
    return "10x"


def run_batches_adaptive(
    infer_fn: Callable[[List[T]], List[R]],
    inputs: List[T],
    batch_size: int,
) -> List[R]:
    """Run ``infer_fn`` on ``inputs`` in batches, halving batch size on OOM.

    This centralises the "try batch, catch CUDA out-of-memory, halve, retry"
    pattern used by patch-based WSI/region inference.  Results are returned in
    the same order as ``inputs``.
    """
    if not inputs:
        return []

    results: List[R] = []
    current_batch_size = batch_size
    i = 0
    while i < len(inputs):
        batch = inputs[i : i + current_batch_size]
        try:
            batch_results = infer_fn(batch)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and current_batch_size > 1:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                current_batch_size = max(1, current_batch_size // 2)
                continue
            raise

        if len(batch_results) != len(batch):
            raise RuntimeError(
                f"inference returned {len(batch_results)} results for {len(batch)} inputs"
            )
        results.extend(batch_results)
        i += len(batch)
    return results


class PatchInference:
    """Run inference on a batch of equally-sized RGB patches."""

    def __init__(
        self,
        model_dir: str | None = None,
        *,
        model: InferenceModel | None = None,
        gpu_ids: List[int] | None = None,
    ) -> None:
        if model is not None and model_dir is not None:
            raise ValueError("Specify either model_dir or model, not both")
        if model is None:
            if model_dir is None:
                raise ValueError("One of model_dir or model is required")
            device = resolve_device(gpu_ids or [])
            model = DeepLIIFModel(model_dir, device)
        self.model = model
        self.model_dir = getattr(model, "model_dir", None)

    def run(
        self,
        patches: List[Image.Image],
        resolution: str | None = None,
        return_marker: bool = False,
        return_images: bool = True,
    ) -> List[Tuple[Dict[str, Image.Image], Dict[str, int | float]]]:
        """Run inference and scoring on *patches*.

        Blank patches are skipped by the model and receive zero scoring.
        Returns a list of ``(images_dict, scoring_dict)`` tuples in the same
        order as the input.

        Args:
            resolution: One of ``10x``, ``20x``, ``40x``.  If None, inferred
                from the first patch width.
            return_marker: If True, the inferred marker image is returned in
                ``images_dict`` and used for marker-based cell reclassification.
                If False (default), the marker image is omitted and scoring is
                computed from the segmentation map only.
            return_images: If True (default), run the full forward pass and
                return segmentation/marker PIL images in ``images_dict``.  If
                False, run a scoring-only fast path that returns an empty
                ``images_dict`` and avoids creating any PIL images.
        """
        if not patches:
            return []

        tile_size = patches[0].width
        resolution = resolution or _resolution_for_tile(tile_size)

        nonblank_indices: List[int] = []
        nonblank_patches: List[Image.Image] = []
        for idx, img in enumerate(patches):
            if is_blank_patch(img):
                continue
            nonblank_indices.append(idx)
            nonblank_patches.append(img)

        if return_images:
            model_outputs: List[ModelOutput] = self.model.forward(nonblank_patches)
            raw_iter = iter(model_outputs)
        else:
            seg_arrays = self.model.forward_arrays(nonblank_patches)
            raw_iter = iter(seg_arrays)  # type: ignore[assignment]

        out: List[Tuple[Dict[str, Image.Image], Dict[str, int | float]]] = [
            ({}, blank_scoring()) for _ in patches
        ]
        for idx in nonblank_indices:
            if return_images:
                output = next(raw_iter)
                seg = output.segmentation
                marker = output.marker if (return_marker and output.marker is not None) else None
                scoring = compute_scoring(seg, marker, resolution=resolution)

                images: Dict[str, Image.Image] = {}
                images[SEG_KEY] = seg
                if marker is not None:
                    images[MARKER_KEY] = marker
            else:
                seg = next(raw_iter)
                scoring = compute_scoring(seg, None, resolution=resolution)
                images = {}

            out[idx] = (images, scoring)
        return out
