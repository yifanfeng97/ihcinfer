"""Cell scoring and contour extraction for DeepLIIF segmentation outputs.

This is a self-contained re-implementation of the DeepLIIF cell-counting
post-processing.  It operates on the RGB segmentation image (``G5``) and the
inferred marker image (``G4``) produced by :class:`ihcinfer.models.DeepLIIFModel`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from skimage.measure import label


LABEL_BACKGROUND = 0
LABEL_POSITIVE = 1
LABEL_NEGATIVE = 2
LABEL_UNKNOWN = 3


def _to_array(img: Image.Image | np.ndarray, grayscale: bool = False) -> np.ndarray:
    """Convert a PIL image or array to a NumPy array."""
    if isinstance(img, Image.Image):
        arr = np.asarray(img.convert("RGB"))
    else:
        arr = np.asarray(img)
    if grayscale and arr.ndim == 3:
        arr = arr.max(axis=-1)
    return arr


def _large_noise_threshold(resolution: str) -> int:
    """Upper size bound for valid cells."""
    if resolution == "10x":
        return 1000
    if resolution == "20x":
        return 4000
    return 16000  # 40x default


def _default_size_threshold(resolution: str) -> int:
    """Default lower size bound for valid cells (squared sqrt defaults)."""
    if resolution == "10x":
        return 4
    if resolution == "20x":
        return 16
    return 49  # 40x default (sqrt 7)


def _default_marker_threshold(marker: np.ndarray) -> int:
    """Compute a default marker intensity threshold from the marker image."""
    vals = marker[marker > 0]
    if vals.size == 0:
        return 128
    # Use Otsu when there is enough bimodal signal; otherwise fall back to mean.
    if len(np.unique(vals)) > 1:
        thresh, _ = cv2.threshold(vals.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return int(thresh)
    return int(vals.mean())


def _create_posneg_mask(seg: np.ndarray, seg_thresh: int) -> np.ndarray:
    """Label each pixel as positive, negative, unknown, or background."""
    r = seg[..., 0].astype(np.int16)
    g = seg[..., 1].astype(np.int16)
    b = seg[..., 2].astype(np.int16)

    mask = np.full(seg.shape[:2], LABEL_UNKNOWN, dtype=np.uint8)
    valid = (r + b > seg_thresh) & (g <= 80)
    mask[valid & (r >= b)] = LABEL_POSITIVE
    mask[valid & (r < b)] = LABEL_NEGATIVE
    return mask


def _mark_background(mask: np.ndarray) -> np.ndarray:
    """Flood-fill unknown pixels connected to the image border as background."""
    unknown = mask == LABEL_UNKNOWN
    labeled, num = label(unknown, connectivity=1, return_num=True)

    if num > 0:
        border_labels = (
            set(labeled[0, :])
            | set(labeled[-1, :])
            | set(labeled[:, 0])
            | set(labeled[:, -1])
        )
        bg_labels = [lbl for lbl in border_labels if lbl != 0]
        if bg_labels:
            mask[np.isin(labeled, bg_labels)] = LABEL_BACKGROUND

    return mask


def _prepare_mask(
    seg: Image.Image | np.ndarray,
    marker: Image.Image | np.ndarray | None,
    resolution: str,
    seg_thresh: int,
) -> Tuple[np.ndarray, np.ndarray | None, int | None]:
    """Create the cell-label mask and marker threshold shared by scoring/cells.

    Returns:
        (mask, marker_array, marker_threshold).
    """
    seg_arr = _to_array(seg)
    marker_arr = _to_array(marker, grayscale=True) if marker is not None else None

    mask = _create_posneg_mask(seg_arr, seg_thresh)
    mask = _mark_background(mask)

    marker_thresh = None
    if marker_arr is not None:
        marker_thresh = _default_marker_threshold(marker_arr)

    return mask, marker_arr, marker_thresh


def _thresholds(resolution: str, large_noise_thresh: int | None, size_thresh: int | None) -> Tuple[int, int]:
    if large_noise_thresh is None:
        large_noise_thresh = _large_noise_threshold(resolution)
    if size_thresh is None:
        size_thresh = _default_size_threshold(resolution)
    return large_noise_thresh, size_thresh


def _count_cells(
    mask: np.ndarray,
    marker_arr: np.ndarray | None,
    marker_thresh: int | None,
    noise_thresh: int,
    large_noise_thresh: int,
    size_thresh: int,
) -> Tuple[int, int]:
    """Count positive/negative cells without extracting contours.

    Marker override (negative cell with high marker intensity becomes positive)
    is applied using vectorised ``np.bincount`` over component labels.
    """
    num_pos = 0
    num_neg = 0

    for label_value, is_positive_label in (
        (LABEL_POSITIVE, True),
        (LABEL_NEGATIVE, False),
    ):
        component_mask = (mask == label_value).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            component_mask, connectivity=8
        )

        if num_labels <= 1:
            continue

        # Vectorised mean marker intensity per component.
        if marker_arr is not None:
            sums = np.bincount(labels.ravel(), weights=marker_arr.ravel(), minlength=num_labels)
        else:
            sums = None

        for i in range(1, num_labels):
            size = int(stats[i, cv2.CC_STAT_AREA])
            if size < noise_thresh or size > large_noise_thresh or size < size_thresh:
                continue

            if marker_arr is not None and marker_thresh is not None and sums is not None:
                mean_marker = sums[i] / size
                # Negative cell with strong marker signal is reclassified as positive.
                if not is_positive_label and mean_marker > marker_thresh:
                    num_pos += 1
                    continue

            if is_positive_label:
                num_pos += 1
            else:
                num_neg += 1

    return num_pos, num_neg


def draw_contours(
    image: Image.Image | np.ndarray,
    cells: List[Dict[str, Any]],
    positive_color: Tuple[int, int, int] = (255, 0, 0),
    negative_color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> Image.Image:
    """Draw cell boundaries on the original *image*.

    Positive cells use *positive_color* (default red) and negative cells use
    *negative_color* (default blue), matching the DeepLIIF visualization
    convention.
    """
    arr = _to_array(image).copy()
    for cell in cells:
        contour = np.array(cell["boundary"], dtype=np.int32).reshape(-1, 1, 2)
        color = positive_color if cell["positive"] else negative_color
        cv2.drawContours(arr, [contour], -1, color, thickness)
    return Image.fromarray(arr)


def extract_cells(
    seg: Image.Image | np.ndarray,
    marker: Image.Image | np.ndarray | None = None,
    resolution: str = "40x",
    seg_thresh: int = 120,
    noise_thresh: int = 4,
    large_noise_thresh: int | None = None,
    marker_thresh: int | None = None,
    size_thresh: int | None = None,
) -> List[Dict[str, Any]]:
    """Extract per-cell information from a DeepLIIF segmentation map.

    Each cell is returned as a dict with ``centroid``, ``boundary``,
    ``positive``, and ``size`` keys, matching the schema used by the
    reference ``cells.json``.
    """
    mask, marker_arr, default_marker_thresh = _prepare_mask(seg, marker, resolution, seg_thresh)
    if marker_arr is not None and marker_thresh is None:
        marker_thresh = default_marker_thresh

    large_noise_thresh, size_thresh = _thresholds(resolution, large_noise_thresh, size_thresh)

    cells: List[Dict[str, Any]] = []

    for label_value in (LABEL_POSITIVE, LABEL_NEGATIVE):
        component_mask = (mask == label_value).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            component_mask, connectivity=8
        )

        for i in range(1, num_labels):
            size = int(stats[i, cv2.CC_STAT_AREA])
            if size < noise_thresh or size > large_noise_thresh or size < size_thresh:
                continue

            is_positive = label_value == LABEL_POSITIVE
            mean_marker = 0.0
            if marker_arr is not None:
                cell_pixels = marker_arr[labels == i]
                if cell_pixels.size > 0:
                    mean_marker = float(cell_pixels.mean())

            if marker_thresh is not None and mean_marker > marker_thresh:
                is_positive = True

            cx, cy = centroids[i]
            centroid = [int(round(cx)), int(round(cy))]

            cell_mask = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(
                cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if not contours:
                continue
            boundary = contours[0].reshape(-1, 2).tolist()

            cells.append(
                {
                    "centroid": centroid,
                    "boundary": boundary,
                    "positive": is_positive,
                    "size": size,
                }
            )

    return cells


def compute_scoring(
    seg: Image.Image | np.ndarray,
    marker: Image.Image | np.ndarray | None,
    resolution: str = "40x",
    seg_thresh: int = 120,
    noise_thresh: int = 4,
    large_noise_thresh: int | None = None,
    marker_thresh: int | None = None,
    size_thresh: int | None = None,
) -> Dict[str, int | float]:
    """Count positive/negative cells from a DeepLIIF segmentation map.

    This is a fast, contour-free implementation: it reuses the same
    foreground/background classification as :func:`extract_cells` but
    only counts connected components, making it suitable for bulk CSV
    scoring and patch-level scoring where boundaries are not required.
    """
    mask, marker_arr, default_marker_thresh = _prepare_mask(seg, marker, resolution, seg_thresh)
    if marker_arr is not None and marker_thresh is None:
        marker_thresh = default_marker_thresh

    large_noise_thresh, size_thresh = _thresholds(resolution, large_noise_thresh, size_thresh)

    num_pos, num_neg = _count_cells(
        mask,
        marker_arr,
        marker_thresh,
        noise_thresh,
        large_noise_thresh,
        size_thresh,
    )
    return compute_scoring_from_counts(num_pos, num_neg)


def compute_scoring_from_counts(num_pos: int, num_neg: int) -> Dict[str, int | float]:
    """Build the scoring dict from raw positive/negative counts."""
    num_total = num_pos + num_neg
    percent_pos = round(num_pos / num_total * 100, 1) if num_total > 0 else 0.0
    return {
        "num_total": num_total,
        "num_pos": num_pos,
        "num_neg": num_neg,
        "percent_pos": percent_pos,
    }


def compute_scoring_from_cells(cells: List[Dict[str, Any]]) -> Dict[str, int | float]:
    """Count positive/negative cells from a pre-computed cell list."""
    num_pos = sum(1 for cell in cells if cell["positive"])
    num_neg = len(cells) - num_pos
    return compute_scoring_from_counts(num_pos, num_neg)