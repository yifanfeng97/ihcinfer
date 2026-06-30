"""Heatmap generation and overlay visualization."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


_CMAP_CODES = {
    "jet": cv2.COLORMAP_JET,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "plasma": cv2.COLORMAP_PLASMA,
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "turbo": cv2.COLORMAP_TURBO,
    "hot": cv2.COLORMAP_HOT,
    "cividis": cv2.COLORMAP_CIVIDIS,
    "twilight": cv2.COLORMAP_TWILIGHT,
}


def _estimate_patch_size(records: List[dict]) -> int:
    """Estimate regular patch spacing from record top-left coordinates."""
    xs = sorted({int(round(rec["x"])) for rec in records})
    ys = sorted({int(round(rec["y"])) for rec in records})
    dxs = np.diff(xs) if len(xs) > 1 else np.array([1])
    dys = np.diff(ys) if len(ys) > 1 else np.array([1])
    dx = int(round(np.median(dxs[dxs > 0]))) if np.any(dxs > 0) else 1
    dy = int(round(np.median(dys[dys > 0]))) if np.any(dys > 0) else 1
    return max(1, (dx + dy) // 2)


def _resize_mask_to_shape(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Resize a binary mask to (h, w) and threshold back to bool."""
    h, w = shape
    mh, mw = mask.shape
    interp = cv2.INTER_AREA if (w < mw and h < mh) else cv2.INTER_LINEAR
    resized = cv2.resize(mask.astype(np.float32), (w, h), interpolation=interp)
    return resized > 0.5


def _extract_mask_array(tissue_mask) -> np.ndarray:
    """Extract a 2D binary ndarray from a ``TissueMask`` or raw array."""
    mask_arr = tissue_mask.mask if hasattr(tissue_mask, "mask") else tissue_mask
    mask_arr = np.asarray(mask_arr, dtype=np.uint8)
    if mask_arr.ndim != 2:
        raise ValueError("tissue_mask must be a 2D binary array")
    return mask_arr


def blend_heatmap_overlay(
    thumbnail: Image.Image,
    heatmap: Image.Image,
    alpha: float = 0.4,
    tissue_mask=None,
) -> Image.Image:
    """Blend a translucent heatmap over an H&E thumbnail.

    Args:
        thumbnail: H&E thumbnail (RGB), same size as *heatmap*.
        heatmap: Colored heatmap (RGB), same size as *thumbnail*.
        alpha: Opacity of the heatmap layer (0 = invisible, 1 = opaque).
        tissue_mask: Optional binary mask or ``TissueMask``. Non-tissue pixels
            keep the thumbnail unchanged.

    Returns:
        A new RGB image of the same size as the inputs.
    """
    thumb_arr = np.asarray(thumbnail, dtype=np.float32)
    heat_arr = np.asarray(heatmap, dtype=np.float32)
    if thumb_arr.shape[:2] != heat_arr.shape[:2]:
        raise ValueError("thumbnail and heatmap must have the same dimensions")

    h, w = thumb_arr.shape[:2]
    if tissue_mask is not None:
        mask_arr = _extract_mask_array(tissue_mask)
        mask = _resize_mask_to_shape(mask_arr, (h, w)).astype(np.float32)
    else:
        mask = np.ones((h, w), dtype=np.float32)

    alpha_channel = mask * float(alpha)
    overlay = thumb_arr * (1.0 - alpha_channel[..., None]) + heat_arr * alpha_channel[
        ..., None
    ]
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def _build_interpolated_heatmap(
    records: List[dict],
    width: int,
    height: int,
    mode: str,
    patch_size: int,
    grid_factor: int,
    radius_factor: float,
    sigma_factor: float,
    max_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian-kernel interpolated heatmap on a fine grid.

    Each grid cell is ``patch_size / grid_factor`` level-0 pixels.  Cells with
    no patch within ``radius_factor * patch_size`` remain NaN.  The returned
    ``valid`` mask marks cells that received a finite value.

    This mirrors the STPath ``build_interpolated_heatmap`` approach and avoids
    the hard patch-block edges that come from assigning one value per patch.
    """
    values = np.array([float(rec.get(mode, 0)) for rec in records], dtype=np.float64)
    centers = np.array(
        [
            [rec["x"] + rec["width"] / 2.0, rec["y"] + rec["height"] / 2.0]
            for rec in records
        ],
        dtype=np.float64,
    )

    cell_size = patch_size / grid_factor
    cols = max(1, int(np.ceil(width / cell_size)))
    rows = max(1, int(np.ceil(height / cell_size)))

    grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
    grid_centers = np.stack(
        [
            (grid_x.ravel() + 0.5) * cell_size,
            (grid_y.ravel() + 0.5) * cell_size,
        ],
        axis=1,
    )

    tree = cKDTree(centers)
    radius = patch_size * radius_factor
    sigma = patch_size * sigma_factor
    k = min(max_neighbors, len(values))

    if k == 0:
        heatmap = np.full((rows, cols), np.nan, dtype=np.float32)
        return heatmap, ~np.isnan(heatmap)

    distances, indices = tree.query(grid_centers, k=k, distance_upper_bound=radius)
    if k == 1:
        distances = distances[:, np.newaxis]
        indices = indices[:, np.newaxis]

    valid = (indices < len(values)) & (distances <= radius)
    weights = np.zeros_like(distances, dtype=np.float64)
    weights[valid] = np.exp(-0.5 * (distances[valid] / sigma) ** 2)

    neighbor_values = np.take(values, indices, mode="clip")
    neighbor_values = np.where(valid, neighbor_values, 0.0)

    weighted_sum = (weights * neighbor_values).sum(axis=1)
    weight_sum = weights.sum(axis=1)

    result = np.full(grid_centers.shape[0], np.nan, dtype=np.float32)
    nonzero = weight_sum > 0
    result[nonzero] = (weighted_sum[nonzero] / weight_sum[nonzero]).astype(np.float32)
    heatmap = result.reshape(rows, cols)
    valid_mask = ~np.isnan(heatmap)
    return heatmap, valid_mask


def _build_grid_heatmap(
    records: List[dict],
    width: int,
    height: int,
    mode: str,
    downsample: int,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Original patch-grid heatmap with optional Gaussian smoothing."""
    hm_h = max(1, height // downsample)
    hm_w = max(1, width // downsample)
    accumulator = np.zeros((hm_h, hm_w), dtype=np.float64)
    counter = np.zeros((hm_h, hm_w), dtype=np.uint32)

    for rec in records:
        x, y, w, h = rec["x"], rec["y"], rec["width"], rec["height"]
        value = float(rec.get(mode, 0))

        x1 = max(0, x // downsample)
        y1 = max(0, y // downsample)
        x2 = min(hm_w, (x + w + downsample - 1) // downsample)
        y2 = min(hm_h, (y + h + downsample - 1) // downsample)
        if x2 <= x1 or y2 <= y1:
            continue

        accumulator[y1:y2, x1:x2] += value
        counter[y1:y2, x1:x2] += 1

    valid = counter > 0
    heatmap = np.zeros((hm_h, hm_w), dtype=np.float32)
    heatmap[valid] = (accumulator[valid] / counter[valid]).astype(np.float32)

    if sigma > 0:
        mask = valid.astype(np.float32)
        filled = heatmap.copy()
        filled[~valid] = 0.0
        ksize = max(3, int(6 * sigma) | 1)
        blurred = cv2.GaussianBlur(filled, (ksize, ksize), sigma)
        blurred_mask = cv2.GaussianBlur(mask, (ksize, ksize), sigma)
        with np.errstate(divide="ignore", invalid="ignore"):
            smoothed = np.where(blurred_mask > 0, blurred / blurred_mask, 0.0)
        heatmap = np.nan_to_num(smoothed, nan=0.0).astype(np.float32)

    return heatmap, valid


def build_heatmap(
    records: List[dict],
    width: int,
    height: int,
    out_path: str,
    mode: str = "percent_pos",
    tile_size: int = 512,
    downsample: int | None = None,
    cmap: str = "viridis",
    vmax: float | None = None,
    sigma: float = 0.75,
    bg_color: Tuple[int, int, int] | None = (255, 255, 255),
    upscale: int | None = 2,
    max_size: int | None = None,
    tissue_mask=None,
    grid_factor: int | None = None,
    radius_factor: float = 1.5,
    sigma_factor: float = 0.75,
    max_neighbors: int = 50,
) -> None:
    """Build a heatmap from patch records.

    Two modes are supported:

    1. **Interpolated mode** (default when ``grid_factor`` is set): build a fine
       grid whose cells are ``tile_size / grid_factor`` pixels and fill each cell
       with a Gaussian-weighted average of nearby patch scores.  This produces the
       smooth, continuous appearance used by STPath and avoids visible patch blocks.
    2. **Grid mode** (legacy): assign each patch score to its footprint cell(s)
       at ``downsample`` resolution and optionally smooth with a Gaussian kernel.

    Args:
        records: Patch scoring records with ``x, y, width, height`` and *mode* value.
        width: Level-0 image width in pixels.
        height: Level-0 image height in pixels.
        out_path: Path to write the heatmap image (format inferred from extension).
        mode: Metric to visualize, e.g. ``percent_pos`` or ``num_total``.
        tile_size: Patch size used to estimate spacing in interpolated mode and
            as the fallback cell size in grid mode.
        downsample: Size in pixels of one heatmap cell in grid mode.  Default
            ``None`` uses ``tile_size``.
        cmap: Colormap name. Supported: jet, viridis, plasma, inferno, magma,
            turbo, hot, cividis, twilight. Defaults to ``viridis``.
        vmax: Fixed upper bound for color scaling.  Values above ``vmax`` are
            clipped.  If ``None``, the maximum value in the heatmap is used.
        sigma: Standard deviation for Gaussian smoothing in grid mode.  Set to
            ``0`` to disable smoothing.
        bg_color: RGB color for non-tissue background pixels.  Use ``None`` to
            keep the colormap's color for background (legacy behaviour).
        upscale: Integer upscaling factor.  If ``None`` and *max_size* is set,
            the heatmap is resized to fit within *max_size* pixels on its long
            edge.  In interpolated mode this is applied after the native grid.
        max_size: Optional target size in pixels for the longest heatmap edge.
            Only used when *upscale* is ``None``.
        tissue_mask: Optional ``TissueMask`` or binary ndarray used to clip the
            heatmap to the tissue boundary.  Non-tissue pixels are filled with
            *bg_color*.
        grid_factor: If set, use interpolated mode with this many grid cells per
            patch.  Typical values are 2–8; ``4`` matches the STPath default.
        radius_factor: Interpolation neighbor radius as a multiple of patch size.
        sigma_factor: Gaussian kernel sigma as a multiple of patch size.
        max_neighbors: Maximum number of neighbors considered per grid cell.
    """
    if upscale is not None and upscale < 1:
        raise ValueError("upscale must be >= 1")
    if max_size is not None and max_size < 1:
        raise ValueError("max_size must be >= 1")
    if grid_factor is not None and grid_factor < 1:
        raise ValueError("grid_factor must be >= 1")

    if grid_factor is not None:
        patch_size = _estimate_patch_size(records) if records else tile_size
        heatmap, valid = _build_interpolated_heatmap(
            records,
            width,
            height,
            mode,
            patch_size,
            grid_factor,
            radius_factor,
            sigma_factor,
            max_neighbors,
        )
        hm_h, hm_w = heatmap.shape
    else:
        downsample = downsample if downsample is not None else tile_size
        if downsample < 1:
            raise ValueError("downsample must be >= 1")
        heatmap, valid = _build_grid_heatmap(
            records, width, height, mode, downsample, sigma
        )
        hm_h, hm_w = heatmap.shape

    # Determine final output size.
    if upscale is not None:
        scale = float(upscale)
    elif max_size is not None:
        scale = max_size / max(hm_w, hm_h)
    else:
        scale = 1.0

    if scale != 1.0:
        out_w = max(1, int(round(hm_w * scale)))
        out_h = max(1, int(round(hm_h * scale)))
        heatmap = cv2.resize(
            heatmap, (out_w, out_h), interpolation=cv2.INTER_CUBIC
        )
        valid = _resize_mask_to_shape(valid.astype(np.uint8), (out_h, out_w))
    else:
        out_w, out_h = hm_w, hm_h

    # Clip to the tissue mask boundary if available for a clean outline.
    if tissue_mask is not None:
        mask_arr = _extract_mask_array(tissue_mask)
        mask_resized = _resize_mask_to_shape(mask_arr, (out_h, out_w))
        valid = valid & mask_resized

    # Colormap and background.
    heatmap = np.nan_to_num(heatmap, nan=0.0)
    if vmax is None:
        vmax = float(heatmap[valid].max()) if valid.any() else 1.0
    if vmax <= 0:
        vmax = 1.0

    norm = np.clip(heatmap / vmax * 255, 0, 255).astype(np.uint8)
    cmap_code = _CMAP_CODES.get(cmap.lower(), cv2.COLORMAP_VIRIDIS)
    colored = cv2.applyColorMap(norm, cmap_code)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    if bg_color is not None:
        bg_arr = np.full_like(colored, bg_color, dtype=np.uint8)
        alpha = valid.astype(np.uint8)[..., None]
        colored = (colored * alpha + bg_arr * (1 - alpha)).astype(np.uint8)

    Image.fromarray(colored).save(out_path)
