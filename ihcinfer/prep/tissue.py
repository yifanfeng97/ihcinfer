"""Tissue segmentation mask and H&E/IHC segmenter."""

from __future__ import annotations

import os
from typing import List, Tuple

import cv2
import numpy as np
import openslide
from PIL import Image


class TissueMask:
    """Binary tissue mask with efficient patch-membership checks."""

    def __init__(
        self,
        mask: np.ndarray,
        scale: Tuple[float, float],
        contours: List[np.ndarray],
        holes: List[List[np.ndarray]],
    ) -> None:
        self.mask = mask.astype(np.uint8)
        self.scale = scale
        self.contours = contours
        self.holes = holes

    def contains_patch(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        min_ratio: float = 0.05,
    ) -> bool:
        """Return True if at least ``min_ratio`` of the patch box is tissue."""
        sx, sy = self.scale
        mh, mw = self.mask.shape
        x1 = max(0, min(mw, int(x / sx)))
        y1 = max(0, min(mh, int(y / sy)))
        x2 = max(0, min(mw, int((x + w) / sx)))
        y2 = max(0, min(mh, int((y + h) / sy)))
        region = self.mask[y1:y2, x1:x2]
        if region.size == 0:
            return False
        return bool(region.mean() >= min_ratio)

    def contains_center(self, x: int, y: int) -> bool:
        sx, sy = self.scale
        mx = int(x / sx)
        my = int(y / sy)
        if 0 <= my < self.mask.shape[0] and 0 <= mx < self.mask.shape[1]:
            return bool(self.mask[my, mx])
        return False

    def bbox(self) -> tuple[int, int, int, int]:
        """Return the bounding box of the tissue mask in level-0 coordinates."""
        ys, xs = np.where(self.mask > 0)
        if len(xs) == 0:
            return (0, 0, 0, 0)
        sx, sy = self.scale
        x_min = int(xs.min() * sx)
        y_min = int(ys.min() * sy)
        x_max = int(xs.max() * sx)
        y_max = int(ys.max() * sy)
        return (x_min, y_min, x_max, y_max)


class TissueSegmenter:
    """Tissue segmentation with IHC or H&E-optimized pipelines."""

    def __init__(
        self,
        seg_level: int | str = "auto",
        mode: str = "ihc",
        sthresh: int = 20,
        sthresh_up: int = 255,
        mthresh: int = 7,
        close: int = 0,
        ihc_close: int = 7,
        use_otsu: bool = False,
        filter_params: dict | None = None,
        ref_patch_size: int = 512,
        white_v_thresh: int = 240,
        white_s_thresh: int = 20,
        black_v_thresh: int = 15,
    ) -> None:
        if mode not in ("he", "ihc"):
            raise ValueError("mode must be 'he' or 'ihc'")
        self.seg_level = seg_level
        self.mode = mode
        self.sthresh = sthresh
        self.sthresh_up = sthresh_up
        self.mthresh = mthresh
        self.close = close
        self.ihc_close = ihc_close
        self.use_otsu = use_otsu
        self.filter_params = filter_params or {"a_t": 100}
        self.ref_patch_size = ref_patch_size
        self.white_v_thresh = white_v_thresh
        self.white_s_thresh = white_s_thresh
        self.black_v_thresh = black_v_thresh

    @staticmethod
    def _filter_contours(contours, hierarchy, filter_params):
        hierarchy = np.squeeze(hierarchy, axis=0)[:, 2:]
        parent_idx = np.flatnonzero(hierarchy[:, 1] == -1)
        filtered = []
        all_holes = []

        a_t = filter_params["a_t"]
        a_h = filter_params.get("a_h", a_t * 0.05)
        max_n_holes = filter_params.get("max_n_holes", 10)

        for cont_idx in parent_idx:
            holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
            area = cv2.contourArea(contours[cont_idx])
            area -= sum(cv2.contourArea(contours[h]) for h in holes)
            if area == 0:
                continue
            if area > a_t:
                filtered.append(cont_idx)
                all_holes.append(holes)

        foreground = [contours[i] for i in filtered]
        hole_contours = []
        for holes in all_holes:
            unfiltered = sorted(
                [contours[h] for h in holes], key=cv2.contourArea, reverse=True
            )[:max_n_holes]
            kept = [h for h in unfiltered if cv2.contourArea(h) > a_h]
            hole_contours.append(kept)

        return foreground, hole_contours

    def _scaled_filter_params(self, scale: tuple[float, float]) -> dict:
        """Return filter params with area thresholds expressed in mask pixels.

        ``filter_params['a_t']`` is interpreted as a number of
        ``ref_patch_size x ref_patch_size`` regions at level 0.  The threshold
        is converted to mask-resolution pixels by multiplying with the area of
        one such region at the downsampled segmentation level.
        """
        if scale[0] <= 1.0 and scale[1] <= 1.0:
            ref_patch_area_at_mask = 1
        else:
            ref_patch_area_at_mask = int(self.ref_patch_size**2 / (scale[0] * scale[1]))
        filter_params = self.filter_params.copy()
        filter_params["a_t"] = filter_params.get("a_t", 100) * ref_patch_area_at_mask
        filter_params.setdefault("a_h", filter_params["a_t"] * 0.05)
        filter_params.setdefault("max_n_holes", 10)
        return filter_params

    def _build_mask(
        self,
        img: np.ndarray,
        scale: tuple[float, float],
        binary: np.ndarray,
        filter_params: dict,
    ) -> TissueMask:
        """Convert a binary tissue mask into a level-0 aligned ``TissueMask``."""
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            empty = np.zeros(
                (img.shape[0], img.shape[1]),
                dtype=np.uint8,
            )
            return TissueMask(empty, scale, [], [])

        foreground, holes = self._filter_contours(contours, hierarchy, filter_params)

        scaled_foreground = [
            (cont * scale).astype(np.int32) for cont in foreground
        ]
        scaled_holes = [
            [(hole * scale).astype(np.int32) for hole in hole_list]
            for hole_list in holes
        ]

        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        cv2.drawContours(mask, foreground, -1, 1, -1)
        for hole_list in holes:
            cv2.drawContours(mask, hole_list, -1, 0, -1)

        return TissueMask(mask, scale, scaled_foreground, scaled_holes)

    def _segment_he(self, img: np.ndarray, scale: tuple[float, float]) -> TissueMask:
        """H&E-style HSV saturation thresholding."""
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        sat = cv2.medianBlur(hsv[:, :, 1], self.mthresh)

        if self.use_otsu:
            _, binary = cv2.threshold(
                sat, 0, self.sthresh_up, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            _, binary = cv2.threshold(
                sat, self.sthresh, self.sthresh_up, cv2.THRESH_BINARY
            )

        if self.close > 0:
            kernel = np.ones((self.close, self.close), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return self._build_mask(img, scale, binary, self._scaled_filter_params(scale))

    def _segment_ihc(self, img: np.ndarray, scale: tuple[float, float]) -> TissueMask:
        """IHC tissue mask: non-white, non-black pixels are treated as tissue.

        The glass background in IHC slides is bright and nearly colourless, so we
        define background in HSV as pixels with high Value and low Saturation.
        Some scanners (e.g. MIRAX .mrxs) also fill out-of-scan areas with black,
        which is treated as background as well.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        sure_bg = (
            ((hsv[:, :, 2] >= self.white_v_thresh) & (hsv[:, :, 1] <= self.white_s_thresh))
            | (hsv[:, :, 2] <= self.black_v_thresh)
        )
        binary = (~sure_bg).astype(np.uint8) * 255

        if self.ihc_close > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.ihc_close, self.ihc_close)
            )
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return self._build_mask(img, scale, binary, self._scaled_filter_params(scale))

    def segment(self, slide_path_or_img) -> TissueMask:
        """Segment tissue and return a level-0 aligned ``TissueMask``."""
        if isinstance(slide_path_or_img, np.ndarray):
            img = np.asarray(slide_path_or_img)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            scale = (1.0, 1.0)
        elif isinstance(slide_path_or_img, Image.Image):
            img = np.array(slide_path_or_img.convert("RGB"))
            scale = (1.0, 1.0)
        elif isinstance(slide_path_or_img, (str, os.PathLike)) and any(
            str(slide_path_or_img).lower().endswith(ext)
            for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        ):
            img = np.array(Image.open(str(slide_path_or_img)).convert("RGB"))
            scale = (1.0, 1.0)
        else:
            slide = openslide.OpenSlide(str(slide_path_or_img))
            level = (
                slide.get_best_level_for_downsample(32)
                if self.seg_level == "auto"
                else int(self.seg_level)
            )
            dims = slide.level_dimensions[level]
            img = np.array(slide.read_region((0, 0), level, dims).convert("RGB"))
            scale = (
                slide.dimensions[0] / dims[0],
                slide.dimensions[1] / dims[1],
            )
            slide.close()

        if self.mode == "ihc":
            return self._segment_ihc(img, scale)
        return self._segment_he(img, scale)


def segment_tissue(
    slide_path_or_img,
    *,
    mode: str = "ihc",
    seg_level: int | str = "auto",
    **kwargs,
) -> TissueMask:
    """Segment tissue and return a level-0 aligned ``TissueMask``.

    This is a convenience wrapper around :class:`TissueSegmenter`.  The default
    ``mode="ihc"`` treats non-white pixels as tissue, which works well for
    IHC whole-slide images with bright glass backgrounds.  Use ``mode="he"``
    for H&E-style saturation thresholding.

    Args:
        slide_path_or_img: WSI file path, image file path, ``PIL.Image``, or
            ``np.ndarray``.
        mode: ``"ihc"`` (default) or ``"he"``.
        seg_level: Pyramid level for WSI segmentation, or ``"auto"``.
        **kwargs: Additional arguments forwarded to ``TissueSegmenter``.

    Returns:
        A :class:`TissueMask` instance.
    """
    return TissueSegmenter(seg_level=seg_level, mode=mode, **kwargs).segment(
        slide_path_or_img
    )
