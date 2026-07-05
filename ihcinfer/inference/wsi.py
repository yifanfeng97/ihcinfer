"""High-efficiency patch-based IHC inference on whole-slide images.

This module exposes the public ``IHCAnalyzer`` class.  The actual
patch-level and region-level logic live in ``patch.py`` and ``region.py``
to keep each file focused.
"""

from __future__ import annotations

import os
import random
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..models import DeepLIIFModel, InferenceModel
from ..models.device import resolve_device
from ..outputs import (
    blend_heatmap_overlay,
    build_heatmap,
    build_patch_output,
    read_slide_thumbnail,
    save_patch_output,
    write_patch_csv,
)
from ..prep import Tiler, TissueMask
from ..readers import create_reader
from .patch import (
    MARKER_KEY,
    SEG_KEY,
    PatchInference,
    _resolution_for_patch_size,
    run_batches_adaptive,
)
from .region import RegionInference


PATCH_EXTENSIONS = (".png", ".jpg", ".jpeg")


def _collect_patch_entries(
    inputs: str | Path | list[str | Path],
) -> list[tuple[str, Image.Image]]:
    """Expand a list of file / directory paths into (stem, image) pairs."""
    if isinstance(inputs, (str, Path)):
        paths = [Path(inputs)]
    else:
        paths = [Path(p) for p in inputs]

    entries: list[tuple[str, Image.Image]] = []
    for path in paths:
        if path.is_dir():
            files = sorted(
                f
                for f in path.iterdir()
                if f.is_file() and f.suffix.lower() in PATCH_EXTENSIONS
            )
        elif path.is_file():
            files = [path]
        else:
            raise ValueError(f"Input path not found: {path}")

        for f in files:
            entries.append((f.stem, Image.open(f).convert("RGB")))
    return entries


def _unique_names(names: list[str]) -> list[str]:
    """Make names unique while preserving order and original stem first."""
    seen: set[str] = set()
    unique: list[str] = []
    for name in names:
        candidate = name
        counter = 0
        while candidate in seen:
            candidate = f"{name}_{counter}"
            counter += 1
        seen.add(candidate)
        unique.append(candidate)
    return unique


def _progress_log(msg: str, progress: bool) -> None:
    """Print a progress message to stderr when *progress* is enabled."""
    if progress:
        print(msg, file=sys.stderr, flush=True)


def _progress_tick(index: int, total: int, progress: bool) -> bool:
    """Return True for the first, last, and roughly every 10% of items."""
    if not progress or total <= 0:
        return False
    if index == 0 or index == total - 1:
        return True
    step = max(1, total // 10)
    return (index + 1) % step == 0


def _run_patches_adaptive(
    patch_infer: PatchInference,
    patches: list[Image.Image],
    *,
    batch_size: int,
    return_marker: bool,
    return_images: bool = True,
) -> list[tuple[dict, dict]]:
    """Run patch inference with adaptive batch sizing on OOM."""
    if not patches:
        return []

    resolution = _resolution_for_patch_size(patches[0].width)

    def _infer(batch: list[Image.Image]) -> list[tuple[dict, dict]]:
        return patch_infer.run(
            batch,
            resolution=resolution,
            return_marker=return_marker,
            return_images=return_images,
        )

    return run_batches_adaptive(_infer, patches, batch_size)


def _plan_patches_and_regions(
    width: int,
    height: int,
    tissue_mask: TissueMask | None,
    patch_size: int,
    region_size: int,
    tissue_min_ratio: float,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int]]]:
    """Plan all patch and complete-region coordinates from the tissue mask.

    Returns:
        - patches: list of ``(x, y, w, h)`` for every tissue patch.
        - regions: list of ``(x, y)`` top-left corners of complete regions.
    """
    patches: list[tuple[int, int, int, int]] = []
    patch_set: set[tuple[int, int]] = set()

    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            w = min(patch_size, width - x)
            h = min(patch_size, height - y)
            if tissue_mask is not None and not tissue_mask.contains_patch(
                x, y, w, h, min_ratio=tissue_min_ratio
            ):
                continue
            patches.append((x, y, w, h))
            patch_set.add((x, y))

    n = region_size // patch_size
    regions: list[tuple[int, int]] = []
    if height >= region_size and width >= region_size:
        for y in range(0, height - region_size + 1, region_size):
            for x in range(0, width - region_size + 1, region_size):
                complete = True
                for ry in range(n):
                    for rx in range(n):
                        if (x + rx * patch_size, y + ry * patch_size) not in patch_set:
                            complete = False
                            break
                    if not complete:
                        break
                if complete:
                    regions.append((x, y))

    return patches, regions


def _group_patches_by_chunk(
    patches: list[tuple[int, int, int, int]],
    chunk_size: int,
) -> dict[tuple[int, int], list[tuple[int, int, int, int]]]:
    """Group planned patch coordinates by the chunk that contains them."""
    chunk_map: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for x, y, w, h in patches:
        cx = (x // chunk_size) * chunk_size
        cy = (y // chunk_size) * chunk_size
        chunk_map.setdefault((cx, cy), []).append((x, y, w, h))
    return chunk_map


def _crop_patch(
    chunk_img: Image.Image,
    local_x: int,
    local_y: int,
    w: int,
    h: int,
    patch_size: int,
) -> Image.Image:
    """Crop a patch from a chunk and pad to ``patch_size`` if it is partial."""
    patch = chunk_img.crop((local_x, local_y, local_x + w, local_y + h))
    if patch.size != (patch_size, patch_size):
        padded = Image.new("RGB", (patch_size, patch_size), (255, 255, 255))
        padded.paste(patch, (0, 0))
        patch = padded
    return patch


class _PatchBuffer:
    """Accumulate patches across chunk boundaries and flush full batches."""

    def __init__(self, batch_size: int, patch_infer: PatchInference) -> None:
        self.batch_size = batch_size
        self.patch_infer = patch_infer
        self._buffer: list[tuple[int, int, int, int, Image.Image]] = []
        self.records: list[dict] = []
        self.scoring_cache: dict[tuple[int, int], dict] = {}

    @property
    def num_scored(self) -> int:
        """Number of patches that have already been scored."""
        return len(self.records)

    def add(self, patches: list[tuple[int, int, int, int, Image.Image]]) -> None:
        """Add patches from a chunk and flush any full batches."""
        self._buffer.extend(patches)
        while len(self._buffer) >= self.batch_size:
            self._flush(self.batch_size)

    def _flush(self, n: int) -> None:
        batch = self._buffer[:n]
        self._buffer = self._buffer[n:]
        imgs = [img for (_, _, _, _, img) in batch]
        results = _run_patches_adaptive(
            self.patch_infer,
            imgs,
            batch_size=self.batch_size,
            return_marker=False,
            return_images=False,
        )
        for (x, y, w, h, _), (_, scoring) in zip(batch, results):
            self.records.append(
                {
                    "patch_id": f"{x}_{y}",
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    **scoring,
                }
            )
            self.scoring_cache[(x, y)] = scoring

    def finalize(self) -> tuple[list[dict], dict[tuple[int, int], dict]]:
        """Flush the remaining partial batch and return all records."""
        if self._buffer:
            imgs = [img for (_, _, _, _, img) in self._buffer]
            results = _run_patches_adaptive(
                self.patch_infer,
                imgs,
                batch_size=self.batch_size,
                return_marker=False,
                return_images=False,
            )
            for (x, y, w, h, _), (_, scoring) in zip(self._buffer, results):
                self.records.append(
                    {
                        "patch_id": f"{x}_{y}",
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        **scoring,
                    }
                )
                self.scoring_cache[(x, y)] = scoring
            self._buffer.clear()
        return self.records, self.scoring_cache


@dataclass
class WSIResult:
    """Result of a whole-slide inference run."""

    slide_name: str
    csv_path: Path
    heatmap_path: Path
    thumbnail_path: Path | None = None
    overlay_path: Path | None = None
    summary: dict = field(default_factory=dict)
    records: list[dict] = field(default_factory=list)
    region_sample_paths: list[Path] = field(default_factory=list)
    patch_sample_dirs: list[Path] = field(default_factory=list)


class IHCAnalyzer:
    """Run IHC inference on a whole-slide image in batched patches."""

    def __init__(
        self,
        model_dir: str | None = None,
        *,
        model: InferenceModel | None = None,
        gpu_ids: list[int] | None = [0],
        batch_size: int = 16,
        image_format: str = "jpg",
        overlay_thickness: int = 2,
        overlay_alpha: float = 0.4,
        auto_download: bool = True,
    ) -> None:
        if model is not None and model_dir is not None:
            raise ValueError("Specify either model_dir or model, not both")
        if model is None:
            gpu_ids = gpu_ids if gpu_ids is not None else [0]
            model = DeepLIIFModel(
                model_dir,
                resolve_device(gpu_ids),
                auto_download=auto_download,
            )
            self.model_dir = getattr(model, "model_dir", None)
        else:
            self.model_dir = getattr(model, "model_dir", None)

        self.batch_size = batch_size
        self.image_format = image_format
        self.overlay_thickness = overlay_thickness
        self.overlay_alpha = overlay_alpha

        self._patch_infer = PatchInference(model=model)
        self._region_infer = RegionInference(self._patch_infer)

    @property
    def model(self) -> InferenceModel:
        return self._patch_infer.model

    def segment_tissue(self, slide_path_or_img, **kwargs) -> TissueMask:
        """Segment IHC tissue and return a level-0 aligned ``TissueMask``.

        This is a convenience wrapper that defaults to ``mode="ihc"``.
        """
        from ..prep.tissue import segment_tissue

        return segment_tissue(slide_path_or_img, mode="ihc", **kwargs)

    @staticmethod
    def _reservoir_add(
        reservoir: list,
        item: object,
        seen: int,
        max_size: int,
    ) -> None:
        """Add *item* to *reservoir* using reservoir sampling.

        *seen* is the 1-based index of the current item among all candidates.
        """
        if len(reservoir) < max_size:
            reservoir.append(item)
        else:
            j = random.randint(0, seen - 1)
            if j < max_size:
                reservoir[j] = item

    def infer_patches(
        self,
        inputs: str | Path | list[str | Path],
        *,
        output_dir: str | Path | None = None,
        save_marker: bool = False,
        overlay_thickness: int | None = None,
        image_format: str | None = None,
    ) -> dict[str, dict]:
        """Run inference on patch image files or directories.

        This is the primary patch-level entry point.  It accepts file paths or
        directory paths, runs inference, builds the standard outputs, and
        optionally saves them to disk.

        Args:
            inputs: A single file path, a single directory path, or a list of
                file / directory paths.  Directories are scanned for
                ``*.png`` / ``*.jpg`` / ``*.jpeg`` files (case-insensitive).
            output_dir: If given, outputs are saved under
                ``{output_dir}/{file_stem}/``.
            save_marker: If True, the inferred marker modality is used for
                scoring and returned/saved as ``marker.{image_format}``.  Default
                is False, which means no marker and scoring from the segmentation
                map only.
            overlay_thickness: Optional override for contour thickness.
                Defaults to ``self.overlay_thickness``.
            image_format: Optional override for saved image extension.
                Defaults to ``self.image_format``.

        Returns:
            A dictionary mapping each patch name to a result dictionary with
            three keys:

            - ``"scoring"``: the cell-count scoring dictionary.
            - ``"images"``: a dictionary of ``PIL.Image.Image`` objects.  Always
              contains ``"segmentation"`` and ``"overlay"``.  Also contains
              ``"marker"`` when *save_marker* is True.
            - ``"cells"``: a list of per-cell dictionaries (centroid,
              boundary, positive, size).  Empty for blank patches.
        """
        entries = _collect_patch_entries(inputs)
        if not entries:
            raise ValueError("No input patches found")

        overlay_thickness = (
            overlay_thickness if overlay_thickness is not None else self.overlay_thickness
        )
        image_format = image_format if image_format is not None else self.image_format

        names = _unique_names([name for name, _ in entries])
        originals = [img for _, img in entries]
        patch_size = originals[0].width
        resolution = _resolution_for_patch_size(patch_size)

        results: dict[str, dict] = {}
        batch_results: list[tuple[dict, dict]] = []
        for i in range(0, len(originals), self.batch_size):
            batch = originals[i : i + self.batch_size]
            batch_results.extend(
                self._patch_infer.run(batch, return_marker=save_marker)
            )

        for name, orig, (images, scoring) in zip(names, originals, batch_results):
            result_images: dict[str, Image.Image] = {}
            cells: list[dict[str, object]] = []

            segmentation = images.get(SEG_KEY)
            if segmentation is not None:
                marker = images.get(MARKER_KEY) if save_marker else None
                output = build_patch_output(
                    name=name,
                    original=orig,
                    segmentation=segmentation,
                    marker=marker,
                    overlay_thickness=overlay_thickness,
                    resolution=resolution,
                )

                if output_dir is not None:
                    save_patch_output(
                        output,
                        Path(output_dir) / name,
                        image_format=image_format,
                        save_marker=save_marker,
                    )

                result_images[SEG_KEY] = output.segmentation
                result_images["overlay"] = output.overlay
                if marker is not None:
                    result_images[MARKER_KEY] = marker

                cells = output.cells

            results[name] = {"scoring": scoring, "images": result_images, "cells": cells}

        return results

    def _run_on_image_patches(
        self,
        patches: list[Image.Image],
        return_marker: bool = False,
    ) -> list[tuple[dict, dict]]:
        """Run inference on a list of equal-sized RGB patches (internal use).

        Most users should call :meth:`infer_patches` instead.
        """
        return self._patch_infer.run(patches, return_marker=return_marker)

    def infer_region(
        self,
        region: Image.Image | np.ndarray,
        x_offset: int = 0,
        y_offset: int = 0,
        patch_size: int = 512,
        overlap_size: int = 32,
        tissue_mask: TissueMask | None = None,
        tissue_min_ratio: float = 0.01,
        save_masks: bool = False,
        patch_output_dir: str | None = None,
        stitch_outputs: bool = True,
        image_format: str | None = None,
    ) -> tuple[list[dict], dict[str, Image.Image] | None]:
        """Infer one WSI region and return patch records + optional stitched outputs."""
        image_format = image_format if image_format is not None else self.image_format
        return self._region_infer.run(
            region,
            x_offset=x_offset,
            y_offset=y_offset,
            patch_size=patch_size,
            overlap_size=overlap_size,
            batch_size=self.batch_size,
            tissue_mask=tissue_mask,
            tissue_min_ratio=tissue_min_ratio,
            save_masks=save_masks,
            patch_output_dir=patch_output_dir,
            stitch_outputs=stitch_outputs,
            image_format=image_format,
        )

    def _visualize_regions(
        self,
        region_coords: list[tuple[int, int]],
        slide_path: str,
        output_dir: Path,
        region_size: int,
        patch_size: int,
        image_format: str,
        save_marker: bool,
    ) -> list[Path]:
        """Re-read selected regions and save segmentation + overlay samples.

        Each region is written into its own coordinate-named subdirectory:
        ``region_samples/{rx}_{ry}/segmentation.{fmt}`` and ``overlay.{fmt}``.
        """
        if not region_coords:
            return []

        region_paths: list[Path] = []
        resolution = _resolution_for_patch_size(patch_size)
        region_samples_dir = output_dir / "region_samples"
        region_samples_dir.mkdir(exist_ok=True)

        with create_reader(slide_path) as reader:
            for rx, ry in region_coords:
                region_arr = reader.read((rx, ry, region_size, region_size))
                region_img = Image.fromarray(region_arr)
                tiler = Tiler(region_img, tile_size=patch_size, overlap_size=0)

                seg_canvas = Image.new("RGB", (region_size, region_size))
                overlay_canvas = Image.new("RGB", (region_size, region_size))
                tiles = list(tiler)

                for i in range(0, len(tiles), self.batch_size):
                    batch_tiles = tiles[i : i + self.batch_size]
                    imgs = [tile.img for tile in batch_tiles]
                    results = self._patch_infer.run(
                        imgs, resolution=resolution, return_marker=save_marker
                    )
                    for tile, (images, _) in zip(batch_tiles, results):
                        segmentation = images.get(SEG_KEY)
                        if segmentation is None:
                            continue
                        marker = images.get(MARKER_KEY) if save_marker else None
                        output = build_patch_output(
                            name=f"{rx + tile.abs_cx}_{ry + tile.abs_cy}",
                            original=tile.img,
                            segmentation=segmentation,
                            marker=marker,
                            overlay_thickness=self.overlay_thickness,
                            resolution=resolution,
                        )
                        seg_canvas.paste(output.segmentation, (tile.abs_cx, tile.abs_cy))
                        overlay_canvas.paste(output.overlay, (tile.abs_cx, tile.abs_cy))

                region_dir = region_samples_dir / f"{rx}_{ry}"
                region_dir.mkdir(exist_ok=True)
                seg_path = region_dir / f"segmentation.{image_format}"
                overlay_path = region_dir / f"overlay.{image_format}"
                seg_canvas.save(seg_path, quality=95)
                overlay_canvas.save(overlay_path, quality=95)
                region_paths.append(seg_path)
                region_paths.append(overlay_path)

        return region_paths

    def _visualize_patches(
        self,
        patch_coords: list[tuple[int, int]],
        slide_path: str,
        output_dir: Path,
        patch_size: int,
        image_format: str,
        save_marker: bool,
    ) -> list[Path]:
        """Re-read selected patches and save standard patch outputs."""
        if not patch_coords:
            return []

        patch_dirs: list[Path] = []
        resolution = _resolution_for_patch_size(patch_size)
        patch_samples_dir = output_dir / "patch_samples"
        patch_samples_dir.mkdir(exist_ok=True)

        with create_reader(slide_path) as reader:
            for px, py in patch_coords:
                patch_arr = reader.read((px, py, patch_size, patch_size))
                patch_img = Image.fromarray(patch_arr)
                images, _ = self._patch_infer.run(
                    [patch_img], resolution=resolution, return_marker=save_marker
                )[0]
                segmentation = images.get(SEG_KEY)
                if segmentation is None:
                    continue

                marker = images.get(MARKER_KEY) if save_marker else None
                output = build_patch_output(
                    name=f"{px}_{py}",
                    original=patch_img,
                    segmentation=segmentation,
                    marker=marker,
                    overlay_thickness=self.overlay_thickness,
                    resolution=resolution,
                )
                patch_dir = patch_samples_dir / f"{px}_{py}"
                save_patch_output(
                    output,
                    patch_dir,
                    image_format=image_format,
                    save_marker=save_marker,
                )
                patch_dirs.append(patch_dir)

        return patch_dirs

    def infer_wsi(
        self,
        slide_path: str | Path,
        output_dir: str | Path,
        *,
        patch_size: int = 512,
        chunk_size: int = 8192,
        region_size: int = 2048,
        tissue_min_ratio: float = 0.05,
        num_region_samples: int = 2,
        num_patch_samples: int = 4,
        save_marker_in_samples: bool = False,
        image_format: str | None = None,
        heatmap_cmap: str = "viridis",
        heatmap_vmax: float | None = 50.0,
        heatmap_sigma: float = 0.75,
        heatmap_upscale: int | None = None,
        heatmap_max_size: int = 1024,
        heatmap_grid_factor: int = 4,
        skip_thumbnail: bool = False,
        overlay_alpha: float | None = None,
        progress: bool = False,
    ) -> WSIResult:
        """Run inference on a whole-slide image and produce CSV/heatmap/samples."""
        if region_size % patch_size != 0:
            raise ValueError("region_size must be a multiple of patch_size")

        slide_path = str(slide_path)
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        image_format = image_format if image_format is not None else self.image_format
        basename = Path(slide_path).stem

        _progress_log(f"Segmenting tissue from {slide_path}...", progress)
        tissue_mask = self.segment_tissue(slide_path)

        with create_reader(slide_path) as reader:
            width, height = reader.width, reader.height

            patches, regions = _plan_patches_and_regions(
                width,
                height,
                tissue_mask,
                patch_size,
                region_size,
                tissue_min_ratio,
            )
            chunk_map = _group_patches_by_chunk(patches, chunk_size)

            _progress_log(
                f"Slide dimensions: {width}x{height}; planned {len(patches)} patches "
                f"across {len(regions)} complete regions.",
                progress,
            )

            # Preselect random regions for visualization before any inference.
            region_sample_coords: list[tuple[int, int]] = []
            if regions and num_region_samples > 0:
                k = min(num_region_samples, len(regions))
                region_sample_coords = random.sample(regions, k=k)

            patch_reservoir: list[tuple[int, int]] = []
            patch_seen = 0
            buffer = _PatchBuffer(self.batch_size, self._patch_infer)
            num_chunks = len(chunk_map)

            _progress_log(
                f"Running inference on {num_chunks} chunks (batch_size={self.batch_size})...",
                progress,
            )

            for chunk_idx, (cx, cy) in enumerate(sorted(chunk_map.keys())):
                cw = min(chunk_size, width - cx)
                ch = min(chunk_size, height - cy)
                chunk_arr = reader.read((cx, cy, cw, ch))
                chunk_img = Image.fromarray(chunk_arr)

                chunk_patches: list[tuple[int, int, int, int, Image.Image]] = []
                for px, py, pw, ph in chunk_map[(cx, cy)]:
                    patch_img = _crop_patch(
                        chunk_img, px - cx, py - cy, pw, ph, patch_size
                    )
                    chunk_patches.append((px, py, pw, ph, patch_img))

                    if pw == patch_size and ph == patch_size:
                        patch_seen += 1
                        self._reservoir_add(
                            patch_reservoir, (px, py), patch_seen, num_patch_samples
                        )

                buffer.add(chunk_patches)
                if _progress_tick(chunk_idx, num_chunks, progress):
                    _progress_log(
                        f"  chunk {chunk_idx + 1}/{num_chunks} @ ({cx}, {cy}) — "
                        f"{len(chunk_patches)} patches, {buffer.num_scored} scored so far",
                        progress,
                    )

            all_records, scoring_cache = buffer.finalize()

        _progress_log(
            f"Inference complete: {len(all_records)} patches scored.", progress
        )

        csv_path = output_dir / "patch_scoring.csv"
        write_patch_csv(all_records, str(csv_path))
        _progress_log(f"CSV saved: {csv_path}", progress)

        # Build the heatmap at patch-grid resolution so each scored patch
        # becomes one cell; resize to ~2048 px on the long edge for a smooth,
        # Nature-style presentation without visible tile blocks.
        heatmap_path = output_dir / f"heatmap.{image_format}"
        build_heatmap(
            all_records,
            width,
            height,
            str(heatmap_path),
            patch_size=patch_size,
            downsample=patch_size,
            cmap=heatmap_cmap,
            vmax=heatmap_vmax,
            sigma=heatmap_sigma,
            upscale=heatmap_upscale,
            max_size=heatmap_max_size,
            tissue_mask=tissue_mask,
            grid_factor=heatmap_grid_factor,
        )

        _progress_log(f"Heatmap saved: {heatmap_path}", progress)

        thumbnail_path: Path | None = None
        overlay_path: Path | None = None
        if not skip_thumbnail:
            try:
                with Image.open(heatmap_path) as heatmap_img:
                    target_size = heatmap_img.size
                    thumbnail_path = output_dir / f"he_thumbnail.{image_format}"
                    thumb = read_slide_thumbnail(slide_path, target_size)
                    thumb.save(thumbnail_path, quality=95)

                    overlay_path = output_dir / f"overlay.{image_format}"
                    alpha = (
                        overlay_alpha
                        if overlay_alpha is not None
                        else self.overlay_alpha
                    )
                    overlay = blend_heatmap_overlay(
                        thumb,
                        heatmap_img,
                        alpha=alpha,
                        tissue_mask=tissue_mask,
                    )
                    overlay.save(overlay_path, quality=95)
                    _progress_log(
                        f"Thumbnail saved: {thumbnail_path}; overlay saved: {overlay_path}",
                        progress,
                    )
            except Exception as exc:
                warnings.warn(
                    f"Thumbnail/overlay generation failed: {exc}", stacklevel=2
                )

        region_paths = self._visualize_regions(
            region_sample_coords,
            slide_path,
            output_dir,
            region_size,
            patch_size,
            image_format,
            save_marker_in_samples,
        )
        _progress_log(
            f"Region samples saved: {len(region_paths) // 2}", progress
        )

        patch_dirs = self._visualize_patches(
            patch_reservoir,
            slide_path,
            output_dir,
            patch_size,
            image_format,
            save_marker_in_samples,
        )
        _progress_log(f"Patch samples saved: {len(patch_dirs)}", progress)

        summary = {
            "num_total": sum(r.get("num_total", 0) for r in all_records),
            "num_pos": sum(r.get("num_pos", 0) for r in all_records),
            "num_neg": sum(r.get("num_neg", 0) for r in all_records),
        }
        total = summary["num_total"]
        summary["percent_pos"] = (
            100.0 * summary["num_pos"] / total if total > 0 else 0.0
        )

        _progress_log(
            f"Summary — total cells: {summary['num_total']}, "
            f"positive: {summary['num_pos']}, negative: {summary['num_neg']}, "
            f"percent_pos: {summary['percent_pos']:.2f}%",
            progress,
        )

        return WSIResult(
            slide_name=basename,
            csv_path=csv_path,
            heatmap_path=heatmap_path,
            thumbnail_path=thumbnail_path,
            overlay_path=overlay_path,
            summary=summary,
            records=all_records,
            region_sample_paths=region_paths,
            patch_sample_dirs=patch_dirs,
        )
