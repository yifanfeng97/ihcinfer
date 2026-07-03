"""Region-level DeepLIIF inference with tiling and stitching."""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
from PIL import Image

from ..outputs import save_patch_outputs
from ..prep import Tiler, TissueMask, is_blank_patch
from .patch import PatchInference, _run_batches_adaptive


class RegionInference:
    """Run inference on one WSI region in batched tiles."""

    def __init__(self, patch_infer: PatchInference) -> None:
        self.patch_infer = patch_infer

    def run(
        self,
        region: Image.Image | np.ndarray,
        x_offset: int = 0,
        y_offset: int = 0,
        tile_size: int = 512,
        overlap_size: int = 32,
        batch_size: int = 16,
        tissue_mask: TissueMask | None = None,
        tissue_min_ratio: float = 0.01,
        save_masks: bool = False,
        patch_output_dir: str | None = None,
        stitch_outputs: bool = True,
        image_format: str = "jpg",
    ) -> Tuple[List[dict], dict[str, Image.Image] | None]:
        """Infer one region and return patch records + optional stitched outputs."""
        if isinstance(region, np.ndarray):
            region = Image.fromarray(region)

        tiler = Tiler(region, tile_size=tile_size, overlap_size=overlap_size)

        filtered_tiles = []
        for tile in tiler:
            if tissue_mask is not None and not tissue_mask.contains_patch(
                x_offset + tile.abs_cx,
                y_offset + tile.abs_cy,
                tile.cw,
                tile.ch,
                min_ratio=tissue_min_ratio,
            ):
                continue
            if is_blank_patch(tile.img):
                continue
            filtered_tiles.append(tile)

        records: List[dict] = []
        if filtered_tiles:
            imgs = [tile.img for tile in filtered_tiles]

            def _infer(batch: List[Image.Image]) -> List[Tuple[dict, dict]]:
                return self.patch_infer.run(batch)

            all_results = _run_batches_adaptive(_infer, imgs, batch_size)

            for tile, (raw, scoring) in zip(filtered_tiles, all_results):
                record = {
                    "patch_id": f"{x_offset}_{y_offset}_{tile.abs_cx}_{tile.abs_cy}",
                    "x": x_offset + tile.abs_cx,
                    "y": y_offset + tile.abs_cy,
                    "width": tile.cw,
                    "height": tile.ch,
                    **scoring,
                }
                records.append(record)
                if save_masks and patch_output_dir:
                    patch_dir = os.path.join(
                        patch_output_dir,
                        f"patch_{x_offset}_{y_offset}_{tile.abs_cx}_{tile.abs_cy}",
                    )
                    os.makedirs(patch_dir, exist_ok=True)
                    save_patch_outputs(record["patch_id"], raw, patch_dir, image_format=image_format)
                if stitch_outputs:
                    tiler.stitch(tile, raw)

        region_results = tiler.results() if (records and stitch_outputs) else None
        return records, region_results
