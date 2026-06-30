"""Benchmark: ihcinfer region inference vs. original DeepLIIF sequential tiles.

Run with:
    uv run python benchmarks/bench_region_inference.py
"""

from __future__ import annotations

import time

import torch
from PIL import Image

from deepliif.models import find_marker_key, get_opt, init_nets, run_dask_batch
from deepliif.postprocessing import compute_final_results
from ihcinfer import SlideInference
from ihcinfer.prep import TissueSegmenter
from ihcinfer.readers import create_reader

MODEL_DIR = "/home/fengyifan/disk/code/DeepLIIF/model-server/DeepLIIF_Latest_Model"
SVS = "tests/data/slides/98140-6 CD3.svs"
CPU = torch.device("cpu")
TILE_SIZE = 512
REGION_SIZE = 1024


def find_first_tissue_region(mask, slide_width, slide_height, region_size: int = REGION_SIZE):
    for y in range(0, slide_height, region_size):
        for x in range(0, slide_width, region_size):
            w = min(region_size, slide_width - x)
            h = min(region_size, slide_height - y)
            if mask.contains_patch(x, y, w, h, min_ratio=0.05):
                return x, y, w, h
    raise RuntimeError("No tissue region found")


def tile_region(region_img: Image.Image, tile_size: int = TILE_SIZE):
    tiles = []
    for y in range(0, region_img.height, tile_size):
        for x in range(0, region_img.width, tile_size):
            crop = region_img.crop((x, y, min(x + tile_size, region_img.width), min(y + tile_size, region_img.height)))
            if crop.width < tile_size or crop.height < tile_size:
                canvas = Image.new("RGB", (tile_size, tile_size), (255, 255, 255))
                canvas.paste(crop, (0, 0))
                crop = canvas
            tiles.append((x, y, crop))
    return tiles


def bench_fast_region(region_img, tissue_mask, x, y, tissue_min_ratio: float = 0.0):
    inf = SlideInference(model_dir=MODEL_DIR, gpu_ids=[], batch_size=4)
    t0 = time.perf_counter()
    records, _ = inf.run_on_region(
        region_img,
        x_offset=x,
        y_offset=y,
        tile_size=TILE_SIZE,
        overlap_size=0,
        tissue_mask=tissue_mask,
        tissue_min_ratio=tissue_min_ratio,
    )
    elapsed = time.perf_counter() - t0
    return elapsed, len(records)


def bench_original_sequential(region_img, x_offset, y_offset):
    opt = get_opt(MODEL_DIR)
    opt.use_dp = False
    nets = init_nets(MODEL_DIR, eager_mode=False, opt=opt, force_device=CPU)
    tiles = tile_region(region_img)
    seg_key = f"G{opt.mod_id_seg}"

    t0 = time.perf_counter()
    for _, _, tile in tiles:
        raw = run_dask_batch([tile], nets=nets, opt=opt, seg_only=True)
        marker_key = find_marker_key(raw[0])
        compute_final_results(
            tile,
            raw[0].get(seg_key),
            raw[0].get(marker_key) if marker_key else None,
            resolution="40x",
        )
    elapsed = time.perf_counter() - t0
    return elapsed, len(tiles)


def main():
    print("Segmenting tissue...")
    tissue_mask = TissueSegmenter(seg_level="auto").segment(SVS)

    with create_reader(SVS) as reader:
        x, y, w, h = find_first_tissue_region(tissue_mask, reader.width, reader.height, REGION_SIZE)
        region_arr = reader.read((x, y, w, h))
        region_img = Image.fromarray(region_arr)

    print(f"Region: ({x}, {y}, {w}, {h}) from {SVS}\n")

    fast_elapsed_all, fast_records_all = bench_fast_region(region_img, tissue_mask, x, y, tissue_min_ratio=0.0)
    print(f"ihcinfer run_on_region ({fast_records_all} tiles, no tissue filter): {fast_elapsed_all:.2f}s")

    fast_elapsed_filt, fast_records_filt = bench_fast_region(region_img, tissue_mask, x, y, tissue_min_ratio=0.05)
    print(f"ihcinfer run_on_region ({fast_records_filt} tiles, min_ratio=0.05): {fast_elapsed_filt:.2f}s")

    orig_elapsed, orig_tiles = bench_original_sequential(region_img, x, y)
    print(f"DeepLIIF original sequential ({orig_tiles} tiles): {orig_elapsed:.2f}s")

    print("\n--- Summary ---")
    print(f"Raw throughput (same #tiles): {orig_elapsed/fast_elapsed_all:.2f}x")
    print(f"With tissue filtering (fewer tiles): {orig_elapsed/fast_elapsed_filt:.2f}x")


if __name__ == "__main__":
    main()
