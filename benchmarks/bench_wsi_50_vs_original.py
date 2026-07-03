"""Benchmark: ihcinfer WSI scoring-only path vs original DeepLIIF sequential.

Run with:
    uv run python outputs/bench_wsi_50_vs_original.py
"""

from __future__ import annotations

import time

import numpy as np
import torch
from PIL import Image

from deepliif.models import find_marker_key, get_opt, init_nets, run_dask_batch
from deepliif.postprocessing import compute_final_results
from ihcinfer import IHCAnalyzer
from ihcinfer.prep import TissueSegmenter
from ihcinfer.readers import create_reader

MODEL_DIR = "/home/fengyifan/disk/code/DeepLIIF/model-server/DeepLIIF_Latest_Model"
SVS = "tests/data/slides/98140-6 CD3.svs"
PATCH_SIZE = 512
NUM_PATCHES = 50


def sample_tissue_patches(slide_path: str, num_patches: int = NUM_PATCHES):
    print("Segmenting tissue...")
    tissue_mask = TissueSegmenter(seg_level="auto").segment(slide_path)

    with create_reader(slide_path) as reader:
        width, height = reader.width, reader.height
        patches = []
        coords = []
        for y in range(0, height - PATCH_SIZE + 1, PATCH_SIZE):
            for x in range(0, width - PATCH_SIZE + 1, PATCH_SIZE):
                if not tissue_mask.contains_patch(x, y, PATCH_SIZE, PATCH_SIZE, min_ratio=0.05):
                    continue
                patches.append((x, y))
                if len(patches) >= num_patches:
                    break
            if len(patches) >= num_patches:
                break

        images = []
        for x, y in patches[:num_patches]:
            arr = reader.read((x, y, PATCH_SIZE, PATCH_SIZE))
            images.append(Image.fromarray(arr))

    print(f"Sampled {len(images)} tissue patches from {slide_path}\n")
    return images


def bench_original_sequential(patches):
    device = torch.device("cuda:0")
    opt = get_opt(MODEL_DIR)
    opt.use_dp = False
    nets = init_nets(MODEL_DIR, eager_mode=False, opt=opt, force_device=device)
    seg_key = f"G{opt.mod_id_seg}"

    t0 = time.perf_counter()
    for img in patches:
        raw = run_dask_batch([img], nets=nets, opt=opt, seg_only=True)
        mk = find_marker_key(raw[0])
        compute_final_results(
            img,
            raw[0].get(seg_key),
            raw[0].get(mk) if mk else None,
            resolution="40x",
        )
    elapsed = time.perf_counter() - t0
    return elapsed


def bench_ihcinfer_scoring_only(patches):
    inf = IHCAnalyzer(model_dir=MODEL_DIR, gpu_ids=[0], batch_size=16)
    t0 = time.perf_counter()
    inf._patch_infer.run(patches, return_images=False)
    elapsed = time.perf_counter() - t0
    return elapsed


def bench_ihcinfer_full(patches):
    inf = IHCAnalyzer(model_dir=MODEL_DIR, gpu_ids=[0], batch_size=16)
    t0 = time.perf_counter()
    inf._run_on_image_patches(patches)
    elapsed = time.perf_counter() - t0
    return elapsed


def main():
    patches = sample_tissue_patches(SVS, NUM_PATCHES)
    n = len(patches)

    t_orig = bench_original_sequential(patches)
    t_fast_scoring = bench_ihcinfer_scoring_only(patches)
    t_fast_full = bench_ihcinfer_full(patches)

    print(f"Original DeepLIIF sequential full pipeline ({n} patches): {t_orig:.2f}s ({t_orig/n:.2f}s/patch)")
    print(f"ihcinfer scoring-only ({n} patches):                    {t_fast_scoring:.2f}s ({t_fast_scoring/n:.2f}s/patch)")
    print(f"ihcinfer full image-returning ({n} patches):            {t_fast_full:.2f}s ({t_fast_full/n:.2f}s/patch)")

    print("\n--- Summary ---")
    print(f"ihcinfer scoring-only vs original full pipeline: {t_orig/t_fast_scoring:.2f}x")
    print(f"ihcinfer full vs original full pipeline:         {t_orig/t_fast_full:.2f}x")

    # Extrapolate to the full 1453 tissue patches count from the WSI benchmark.
    full_count = 1453
    print(f"\nExtrapolated to {full_count} patches:")
    print(f"  Original full pipeline: ~{t_orig/n*full_count:.0f}s ({t_orig/n*full_count/60:.1f}min)")
    print(f"  ihcinfer scoring-only (WSI core): ~{t_fast_scoring/n*full_count:.0f}s ({t_fast_scoring/n*full_count/60:.1f}min)")


if __name__ == "__main__":
    main()
