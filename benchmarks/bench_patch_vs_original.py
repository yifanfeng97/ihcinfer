"""Benchmark: ihcinfer patch inference vs. original DeepLIIF.

Run with:
    PYTHONPATH=/path/to/DeepLIIF uv run python benchmarks/bench_patch_vs_original.py [--device cpu|cuda:0]

If --model_dir is omitted, the pretrained DeepLIIF model is downloaded automatically
on first use.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from PIL import Image

from deepliif.models import find_marker_key, get_opt, init_nets, run_dask_batch
from deepliif.postprocessing import compute_final_results
from ihcinfer import IHCAnalyzer
from ihcinfer.models import DeepLIIFModel

PATCH_DIR = Path(__file__).resolve().parent.parent / "tests" / "data" / "patches"


def load_patches(n: int = 4):
    paths = sorted(PATCH_DIR.glob("*.png"))[:n]
    return [Image.open(p).convert("RGB") for p in paths]


def load_original(model_dir: str | None, device):
    """Load original DeepLIIF nets and return (opt, nets, load_time)."""
    opt = get_opt(model_dir)
    opt.use_dp = False
    t0 = time.perf_counter()
    nets = init_nets(model_dir, eager_mode=False, opt=opt, force_device=device)
    load_time = time.perf_counter() - t0
    return opt, nets, load_time


def load_custom(model_dir: str | None, device):
    """Load custom ihcinfer model and return (model, load_time)."""
    t0 = time.perf_counter()
    model = DeepLIIFModel(model_dir, device)
    load_time = time.perf_counter() - t0
    return model, load_time


def bench_original_raw(patches, opt, nets):
    """Original DeepLIIF: raw inference only (model already loaded)."""
    t0 = time.perf_counter()
    run_dask_batch(patches, nets=nets, opt=opt, seg_only=True)
    return time.perf_counter() - t0


def bench_original_full(patches, opt, nets):
    """Original DeepLIIF: inference + postprocessing (model already loaded)."""
    seg_key = f"G{opt.mod_id_seg}"
    t0 = time.perf_counter()
    raw_results = run_dask_batch(patches, nets=nets, opt=opt, seg_only=True)
    for img, raw in zip(patches, raw_results):
        mk = find_marker_key(raw)
        compute_final_results(
            img,
            raw.get(seg_key),
            raw.get(mk) if mk else None,
            resolution="40x",
        )
    return time.perf_counter() - t0


def bench_ihcinfer_raw(patches, model):
    """ihcinfer custom engine: raw inference only (tensors, no PIL conversion)."""
    t0 = time.perf_counter()
    model.forward_tensors(patches)
    return time.perf_counter() - t0


def bench_ihcinfer_full(patches, inf):
    """ihcinfer custom engine: inference + scoring (model already loaded)."""
    t0 = time.perf_counter()
    inf._run_on_image_patches(patches)
    return time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(description="Benchmark ihcinfer patch inference vs original DeepLIIF")
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Path to DeepLIIF model directory (auto-downloaded if omitted)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="torch device to use, e.g. cpu or cuda:0",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    gpu_ids = [0] if device.type == "cuda" else []

    patches = load_patches(n=4)
    n = len(patches)
    print(f"Benchmarking on {n} patches ({patches[0].size[0]}x{patches[0].size[1]})")
    print(f"Model: {args.model_dir or '<auto-download>'}")
    print(f"Device: {device}\n")

    opt, nets, t_load_orig = load_original(args.model_dir, device)
    model, t_load_fast = load_custom(args.model_dir, device)
    inf = IHCAnalyzer(model_dir=args.model_dir, gpu_ids=gpu_ids, batch_size=len(patches))

    print(f"DeepLIIF original model load: {t_load_orig:.2f}s")
    print(f"ihcinfer model load:          {t_load_fast:.2f}s\n")

    t_orig_raw = bench_original_raw(patches, opt, nets)
    t_fast_raw = bench_ihcinfer_raw(patches, model)
    t_orig_full = bench_original_full(patches, opt, nets)
    t_fast_full = bench_ihcinfer_full(patches, inf)

    print(f"DeepLIIF original raw inference: {t_orig_raw:.2f}s ({t_orig_raw/n:.2f}s/patch)")
    print(f"ihcinfer raw inference:          {t_fast_raw:.2f}s ({t_fast_raw/n:.2f}s/patch)")
    print(f"DeepLIIF original full pipeline: {t_orig_full:.2f}s ({t_orig_full/n:.2f}s/patch)")
    print(f"ihcinfer full pipeline:          {t_fast_full:.2f}s ({t_fast_full/n:.2f}s/patch)")

    print("\n--- Summary (inference only) ---")
    print(f"Raw inference speedup:  {t_orig_raw/t_fast_raw:.2f}x")
    print(f"Full pipeline speedup:  {t_orig_full/t_fast_full:.2f}x")
    print(f"Model load speedup:     {t_load_orig/t_load_fast:.2f}x")


if __name__ == "__main__":
    main()
