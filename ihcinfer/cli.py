"""Command-line interface for ihcinfer.

The installed command is ``ihc`` and provides three subcommands:

- ``ihc tissue_seg``  : segment tissue from a WSI or ordinary image.
- ``ihc patch_infer`` : run DeepLIIF inference on PNG/JPEG patches.
- ``ihc infer``       : run DeepLIIF inference on a whole-slide IHC image.

Examples
--------
    # Tissue segmentation
    ihc tissue_seg --input slide.svs --output_dir ./mask_out --overlay

    # Patch inference
    ihc patch_infer --input tests/data/patches/22_2.png --output_dir ./out

    # WSI inference
    ihc infer --slide_path slide.svs --output_dir ./out --gpu_ids 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from ihcinfer import IHCAnalyzer, segment_tissue
from ihcinfer.readers import create_reader


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Path to DeepLIIF model directory (auto-downloaded if omitted)",
    )
    parser.add_argument(
        "--gpu_ids",
        type=int,
        nargs="+",
        default=None,
        help="CUDA device ids (default: [0] if available)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Inference batch size",
    )


def _cmd_tissue_seg(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Segmenting tissue from {args.input} (mode={args.mode}) ...")
    mask = segment_tissue(
        args.input,
        mode=args.mode,
        seg_level=args.seg_level,
    )

    mask_path = output_dir / "mask.png"
    mask_img = Image.fromarray((mask.mask * 255).astype(np.uint8))
    mask_img.save(mask_path)
    print(f"Mask saved: {mask_path} (shape: {mask.mask.shape})")

    bbox = mask.bbox()
    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
        print(f"Tissue bbox (level-0): {bbox}")
    else:
        print("Warning: no tissue detected")

    if args.overlay:
        overlay_path = output_dir / "overlay.png"
        base = _load_thumbnail(args.input, args.max_size)
        overlay = _blend_mask_overlay(base, mask_img.resize(base.size, Image.Resampling.NEAREST))
        overlay.save(overlay_path)
        print(f"Overlay saved: {overlay_path}")

    return 0


def _cmd_patch_infer(args: argparse.Namespace) -> int:
    analyzer = IHCAnalyzer(
        model_dir=args.model_dir,
        gpu_ids=args.gpu_ids,
        batch_size=args.batch_size or 4,
    )
    results = analyzer.infer_patches(
        args.input,
        output_dir=args.output_dir,
        save_marker=args.save_marker,
    )
    for name, result in results.items():
        print(f"{name}: {result['scoring']}")
    print(f"Done. Outputs in {args.output_dir}")
    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = IHCAnalyzer(
        model_dir=args.model_dir,
        gpu_ids=args.gpu_ids,
        batch_size=args.batch_size or 8,
    )

    kwargs: dict = {
        "patch_size": args.patch_size,
        "region_size": args.region_size,
    }
    if args.chunk_size is not None:
        kwargs["chunk_size"] = args.chunk_size

    result = analyzer.infer_wsi(
        slide_path=args.slide_path,
        output_dir=str(output_dir),
        num_region_samples=args.num_region_samples,
        num_patch_samples=args.num_patch_samples,
        save_marker_in_samples=args.save_marker_in_samples,
        heatmap_cmap=args.heatmap_cmap,
        heatmap_vmax=args.heatmap_vmax,
        heatmap_sigma=args.heatmap_sigma,
        heatmap_upscale=args.heatmap_upscale,
        heatmap_max_size=args.heatmap_max_size,
        heatmap_grid_factor=args.heatmap_grid_factor,
        skip_thumbnail=args.skip_thumbnail,
        overlay_alpha=args.overlay_alpha,
        **kwargs,
    )

    print("IHC WSI inference complete.")
    print(f"Patch size: {args.patch_size}x{args.patch_size}")
    print(f"Region size: {args.region_size}x{args.region_size}")
    print(f"CSV: {result.csv_path}")
    print(f"Heatmap: {result.heatmap_path}")
    if result.thumbnail_path:
        print(f"Thumbnail: {result.thumbnail_path}")
    if result.overlay_path:
        print(f"Overlay: {result.overlay_path}")
    print(f"Region samples: {len(result.region_sample_paths) // 2}")
    print(f"Patch samples: {len(result.patch_sample_dirs)}")
    return 0


def _load_thumbnail(path: str, max_size: int) -> Image.Image:
    """Load a thumbnail for *path*, falling back to PIL for ordinary images."""
    try:
        with create_reader(path) as reader:
            thumb = reader.read_thumbnail((max_size, max_size))
    except Exception:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        thumb = img
    return thumb


def _blend_mask_overlay(base: Image.Image, mask: Image.Image) -> Image.Image:
    """Blend a binary mask as a red overlay on top of *base*."""
    mask_l = mask.convert("L")
    red = Image.new("RGB", base.size, (255, 0, 0))
    overlay = Image.composite(red, base, mask_l)
    return Image.blend(base, overlay, alpha=0.4)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ihc",
        description="IHC inference and tissue segmentation CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # tissue_seg
    tissue_parser = subparsers.add_parser(
        "tissue_seg",
        help="Segment tissue from a WSI or ordinary image",
    )
    tissue_parser.add_argument(
        "--input",
        required=True,
        help="Path to WSI (SVS/KFB) or image file",
    )
    tissue_parser.add_argument(
        "--output_dir",
        default="./segment_outputs",
        help="Where to save the mask (and optional overlay)",
    )
    tissue_parser.add_argument(
        "--mode",
        choices=["ihc", "clam"],
        default="ihc",
        help="Segmentation mode (default: ihc)",
    )
    tissue_parser.add_argument(
        "--seg_level",
        default="auto",
        help="WSI pyramid level for segmentation (default: auto)",
    )
    tissue_parser.add_argument(
        "--overlay",
        action="store_true",
        help="Also save a red tissue overlay on a thumbnail",
    )
    tissue_parser.add_argument(
        "--max_size",
        type=int,
        default=2048,
        help="Maximum thumbnail edge length for the overlay (default 2048)",
    )
    tissue_parser.set_defaults(func=_cmd_tissue_seg)

    # patch_infer
    patch_parser = subparsers.add_parser(
        "patch_infer",
        help="Run DeepLIIF inference on PNG/JPEG patches",
    )
    patch_parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Patch image file(s) or directory(ies)",
    )
    patch_parser.add_argument(
        "--output_dir",
        default="./patch_outputs",
        help="Where to save outputs",
    )
    patch_parser.add_argument(
        "--save_marker",
        action="store_true",
        help="Also save the inferred marker modality",
    )
    _add_model_args(patch_parser)
    patch_parser.set_defaults(func=_cmd_patch_infer)

    # infer
    infer_parser = subparsers.add_parser(
        "infer",
        help="Run DeepLIIF inference on an IHC whole-slide image",
    )
    infer_parser.add_argument(
        "--slide_path",
        required=True,
        help="Path to IHC WSI file",
    )
    infer_parser.add_argument(
        "--output_dir",
        default="./ihc_outputs",
        help="Where to save outputs",
    )
    infer_parser.add_argument(
        "--patch_size",
        type=int,
        default=512,
        help="Patch size in pixels (default 512)",
    )
    infer_parser.add_argument(
        "--region_size",
        type=int,
        default=2048,
        help="Region size in pixels; must be a multiple of --patch_size (default 2048)",
    )
    infer_parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="WSI chunk read size (default 8192)",
    )
    infer_parser.add_argument(
        "--num_region_samples",
        type=int,
        default=2,
        help="Number of region samples to save",
    )
    infer_parser.add_argument(
        "--num_patch_samples",
        type=int,
        default=4,
        help="Number of patch samples to save",
    )
    infer_parser.add_argument(
        "--heatmap_cmap",
        type=str,
        default="viridis",
        help="Heatmap colormap name",
    )
    infer_parser.add_argument(
        "--heatmap_vmax",
        type=float,
        default=50.0,
        help="Heatmap color scale upper bound",
    )
    infer_parser.add_argument(
        "--heatmap_sigma",
        type=float,
        default=0.75,
        help="Gaussian smoothing sigma",
    )
    infer_parser.add_argument(
        "--heatmap_upscale",
        type=int,
        default=None,
        help="Integer heatmap upscaling factor",
    )
    infer_parser.add_argument(
        "--heatmap_max_size",
        type=int,
        default=1024,
        help="Maximum heatmap edge length when --heatmap_upscale is not set",
    )
    infer_parser.add_argument(
        "--heatmap_grid_factor",
        type=int,
        default=4,
        help="Grid cells per patch for interpolated heatmap",
    )
    infer_parser.add_argument(
        "--skip_thumbnail",
        action="store_true",
        help="Skip H&E thumbnail and heatmap overlay generation",
    )
    infer_parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.4,
        help="Heatmap overlay opacity on H&E thumbnail",
    )
    infer_parser.add_argument(
        "--save_marker_in_samples",
        action="store_true",
        help="Save marker images in samples",
    )
    _add_model_args(infer_parser)
    infer_parser.set_defaults(func=_cmd_infer)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
