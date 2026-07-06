"""Example: run DeepLIIF inference on an IHC whole-slide image (SVS/KFB/etc.)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ihcinfer import IHCAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepLIIF inference on an IHC whole-slide image"
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Path to DeepLIIF model directory (auto-downloaded if omitted)",
    )
    parser.add_argument("--slide_path", required=True, help="Path to IHC WSI file")
    parser.add_argument("--output_dir", default="./ihc_outputs", help="Where to save outputs")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0], help="CUDA device ids")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    parser.add_argument(
        "--patch_size", type=int, default=512, help="Patch size in pixels (default 512)"
    )
    parser.add_argument(
        "--region_size",
        type=int,
        default=2048,
        help="Region size in pixels; must be a multiple of --patch_size (default 2048)",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=None, help="WSI chunk read size (default 8192)"
    )
    parser.add_argument(
        "--num_region_samples", type=int, default=2, help="Number of region samples to save"
    )
    parser.add_argument(
        "--num_patch_samples", type=int, default=4, help="Number of patch samples to save"
    )
    parser.add_argument(
        "--heatmap_cmap", type=str, default="viridis", help="Heatmap colormap name"
    )
    parser.add_argument(
        "--heatmap_vmax", type=float, default=50.0, help="Heatmap color scale upper bound"
    )
    parser.add_argument(
        "--heatmap_sigma", type=float, default=0.5, help="Gaussian smoothing sigma (grid mode only)"
    )
    parser.add_argument(
        "--heatmap_sigma_factor",
        type=float,
        default=0.45,
        help="Interpolated heatmap Gaussian sigma as a multiple of patch size (default 0.45)",
    )
    parser.add_argument(
        "--heatmap_radius_factor",
        type=float,
        default=1.5,
        help="Interpolated heatmap neighbor radius as a multiple of patch size (default 1.5)",
    )
    parser.add_argument(
        "--heatmap_upscale",
        type=int,
        default=None,
        help="Integer heatmap upscaling factor (default: auto-fit to --heatmap_max_size)",
    )
    parser.add_argument(
        "--heatmap_max_size",
        type=int,
        default=1024,
        help="Maximum heatmap edge length in pixels when --heatmap_upscale is not set",
    )
    parser.add_argument(
        "--heatmap_grid_factor",
        type=int,
        default=4,
        help="Grid cells per patch for interpolated heatmap (default 4)",
    )
    parser.add_argument(
        "--skip_thumbnail",
        action="store_true",
        help="Skip H&E thumbnail and heatmap overlay generation",
    )
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.4,
        help="Heatmap overlay opacity on H&E thumbnail (default 0.4)",
    )
    parser.add_argument(
        "--save_marker_in_samples", action="store_true", help="Save marker images in samples"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages (only print final paths)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inf = IHCAnalyzer(
        model_dir=args.model_dir,
        gpu_ids=args.gpu_ids,
        batch_size=args.batch_size,
    )

    kwargs = {"patch_size": args.patch_size, "region_size": args.region_size}
    if args.chunk_size is not None:
        kwargs["chunk_size"] = args.chunk_size

    result = inf.infer_wsi(
        slide_path=args.slide_path,
        output_dir=str(output_dir),
        num_region_samples=args.num_region_samples,
        num_patch_samples=args.num_patch_samples,
        save_marker_in_samples=args.save_marker_in_samples,
        heatmap_cmap=args.heatmap_cmap,
        heatmap_vmax=args.heatmap_vmax,
        heatmap_sigma=args.heatmap_sigma,
        heatmap_sigma_factor=args.heatmap_sigma_factor,
        heatmap_radius_factor=args.heatmap_radius_factor,
        heatmap_upscale=args.heatmap_upscale,
        heatmap_max_size=args.heatmap_max_size,
        heatmap_grid_factor=args.heatmap_grid_factor,
        skip_thumbnail=args.skip_thumbnail,
        overlay_alpha=args.overlay_alpha,
        progress=not args.quiet,
        **kwargs,
    )

    if args.quiet:
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
    else:
        # Progress messages already printed a summary; just echo the key paths.
        print(f"CSV: {result.csv_path}")
        print(f"Heatmap: {result.heatmap_path}")


if __name__ == "__main__":
    main()
