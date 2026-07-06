"""Example: segment tissue from an IHC whole-slide image or patch image.

Usage:
    # WSI (auto-detects a suitable downsampled segmentation level)
    uv run python examples/segment_tissue.py \
        --input "tests/data/slides/98140-6 CD3.svs" \
        --output_dir ./tissue_mask \
        --overlay

    # Ordinary image
    uv run python examples/segment_tissue.py \
        --input tests/data/patches/22_2.png \
        --output_dir ./tissue_mask \
        --overlay

Outputs:
    - mask.png   : binary tissue mask at the segmentation level
    - overlay.png: red tissue overlay on a thumbnail (if --overlay)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from ihcinfer import segment_tissue
from ihcinfer.readers import create_reader


def _load_thumbnail(path: str, max_size: int) -> Image.Image:
    """Return a thumbnail for a WSI or an ordinary image."""
    try:
        with create_reader(path) as reader:
            return reader.read_thumbnail((max_size, max_size))
    except Exception:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment tissue from an IHC image")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to WSI (SVS/KFB) or image file",
    )
    parser.add_argument(
        "--output_dir",
        default="./tissue_mask",
        help="Where to save mask.png and overlay.png",
    )
    parser.add_argument(
        "--mode",
        choices=["ihc", "he"],
        default="ihc",
        help="Segmentation mode (default: ihc)",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Also save a red tissue overlay on a thumbnail",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=2048,
        help="Maximum thumbnail edge length for the overlay",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Segmenting tissue from {args.input} (mode={args.mode}) ...")
    mask = segment_tissue(args.input, mode=args.mode)

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
        base = _load_thumbnail(args.input, args.max_size)
        mask_resized = mask_img.resize(base.size, Image.Resampling.NEAREST)
        red = Image.new("RGB", base.size, (255, 0, 0))
        overlay = Image.composite(red, base, mask_resized.convert("L"))
        overlay = Image.blend(base, overlay, alpha=0.4)

        overlay_path = output_dir / "overlay.png"
        overlay.save(overlay_path)
        print(f"Overlay saved: {overlay_path}")


if __name__ == "__main__":
    main()
