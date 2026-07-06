"""Compare IHC tissue segmentation strategies and their downstream WSI outputs.

This script crops a tissue-rich region from a WSI, then runs the same IHC
inference pipeline with several segmentation configurations side-by-side:

- ``ihc_default``       current default Otsu-based IHC segmentation
- ``ihc_white_filter``  Otsu + HSV white-background suppression
- ``ihc_white_loose``   same but with looser white definition
- ``clam``              CLAM-style saturation thresholding

For each strategy it writes:

1. A tissue-mask overlay on the cropped slide.
2. A full ``IHCAnalyzer.infer_wsi`` output directory with CSV, heatmap,
   overlay, H&E thumbnail, and sampled patches/regions.
3. A side-by-side summary image of all masks and all heatmaps.

Usage
-----
    uv run python examples/compare_tissue_seg.py \
        --slide_path /path/to/CD3.svs \
        --model_dir /path/to/DeepLIIF_Latest_Model \
        --output_dir ./outputs/compare_seg
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import openslide
from PIL import Image, ImageDraw, ImageFont

from ihcinfer import IHCAnalyzer
from ihcinfer.prep import segment_tissue


MASK_CONFIGS = [
    ("ihc_default", {"mode": "ihc"}, "IHC default (Otsu)"),
    (
        "ihc_white_filter",
        {"mode": "ihc", "white_filter": True, "white_v_thresh": 240, "white_s_thresh": 20},
        "IHC + white filter (V>=240, S<=20)",
    ),
    (
        "ihc_white_loose",
        {"mode": "ihc", "white_filter": True, "white_v_thresh": 230, "white_s_thresh": 30},
        "IHC + white filter (V>=230, S<=30)",
    ),
    ("clam", {"mode": "clam"}, "CLAM (saturation)"),
]


def _crop_region(
    slide_path: str,
    output_path: Path,
    x: int,
    y: int,
    size: int,
) -> None:
    """Crop a level-0 region from a WSI and save it as a high-quality JPEG."""
    slide = openslide.OpenSlide(slide_path)
    region = slide.read_region((x, y), 0, (size, size))
    slide.close()
    region.convert("RGB").save(output_path, quality=95)


def _mask_overlay(img: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    """Return an RGB overlay where the binary mask is blended in ``color``."""
    overlay = img.copy()
    alpha = 0.4
    for c in range(3):
        overlay[:, :, c] = np.where(
            mask,
            (1 - alpha) * overlay[:, :, c] + alpha * color[c],
            overlay[:, :, c],
        ).astype(np.uint8)
    return overlay


def _resize_to_height(img: Image.Image, target_height: int) -> Image.Image:
    ratio = target_height / img.height
    new_w = int(img.width * ratio)
    return img.resize((new_w, target_height), Image.Resampling.LANCZOS)


def _draw_label(canvas: Image.Image, label: str) -> None:
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    x = max(5, (canvas.width - text_w) // 2)
    draw.text((x, 5), label, fill=(0, 0, 0), font=font)


def _build_horizontal_summary(
    panels: list[tuple[np.ndarray, str]],
    target_height: int = 1024,
) -> Image.Image:
    """Concatenate labelled panels horizontally."""
    banner_h = 28
    resized: list[Image.Image] = []
    for arr, label in panels:
        img = _resize_to_height(Image.fromarray(arr), target_height)
        canvas = Image.new("RGB", (img.width, target_height + banner_h), (255, 255, 255))
        canvas.paste(img, (0, banner_h))
        _draw_label(canvas, label)
        resized.append(canvas)

    total_w = sum(c.width for c in resized)
    summary = Image.new("RGB", (total_w, target_height + banner_h), (255, 255, 255))
    x = 0
    for canvas in resized:
        summary.paste(canvas, (x, 0))
        x += canvas.width
    return summary


def _build_heatmap_grid(
    panels: list[tuple[np.ndarray, np.ndarray, str]],
    target_height: int = 512,
) -> Image.Image:
    """Build a grid with one column per strategy and two rows (heatmap, overlay)."""
    banner_h = 28
    rows = 2
    cols = len(panels)

    cell_h = target_height
    cell_w = max(_resize_to_height(Image.fromarray(p[0]), target_height).width for p in panels)

    summary = Image.new(
        "RGB",
        (cols * cell_w, rows * cell_h + banner_h),
        (255, 255, 255),
    )

    for col, (hm_arr, ov_arr, label) in enumerate(panels):
        hm = _resize_to_height(Image.fromarray(hm_arr), target_height)
        ov = _resize_to_height(Image.fromarray(ov_arr), target_height)

        x = col * cell_w + (cell_w - hm.width) // 2
        summary.paste(hm, (x, banner_h))
        x = col * cell_w + (cell_w - ov.width) // 2
        summary.paste(ov, (x, banner_h + cell_h))

        # Column label spans both rows.
        label_canvas = Image.new("RGB", (cell_w, banner_h), (255, 255, 255))
        _draw_label(label_canvas, label)
        summary.paste(label_canvas, (col * cell_w, 0))

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare tissue segmentation strategies")
    parser.add_argument("--slide_path", default=os.environ.get("IHCINFER_TEST_SLIDE"))
    parser.add_argument("--model_dir", default=os.environ.get("IHCINFER_MODEL_DIR"))
    parser.add_argument("--output_dir", default="outputs/compare_seg")
    parser.add_argument("--crop_x", type=int, default=16384)
    parser.add_argument("--crop_y", type=int, default=16384)
    parser.add_argument("--crop_size", type=int, default=4096)
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    if not args.slide_path or not Path(args.slide_path).exists():
        raise FileNotFoundError(f"slide not found: {args.slide_path}")
    if not args.model_dir or not Path(args.model_dir).exists():
        raise FileNotFoundError(f"model_dir not found: {args.model_dir}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cropped_path = output_dir / "cropped_slide.jpg"
    print(f"Cropping {args.crop_size}x{args.crop_size} region from {args.slide_path}...")
    _crop_region(
        args.slide_path,
        cropped_path,
        args.crop_x,
        args.crop_y,
        args.crop_size,
    )

    analyzer = IHCAnalyzer(
        model_dir=args.model_dir,
        gpu_ids=args.gpu_ids,
        batch_size=args.batch_size,
        auto_download=False,
    )

    img = np.array(Image.open(cropped_path).convert("RGB"))
    mask_panels: list[tuple[np.ndarray, str]] = [(img, "Original")]
    heatmap_panels: list[tuple[np.ndarray, np.ndarray, str]] = []

    print("\nRunning segmentations and inference...")
    for name, seg_kwargs, label in MASK_CONFIGS:
        print(f"\n  [{name}]")
        seg_dir = output_dir / name
        seg_dir.mkdir(parents=True, exist_ok=True)

        # 1. Tissue mask overlay.
        mask = segment_tissue(cropped_path, **seg_kwargs)
        overlay = _mask_overlay(img, mask.mask > 0)
        overlay_path = seg_dir / "tissue_mask_overlay.jpg"
        Image.fromarray(overlay).save(overlay_path, quality=95)
        print(f"    mask foreground ratio: {mask.mask.mean():.4f}")
        print(f"    mask overlay: {overlay_path}")
        mask_panels.append((overlay, label))

        # 2. Full WSI inference with this segmentation strategy.
        result = analyzer.infer_wsi(
            cropped_path,
            seg_dir / "inference",
            tissue_seg_kwargs=seg_kwargs,
            patch_size=512,
            region_size=2048,
            tissue_min_ratio=0.01,
            num_region_samples=2,
            num_patch_samples=4,
            progress=True,
        )
        print(f"    patches scored: {len(result.records)}")
        print(f"    total cells: {result.summary['num_total']}")
        print(f"    positive: {result.summary['num_pos']} ({result.summary['percent_pos']:.2f}%)")

        # 3. Collect heatmap + overlay for the summary image.
        heatmap = np.array(Image.open(result.heatmap_path).convert("RGB"))
        overlay_pred = (
            np.array(Image.open(result.overlay_path).convert("RGB"))
            if result.overlay_path
            else heatmap
        )
        heatmap_panels.append((heatmap, overlay_pred, label))

    # 4. Summary images.
    mask_summary = _build_horizontal_summary(mask_panels)
    mask_summary_path = output_dir / "mask_comparison.jpg"
    mask_summary.save(mask_summary_path, quality=95)
    print(f"\nMask comparison saved: {mask_summary_path}")

    heatmap_summary = _build_heatmap_grid(heatmap_panels)
    heatmap_summary_path = output_dir / "heatmap_comparison.jpg"
    heatmap_summary.save(heatmap_summary_path, quality=95)
    print(f"Heatmap/overlay comparison saved: {heatmap_summary_path}")


if __name__ == "__main__":
    main()
