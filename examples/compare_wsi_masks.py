"""Compare whole-slide tissue segmentation masks side-by-side.

This script segments the same WSI with several strategies and writes a single
summary image showing the original thumbnail plus each mask overlay.  It is
much faster than running full inference because it only computes masks.

Usage
-----
    uv run python examples/compare_wsi_masks.py \
        --slide_path /path/to/CD3.svs \
        --output_dir ./outputs/compare_wsi_masks
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import openslide
from PIL import Image, ImageDraw, ImageFont

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


def _mask_overlay(img: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
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
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    x = max(5, (canvas.width - text_w) // 2)
    draw.text((x, 5), label, fill=(0, 0, 0), font=font)


def _build_summary(
    panels: list[tuple[np.ndarray, str]],
    target_height: int = 1200,
) -> Image.Image:
    banner_h = 32
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare whole-slide tissue masks")
    parser.add_argument("--slide_path", default=os.environ.get("IHCINFER_TEST_SLIDE"))
    parser.add_argument("--output_dir", default="outputs/compare_wsi_masks")
    parser.add_argument("--target_height", type=int, default=1200)
    args = parser.parse_args()

    if not args.slide_path or not Path(args.slide_path).exists():
        raise FileNotFoundError(f"slide not found: {args.slide_path}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    slide = openslide.OpenSlide(args.slide_path)
    level = slide.get_best_level_for_downsample(32)
    dims = slide.level_dimensions[level]
    thumb = np.array(slide.read_region((0, 0), level, dims).convert("RGB"))
    slide.close()

    print(f"Thumbnail level={level}, dims={dims}, shape={thumb.shape}")

    panels: list[tuple[np.ndarray, str]] = [(thumb, "Original")]
    per_config: list[tuple[str, float, int, Path]] = []

    for name, seg_kwargs, label in MASK_CONFIGS:
        print(f"\n  [{name}] segmenting...")
        mask = segment_tissue(args.slide_path, seg_level="auto", **seg_kwargs)
        fg_ratio = mask.mask.mean()

        overlay = _mask_overlay(thumb, mask.mask > 0)
        overlay_path = output_dir / f"{name}_overlay.jpg"
        Image.fromarray(overlay).save(overlay_path, quality=95)

        # Rough patch count at 512x512 level 0 using the mask.
        sx, sy = mask.scale
        patch_size = 512
        n_patches = int(mask.mask.sum() * sx * sy / (patch_size * patch_size))

        print(f"    foreground ratio: {fg_ratio:.4f}")
        print(f"    estimated 512-patches: {n_patches}")
        print(f"    overlay: {overlay_path}")

        panels.append((overlay, label))
        per_config.append((name, fg_ratio, n_patches, overlay_path))

    summary = _build_summary(panels, target_height=args.target_height)
    summary_path = output_dir / "wsi_mask_comparison.jpg"
    summary.save(summary_path, quality=95)
    print(f"\nSummary image saved: {summary_path}")

    # Write a small text report.
    report_path = output_dir / "comparison_report.txt"
    lines = ["Strategy | foreground_ratio | estimated_512_patches"]
    lines.append("-" * 55)
    for name, fg, n_patches, _ in per_config:
        lines.append(f"{name:20s} | {fg:.4f} | {n_patches}")
    report_path.write_text("\n".join(lines) + "\n")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
