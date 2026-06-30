"""Example: run DeepLIIF inference on one or more PNG/JPEG patches.

Usage:
    uv run python examples/infer_patch.py \
        --model_dir /path/to/DeepLIIF_Latest_Model \
        --input patch_1.png patch_2.png \
        --output_dir ./patch_outputs

Outputs per patch (under ``output_dir/{file_stem}/``):
    - segmentation.jpg    : final cell segmentation
    - overlay.jpg         : original patch with red/blue cell contours
    - cells.json          : per-cell centroid/boundary/positive/size
    - scoring.json        : total/positive/negative counts and percent_pos

Optional:
    --save_marker         : also save the inferred marker modality
"""

from __future__ import annotations

import argparse

from ihcinfer import SlideInference


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepLIIF inference on small patches")
    parser.add_argument("--model_dir", required=True, help="Path to DeepLIIF model directory")
    parser.add_argument("--input", nargs="+", required=True, help="Patch image file(s) or directory(ies)")
    parser.add_argument("--output_dir", default="./patch_outputs", help="Where to save outputs")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=None, help="CUDA device ids")
    parser.add_argument("--batch_size", type=int, default=4, help="Inference batch size")
    parser.add_argument("--save_marker", action="store_true", help="Also save the inferred marker modality")
    args = parser.parse_args()

    inf = SlideInference(
        model_dir=args.model_dir,
        gpu_ids=args.gpu_ids,
        batch_size=args.batch_size,
    )

    results = inf.run_on_patches(
        args.input,
        output_dir=args.output_dir,
        save_marker=args.save_marker,
    )

    for name, result in results.items():
        print(f"{name}: {result['scoring']}")

    print(f"Done. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
