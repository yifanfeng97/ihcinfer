# ihcinfer examples

This directory contains runnable examples for the most common use cases.

## Requirements

All examples assume you have installed `ihcinfer` and have a DeepLIIF
TorchScript model directory containing `G1.pt` ... `G4.pt` and `G51.pt` ... `G55.pt`.

You can install the package with:

```bash
uv sync
```

## Patch inference

Run on one or more PNG/JPEG patches or directories:

```bash
uv run python examples/infer_patch.py \
    --model_dir /path/to/DeepLIIF_Latest_Model \
    --input tests/data/patches/22_2.png tests/data/patches \
    --output_dir ./patch_outputs
```

The script prints cell-count scoring and writes the following files per patch
under `./patch_outputs/{file_stem}/`:

```
22_2/
├── segmentation.jpg    # final cell segmentation
├── overlay.jpg         # original patch with red/blue cell contours
├── cells.json          # per-cell centroid/boundary/positive/size
└── scoring.json        # total/positive/negative counts and percent_pos
```

- Red contours = positive cells
- Blue contours = negative cells

Each patch gets its own directory named after the original file stem
(e.g. `22_2/`).  If two inputs share the same stem, a suffix is appended
(e.g. `22_2_1/`).

The inferred marker modality is **not** saved by default.  To also save it,
add `--save_marker`:

```bash
uv run python examples/infer_patch.py \
    --model_dir /path/to/DeepLIIF_Latest_Model \
    --input tests/data/patches/22_2.png \
    --output_dir ./patch_outputs \
    --save_marker
```

## Whole-slide inference

Run on a whole-slide image (SVS, KFB, TIFF, etc.):

```bash
uv run python examples/infer_wsi.py \
    --model_dir /path/to/DeepLIIF_Latest_Model \
    --slide_path /path/to/slide.svs \
    --output_dir ./wsi_outputs \
    --batch_size 8
```

Outputs:

- `patch_scoring.csv` — per-patch cell counts
- `heatmap.jpg` — `percent_pos` heatmap using Gaussian-kernel interpolation on a fine grid (`grid_factor=4` cells per 512×512 patch), resized to fit a 1024 px long edge (default `viridis` colormap, white background, tissue-mask boundary). Use `--heatmap_grid_factor`, `--heatmap_max_size`, `--heatmap_sigma`, or `--heatmap_upscale` to tune appearance.
- `he_thumbnail.jpg` — low-resolution H&E thumbnail of the whole slide, resized to the same dimensions as `heatmap.jpg`.
- `overlay.jpg` — translucent heatmap blended over `he_thumbnail.jpg`, same dimensions. Tissue-mask regions carry the heatmap at `--overlay_alpha` opacity (default 0.4); non-tissue regions keep the original H&E thumbnail.
- `region_samples/` — up to 2 complete `2048×2048` regions selected randomly from all tissue-rich regions; each region is saved under a `{x}_{y}` folder with `overlay.jpg` and `segmentation.jpg`
- `patch_samples/` — up to 4 sampled patches in standard patch layout

Use `--skip_thumbnail` to skip `he_thumbnail.jpg` and `overlay.jpg` generation.

The pipeline now pre-plans all tissue patches and complete regions from the initial tissue mask, batches patches across chunk boundaries for better GPU utilization, and handles visualization (region/patch samples) in a separate pass. Use `--batch_size` to tune GPU throughput, `--chunk_size` to adjust the WSI read size (default `8192`), and `--num_region_samples` / `--num_patch_samples` to adjust the number of saved visual samples.
