# ihcinfer examples

This directory contains runnable examples for the most common use cases.

## Requirements

All examples assume you have installed `ihcinfer`.  If a DeepLIIF TorchScript
model directory is not provided via `--model_dir`, the model will be downloaded
automatically on first use from the official Zenodo record and cached locally.

You can install the package with:

```bash
uv sync
```

## Patch inference

Run on one or more PNG/JPEG patches or directories.  The first run without
`--model_dir` will download the pretrained model (~3 GB):

```bash
uv run python examples/infer_patch.py \
    --input tests/data/patches/22_2.png tests/data/patches \
    --output_dir ./patch_outputs
```

To use a local model directory instead:

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
    --input tests/data/patches/22_2.png \
    --output_dir ./patch_outputs \
    --save_marker
```

## Whole-slide IHC inference

Run on an IHC whole-slide image (SVS, KFB, TIFF, etc.).  The first run without
`--model_dir` will download the pretrained model (~3 GB):

```bash
uv run python examples/infer_ihc.py \
    --slide_path /path/to/slide.svs \
    --output_dir ./ihc_outputs \
    --batch_size 8 \
    --patch_size 512 \
    --region_size 2048
```

To use a local model directory:

```bash
uv run python examples/infer_ihc.py \
    --model_dir /path/to/DeepLIIF_Latest_Model \
    --slide_path /path/to/slide.svs \
    --output_dir ./ihc_outputs \
    --batch_size 8 \
    --patch_size 512 \
    --region_size 2048
```

Outputs:

- `patch_scoring.csv` — per-patch cell counts
- `heatmap.jpg` — `percent_pos` heatmap using Gaussian-kernel interpolation on a fine grid (`grid_factor=4` cells per 512×512 patch), resized to fit a 1024 px long edge (default `viridis` colormap, white background, tissue-mask boundary). Use `--heatmap_grid_factor`, `--heatmap_max_size`, `--heatmap_sigma`, or `--heatmap_upscale` to tune appearance.
- `he_thumbnail.jpg` — low-resolution H&E thumbnail of the whole slide, resized to the same dimensions as `heatmap.jpg`.
- `overlay.jpg` — translucent heatmap blended over `he_thumbnail.jpg`, same dimensions. Tissue-mask regions carry the heatmap at `--overlay_alpha` opacity (default 0.4); non-tissue regions keep the original H&E thumbnail.
- `region_samples/` — up to 2 complete `2048×2048` regions selected randomly from all tissue-rich regions; each region is saved under a `{x}_{y}` folder with `overlay.jpg` and `segmentation.jpg`
- `patch_samples/` — up to 4 sampled patches in standard patch layout

Use `--skip_thumbnail` to skip `he_thumbnail.jpg` and `overlay.jpg` generation.

The pipeline now pre-plans all tissue patches and complete regions from the initial tissue mask, batches patches across chunk boundaries for better GPU utilization, and handles visualization (region/patch samples) in a separate pass. Use `--batch_size` to tune GPU throughput, `--chunk_size` to adjust the WSI read size (default `8192`), `--patch_size` / `--region_size` to change the patch/region geometry, and `--num_region_samples` / `--num_patch_samples` to adjust the number of saved visual samples.

## Tissue segmentation

Segment tissue from an IHC whole-slide image or a regular image.  This does **not**
need a DeepLIIF model.

```bash
uv run python examples/segment_tissue.py \
    --input "tests/data/slides/98140-6 CD3.svs" \
    --output_dir ./tissue_mask \
    --overlay
```

Outputs:

- `mask.png` — binary tissue mask at the segmentation level
- `overlay.png` — red tissue overlay on a thumbnail (if `--overlay`)

Add `--mode clam` to use the CLAM-style H&E segmenter instead of the default IHC mode.

## Unified CLI

After installing the package, a single `ihcinfer` command is available with three
subcommands: `patch`, `ihc`, and `segment`.

```bash
# Show all subcommands
ihcinfer --help

# Patch inference
ihcinfer patch \
    --input tests/data/patches/22_2.png \
    --output_dir ./patch_outputs

# WSI inference
ihcinfer ihc \
    --slide_path /path/to/slide.svs \
    --output_dir ./ihc_outputs \
    --gpu_ids 0 \
    --batch_size 8

# Tissue segmentation
ihcinfer segment \
    --input "tests/data/slides/98140-6 CD3.svs" \
    --output_dir ./tissue_mask \
    --overlay
```
