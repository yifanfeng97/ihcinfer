"""Tests for patch-level inference."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ihcinfer import SlideInference
from ihcinfer.inference import run_batches_adaptive

MODEL_DIR = "/home/fengyifan/disk/code/DeepLIIF/model-server/DeepLIIF_Latest_Model"
PATCH = "tests/data/patches/22_2.png"


@pytest.fixture(scope="module")
def inference():
    return SlideInference(model_dir=MODEL_DIR, gpu_ids=[], batch_size=2)


def _assert_scoring(score: dict) -> None:
    assert score["num_total"] >= 0
    assert score["num_pos"] >= 0
    assert score["num_neg"] >= 0
    assert 0.0 <= score["percent_pos"] <= 100.0


def test_run_on_patches_returns_results(inference):
    results = inference.run_on_patches([PATCH])
    assert isinstance(results, dict)
    assert "22_2" in results
    result = results["22_2"]
    assert "scoring" in result
    assert "images" in result
    assert "cells" in result
    assert isinstance(result["cells"], list)
    assert len(result["cells"]) > 0
    _assert_scoring(result["scoring"])

    images = result["images"]
    assert isinstance(images["segmentation"], Image.Image)
    assert isinstance(images["overlay"], Image.Image)
    assert "marker" not in images


@pytest.mark.slow
def test_run_on_patches_supports_directory(inference):
    results = inference.run_on_patches("tests/data/patches")
    assert len(results) == 4
    for name, result in results.items():
        assert name
        assert "scoring" in result
        assert "images" in result
        assert isinstance(result["images"]["segmentation"], Image.Image)
        assert isinstance(result["images"]["overlay"], Image.Image)
        _assert_scoring(result["scoring"])


def test_run_on_patches_saves_outputs(inference, tmp_path: Path):
    results = inference.run_on_patches([PATCH], output_dir=tmp_path)
    result = results["22_2"]

    assert isinstance(result["images"]["segmentation"], Image.Image)
    assert isinstance(result["images"]["overlay"], Image.Image)
    assert "marker" not in result["images"]

    assert (tmp_path / "22_2" / "segmentation.jpg").exists()
    assert (tmp_path / "22_2" / "overlay.jpg").exists()
    assert (tmp_path / "22_2" / "cells.json").exists()
    assert (tmp_path / "22_2" / "scoring.json").exists()

    saved_scoring = json.loads((tmp_path / "22_2" / "scoring.json").read_text())
    assert saved_scoring == result["scoring"]


def test_run_on_patches_can_save_marker(inference, tmp_path: Path):
    results = inference.run_on_patches([PATCH], output_dir=tmp_path, save_marker=True)
    marker = results["22_2"]["images"].get("marker")
    assert isinstance(marker, Image.Image)
    assert (tmp_path / "22_2" / "marker.jpg").exists()


def test_run_on_patches_blank_patch(inference, tmp_path: Path):
    blank = tmp_path / "blank.png"
    Image.new("RGB", (512, 512), (255, 255, 255)).save(blank)

    results = inference.run_on_patches([blank], output_dir=tmp_path)
    assert results["blank"]["scoring"] == {
        "num_total": 0,
        "num_pos": 0,
        "num_neg": 0,
        "percent_pos": 0.0,
    }
    assert results["blank"]["images"] == {}
    assert results["blank"]["cells"] == []
    assert not (tmp_path / "blank" / "segmentation.jpg").exists()


@pytest.mark.slow
def test_run_on_patches_default_batch_size():
    # Different instance-level batch sizes should produce the same scoring.
    inf_default = SlideInference(model_dir=MODEL_DIR, gpu_ids=[], batch_size=2)
    inf_small = SlideInference(model_dir=MODEL_DIR, gpu_ids=[], batch_size=1)

    default_results = inf_default.run_on_patches([PATCH])
    small_batch_results = inf_small.run_on_patches([PATCH])
    assert (
        default_results["22_2"]["scoring"]
        == small_batch_results["22_2"]["scoring"]
    )


def test_run_on_image_patches_returns_scoring(inference):
    img = Image.open(PATCH).convert("RGB")
    results = inference._run_on_image_patches([img])
    assert len(results) == 1
    images, scoring = results[0]
    assert "segmentation" in images
    _assert_scoring(scoring)


@pytest.mark.slow
def test_batch_consistency(inference):
    img = Image.open(PATCH).convert("RGB")
    r1 = inference._run_on_image_patches([img])
    r2 = inference._run_on_image_patches([img, img])
    assert r1[0][1] == r2[0][1]


def test_blank_image_patch(inference):
    white = Image.new("RGB", (512, 512), (255, 255, 255))
    results = inference._run_on_image_patches([white])
    assert results[0][1] == {
        "num_total": 0,
        "num_pos": 0,
        "num_neg": 0,
        "percent_pos": 0.0,
    }


@pytest.mark.slow
def test_run_on_region(inference, tmp_path):
    arr = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    region = Image.fromarray(arr)
    records, region_results = inference.run_on_region(
        region, tile_size=512, overlap_size=0
    )
    assert len(records) == 4
    for rec in records:
        assert "patch_id" in rec
        assert "x" in rec and "y" in rec
        assert "num_total" in rec
    assert region_results is not None
    assert "segmentation" in region_results


def testrun_batches_adaptive_runs_all_inputs():
    calls = []

    def infer(batch):
        calls.append(len(batch))
        return [x * 2 for x in batch]

    inputs = list(range(10))
    results = run_batches_adaptive(infer, inputs, batch_size=3)
    assert results == [x * 2 for x in inputs]
    assert sum(calls) == len(inputs)


def testrun_batches_adaptive_empty():
    assert run_batches_adaptive(lambda b: b, [], batch_size=4) == []


def testrun_batches_adaptive_halves_on_oom():
    """Simulate OOM once to verify the helper retries with a smaller batch."""
    oom_count = 0

    def infer(batch):
        nonlocal oom_count
        if len(batch) > 1 and oom_count == 0:
            oom_count += 1
            raise RuntimeError("CUDA out of memory")
        return [x + 1 for x in batch]

    results = run_batches_adaptive(infer, list(range(5)), batch_size=4)
    assert results == list(range(1, 6))
    assert oom_count == 1


def testrun_batches_adaptive_mismatched_length_raises():
    def infer(batch):
        return batch[:-1]

    with pytest.raises(RuntimeError, match="inference returned"):
        run_batches_adaptive(infer, [1, 2, 3], batch_size=2)


def test_forward_arrays_matches_tensor_to_pil(inference):
    img = Image.open(PATCH).convert("RGB")
    pil_results = inference._patch_infer.model.forward([img])
    array_results = inference._patch_infer.model.forward_arrays([img])

    assert len(pil_results) == 1
    assert len(array_results) == 1

    pil_arr = np.asarray(pil_results[0].segmentation)
    assert np.array_equal(pil_arr, array_results[0])
    assert array_results[0].dtype == np.uint8
    assert array_results[0].shape == (img.height, img.width, 3)


def test_patch_inference_run_return_images_false(inference):
    img = Image.open(PATCH).convert("RGB")
    full_results = inference._patch_infer.run([img])
    scoring_results = inference._patch_infer.run([img], return_images=False)

    assert len(full_results) == 1
    assert len(scoring_results) == 1

    full_images, full_scoring = full_results[0]
    scoring_images, scoring_only = scoring_results[0]

    assert "segmentation" in full_images
    assert scoring_images == {}
    assert scoring_only == full_scoring
    _assert_scoring(scoring_only)


def test_patch_inference_run_return_images_false_blank(inference):
    white = Image.new("RGB", (512, 512), (255, 255, 255))
    results = inference._patch_infer.run([white], return_images=False)
    assert len(results) == 1
    images, scoring = results[0]
    assert images == {}
    assert scoring == {
        "num_total": 0,
        "num_pos": 0,
        "num_neg": 0,
        "percent_pos": 0.0,
    }
