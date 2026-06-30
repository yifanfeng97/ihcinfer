"""Tests for base utility helpers."""

import numpy as np
import pytest
import torch
from PIL import Image

from ihcinfer.models.device import resolve_device
from ihcinfer.prep import blank_scoring, is_blank_patch


def test_resolve_device_cuda_if_available():
    dev = resolve_device([0])
    if torch.cuda.is_available():
        assert dev.type == "cuda"
        assert dev.index == 0
    else:
        assert dev.type == "cpu"


def test_resolve_device_cpu_explicit():
    assert resolve_device([]).type == "cpu"


def test_is_blank_patch_white():
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    assert is_blank_patch(img, threshold=0.95) is True


def test_is_blank_patch_black():
    img = Image.new("RGB", (64, 64), (0, 0, 0))
    assert is_blank_patch(img, threshold=0.95) is True


def test_is_blank_patch_tissue():
    arr = np.random.randint(40, 200, (64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    assert is_blank_patch(img, threshold=0.95) is False


def test_blank_scoring():
    assert blank_scoring() == {
        "num_total": 0,
        "num_pos": 0,
        "num_neg": 0,
        "percent_pos": 0.0,
    }
