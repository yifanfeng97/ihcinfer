"""Tests for the automatic model download helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from ihcinfer.models.download import (
    DEFAULT_MODEL_URL,
    _find_model_root,
    default_cache_dir,
    download_model,
    is_model_complete,
)


class TestDefaultCacheDir:
    def test_returns_ihcinfer_subpath(self, monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/xdg-cache")
        cache = default_cache_dir()
        assert cache == Path("/tmp/xdg-cache/ihcinfer/models/DeepLIIF_Latest_Model")

    def test_falls_back_to_dot_cache_without_xdg(self, monkeypatch):
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        cache = default_cache_dir()
        assert cache == Path.home() / ".cache" / "ihcinfer" / "models" / "DeepLIIF_Latest_Model"


class TestIsModelComplete:
    def test_complete_when_required_files_present(self, tmp_path: Path):
        for name in ("train_opt.txt", "G1.pt", "G51.pt"):
            (tmp_path / name).write_text("dummy")
        assert is_model_complete(tmp_path) is True

    def test_incomplete_when_files_missing(self, tmp_path: Path):
        (tmp_path / "train_opt.txt").write_text("dummy")
        (tmp_path / "G1.pt").write_text("dummy")
        assert is_model_complete(tmp_path) is False

    def test_incomplete_for_empty_dir(self, tmp_path: Path):
        assert is_model_complete(tmp_path) is False


class TestFindModelRoot:
    def test_finds_root_in_nested_archive(self, tmp_path: Path):
        inner = tmp_path / "DeepLIIF_Latest_Model"
        inner.mkdir()
        for name in ("train_opt.txt", "G1.pt", "G51.pt"):
            (inner / name).write_text("dummy")
        assert _find_model_root(tmp_path) == inner

    def test_returns_same_path_when_complete(self, tmp_path: Path):
        for name in ("train_opt.txt", "G1.pt", "G51.pt"):
            (tmp_path / name).write_text("dummy")
        assert _find_model_root(tmp_path) == tmp_path


class TestDownloadModel:
    def test_download_model_skips_when_complete(self, tmp_path: Path, capsys):
        for name in ("train_opt.txt", "G1.pt", "G51.pt"):
            (tmp_path / name).write_text("dummy")
        result = download_model(tmp_path, progress=None)
        assert result == tmp_path
        captured = capsys.readouterr()
        assert "already present" in captured.out

    def test_download_model_raises_for_bogus_url(self, tmp_path: Path):
        with pytest.raises(Exception):
            download_model(tmp_path, url="https://localhost:9/invalid.zip", progress=None)


def test_default_url_points_to_zenodo():
    assert "zenodo.org/records/4751737" in DEFAULT_MODEL_URL
