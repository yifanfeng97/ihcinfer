"""Download the pretrained DeepLIIF model from Zenodo.

This module provides a small, dependency-free helper that fetches the official
``DeepLIIF_Latest_Model.zip`` archive the first time a user instantiates
:class:`IHCAnalyzer` without a local model directory.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Callable
from urllib.request import urlretrieve


# Official DeepLIIF model archive (Zenodo record for the Nature MI 2022 paper).
DEFAULT_MODEL_URL = (
    "https://zenodo.org/records/4751737/files/DeepLIIF_Latest_Model.zip"
)

# Files that must exist for a model directory to be considered complete.
_REQUIRED_FILES = ["train_opt.txt", "G1.pt", "G51.pt"]


def default_cache_dir() -> Path:
    """Return the default directory where the model should be cached."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "ihcinfer" / "models" / "DeepLIIF_Latest_Model"


def is_model_complete(model_dir: str | os.PathLike) -> bool:
    """Return True if *model_dir* contains the expected DeepLIIF checkpoints."""
    model_path = Path(model_dir)
    return all((model_path / name).exists() for name in _REQUIRED_FILES)


def _find_model_root(path: Path) -> Path | None:
    """Search *path* and one level of subdirectories for a complete model."""
    if is_model_complete(path):
        return path
    for child in path.iterdir():
        if child.is_dir() and is_model_complete(child):
            return child
    return None


def _default_progress(block_num: int, block_size: int, total_size: int) -> None:
    """Print a simple progress bar to stdout."""
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 // total_size)
    mb = downloaded / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    print(
        f"\rDownloading DeepLIIF model... {pct:3d}% "
        f"({mb:.1f}/{total_mb:.1f} MB)",
        end="",
        flush=True,
    )


def download_model(
    model_dir: str | os.PathLike | None = None,
    *,
    url: str | None = None,
    progress: Callable[[int, int, int], None] | None = _default_progress,
) -> Path:
    """Download and extract the pretrained DeepLIIF model.

    Args:
        model_dir: Directory where the model should be saved.  Defaults to a
            platform-specific cache directory.
        url: Direct download URL for the model zip archive.  Defaults to the
            official Zenodo record.
        progress: Optional callback for download progress.  Pass ``None`` to
            silence output.

    Returns:
        Path to the extracted model directory.
    """
    model_dir = Path(model_dir) if model_dir is not None else default_cache_dir()
    model_dir.mkdir(parents=True, exist_ok=True)

    if is_model_complete(model_dir):
        print(f"Model already present at {model_dir}")
        return model_dir

    url = url or os.environ.get("IHCINFER_MODEL_URL") or DEFAULT_MODEL_URL
    archive_name = "DeepLIIF_Latest_Model.zip"

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / archive_name
        print(f"Downloading DeepLIIF model from {url}")
        urlretrieve(url, str(archive_path), reporthook=progress)
        if progress is not None:
            print()  # newline after progress bar

        print(f"Extracting model to {model_dir} ...")
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(model_dir)

    # The archive may contain a top-level folder; find the actual checkpoint root.
    root = _find_model_root(model_dir)
    if root is None:
        raise RuntimeError(
            f"Downloaded archive does not contain required model files in {model_dir}"
        )

    # If the model ended up in a nested directory, move its contents up.
    if root != model_dir:
        for item in root.iterdir():
            dest = model_dir / item.name
            if dest.exists():
                shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
            shutil.move(str(item), str(dest))
        shutil.rmtree(root)

    if not is_model_complete(model_dir):
        raise RuntimeError(
            f"Model download completed but required files are missing in {model_dir}"
        )

    print(f"Model ready at {model_dir}")
    return model_dir
