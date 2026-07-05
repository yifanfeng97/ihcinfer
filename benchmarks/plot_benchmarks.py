"""Generate polished benchmark comparison charts for the README.

This script uses the timing numbers reported in the README benchmark table and
writes PNG images to ``docs/assets/``.  It does not require the original
DeepLIIF repository or a GPU.

Run with ``--lang en|zh|all`` to control which language variants are generated.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

ASSETS_DIR = Path(__file__).resolve().parent.parent / "docs" / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Brand colours
COLOR_ORIGINAL = "#94a3b8"      # slate-400
COLOR_IHC = "#3776ab"         # brand blue
COLOR_ACCENT = "#06b6d4"      # cyan accent
COLOR_WSI_ORIG = "#64748b"    # darker slate for WSI estimate
COLOR_WSI_IHC = "#0ea5e9"     # sky blue

LABELS = {
    "en": {
        "cpu_full_pipeline": "CPU full pipeline\n(4 patches)",
        "gpu_inference_only": "GPU inference only",
        "gpu_full_pipeline": "GPU full pipeline",
        "wsi_end_to_end": "WSI end-to-end\n(1453 patches)",
        "original": "Original DeepLIIF",
        "ihcinfer": "ihcinfer",
        "time_seconds": "Time (seconds)",
        "patch_time_title": "Patch-level inference time comparison",
        "speedup_factor": "Speedup factor (× faster)",
        "speedup_title": "ihcinfer speedup over original DeepLIIF",
        "patches_per_minute": "Patches per minute",
        "wsi_throughput_title": "WSI throughput (1453 patches, GPU)",
    },
    "zh": {
        "cpu_full_pipeline": "CPU 完整流程\n(4 张 patch)",
        "gpu_inference_only": "GPU 纯推理",
        "gpu_full_pipeline": "GPU 完整流程",
        "wsi_end_to_end": "WSI 端到端\n(1453 张 patch)",
        "original": "原始 DeepLIIF",
        "ihcinfer": "ihcinfer",
        "time_seconds": "时间（秒）",
        "patch_time_title": "Patch 级推理耗时对比",
        "speedup_factor": "加速比（× 更快）",
        "speedup_title": "ihcinfer 相比原始 DeepLIIF 的加速比",
        "patches_per_minute": "每分钟处理 patch 数",
        "wsi_throughput_title": "WSI 吞吐（1453 张 patch，GPU）",
    },
}


def _find_cjk_font() -> Path | None:
    """Return the first available system CJK font, or None."""
    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK SC Regular",
        "NotoSansCJK-Regular.ttc",
        "WenQuanYi Micro Hei",
        "Source Han Sans SC",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
    ]
    for name in candidates:
        try:
            path = Path(fm.findfont(name))
            if path.exists():
                return path
        except Exception:
            continue
    return None


def _ensure_cjk_font(cjk_font: Path | None = None) -> Path:
    """Return a usable CJK font path, auto-downloading if necessary."""
    if cjk_font is not None:
        return cjk_font
    system_font = _find_cjk_font()
    if system_font is not None:
        return system_font

    # Try to fetch via apt-get download (works on Debian/Ubuntu without root).
    cache_dir = Path.home() / ".cache" / "ihcinfer" / "fonts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ttc = cache_dir / "NotoSansCJK-Regular.ttc"
    if ttc.exists():
        return ttc

    if shutil.which("apt-get"):
        try:
            subprocess.run(
                ["apt-get", "download", "fonts-noto-cjk", "-y"],
                cwd=cache_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            deb = next(cache_dir.glob("fonts-noto-cjk_*.deb"))
            extract_dir = cache_dir / "extracted"
            subprocess.run(
                ["dpkg-deb", "-x", str(deb), str(extract_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
            extracted = next(extract_dir.rglob("NotoSansCJK-Regular.ttc"))
            ttc.write_bytes(extracted.read_bytes())
            return ttc
        except Exception as exc:
            raise RuntimeError(
                "No CJK font found and automatic download failed. "
                "Install fonts-noto-cjk or pass --cjk-font manually."
            ) from exc

    raise RuntimeError(
        "No CJK font found. Install fonts-noto-cjk or pass --cjk-font manually."
    )


def _configure_font(lang: str, cjk_font: Path | None) -> None:
    """Configure matplotlib fonts for the requested language."""
    if lang == "zh":
        font_path = _ensure_cjk_font(cjk_font)
        # Register the font file with matplotlib so that findfont can locate it
        # by its family name.
        fm.fontManager.addfont(str(font_path))
        prop = fm.FontProperties(fname=str(font_path))
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [prop.get_name(), "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    else:
        matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = True


def _save(fig: plt.Figure, name: str, lang: str) -> Path:
    path = ASSETS_DIR / f"{name}_{lang}.png"
    # Use a fixed bounding box so all charts share the exact same output dimensions.
    fig.savefig(path, dpi=300, bbox_inches=None, facecolor="white", edgecolor="none")
    plt.close(fig)
    return path


def plot_patch_times(lang: str) -> Path:
    """Grouped bar chart comparing patch-level execution times."""
    t = LABELS[lang]
    labels = [
        t["cpu_full_pipeline"],
        t["gpu_inference_only"],
        t["gpu_full_pipeline"],
    ]
    original = np.array([28.54, 1.75, 10.21])
    ihcinfer = np.array([19.50, 0.55, 0.69])

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars1 = ax.bar(x - width / 2, original, width, label=t["original"], color=COLOR_ORIGINAL)
    bars2 = ax.bar(x + width / 2, ihcinfer, width, label=t["ihcinfer"], color=COLOR_IHC)

    ax.set_ylabel(t["time_seconds"], fontsize=12, fontweight="bold")
    ax.set_title(t["patch_time_title"], fontsize=15, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(frameon=False, fontsize=11)
    ax.set_yscale("log")
    ax.set_ylim(top=max(original.max(), ihcinfer.max()) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}s",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color="#475569")

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}s",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=COLOR_IHC, fontweight="bold")

    fig.tight_layout()
    return _save(fig, "bench_patch_times", lang)


def plot_speedup(lang: str) -> Path:
    """Horizontal bar chart of speedup factors."""
    t = LABELS[lang]
    metrics = [
        t["cpu_full_pipeline"],
        t["gpu_inference_only"],
        t["gpu_full_pipeline"],
        t["wsi_end_to_end"],
    ]
    speedups = np.array([1.46, 3.17, 14.79, 2.75])
    colors = [COLOR_ACCENT, COLOR_ACCENT, COLOR_IHC, COLOR_WSI_IHC]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_pos = np.arange(len(metrics))
    bars = ax.barh(y_pos, speedups, color=colors, height=0.55)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(t["speedup_factor"], fontsize=12, fontweight="bold")
    ax.set_title(t["speedup_title"], fontsize=15, fontweight="bold", pad=20)
    ax.set_xlim(0, speedups.max() * 1.25)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, val in zip(bars, speedups):
        width = bar.get_width()
        ax.annotate(f"{val:.2f}×",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(6, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=11, fontweight="bold", color="#1e293b")

    fig.tight_layout()
    return _save(fig, "bench_speedup", lang)


def plot_wsi_throughput(lang: str) -> Path:
    """Bar chart comparing WSI throughput in patches per minute."""
    t = LABELS[lang]
    labels = [t["original"], t["ihcinfer"]]
    # 1453 patches in ~10 min vs ~4 min
    patches_per_min = np.array([1453 / 10, 1453 / 4])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(labels, patches_per_min, color=[COLOR_WSI_ORIG, COLOR_WSI_IHC], width=0.5)

    ax.set_ylabel(t["patches_per_minute"], fontsize=12, fontweight="bold")
    ax.set_title(t["wsi_throughput_title"], fontsize=15, fontweight="bold", pad=20)
    ax.set_ylim(0, patches_per_min.max() * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, val in zip(bars, patches_per_min):
        height = bar.get_height()
        ax.annotate(f"{val:.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=12, fontweight="bold", color="#1e293b")

    fig.tight_layout()
    return _save(fig, "bench_wsi_throughput", lang)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate benchmark charts for the ihcinfer README."
    )
    parser.add_argument(
        "--lang",
        choices=["en", "zh", "all"],
        default="all",
        help="Language variant(s) to generate (default: all).",
    )
    parser.add_argument(
        "--cjk-font",
        type=Path,
        default=None,
        help="Path to a CJK font file for Chinese charts. "
             "If omitted, the script tries common system fonts and "
             "falls back to downloading fonts-noto-cjk via apt-get.",
    )
    args = parser.parse_args(argv)

    langs = ["en", "zh"] if args.lang == "all" else [args.lang]
    paths: list[Path] = []
    for lang in langs:
        _configure_font(lang, args.cjk_font)
        paths.extend([
            plot_patch_times(lang),
            plot_speedup(lang),
            plot_wsi_throughput(lang),
        ])

    for p in paths:
        print(f"Saved: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
