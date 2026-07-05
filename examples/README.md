# ihcinfer 示例索引

这里集中了 `ihcinfer` 的常用示例与可执行命令。所有脚本都假设你已经安装好包：

```bash
uv sync
```

如果你省略 `--model_dir`，DeepLIIF 模型会在首次使用时自动从 Zenodo 下载（约 3 GB）。

---

## 快速命令

```bash
# 1. 组织分割（无需模型）
ihc tissue_seg \
    --input "tests/data/slides/98140-6 CD3.svs" \
    --output_dir ./tissue_mask \
    --overlay

# 2. Patch 推理
ihc patch_infer \
    --input tests/data/patches/22_2.png \
    --output_dir ./patch_outputs

# 3. 全片 IHC 推理
ihc infer \
    --slide_path /path/to/slide.svs \
    --output_dir ./ihc_outputs \
    --gpu_ids 0 \
    --batch_size 8
```

---

## 示例脚本

| 脚本 | 功能 | 关键参数 |
|---|---|---|
| [`infer_patch.py`](infer_patch.py) | Patch / 目录批量推理 | `--input`, `--output_dir`, `--save_marker` |
| [`infer_ihc.py`](infer_ihc.py) | 全片 IHC 推理 + 热力图/叠加图 | `--slide_path`, `--batch_size`, `--patch_size`, `--region_size` |
| [`segment_tissue.py`](segment_tissue.py) | WSI / 图像组织分割 | `--input`, `--output_dir`, `--overlay`, `--mode ihc\|clam` |

---

## 更多资源

- **主 README**：安装、功能概览、架构图、输出图库 — [`../README.md`](../README.md)
- **CLI 帮助**：`ihc --help`、`ihc tissue_seg --help` 等
- **性能基准**：[`../benchmarks/`](../benchmarks/)
- **API 详细说明**：见主 README 的 “Python API” 与 “Advanced Usage” 部分
