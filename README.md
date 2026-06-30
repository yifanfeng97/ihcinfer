# ihcinfer

`ihcinfer` 是一个基于 DeepLIIF 的轻量级快速推理库，专门用于大尺寸数字切片（SVS / KFB）的 patch 级批量推理。

PyPI 发行包名为 `ihcinfer`，安装后仍通过 `import ihcinfer` 使用。

## 目标

- 读取 SVS / KFB 格式全片图像
- 按 patch 批量推理，提高 GPU 利用率
- 输出 patch 坐标 + 细胞计数 CSV
- 可选输出 patch 级分割 mask / 热力图

## 安装

```bash
pip install ihcinfer
# 或从源码安装
cd /home/fengyifan/disk/code/ihcinfer
uv sync
```

## 快速开始

```python
from ihcinfer import SlideInference

inf = SlideInference(
    model_dir="/path/to/DeepLIIF/model-server/DeepLIIF_Latest_Model",
    gpu_ids=[0],
    batch_size=16,
)

result = inf.run_on_wsi(
    slide_path="/path/to/slide.svs",
    output_dir="/path/to/output",
)

print(result.csv_path)
print(result.heatmap_path)
print(f"Region samples: {len(result.region_sample_paths) // 2}")
print(f"Patch samples: {len(result.patch_sample_dirs)}")
```

## 项目结构

```
ihcinfer/
├── ihcinfer/
│   ├── __init__.py       # 公共 API：SlideInference
│   ├── inference/        # 推理入口（SlideInference / PatchInference / RegionInference）
│   ├── models/           # DeepLIIF 模型加载与运行
│   ├── prep/             # 切片、组织 mask、空白 patch 过滤
│   ├── readers/          # WSI 读取抽象层（OpenSlide / KFB / PIL）
│   ├── scoring/          # 细胞计数与后处理
│   └── outputs/          # CSV / 热力图 / patch 输出保存
├── tests/                # 测试
├── SPEC.md               # 需求规格
└── PLAN.md               # 实施计划
```

## 高级用法

除公共 API `SlideInference` 外，高级类/函数仍可通过子模块显式导入：

```python
from ihcinfer.inference import PatchInference, RegionInference
from ihcinfer.models import DeepLIIFModel
from ihcinfer.prep import Tiler, TissueSegmenter, TissueMask
from ihcinfer.readers import create_reader
from ihcinfer.scoring import compute_scoring, extract_cells
from ihcinfer.outputs import build_patch_output, save_patch_output, build_heatmap
```

## 测试

```bash
uv run pytest tests/ -v
```

## 性能对比

测试环境：6× NVIDIA RTX 3090 / 24 GiB，DeepLIIF TorchScript 模型，4 张 `512×512` patch（patch 级 benchmark）和 `tests/data/slides/98140-6 CD3.svs` 全片（WSI benchmark）。

### Patch 级推理（CPU）

| 指标 | DeepLIIF original | ihcinfer | 加速比 |
|------|-------------------|--------------|--------|
| 模型加载 | 1.47s | 1.42s | **1.04x** |
| 纯推理 | 18.76s (4.69s/patch) | 19.55s (4.89s/patch) | 0.96x |
| 完整流程（推理 + 后处理） | 28.54s (7.13s/patch) | 19.50s (4.88s/patch) | **1.46x** |

### Patch 级推理（GPU: cuda:0）

| 指标 | DeepLIIF original | ihcinfer | 加速比 |
|------|-------------------|--------------|--------|
| 模型加载 | 1.94s | 1.44s | **1.35x** |
| 纯推理 | 1.75s (0.44s/patch) | 0.55s (0.14s/patch) | **3.17x** |
| 完整流程（推理 + 后处理） | 10.21s (2.55s/patch) | 0.69s (0.17s/patch) | **14.79x** |

### WSI / Region 级推理

- **Region（1024×1024，CPU）**：DeepLIIF original 顺序处理 4 张 tile 约 25.60s；ihcinfer `run_on_region` 约 19.83s（不过滤 tissue）/ 14.49s（按 `min_ratio=0.05` 过滤），分别快 **1.29x** / **1.77x**。
- **完整 WSI（GPU: cuda:0，1453 张 512×512 tissue patches）**：
  - ihcinfer 完整流程（CSV + heatmap + thumbnail + overlay + 2 region samples + 4 patch samples）：**3m46s**（≈0.16s/patch）。
  - ihcinfer 推理核心（CSV + heatmap，跳过可视化采样和 thumbnail）：**210s**（≈0.14s/patch）。
  - 用同一张切片的 50 张实际 patch 对比：DeepLIIF original 顺序完整流程约 **0.41s/patch**，ihcinfer scoring-only 路径约 **0.11s/patch**、完整图片返回路径约 **0.12s/patch**，单 patch 加速约 **3.3–3.7x**。
  - 按此比例推算，相同 1453 张 patch 的 original 顺序流程约需 **10 分钟**（仅推理+后处理），ihcinfer 端到端约 **4 分钟**，WSI 端到端加速约 **2.5–3x**。

本次重构后在 WSI 主流程启用了 scoring-only 快速路径：`_PatchBuffer` 在计算 scoring 时不再生成 seg/marker PIL 图片，进一步降低了 host 内存峰值，同时保持 patch/region 可视化路径不变。完整流程在 GPU 下提升最明显，主要因为 ihcinfer 把推理和后处理串得更紧凑，减少了原始 DeepLIIF pipeline 中的数据流转开销；WSI 场景下还通过 tissue mask 预过滤、chunk 批量读取、跨 chunk 的 patch buffer 以及 scoring-only 路径进一步减少冗余工作。

复现方式（需要原始 DeepLIIF 仓库在 `PYTHONPATH` 中）：

```bash
# patch 级对比
PYTHONPATH=/path/to/DeepLIIF uv run python benchmarks/bench_custom_vs_original.py --device cuda:0

# region 级对比
PYTHONPATH=/path/to/DeepLIIF uv run python benchmarks/bench_region_inference.py

# WSI 完整流程计时
uv run python examples/infer_wsi.py \
  --model_dir /path/to/DeepLIIF_Latest_Model \
  --slide_path /path/to/slide.svs \
  --output_dir ./wsi_outputs \
  --gpu_ids 0 --batch_size 8
```
