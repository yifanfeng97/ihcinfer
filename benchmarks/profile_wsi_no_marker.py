import cProfile, time
from PIL import Image
import numpy as np
import torch

# Monkey-patch DeepLIIFModel.forward to skip marker-to-PIL conversion.
from ihcinfer.models import DeepLIIFModel, tensor_to_pil

_original_forward = DeepLIIFModel.forward

def _seg_only_forward(self, images, return_modalities=False):
    if not images:
        return []
    gens, seg_tensor = self.forward_tensors(images)
    results = []
    for b in range(seg_tensor.size(0)):
        results.append({self.seg_key: tensor_to_pil(seg_tensor[b])})
    return results

DeepLIIFModel.forward = _seg_only_forward

from ihcinfer import SlideInference

MODEL_DIR = "/home/fengyifan/disk/code/DeepLIIF/model-server/DeepLIIF_Latest_Model"
SVS = "tests/data/slides/98140-6 CD3.svs"

inf = SlideInference(model_dir=MODEL_DIR, gpu_ids=[3], batch_size=8)
t0 = time.perf_counter()
result = inf.run_on_wsi(
    SVS,
    "outputs/wsi_profile_out2",
    num_region_samples=0,
    num_patch_samples=0,
    skip_thumbnail=True,
)
elapsed = time.perf_counter() - t0
print(f"WSI core time (no marker conversion): {elapsed:.2f}s, records: {len(result.records)}")
