"""Standalone DeepLIIF TorchScript inference engine.

This module re-implements the DeepLIIF forward pipeline without importing
``deepliif.models``.  It loads the published TorchScript ``.pt`` checkpoints
from a model directory and runs the multi-generator / multi-segmenter forward
pass directly.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from .base import InferenceModel
from .options import DeepLIIFOptions


def _make_power_of_2(
    img: Image.Image,
    base: int = 4,
    method=Image.Resampling.BICUBIC,
) -> Image.Image:
    """Resize image dimensions to the nearest multiple of *base*."""
    w, h = img.size
    new_w = int(round(w / base) * base)
    new_h = int(round(h / base) * base)
    if new_w == w and new_h == h:
        return img
    return img.resize((new_w, new_h), method)


def preprocess_images(
    images: List[Image.Image],
    scale_size: int = 512,
) -> torch.Tensor:
    """Convert a list of RGB PIL images to a normalized float batch tensor."""
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    tensors = []
    for img in images:
        img = img.convert("RGB").resize(
            (scale_size, scale_size), Image.Resampling.BICUBIC
        )
        img = _make_power_of_2(img, base=4)
        tensors.append(normalize(to_tensor(img)))

    return torch.stack(tensors)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a single ``(C, H, W)`` or ``(1, C, H, W)`` tensor to a PIL image."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    arr = (tensor * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
    arr = (arr.transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


class DeepLIIFModel(InferenceModel):
    """Load and run DeepLIIF TorchScript checkpoints directly.

    Args:
        model_dir: Path containing ``G1.pt`` ... ``G55.pt`` and ``train_opt.txt``.
        device: ``torch.device`` on which to run inference.
    """

    def __init__(self, model_dir: str, device: torch.device) -> None:
        self.device = device
        self.opt = DeepLIIFOptions(model_dir)
        if self.opt.model not in ("DeepLIIF", "DeepLIIFKD"):
            raise NotImplementedError(f"Model {self.opt.model!r} is not supported")

        self.generators: Dict[str, torch.jit.ScriptModule] = {}
        self.segmenters: Dict[str, torch.jit.ScriptModule] = {}

        for i in range(1, self.opt.modalities_no + 1):
            key = f"G{i}"
            self.generators[key] = self._load(key)

        for i in range(1, self.opt.modalities_no + 2):
            key = f"G{self.opt.mod_id_seg}{i}"
            self.segmenters[key] = self._load(key)

        self.num_seg_inputs = self.opt.modalities_no + 1
        self.seg_weight = 1.0 / self.num_seg_inputs

    def _load(self, name: str) -> torch.jit.ScriptModule:
        path = os.path.join(self.opt.model_dir, f"{name}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing TorchScript checkpoint: {path}")
        model = torch.jit.load(path, map_location=self.device)
        model.eval()
        return model

    @property
    def seg_key(self) -> str:
        return f"G{self.opt.mod_id_seg}"

    @property
    def marker_key(self) -> str | None:
        """Return the generator key that corresponds to the Marker modality."""
        for idx, mod_name in enumerate(self.opt.modalities_names):
            if mod_name == "Marker":
                # output generator keys start at G1 for modalities_names[1]
                return f"G{idx}"
        return None

    @torch.inference_mode()
    def forward_tensors(
        self,
        images: List[Image.Image],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Run the DeepLIIF pipeline and return raw tensors instead of PIL images.

        Returns:
            A tuple of (generators_dict, seg_tensor) where ``seg_tensor`` has
            shape ``(B, C, H, W)``.
        """
        if not images:
            return {}, torch.empty(0)

        x = preprocess_images(images, self.opt.scale_size).to(self.device)

        gens: Dict[str, torch.Tensor] = {}
        for i in range(1, self.opt.modalities_no + 1):
            key = f"G{i}"
            gens[key] = self.generators[key](x)

        segs: List[torch.Tensor] = []
        for i in range(1, self.num_seg_inputs + 1):
            key = f"G{self.opt.mod_id_seg}{i}"
            inp = x if i == 1 else gens[f"G{i - 1}"]
            segs.append(self.segmenters[key](inp))

        seg_tensor = torch.stack(segs, dim=0).sum(dim=0) * self.seg_weight
        return gens, seg_tensor

    @torch.inference_mode()
    def forward_arrays(
        self,
        images: List[Image.Image],
    ) -> List[np.ndarray]:
        """Run the DeepLIIF pipeline and return uint8 RGB segmentation arrays.

        This is a scoring-only fast path: it skips PIL creation for both the
        segmentation and marker tensors, reducing host memory and CPU overhead.

        Returns:
            A list of ``(H, W, 3)`` uint8 numpy arrays, one per input image.
        """
        if not images:
            return []

        _, seg_tensor = self.forward_tensors(images)
        seg_tensor = (seg_tensor * 0.5 + 0.5).clamp(0, 1)
        seg_tensor = (seg_tensor.permute(0, 2, 3, 1) * 255).to(torch.uint8)
        return [arr.cpu().numpy() for arr in seg_tensor]

    @torch.inference_mode()
    def forward(
        self,
        images: List[Image.Image],
        return_modalities: bool = False,
    ) -> List[Dict[str, Image.Image]]:
        """Run the full DeepLIIF pipeline on a batch of RGB images.

        Args:
            images: List of equally-sized RGB PIL images.
            return_modalities: If True, include all G1..G4 modality images in
                each result dict.  Otherwise only the Marker (G4) and final
                segmentation (G5) images are returned.

        Returns:
            A list of result dicts, one per input image.  Each dict contains
            at least the keys ``G5`` and ``G4`` (when Marker is present).
        """
        if not images:
            return []

        gens, seg_tensor = self.forward_tensors(images)
        marker_key = self.marker_key
        results: List[Dict[str, Image.Image]] = []
        for b in range(seg_tensor.size(0)):
            res: Dict[str, Image.Image] = {}
            if return_modalities:
                for i in range(1, self.opt.modalities_no + 1):
                    key = f"G{i}"
                    res[key] = tensor_to_pil(gens[key][b])
            elif marker_key is not None:
                res[marker_key] = tensor_to_pil(gens[marker_key][b])

            res[self.seg_key] = tensor_to_pil(seg_tensor[b])
            results.append(res)

        return results
