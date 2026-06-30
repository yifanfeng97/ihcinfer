"""DeepLIIF model option container."""

from __future__ import annotations

import os
import re


class DeepLIIFOptions:
    """Minimal option container parsed from a DeepLIIF model directory."""

    def __init__(self, model_dir: str) -> None:
        self.model_dir = os.path.abspath(model_dir)
        self.model = "DeepLIIF"
        self.scale_size = 512
        self.input_no = 1
        self.mod_id_seg = 5
        self.modalities_no = 4
        self.modalities_names = ["IHC", "Hema", "DAPI", "Lap2", "Marker"]
        self.seg_gen = True
        self._parse_opt_file()

    def _parse_opt_file(self) -> None:
        """Read ``train_opt.txt`` / ``test_opt.txt`` for known fields."""
        for fname in ("test_opt.txt", "train_opt.txt"):
            path = os.path.join(self.model_dir, fname)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()

                m = re.search(r"^\s*load_size:\s*(\d+)", text, re.MULTILINE)
                if m:
                    self.scale_size = int(m.group(1))

                m = re.search(r"^\s*model:\s*(\S+)", text, re.MULTILINE)
                if m:
                    self.model = m.group(1)

                m = re.search(r"^\s*targets_no:\s*(\d+)", text, re.MULTILINE)
                if m:
                    self.modalities_no = int(m.group(1)) - self.input_no

                break
