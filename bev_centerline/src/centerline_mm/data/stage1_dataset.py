from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from centerline_mm.utils.runtime import read_jsonl

from .transforms import image_transform, load_rgb, load_mask, mask_transform


class Stage1SegmentationDataset(Dataset):
    """JSONL rows: {"image": "...", "mask": "..."}."""

    def __init__(self, manifest: str | Path, image_size: int = 512, norm_preset: str = "lvd1689m") -> None:
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.rows = read_jsonl(self.manifest)
        self.image_tf = image_transform(image_size, norm_preset=norm_preset)
        self.mask_tf = mask_transform(image_size)

    def _resolve(self, value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            path = (self.root / path).resolve()
        return str(path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        image_path = self._resolve(row["image"])
        mask_path = self._resolve(row["mask"])
        return {
            "image": self.image_tf(load_rgb(image_path)),
            "mask": self.mask_tf(load_mask(mask_path)),
            "image_path": image_path,
            "mask_path": mask_path,
        }
