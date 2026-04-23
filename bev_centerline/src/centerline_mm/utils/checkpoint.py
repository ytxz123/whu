from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .runtime import ensure_dir


def save_checkpoint(path: str | Path, **items: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(items, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=device)

