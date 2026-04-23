from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SegmentationHead(nn.Module):
    """Lightweight stage-1-only head. It is discarded after supervising DINOv3."""

    def __init__(self, in_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, 1),
        )

    def forward(self, tokens: torch.Tensor, grid_size: tuple[int, int], out_size: tuple[int, int]) -> torch.Tensor:
        b, _, c = tokens.shape
        h, w = grid_size
        x = tokens.transpose(1, 2).reshape(b, c, h, w)
        logits = self.head(x)
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)


class Stage1SegmentationModel(nn.Module):
    def __init__(self, task_encoder: nn.Module, feature_dim: int) -> None:
        super().__init__()
        self.task_encoder = task_encoder
        self.seg_head = SegmentationHead(feature_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        out = self.task_encoder(images)
        return self.seg_head(out["tokens"], out["grid_size"], images.shape[-2:])

