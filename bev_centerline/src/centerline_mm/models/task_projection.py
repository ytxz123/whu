from __future__ import annotations

import math

import torch
from torch import nn


class TwoDimPositionEncoding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, grid_size: tuple[int, int], batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        h, w = grid_size
        ys = torch.linspace(0, 1, h, device=device, dtype=dtype)
        xs = torch.linspace(0, 1, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([xx, yy], dim=-1).reshape(1, h * w, 2)
        return self.proj(coords).expand(batch, -1, -1)


class TokenResampler(nn.Module):
    def __init__(self, dim: int, token_count: int, num_heads: int = 8) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(token_count, dim) / math.sqrt(dim))
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b = tokens.shape[0]
        queries = self.queries.unsqueeze(0).expand(b, -1, -1)
        out, _ = self.attn(queries, tokens, tokens, need_weights=False)
        return self.norm(out)


class TaskVisualProjector(nn.Module):
    """Maps DINOv3 task tokens into Qwen language hidden space."""

    def __init__(
        self,
        dino_dim: int,
        qwen_dim: int,
        hidden_dim: int = 1024,
        task_token_count: int = 128,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dino_dim)
        self.pos = TwoDimPositionEncoding(dino_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dino_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, qwen_dim),
        )
        self.resampler = TokenResampler(qwen_dim, task_token_count)
        self.out_norm = nn.LayerNorm(qwen_dim)

    def forward(self, dino_tokens: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
        pos = self.pos(grid_size, dino_tokens.shape[0], dino_tokens.device, dino_tokens.dtype)
        x = self.norm(dino_tokens + pos)
        x = self.mlp(x)
        x = self.resampler(x)
        return self.out_norm(x)

