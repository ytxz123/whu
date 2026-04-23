from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class DINOv3TaskEncoder(nn.Module):
    """External task vision path. It never touches Qwen's native visual tower."""

    def __init__(
        self,
        dinov3_repo: str | Path,
        arch: str = "dinov3_vitl16",
        pretrained: bool = False,
        weights: str | None = None,
        freeze_backbone: bool = False,
        trainable_last_blocks: int = 0,
    ) -> None:
        super().__init__()
        if "convnext" in arch:
            raise ValueError(
                "This project expects DINOv3 ViT patch tokens from forward_features(). "
                "Use a ViT backbone such as dinov3_vits16, dinov3_vitb16, or dinov3_vitl16."
            )
        kwargs = {"pretrained": pretrained}
        if weights:
            kwargs["weights"] = weights
            kwargs["pretrained"] = True
        self.backbone = torch.hub.load(str(Path(dinov3_repo).resolve()), arch, source="local", **kwargs)
        self.patch_size = int(getattr(self.backbone, "patch_size", 16))
        self.embed_dim = int(getattr(self.backbone, "embed_dim", 384))

        if freeze_backbone:
            self.freeze_all()
        elif trainable_last_blocks > 0:
            self.freeze_except_last_blocks(trainable_last_blocks)

    def freeze_all(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def freeze_except_last_blocks(self, n: int) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is not None:
            for block in list(blocks)[-n:]:
                for p in block.parameters():
                    p.requires_grad = True
        for name in ("norm", "cls_norm", "local_cls_norm"):
            mod = getattr(self.backbone, name, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = True

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor | tuple[int, int]]:
        features = self.backbone.forward_features(images)
        patch_tokens = features["x_norm_patchtokens"]
        h = images.shape[-2] // self.patch_size
        w = images.shape[-1] // self.patch_size
        return {
            "tokens": patch_tokens,
            "grid_size": (h, w),
            "cls": features["x_norm_clstoken"],
        }

    def load_task_checkpoint(self, path: str | Path, strict: bool = False) -> None:
        state = torch.load(path, map_location="cpu")
        state_dict = state.get("task_encoder", state.get("model", state))
        self.load_state_dict(state_dict, strict=strict)


class TaskEncoderWithAdapter(nn.Module):
    """Small trainable tail around DINOv3 tokens for stage 2/3 reuse."""

    def __init__(self, encoder: DINOv3TaskEncoder, dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.adapter = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor | tuple[int, int]]:
        out = self.encoder(images)
        tokens = out["tokens"]
        out["tokens"] = tokens + self.adapter(tokens)
        return out
