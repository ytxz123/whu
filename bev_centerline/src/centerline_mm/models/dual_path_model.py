from __future__ import annotations

import torch
from torch import nn

from .dinov3_task_encoder import DINOv3TaskEncoder, TaskEncoderWithAdapter
from .qwen_mm import Qwen35MMBackbone
from .task_projection import TaskVisualProjector
from .token_injection import TaskTokenInjector


class DualVisionCenterlineModel(nn.Module):
    def __init__(
        self,
        qwen: Qwen35MMBackbone,
        task_encoder: DINOv3TaskEncoder | TaskEncoderWithAdapter,
        projector: TaskVisualProjector,
        injector: TaskTokenInjector | None = None,
    ) -> None:
        super().__init__()
        self.qwen = qwen
        self.task_encoder = task_encoder
        self.projector = projector
        self.injector = injector or TaskTokenInjector()

    def task_tokens(self, dino_images: torch.Tensor) -> torch.Tensor:
        out = self.task_encoder(dino_images)
        return self.projector(out["tokens"], out["grid_size"])

    def forward_loss(self, qwen_inputs: dict, labels: torch.Tensor, dino_images: torch.Tensor) -> torch.Tensor:
        task_embeds = self.task_tokens(dino_images)
        out = self.qwen.forward_with_task_tokens(qwen_inputs, task_embeds, labels, self.injector)
        return out.loss

    @torch.no_grad()
    def generate(self, qwen_inputs: dict, dino_images: torch.Tensor, max_new_tokens: int, temperature: float) -> torch.Tensor:
        task_embeds = self.task_tokens(dino_images)
        return self.qwen.generate_with_task_tokens(
            qwen_inputs,
            task_embeds,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            injector=self.injector,
        )

