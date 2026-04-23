from __future__ import annotations

import torch


class BinarySegMetrics:
    def __init__(self, threshold: float = 0.5, eps: float = 1e-7) -> None:
        self.threshold = threshold
        self.eps = eps
        self.reset()

    def reset(self) -> None:
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, masks: torch.Tensor) -> None:
        pred = (torch.sigmoid(logits) >= self.threshold).float()
        gt = (masks >= 0.5).float()
        self.tp += float((pred * gt).sum().item())
        self.fp += float((pred * (1 - gt)).sum().item())
        self.fn += float(((1 - pred) * gt).sum().item())

    def compute(self) -> dict[str, float]:
        precision = self.tp / (self.tp + self.fp + self.eps)
        recall = self.tp / (self.tp + self.fn + self.eps)
        iou = self.tp / (self.tp + self.fp + self.fn + self.eps)
        dice = (2 * self.tp) / (2 * self.tp + self.fp + self.fn + self.eps)
        return {
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
        }

