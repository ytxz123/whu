from __future__ import annotations

import torch
from torch import nn


class TaskTokenInjector(nn.Module):
    """Injects task visual tokens as extra language-side context tokens."""

    def __init__(self, insert: str = "after_vision") -> None:
        super().__init__()
        if insert not in {"after_vision", "prefix"}:
            raise ValueError("insert must be 'after_vision' or 'prefix'")
        self.insert = insert

    def forward(
        self,
        text_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
        task_embeds: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        vision_end_token_id: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        b, _, d = text_embeds.shape
        task_len = task_embeds.shape[1]
        ignore = torch.full((b, task_len), -100, dtype=torch.long, device=text_embeds.device)
        task_mask = torch.ones((b, task_len), dtype=attention_mask.dtype, device=text_embeds.device)

        new_embeds, new_masks, new_labels = [], [], []
        for i in range(b):
            pos = 0
            if self.insert == "after_vision" and input_ids is not None and vision_end_token_id is not None:
                matches = (input_ids[i] == vision_end_token_id).nonzero(as_tuple=False).flatten()
                pos = int(matches[-1].item() + 1) if matches.numel() else 0
            row_embed = torch.cat([text_embeds[i, :pos], task_embeds[i], text_embeds[i, pos:]], dim=0)
            row_mask = torch.cat([attention_mask[i, :pos], task_mask[i], attention_mask[i, pos:]], dim=0)
            new_embeds.append(row_embed)
            new_masks.append(row_mask)
            if labels is not None:
                row_label = torch.cat([labels[i, :pos], ignore[i], labels[i, pos:]], dim=0)
                new_labels.append(row_label)

        return (
            torch.stack(new_embeds, dim=0),
            torch.stack(new_masks, dim=0),
            torch.stack(new_labels, dim=0) if labels is not None else None,
        )

