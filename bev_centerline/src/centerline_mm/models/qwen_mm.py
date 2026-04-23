from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn

from .token_injection import TaskTokenInjector


class Qwen35MMBackbone(nn.Module):
    """Qwen3.5 multimodal wrapper preserving the native visual path unchanged."""

    def __init__(self, model_path: str | Path, dtype: str = "bfloat16", device: torch.device | None = None) -> None:
        super().__init__()
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype, torch.bfloat16)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if device is not None:
            self.model.to(device)

        cfg = self.model.config
        self.image_token_id = int(getattr(cfg, "image_token_id"))
        self.vision_end_token_id = int(getattr(cfg, "vision_end_token_id"))
        text_cfg = getattr(cfg, "text_config", cfg)
        self.hidden_size = int(getattr(text_cfg, "hidden_size"))

    def freeze_language_and_native_vision(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False

    def enable_input_grads(self) -> None:
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

    def _messages(self, prompt: str, image: Image.Image | str | Path) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def build_inputs(
        self,
        images: list[Image.Image | str | Path],
        prompts: list[str],
        targets: list[str] | None = None,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        texts = []
        for image, prompt, target in zip(images, prompts, targets or [None] * len(images)):
            text = self.processor.apply_chat_template(
                self._messages(prompt, image),
                tokenize=False,
                add_generation_prompt=True,
            )
            if target is not None:
                text = text + target
            texts.append(text)
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        if device is not None:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        return batch

    def build_labels(
        self,
        images: list[Image.Image | str | Path],
        prompts: list[str],
        targets: list[str],
        full_inputs: dict[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        prompt_inputs = self.build_inputs(images, prompts, targets=None, device=device)
        labels = full_inputs["input_ids"].clone()
        labels[full_inputs["attention_mask"] == 0] = -100
        for i in range(labels.shape[0]):
            prompt_len = int(prompt_inputs["attention_mask"][i].sum().item())
            labels[i, :prompt_len] = -100
        return labels

    def native_inputs_embeds(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"]
        embeds = self.model.get_input_embeddings()(input_ids)
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        if pixel_values is None:
            return embeds

        visual = getattr(self.model, "visual", None)
        if visual is None:
            raise AttributeError("Qwen model does not expose a native visual tower as .visual")
        image_embeds = visual(pixel_values, grid_thw=image_grid_thw)
        image_mask = (input_ids == self.image_token_id).unsqueeze(-1).expand_as(embeds)
        if image_mask.sum().item() != image_embeds.numel():
            raise RuntimeError(
                "Image token count does not match Qwen native visual embeddings. "
                "Check processor/model compatibility."
            )
        return embeds.masked_scatter(image_mask, image_embeds.to(embeds.dtype))

    def forward_with_task_tokens(
        self,
        inputs: dict[str, torch.Tensor],
        task_embeds: torch.Tensor,
        labels: torch.Tensor | None = None,
        injector: TaskTokenInjector | None = None,
    ) -> Any:
        injector = injector or TaskTokenInjector()
        text_embeds = self.native_inputs_embeds(inputs)
        embeds, attn, labels = injector(
            text_embeds=text_embeds,
            attention_mask=inputs["attention_mask"],
            labels=labels,
            task_embeds=task_embeds.to(text_embeds.dtype),
            input_ids=inputs["input_ids"],
            vision_end_token_id=self.vision_end_token_id,
        )
        return self.model(inputs_embeds=embeds, attention_mask=attn, labels=labels, use_cache=False)

    @torch.no_grad()
    def generate_with_task_tokens(
        self,
        inputs: dict[str, torch.Tensor],
        task_embeds: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        injector: TaskTokenInjector | None = None,
    ) -> torch.Tensor:
        injector = injector or TaskTokenInjector()
        text_embeds = self.native_inputs_embeds(inputs)
        embeds, attn, _ = injector(
            text_embeds=text_embeds,
            attention_mask=inputs["attention_mask"],
            labels=None,
            task_embeds=task_embeds.to(text_embeds.dtype),
            input_ids=inputs["input_ids"],
            vision_end_token_id=self.vision_end_token_id,
        )
        do_sample = temperature > 0
        return self.model.generate(
            inputs_embeds=embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
        )
