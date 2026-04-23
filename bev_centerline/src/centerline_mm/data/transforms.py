from __future__ import annotations

import torch
from PIL import Image
from torchvision import transforms

DINO_NORM_PRESETS = {
    "lvd1689m": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    "sat493m": {"mean": (0.430, 0.411, 0.296), "std": (0.213, 0.156, 0.143)},
    "qwen": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
}


def load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask(path: str) -> Image.Image:
    return Image.open(path).convert("L")


def image_transform(size: int = 512, norm_preset: str = "lvd1689m"):
    preset = DINO_NORM_PRESETS.get(str(norm_preset).lower())
    if preset is None:
        raise ValueError(f"Unknown image norm preset: {norm_preset}. Choose one of {sorted(DINO_NORM_PRESETS)}")
    return transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=preset["mean"], std=preset["std"]),
        ]
    )


def mask_transform(size: int = 512):
    return transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float()),
        ]
    )


def pil_for_qwen(path: str, size: int = 512) -> Image.Image:
    image = load_rgb(path)
    return image.resize((size, size), Image.Resampling.BICUBIC)


def stack_images(images: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(images, dim=0)
