from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from centerline_mm.utils.geometry_text import to_coordinate_text
from centerline_mm.utils.json_format import dumps_strict_centerline_json
from centerline_mm.utils.runtime import read_jsonl

from .sharegpt import extract_assistant_payload, extract_image_path, extract_prompt
from .transforms import image_transform, load_rgb, pil_for_qwen


STAGE2_PROMPT = (
    "You are given a 512x512 black-background BEV road-structure image. "
    "Use the native image context and the injected task visual tokens. "
    "Output only the coordinate-level intermediate representation. "
    "Use lines like 'LINE x,y x,y'. Use 'NO_LINES' when no valid path exists."
)

STAGE3_PROMPT = (
    "You are given a 512x512 black-background BEV road-structure image. "
    "Recover every valid road centerline. Output only strict JSON with keys "
    "role, content, lines, category, points. Coordinates must be integers in [0,512]."
)


class CenterlineGenerationDataset(Dataset):
    """Supports dataset_builder ShareGPT rows and simple rows.

    dataset_builder line rows:
    {"messages": [...], "images": ["images/000001.png"]}

    Simple rows:
    {"image": "...", "target": {... or str}}

    Stage 2 converts the target to a coordinate text intermediate.
    Stage 3 normalizes the target to strict JSON.
    """

    def __init__(
        self,
        manifest: str | Path,
        image_size: int = 512,
        stage: int = 3,
        norm_preset: str = "lvd1689m",
        use_dataset_prompt: bool = False,
    ) -> None:
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.rows = read_jsonl(self.manifest)
        self.image_tf = image_transform(image_size, norm_preset=norm_preset)
        self.image_size = image_size
        self.stage = stage
        self.use_dataset_prompt = use_dataset_prompt

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        image_path = extract_image_path(row, self.root)
        target = extract_assistant_payload(row)
        if self.stage == 2:
            prompt = extract_prompt(row, STAGE2_PROMPT) if self.use_dataset_prompt else STAGE2_PROMPT
            target_text = row.get("target_text", to_coordinate_text(target))
        else:
            prompt = extract_prompt(row, STAGE3_PROMPT) if self.use_dataset_prompt else STAGE3_PROMPT
            target_text = dumps_strict_centerline_json(target)
        return {
            "image_path": image_path,
            "qwen_image": pil_for_qwen(image_path, self.image_size),
            "dino_image": self.image_tf(load_rgb(image_path)),
            "prompt": prompt,
            "target": target_text,
        }


def generation_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image_paths": [x["image_path"] for x in batch],
        "qwen_images": [x["qwen_image"] for x in batch],
        "dino_images": __import__("torch").stack([x["dino_image"] for x in batch], dim=0),
        "prompts": [x["prompt"] for x in batch],
        "targets": [x["target"] for x in batch],
    }
