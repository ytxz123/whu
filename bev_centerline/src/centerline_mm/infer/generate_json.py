from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from centerline_mm.data.generation_dataset import STAGE3_PROMPT
from centerline_mm.data.transforms import image_transform, load_rgb, pil_for_qwen
from centerline_mm.models.dual_path_model import DualVisionCenterlineModel
from centerline_mm.train.common import build_dual_model
from centerline_mm.train.stage3_train_json_lora import enable_lora
from centerline_mm.utils.checkpoint import load_checkpoint
from centerline_mm.utils.config import get_by_path, load_yaml, resolve_paths
from centerline_mm.utils.json_format import dumps_strict_centerline_json, empty_result
from centerline_mm.utils.runtime import get_device


def extract_json(text: str) -> str:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        return text[start : end + 1]
    return empty_result()


def build_model_for_infer(infer_cfg: dict, checkpoint: dict, device: torch.device) -> DualVisionCenterlineModel:
    train_cfg = checkpoint.get("config", infer_cfg)
    train_cfg.setdefault("paths", {})
    paths = resolve_paths(infer_cfg)
    train_cfg["paths"]["qwen_path"] = str(paths.qwen_path)
    train_cfg["paths"]["dinov3_repo"] = str(paths.dinov3_repo)
    model = build_dual_model(train_cfg, device=device, load_stage2=False)
    enable_lora(model, train_cfg)
    model.task_encoder.load_state_dict(checkpoint["task_encoder"], strict=False)
    model.projector.load_state_dict(checkpoint["projector"], strict=True)
    if "injector" in checkpoint:
        model.injector.load_state_dict(checkpoint["injector"], strict=False)
    if "qwen_lora" in checkpoint:
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(model.qwen.model, checkpoint["qwen_lora"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/infer.yaml")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt", default=STAGE3_PROMPT)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = resolve_paths(cfg)
    device = get_device(cfg.get("device", "auto"))
    ckpt_path = Path(args.checkpoint or get_by_path(cfg, "paths.checkpoint"))
    if not ckpt_path.is_absolute():
        ckpt_path = paths.project_root / ckpt_path
    checkpoint = load_checkpoint(ckpt_path, device="cpu")
    model = build_model_for_infer(cfg, checkpoint, device)

    image_size = int(cfg.get("image_size", 512))
    qwen_image = pil_for_qwen(args.image, image_size)
    train_cfg = checkpoint.get("config", cfg)
    dino_image = image_transform(image_size, norm_preset=get_by_path(train_cfg, "data.dino_norm_preset", "lvd1689m"))(load_rgb(args.image)).unsqueeze(0).to(device)
    qwen_inputs = model.qwen.build_inputs([qwen_image], [args.prompt], targets=None, device=device)
    with torch.no_grad():
        ids = model.generate(
            qwen_inputs,
            dino_image,
            max_new_tokens=int(cfg.get("max_new_tokens", 512)),
            temperature=float(cfg.get("temperature", 0.0)),
        )
    text = model.qwen.processor.batch_decode(ids, skip_special_tokens=True)[0]
    try:
        print(dumps_strict_centerline_json(json.loads(extract_json(text))))
    except Exception:
        print(empty_result())


if __name__ == "__main__":
    main()
