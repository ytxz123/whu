from __future__ import annotations

from pathlib import Path

import torch

from centerline_mm.models.dinov3_task_encoder import DINOv3TaskEncoder, TaskEncoderWithAdapter
from centerline_mm.models.dual_path_model import DualVisionCenterlineModel
from centerline_mm.models.qwen_mm import Qwen35MMBackbone
from centerline_mm.models.task_projection import TaskVisualProjector
from centerline_mm.models.token_injection import TaskTokenInjector
from centerline_mm.utils.checkpoint import load_checkpoint
from centerline_mm.utils.config import get_by_path, resolve_paths


def build_dual_model(cfg: dict, device: torch.device, load_stage2: bool = False) -> DualVisionCenterlineModel:
    paths = resolve_paths(cfg)
    qwen = Qwen35MMBackbone(paths.qwen_path, dtype=get_by_path(cfg, "model.qwen_dtype", "bfloat16"), device=device)
    encoder = DINOv3TaskEncoder(
        dinov3_repo=paths.dinov3_repo,
        arch=get_by_path(cfg, "model.dinov3_arch", "dinov3_vits16"),
        pretrained=get_by_path(cfg, "model.dinov3_pretrained", False),
        weights=get_by_path(cfg, "model.dinov3_weights"),
        freeze_backbone=False,
        trainable_last_blocks=get_by_path(cfg, "model.trainable_last_blocks", 0),
    )
    stage1_ckpt = get_by_path(cfg, "model.dinov3_checkpoint")
    if stage1_ckpt and not load_stage2:
        stage1_path = Path(stage1_ckpt)
        if not stage1_path.is_absolute():
            stage1_path = paths.project_root / stage1_path
        state = load_checkpoint(stage1_path)
        encoder.load_state_dict(state["task_encoder"], strict=False)

    dino_dim = get_by_path(cfg, "model.dino_feature_dim", encoder.embed_dim)
    task_encoder = TaskEncoderWithAdapter(encoder, dino_dim)
    projector = TaskVisualProjector(
        dino_dim=dino_dim,
        qwen_dim=get_by_path(cfg, "model.qwen_hidden_dim", qwen.hidden_size),
        hidden_dim=get_by_path(cfg, "model.projection_hidden_dim", 1024),
        task_token_count=get_by_path(cfg, "model.task_token_count", 128),
    )
    model = DualVisionCenterlineModel(qwen, task_encoder, projector, TaskTokenInjector()).to(device)

    if load_stage2:
        ckpt_path = Path(get_by_path(cfg, "model.stage2_checkpoint"))
        if not ckpt_path.is_absolute():
            ckpt_path = paths.project_root / ckpt_path
        state = load_checkpoint(ckpt_path)
        model.task_encoder.load_state_dict(state["task_encoder"], strict=False)
        model.projector.load_state_dict(state["projector"], strict=True)
        if "injector" in state:
            model.injector.load_state_dict(state["injector"], strict=False)
    return model


def trainable_parameters(model: torch.nn.Module):
    return [p for p in model.parameters() if p.requires_grad]

