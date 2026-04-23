from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from centerline_mm.data.stage1_dataset import Stage1SegmentationDataset
from centerline_mm.eval.seg_metrics import BinarySegMetrics
from centerline_mm.models.dinov3_task_encoder import DINOv3TaskEncoder
from centerline_mm.models.segmentation import Stage1SegmentationModel
from centerline_mm.train.common import resolve_project_path
from centerline_mm.utils.checkpoint import save_checkpoint
from centerline_mm.utils.config import get_by_path, load_yaml, resolve_paths
from centerline_mm.utils.runtime import ensure_dir, get_device, seed_everything


def evaluate(model: Stage1SegmentationModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    metrics = BinarySegMetrics()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            metrics.update(logits, masks)
    return metrics.compute()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1_segmentation.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = resolve_paths(cfg)
    seed_everything(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    out_dir = ensure_dir(paths.output_dir)

    norm_preset = get_by_path(cfg, "data.dino_norm_preset", "lvd1689m")
    train_ds = Stage1SegmentationDataset(paths.project_root / get_by_path(cfg, "data.train_manifest"), get_by_path(cfg, "data.image_size", 512), norm_preset=norm_preset)
    val_ds = Stage1SegmentationDataset(paths.project_root / get_by_path(cfg, "data.val_manifest"), get_by_path(cfg, "data.image_size", 512), norm_preset=norm_preset)
    train_loader = DataLoader(train_ds, batch_size=get_by_path(cfg, "data.batch_size", 2), shuffle=True, num_workers=get_by_path(cfg, "data.num_workers", 2))
    val_loader = DataLoader(val_ds, batch_size=get_by_path(cfg, "data.batch_size", 2), shuffle=False, num_workers=get_by_path(cfg, "data.num_workers", 2))

    encoder = DINOv3TaskEncoder(
        dinov3_repo=paths.dinov3_repo,
        arch=get_by_path(cfg, "model.dinov3_arch", "dinov3_vitl16"),
        pretrained=get_by_path(cfg, "model.dinov3_pretrained", False),
        weights=resolve_project_path(paths.project_root, get_by_path(cfg, "model.dinov3_weights")),
        freeze_backbone=get_by_path(cfg, "model.freeze_backbone", False),
        trainable_last_blocks=get_by_path(cfg, "model.trainable_last_blocks", 0),
    )
    model = Stage1SegmentationModel(encoder, feature_dim=get_by_path(cfg, "model.feature_dim", encoder.embed_dim)).to(device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=get_by_path(cfg, "optim.lr", 1e-4),
        weight_decay=get_by_path(cfg, "optim.weight_decay", 0.01),
    )

    best_iou = -1.0
    epochs = int(get_by_path(cfg, "optim.epochs", 5))
    log_every = int(get_by_path(cfg, "optim.log_every", 20))
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"stage1 epoch {epoch}/{epochs}")
        for step, batch in enumerate(pbar, 1):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, masks)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item())
            if step % log_every == 0:
                pbar.set_postfix(loss=running / step)

        val_metrics = evaluate(model, val_loader, device)
        print(f"epoch={epoch} val={val_metrics}")
        state = {
            "epoch": epoch,
            "task_encoder": model.task_encoder.state_dict(),
            "seg_head": model.seg_head.state_dict(),
            "metrics": val_metrics,
            "config": cfg,
        }
        save_checkpoint(out_dir / "latest.pt", **state)
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            save_checkpoint(out_dir / "best.pt", **state)


if __name__ == "__main__":
    main()
