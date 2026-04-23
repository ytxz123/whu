from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from centerline_mm.data.stage1_dataset import Stage1SegmentationDataset
from centerline_mm.eval.seg_metrics import BinarySegMetrics
from centerline_mm.models.dinov3_task_encoder import DINOv3TaskEncoder
from centerline_mm.models.segmentation import Stage1SegmentationModel
from centerline_mm.utils.checkpoint import load_checkpoint
from centerline_mm.utils.config import get_by_path, load_yaml, resolve_paths
from centerline_mm.utils.runtime import ensure_dir, get_device, save_json


def denorm(img: torch.Tensor) -> torch.Tensor:
    return (img.detach().cpu() * 0.5 + 0.5).clamp(0, 1)


def save_vis(path: Path, image: torch.Tensor, gt: torch.Tensor, pred: torch.Tensor) -> None:
    image_np = denorm(image).permute(1, 2, 0).numpy()
    gt_np = gt.detach().cpu().squeeze().numpy()
    pred_np = pred.detach().cpu().squeeze().numpy()
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(image_np)
    axes[0].set_title("Input")
    axes[1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("GT")
    axes[2].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Pred")
    axes[3].imshow(image_np)
    axes[3].imshow(pred_np, cmap="Reds", alpha=0.45, vmin=0, vmax=1)
    axes[3].imshow(gt_np, cmap="Greens", alpha=0.25, vmin=0, vmax=1)
    axes[3].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1_segmentation.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-vis", type=int, default=32)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = resolve_paths(cfg)
    device = get_device(cfg.get("device", "auto"))
    ckpt_path = Path(args.checkpoint or (paths.project_root / get_by_path(cfg, "paths.output_dir", "outputs/stage1") / "best.pt"))
    if not ckpt_path.is_absolute():
        ckpt_path = paths.project_root / ckpt_path
    manifest = Path(args.manifest or (paths.project_root / get_by_path(cfg, "data.val_manifest")))
    if not manifest.is_absolute():
        manifest = paths.project_root / manifest
    out_dir = ensure_dir(Path(args.out_dir or (ckpt_path.parent / "eval")))
    vis_dir = ensure_dir(out_dir / "visualizations")

    ds = Stage1SegmentationDataset(manifest, get_by_path(cfg, "data.image_size", 512))
    loader = DataLoader(ds, batch_size=get_by_path(cfg, "data.batch_size", 2), shuffle=False, num_workers=get_by_path(cfg, "data.num_workers", 2))

    encoder = DINOv3TaskEncoder(
        dinov3_repo=paths.dinov3_repo,
        arch=get_by_path(cfg, "model.dinov3_arch", "dinov3_vits16"),
        pretrained=False,
        freeze_backbone=False,
    )
    model = Stage1SegmentationModel(encoder, feature_dim=get_by_path(cfg, "model.feature_dim", encoder.embed_dim)).to(device)
    state = load_checkpoint(ckpt_path)
    model.task_encoder.load_state_dict(state["task_encoder"], strict=False)
    model.seg_head.load_state_dict(state["seg_head"], strict=True)
    model.eval()

    metrics = BinarySegMetrics()
    vis_count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="stage1 eval"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            metrics.update(logits, masks)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            for i in range(images.shape[0]):
                if vis_count >= args.max_vis:
                    break
                save_vis(vis_dir / f"{vis_count:05d}.png", images[i], masks[i], preds[i])
                vis_count += 1

    result = metrics.compute()
    save_json(out_dir / "metrics.json", result)
    print(result)


if __name__ == "__main__":
    main()

