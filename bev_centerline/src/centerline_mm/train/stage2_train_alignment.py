from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from centerline_mm.data.generation_dataset import CenterlineGenerationDataset, generation_collate
from centerline_mm.train.common import build_dual_model, trainable_parameters
from centerline_mm.utils.checkpoint import save_checkpoint
from centerline_mm.utils.config import get_by_path, load_yaml, resolve_paths
from centerline_mm.utils.runtime import ensure_dir, get_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2_alignment.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = resolve_paths(cfg)
    seed_everything(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    out_dir = ensure_dir(paths.output_dir)

    ds = CenterlineGenerationDataset(
        paths.project_root / get_by_path(cfg, "data.train_manifest"),
        get_by_path(cfg, "data.image_size", 512),
        stage=2,
        norm_preset=get_by_path(cfg, "data.dino_norm_preset", "lvd1689m"),
        use_dataset_prompt=get_by_path(cfg, "data.use_dataset_prompt", False),
    )
    loader = DataLoader(ds, batch_size=get_by_path(cfg, "data.batch_size", 1), shuffle=True, num_workers=get_by_path(cfg, "data.num_workers", 1), collate_fn=generation_collate)

    model = build_dual_model(cfg, device=device, load_stage2=False)
    model.qwen.freeze_language_and_native_vision()
    model.qwen.enable_input_grads()

    opt = torch.optim.AdamW(trainable_parameters(model), lr=get_by_path(cfg, "optim.lr", 5e-5), weight_decay=get_by_path(cfg, "optim.weight_decay", 0.01))
    epochs = int(get_by_path(cfg, "optim.epochs", 2))
    log_every = int(get_by_path(cfg, "optim.log_every", 10))
    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"stage2 epoch {epoch}/{epochs}")
        for step, batch in enumerate(pbar, 1):
            dino_images = batch["dino_images"].to(device)
            qwen_inputs = model.qwen.build_inputs(batch["qwen_images"], batch["prompts"], batch["targets"], device=device)
            labels = model.qwen.build_labels(batch["qwen_images"], batch["prompts"], batch["targets"], qwen_inputs, device)
            loss = model.forward_loss(qwen_inputs, labels, dino_images)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % log_every == 0:
                pbar.set_postfix(loss=float(loss.item()))

        save_checkpoint(
            out_dir / "latest.pt",
            epoch=epoch,
            task_encoder=model.task_encoder.state_dict(),
            projector=model.projector.state_dict(),
            injector=model.injector.state_dict(),
            config=cfg,
        )


if __name__ == "__main__":
    main()
