# 配置说明

配置文件位于 `configs/`。

## 公共路径

```yaml
project_root: ..
paths:
  qwen_path: ../Qwen3.5_2B
  dinov3_repo: ../dinov3
  output_dir: outputs/stage1
```

说明：

- `project_root` 相对配置文件目录解析。默认 `..` 指向 `bev_centerline/`。
- `qwen_path` 相对 `project_root` 解析，默认指向工作区中的 `Qwen3.5_2B`。
- `dinov3_repo` 相对 `project_root` 解析，默认指向工作区中的 `dinov3`。
- `output_dir` 相对 `project_root` 解析。

## 阶段一关键配置

```yaml
data:
  train_manifest: data/dataset_test/data_seg/train.jsonl
  val_manifest: data/dataset_test/data_seg/val.jsonl
  image_size: 512
  dino_norm_preset: lvd1689m

model:
  dinov3_arch: dinov3_vitl16
  dinov3_weights: weights/dinov3/dinov3_vitl16_pretrain_lvd1689m.pth
  dinov3_pretrained: false
  freeze_backbone: false
  trainable_last_blocks: 2
  feature_dim: 1024
```

`dinov3_weights` 是相对 `bev_centerline/` 的路径。代码使用 DINOv3 官方方式：

```python
torch.hub.load(dinov3_repo, dinov3_arch, source="local", weights=dinov3_weights)
```

如果换成 SAT-493M 权重，同时改：

```yaml
data:
  dino_norm_preset: sat493m
```

## 阶段二关键配置

```yaml
data:
  train_manifest: data/dataset_test/data_line/train.jsonl
  use_dataset_prompt: false

model:
  dinov3_checkpoint: outputs/stage1/best.pt
  dino_feature_dim: 1024
  qwen_hidden_dim: 2048
  task_token_count: 128
  projection_hidden_dim: 1024
```

阶段二默认冻结 Qwen，只训练任务视觉路径尾部和投影/重采样模块。`use_dataset_prompt: false` 表示使用项目内置坐标级对齐 prompt；dataset_builder 的 prompt 仍可读取，但默认不直接使用。

## 阶段三关键配置

```yaml
model:
  stage2_checkpoint: outputs/stage2/latest.pt
  dinov3_arch: dinov3_vitl16
  dino_feature_dim: 1024
  lora:
    r: 16
    alpha: 32
    dropout: 0.05
```

阶段三默认开启 LoRA，用于学习严格 JSON 输出形式。

## 体检配置

训练前运行：

```bash
./scripts/check_setup.sh
```

它会同时读取：

```text
configs/stage1_segmentation.yaml
configs/stage2_alignment.yaml
configs/stage3_json_lora.yaml
```

并检查数据、权重、Qwen 目录、DINOv3 目录是否能对上。
