# 配置说明

配置文件位于 `configs/`。

## 公共路径

```yaml
project_root: ..
paths:
  qwen_path: ../qwen3.5_2B
  dinov3_repo: ../dinov3
  output_dir: outputs/stage1
```

说明：

- `project_root` 相对配置文件目录解析。默认 `..` 指向 `bev_centerline/`。
- `qwen_path` 相对 `project_root` 解析，默认指向工作区中的 `qwen3.5_2B`。
- `dinov3_repo` 相对 `project_root` 解析，默认指向工作区中的 `dinov3`。
- `output_dir` 相对 `project_root` 解析。

## 阶段一关键配置

```yaml
model:
  dinov3_arch: dinov3_vits16
  dinov3_weights: null
  dinov3_pretrained: false
  freeze_backbone: false
  trainable_last_blocks: 2
  feature_dim: 384
```

如果有本地 DINOv3 权重，将 `dinov3_weights` 改成权重路径，并将 `dinov3_pretrained` 保持 false 或省略；代码会在提供 weights 时自动按本地权重加载。

## 阶段二关键配置

```yaml
model:
  dinov3_checkpoint: outputs/stage1/best.pt
  dino_feature_dim: 384
  qwen_hidden_dim: 2048
  task_token_count: 128
  projection_hidden_dim: 1024
```

阶段二默认冻结 Qwen，只训练任务视觉路径尾部和投影/重采样模块。

## 阶段三关键配置

```yaml
model:
  stage2_checkpoint: outputs/stage2/latest.pt
  lora:
    r: 16
    alpha: 32
    dropout: 0.05
```

阶段三默认开启 LoRA，用于学习严格 JSON 输出形式。

