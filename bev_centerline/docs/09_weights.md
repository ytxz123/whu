# 模型权重放置说明

本项目需要两类权重：

1. Qwen3.5 模型本体。
2. DINOv3 backbone 权重。

## 1. Qwen3.5 放在哪里

当前默认路径：

```text
/Users/tzy/PT/whu/Qwen3.5_2B
```

目录里应该至少有：

```text
Qwen3.5_2B/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── preprocessor_config.json
├── model.safetensors.index.json
└── model.safetensors-00001-of-00001.safetensors
```

不用复制到 `bev_centerline` 里。配置默认通过相对路径读取：

```yaml
paths:
  qwen_path: ../Qwen3.5_2B
```

如果你换位置，例如：

```text
/data/models/Qwen3.5_2B
```

就把配置改成绝对路径：

```yaml
paths:
  qwen_path: /data/models/Qwen3.5_2B
```

需要改这些文件：

```text
configs/stage2_alignment.yaml
configs/stage3_json_lora.yaml
configs/infer.yaml
```

阶段一不需要 Qwen，但保留了公共路径字段。

## 2. DINOv3 官方代码放在哪里

当前默认路径：

```text
/Users/tzy/PT/whu/dinov3
```

里面应该有：

```text
dinov3/
├── hubconf.py
├── dinov3/
└── README.md
```

本项目按 DINOv3 官方推荐方式加载：

```python
torch.hub.load(REPO_DIR, "dinov3_vitl16", source="local", weights=WEIGHT_PATH)
```

也就是说：

- `REPO_DIR` 是 `/Users/tzy/PT/whu/dinov3`。
- `WEIGHT_PATH` 是你下载的 DINOv3 backbone `.pth`。

## 3. DINOv3 权重放在哪里

默认使用 ViT-L/16 LVD-1689M：

```text
bev_centerline/weights/dinov3/dinov3_vitl16_pretrain_lvd1689m.pth
```

创建目录：

```bash
cd /Users/tzy/PT/whu/bev_centerline
mkdir -p weights/dinov3
```

把下载好的权重文件放进去后检查：

```bash
ls -lh weights/dinov3/dinov3_vitl16_pretrain_lvd1689m.pth
```

配置位置：

```yaml
model:
  dinov3_arch: dinov3_vitl16
  dinov3_weights: weights/dinov3/dinov3_vitl16_pretrain_lvd1689m.pth
```

需要改这些文件：

```text
configs/stage1_segmentation.yaml
configs/stage2_alignment.yaml
```

阶段三从阶段二 checkpoint 恢复 DINOv3 分支，通常不用再直接读原始 DINOv3 权重。

## 4. DINOv3 模型和维度对应表

常用 ViT backbone：

```text
dinov3_vits16       feature_dim=384
dinov3_vits16plus   feature_dim=384
dinov3_vitb16       feature_dim=768
dinov3_vitl16       feature_dim=1024
dinov3_vith16plus   feature_dim=1280
dinov3_vit7b16      feature_dim=4096
```

如果改 backbone：

阶段一改：

```yaml
model:
  dinov3_arch: dinov3_vitb16
  dinov3_weights: weights/dinov3/your_vitb16.pth
  feature_dim: 768
```

阶段二改：

```yaml
model:
  dinov3_arch: dinov3_vitb16
  dinov3_weights: weights/dinov3/your_vitb16.pth
  dino_feature_dim: 768
```

阶段三改：

```yaml
model:
  dino_feature_dim: 768
```

## 5. DINOv3 图像归一化

DINOv3 官方 README 给了两套归一化：

LVD-1689M：

```text
mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)
```

SAT-493M：

```text
mean=(0.430, 0.411, 0.296)
std=(0.213, 0.156, 0.143)
```

本项目配置：

```yaml
data:
  dino_norm_preset: lvd1689m
```

如果你使用 SAT 权重，改成：

```yaml
data:
  dino_norm_preset: sat493m
```

阶段一、阶段二、阶段三的配置都要保持一致。
