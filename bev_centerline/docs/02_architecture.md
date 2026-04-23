# 架构设计说明

本项目采用双视觉路径加单语言主干。

## Qwen3.5 原生视觉路径

文件：`src/centerline_mm/models/qwen_mm.py`

职责：

- 使用 `AutoProcessor` 和 `AutoModelForImageTextToText` 加载 Qwen3.5。
- 保留 Qwen3.5 自带视觉编码器 `.visual`。
- 不替换、不改造、不把外部 DINOv3 特征融合进 Qwen 视觉塔。
- 在构造 `inputs_embeds` 时调用 Qwen 自己的 `.visual(pixel_values, grid_thw=...)`，再把图像 token 写回 image token 位置。

## DINOv3 外部任务视觉路径

文件：`src/centerline_mm/models/dinov3_task_encoder.py`

职责：

- 从本地 DINOv3 仓库加载指定主干，例如 `dinov3_vitl16`。
- 输出 `x_norm_patchtokens` 作为任务视觉 token。
- 支持冻结整个主干或只开放最后若干 blocks。
- 阶段一使用分割监督训练，阶段二和阶段三复用训练后的任务视觉能力。

## 第一阶段分割头

文件：`src/centerline_mm/models/segmentation.py`

职责：

- 只在阶段一使用。
- 将 DINOv3 patch token reshape 成二维 feature map。
- 通过轻量卷积头输出中心线或道路结构监督 mask。
- 阶段一结束后保存 DINOv3 任务编码器，最终 JSON 生成不使用该分割头。

## 任务视觉投影模块

文件：`src/centerline_mm/models/task_projection.py`

包含：

- `LayerNorm` 归一化。
- 二维位置编码 `TwoDimPositionEncoding`。
- 轻量 MLP 投影到 Qwen hidden size。
- `TokenResampler` 将 DINOv3 patch token 压缩成固定数量任务上下文 token。

## 任务视觉 token 注入模块

文件：`src/centerline_mm/models/token_injection.py`

职责：

- 将投影后的 DINOv3 任务 token 插入语言模型输入序列。
- 默认插入到 Qwen 原生视觉片段之后，即 `vision_end_token_id` 后。
- 自动扩展 `attention_mask`。
- 训练时给任务 token 的 label 置为 `-100`，不参与 CE。

## 双路径总模型

文件：`src/centerline_mm/models/dual_path_model.py`

数据流：

```text
BEV image
  ├─ Qwen3.5 native vision tower ───────────┐
  └─ DINOv3 task encoder -> projector -> task context tokens
                                             ↓
Qwen3.5 language model input embeddings <- token injector
                                             ↓
coordinate text or strict JSON
```

