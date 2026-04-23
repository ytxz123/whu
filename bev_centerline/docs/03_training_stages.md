# 三阶段训练说明

## 阶段一：任务视觉编码器训练

入口：

```bash
cd bev_centerline
PYTHONPATH=src python -m centerline_mm.train.stage1_train_seg --config configs/stage1_segmentation.yaml
```

目标：

- 用分割任务训练 DINOv3 外部分支。
- 让外部任务视觉路径学习道路结构和中心线恢复相关的细粒度表示。

训练内容：

- DINOv3 主干。
- 阶段一专用轻量分割头。

产物：

- `outputs/stage1/latest.pt`
- `outputs/stage1/best.pt`

checkpoint 内包含：

- `task_encoder`：后续阶段复用。
- `seg_head`：仅用于阶段一评估或继续训练。
- `metrics`：验证集指标。

## 阶段二：一次坐标级语义对齐

入口：

```bash
PYTHONPATH=src python -m centerline_mm.train.stage2_train_alignment --config configs/stage2_alignment.yaml
```

默认写死训练原则：

- 冻结 Qwen3.5 语言主体。
- 冻结 Qwen3.5 原生视觉塔。
- Qwen3.5 仍参与前向计算，额外视觉 token 通过语言模型条件建模产生梯度。
- 主要训练 DINOv3 任务分支少量尾部 adapter、任务视觉投影层、二维位置编码、token 重采样模块和 token 注入相关模块。

训练方式：

- teacher forcing。
- 目标文本是坐标级中间表示。
- 只对目标输出部分计算交叉熵。
- 不做自由采样式推理训练。

中间表示示例：

```text
LINE 56,420 88,396 126,365
LINE 260,120 258,176
```

无有效路径：

```text
NO_LINES
```

产物：

- `outputs/stage2/latest.pt`

## 阶段三：最终 JSON 生成

入口：

```bash
PYTHONPATH=src python -m centerline_mm.train.stage3_train_json_lora --config configs/stage3_json_lora.yaml
```

目标：

- 在阶段二基础上学习最终严格 JSON 输出。
- 视觉对齐机制保持不变。
- Qwen3.5 原生视觉路径保持不变。
- 默认对 Qwen3.5 开启 LoRA。

训练规范化：

- 训练前将目标 JSON 归一化为固定格式。
- 所有坐标转整数。
- 所有坐标裁剪到 `[0,512]`。
- 所有 line 的 `category` 固定为 `road_centerline`。
- 少于 2 个点的线被丢弃。

产物：

- `outputs/stage3/latest.pt`

