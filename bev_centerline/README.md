# BEV Centerline MM

这是一个从零搭建的多模态项目，用于从 512x512 黑底 BEV 道路结构图生成道路中心线 JSON。

核心设计：

- 保留 Qwen3.5 原生视觉编码器，不改内部结构，用于整图全局理解。
- 额外引入 DINOv3 任务视觉编码器，用于学习细粒度道路结构 token。
- DINOv3 token 经过归一化、二维位置编码、轻量投影和重采样后，作为额外上下文 token 注入 Qwen3.5 语言侧。
- 三阶段训练：DINOv3 分割监督、坐标级语义对齐、最终严格 JSON 生成。

入口：

```bash
cd bev_centerline
PYTHONPATH=src python -m centerline_mm.train.stage1_train_seg --config configs/stage1_segmentation.yaml
PYTHONPATH=src python -m centerline_mm.eval.stage1_eval_seg --config configs/stage1_segmentation.yaml
PYTHONPATH=src python -m centerline_mm.train.stage2_train_alignment --config configs/stage2_alignment.yaml
PYTHONPATH=src python -m centerline_mm.train.stage3_train_json_lora --config configs/stage3_json_lora.yaml
PYTHONPATH=src python -m centerline_mm.infer.generate_json --config configs/infer.yaml --image data/demo.png
```

详细说明见 `docs/`。

