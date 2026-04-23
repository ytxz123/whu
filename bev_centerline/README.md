# BEV Centerline MM

这是一个可运行的新项目：用 Qwen3.5 原生视觉路径 + DINOv3 外部任务视觉路径，从 512x512 黑底 BEV 道路结构图生成道路中心线 JSON。

如果你是第一次使用，直接按这个顺序看：

1. [新手从零开始教程](docs/00_beginner_guide.md)
2. [数据集格式和 dataset_builder 对接](docs/08_dataset_format.md)
3. [模型权重放置说明](docs/09_weights.md)
4. [三阶段训练说明](docs/03_training_stages.md)
5. [推理说明](docs/05_inference.md)

最短命令顺序：

```bash
cd /Users/tzy/PT/whu/bev_centerline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 数据和权重放好后先体检
./scripts/check_setup.sh

# 训练与评估
./scripts/train_stage1.sh
./scripts/eval_stage1.sh
./scripts/train_stage2.sh
./scripts/train_stage3.sh

# 推理，只输出 JSON
./scripts/infer.sh --image data/dataset_test/data_line/images/000001.png
```

项目默认读取：

- DINOv3 官方代码仓库：`/Users/tzy/PT/whu/dinov3`
- Qwen3.5 模型目录：`/Users/tzy/PT/whu/Qwen3.5_2B`
- dataset_builder 生成数据：`/Users/tzy/PT/whu/bev_centerline/data/dataset_test`
- DINOv3 权重：`/Users/tzy/PT/whu/bev_centerline/weights/dinov3/dinov3_vitl16_pretrain_lvd1689m.pth`

