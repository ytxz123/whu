# 快速开始

这是极简版流程。第一次使用请优先看 `docs/00_beginner_guide.md`，那里每一步都有检查方法和目录说明。

## 1. 进入项目

```bash
cd /Users/tzy/PT/whu/bev_centerline
```

## 2. 安装环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. 生成数据

在工作区根目录运行 dataset_builder：

```bash
cd /Users/tzy/PT/whu
python dataset_builder/scripts/build_dataset.py \
  --dataset-root /path/to/rc_dataset \
  --output-root /Users/tzy/PT/whu/bev_centerline/data/dataset_test
```

生成后必须有：

```text
bev_centerline/data/dataset_test/data_line/train.jsonl
bev_centerline/data/dataset_test/data_seg/train.jsonl
bev_centerline/data/dataset_test/data_seg/val.jsonl
```

## 4. 放 DINOv3 权重

```bash
cd /Users/tzy/PT/whu/bev_centerline
mkdir -p weights/dinov3
```

把官方 ViT-L/16 LVD-1689M 权重放到：

```text
weights/dinov3/dinov3_vitl16_pretrain_lvd1689m.pth
```

## 5. 训练前检查

```bash
./scripts/check_setup.sh
```

看到 `[OK] Setup check passed.` 再继续。

## 6. 三阶段训练

```bash
./scripts/train_stage1.sh
./scripts/eval_stage1.sh
./scripts/train_stage2.sh
./scripts/train_stage3.sh
```

## 7. 推理

```bash
./scripts/infer.sh --image data/dataset_test/data_line/images/000001.png
```

输出只会是 JSON。
