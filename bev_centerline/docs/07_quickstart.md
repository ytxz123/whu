# 快速开始

## 1. 准备环境

```bash
cd /Users/tzy/PT/whu/bev_centerline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果已经有可用的 PyTorch/Transformers 环境，可以只安装缺失依赖。

## 2. 准备数据

创建：

```text
data/stage1/train.jsonl
data/stage1/val.jsonl
data/stage2/train.jsonl
data/stage3/train.jsonl
```

具体格式见 `docs/08_dataset_format.md`。

## 3. 训练阶段一

```bash
PYTHONPATH=src python -m centerline_mm.train.stage1_train_seg --config configs/stage1_segmentation.yaml
```

## 4. 评估阶段一

```bash
PYTHONPATH=src python -m centerline_mm.eval.stage1_eval_seg --config configs/stage1_segmentation.yaml
```

## 5. 训练阶段二

```bash
PYTHONPATH=src python -m centerline_mm.train.stage2_train_alignment --config configs/stage2_alignment.yaml
```

## 6. 训练阶段三

```bash
PYTHONPATH=src python -m centerline_mm.train.stage3_train_json_lora --config configs/stage3_json_lora.yaml
```

## 7. 推理

```bash
PYTHONPATH=src python -m centerline_mm.infer.generate_json --config configs/infer.yaml --image data/demo.png
```

