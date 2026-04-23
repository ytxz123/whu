# 第一阶段分割评估说明

入口：

```bash
cd bev_centerline
PYTHONPATH=src python -m centerline_mm.eval.stage1_eval_seg \
  --config configs/stage1_segmentation.yaml \
  --checkpoint outputs/stage1/best.pt
```

可选参数：

```bash
--manifest data/stage1/val.jsonl
--out-dir outputs/stage1/eval
--max-vis 32
```

评估指标：

- IoU
- Dice
- Precision
- Recall

输出文件：

```text
outputs/stage1/eval/
  metrics.json
  visualizations/
    00000.png
    00001.png
```

可视化图包含四列：

- Input
- GT
- Pred
- Overlay

Overlay 中红色是预测，绿色是 GT。

