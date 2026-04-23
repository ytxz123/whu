# 推理说明

入口：

```bash
cd bev_centerline
PYTHONPATH=src python -m centerline_mm.infer.generate_json \
  --config configs/infer.yaml \
  --image data/demo.png
```

指定 checkpoint：

```bash
PYTHONPATH=src python -m centerline_mm.infer.generate_json \
  --config configs/infer.yaml \
  --checkpoint outputs/stage3/latest.pt \
  --image data/demo.png
```

输出约束：

- stdout 只打印最终 JSON。
- 若模型输出不可解析，入口会返回空结果：

```json
{"role":"assistant","content":{"lines":[]}}
```

推理入口会做最终规范化：

- 保留 `role=assistant`。
- 保留 `content.lines`。
- 坐标整数化。
- 坐标裁剪到 `[0,512]`。
- 过滤少于 2 个点的线。
- `category` 固定为 `road_centerline`。

