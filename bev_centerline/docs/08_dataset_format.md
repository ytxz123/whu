# 数据集格式要求

所有 manifest 都是 JSONL，每行一个样本。相对路径以 manifest 文件所在目录解析。

## 阶段一：分割训练

默认位置：

```text
bev_centerline/data/stage1/train.jsonl
bev_centerline/data/stage1/val.jsonl
```

每行格式：

```json
{"image":"images/000001.png","mask":"masks/000001.png"}
```

推荐目录：

```text
data/stage1/
  train.jsonl
  val.jsonl
  images/
    000001.png
  masks/
    000001.png
```

要求：

- `image` 是黑底 BEV 道路结构图。
- `mask` 是二值分割监督图。
- mask 非零像素表示道路中心线或与中心线恢复直接相关的监督区域。
- 读取时会 resize 到 `512x512`，mask 使用 nearest 插值。

## 阶段二：坐标级语义对齐

默认位置：

```text
bev_centerline/data/stage2/train.jsonl
```

每行格式：

```json
{"image":"images/000001.png","target":{"lines":[{"points":[[56,420],[88,396],[126,365]]}]}}
```

也可以直接提供最终 JSON：

```json
{"image":"images/000001.png","target":{"role":"assistant","content":{"lines":[{"category":"road_centerline","points":[[260,120],[258,176]]}]}}}
```

阶段二会把目标转换为固定中间文本：

```text
LINE 56,420 88,396 126,365
```

无中心线样本：

```json
{"image":"images/000002.png","target":{"lines":[]}}
```

会转换为：

```text
NO_LINES
```

## 阶段三：最终 JSON 训练

默认位置：

```text
bev_centerline/data/stage3/train.jsonl
```

每行格式同阶段二，但阶段三会把目标规范化为严格 JSON 字符串：

```json
{"role":"assistant","content":{"lines":[{"category":"road_centerline","points":[[56,420],[88,396]]}]}}
```

规范化规则：

- 坐标转整数。
- 坐标裁剪到 `0..512`。
- 线至少 2 个点。
- `category` 固定为 `road_centerline`。
- 最终训练文本不包含解释。

## 推理数据

推理入口只需要单张图片：

```bash
PYTHONPATH=src python -m centerline_mm.infer.generate_json --config configs/infer.yaml --image path/to/image.png
```

