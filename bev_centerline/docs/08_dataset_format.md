# 数据集格式要求

本项目已经按 `dataset_builder` 当前输出格式完成适配。新手只需要记住一句话：

```text
dataset_builder 输出到 bev_centerline/data/dataset_test
```

默认训练配置会直接读取这个目录。

## 1. dataset_builder 输出目录

用 dataset_builder 生成后，应得到：

```text
bev_centerline/data/dataset_test/
├── data_line/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── images/
│       ├── 000001.png
│       └── ...
├── data_seg/
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── images/
│   │   ├── 000001.png
│   │   └── ...
│   └── masks/
│       ├── 000001.png
│       └── ...
└── artifacts/
    ├── family_manifest.jsonl
    ├── meta_train.jsonl
    ├── meta_val.jsonl
    └── build_summary.json
```

如果你的输出目录不是 `data/dataset_test`，有两种做法：

1. 复制或软链接到 `bev_centerline/data/dataset_test`。
2. 修改 `configs/*.yaml` 里的 manifest 路径。

推荐新手用第一种。

## 2. 阶段一使用 segmentation 数据

默认配置：

```yaml
data:
  train_manifest: data/dataset_test/data_seg/train.jsonl
  val_manifest: data/dataset_test/data_seg/val.jsonl
```

`data_seg/train.jsonl` 每行格式：

```json
{"id":"000001","image":"images/000001.png","mask":"masks/000001.png"}
```

路径解释：

- `image` 相对 `data_seg/train.jsonl` 所在目录。
- `mask` 相对 `data_seg/train.jsonl` 所在目录。
- 所以上面的图片实际是 `data_seg/images/000001.png`。
- mask 实际是 `data_seg/masks/000001.png`。

mask 要求：

- 单通道 PNG。
- 道路中心线像素值为 255。
- 背景为 0。
- 训练时会自动变成二值 mask。

## 3. 阶段二和阶段三使用 line 数据

默认配置：

```yaml
data:
  train_manifest: data/dataset_test/data_line/train.jsonl
```

`data_line/train.jsonl` 是 ShareGPT 风格。每行类似：

```json
{
  "id": "000001",
  "messages": [
    {
      "role": "user",
      "content": "<image>\nTask: reconstruct all visible road centerlines inside box [0,0,512,512].\nReturn strict JSON only with top-level key lines."
    },
    {
      "role": "assistant",
      "content": {
        "lines": [
          {
            "category": "road_centerline",
            "points": [[256,200],[256,300]]
          }
        ]
      }
    }
  ],
  "images": ["images/000001.png"]
}
```

本项目会自动做这些适配：

- 从 `images[0]` 找图片。
- 从 user message 读取 prompt，但默认训练会使用项目内置 prompt，避免 dataset_builder prompt 和最终 role/content JSON 目标冲突。
- 从 assistant message 读取 `content.lines`。
- 阶段二把 `lines` 转成坐标中间文本。
- 阶段三把 `lines` 规范化成最终严格 JSON。

## 4. 阶段二目标如何生成

dataset_builder 给的是：

```json
{"lines":[{"category":"road_centerline","points":[[56,420],[88,396],[126,365]]}]}
```

阶段二训练时自动变成：

```text
LINE 56,420 88,396 126,365
```

如果没有线：

```json
{"lines":[]}
```

自动变成：

```text
NO_LINES
```

## 5. 阶段三目标如何生成

dataset_builder 给的是 top-level `lines`：

```json
{"lines":[{"category":"road_centerline","points":[[56,420],[88,396]]}]}
```

阶段三自动规范化成最终要求：

```json
{"role":"assistant","content":{"lines":[{"category":"road_centerline","points":[[56,420],[88,396]]}]}}
```

规范化规则：

- 坐标转整数。
- 坐标裁剪到 `0..512`。
- 少于 2 个点的线被丢弃。
- `category` 固定为 `road_centerline`。
- JSON 使用紧凑格式，减少等价写法。

## 6. 训练前必须检查

数据生成好以后运行：

```bash
cd /Users/tzy/PT/whu/bev_centerline
./scripts/check_setup.sh
```

如果数据格式正确，会看到多行 `[OK]`。

如果看到：

```text
[FAIL] line sample 1 invalid
```

通常是：

- `images` 为空。
- 图片文件不存在。
- `messages` 里没有 assistant。
- assistant content 不是 JSON 对象。

## 7. 仍然兼容的简化格式

除了 dataset_builder 格式，本项目也兼容下面这种简单 JSONL：

```json
{"image":"images/000001.png","target":{"lines":[{"points":[[1,2],[3,4]]}]}}
```

但正式训练推荐直接使用 dataset_builder 输出。

