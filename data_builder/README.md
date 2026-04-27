# data_builder

这是一个轻量的数据集构建与可视化工具。

它主要提供两部分能力：

1. 从原始 RC 数据构建 patch 级数据集
2. 对生成后的 `jsonl` 标注做可视化对比

## 构建输出

构建完成后，输出目录结构类似：

```text
data_0427/
├── img_train/
│   ├── sample_a/
│   │   ├── r0_c0_p01.png
│   │   ├── r0_c1_p02.png
│   │   └── ...
│   └── ...
├── img_val/
│   └── ...
├── train.jsonl
└── val.jsonl
```

`jsonl` 每行格式类似：

```json
{
  "id": "A170_xxx_r2_c7_p01",
  "messages": [
    {
      "role": "system",
      "content": "..."
    },
    {
      "role": "user",
      "content": "<image>\nPlease construct the complete road map in the current BEV (Bird's Eye View) image patch."
    },
    {
      "role": "assistant",
      "content": "[{\"points\":[[151,143],[225,210]]}]"
    }
  ],
  "images": [
    "img_train/A170_xxx/r2_c7_p01.png"
  ]
}
```

注意：

- `assistant.content` 是字符串，字符串内容本身是一个 JSON 数组
- 每个元素当前只保留 `points`
- 坐标使用 patch 局部坐标系
- `images` 指向的是原始图裁出来的 patch，不会指向 `_ground`、`_lane`、`_pose` 这些派生图

## 原始数据目录要求

默认每个样本目录结构：

```text
sample_xxx/
├── patch_tif/
│   ├── 0.tif
│   ├── 0_edit_poly.tif
│   ├── 1.tif
│   ├── 1_edit_poly.tif
│   └── ...
└── label_check_crop/
    └── Lane.geojson
```

其中：

- `*.tif` 是原始图像
- `*_edit_poly.tif` 是 review mask
- `Lane.geojson` 是道路中心线

如果同一目录下还有派生图，例如：

- `0_ground.tif`
- `0_lane.tif`
- `0_pose.tif`

当前默认配置会排除这些派生图，只使用原始图，例如 `0.tif`、`1.tif`。

## 快速开始

### 1. 构建数据集

先编辑默认配置文件：

```text
data_builder/configs/build.yaml
```

最少需要确认这些路径：

```yaml
dataset_root: "/path/to/rc_dataset"
output_root: "/path/to/output/data_0427"
```

如果 train / val 已经分开放置，也可以这样配：

```yaml
dataset_root: ""
train_root: "/path/to/train"
val_root: "/path/to/val"
output_root: "/path/to/output/data_0427"
```

然后直接运行：

```bash
python data_builder/scripts/build_dataset.py
```

默认会自动读取：

```text
data_builder/configs/build.yaml
```

不需要手动写 `--config`。

### 2. 可视化对比

先编辑默认配置文件：

```text
data_builder/configs/visualize.yaml
```

最少需要确认：

```yaml
dataset_root: "/path/to/data_0427"
output_root: "/path/to/vis_compare"
```

这里的 `dataset_root` 应该指向已经生成好的数据集目录，也就是包含：

- `train.jsonl`
- `val.jsonl`
- `img_train/`
- `img_val/`

的那一层目录。

然后直接运行：

```bash
python data_builder/scripts/visualize_dataset.py
```

默认会自动读取：

```text
data_builder/configs/visualize.yaml
```


## 构建脚本使用方法

### 最常用命令

直接使用默认配置：

```bash
python data_builder/scripts/build_dataset.py
```

临时切换配置文件：

```bash
python data_builder/scripts/build_dataset.py \
  --config /path/to/another_build.yaml
```

临时覆盖输出目录：

```bash
python data_builder/scripts/build_dataset.py \
  --output-root /tmp/data_0427
```

只跑少量样本做检查：

```bash
python data_builder/scripts/build_dataset.py \
  --max-samples-per-split 50
```

调大 Douglas–Peucker 简化强度：

```bash
python data_builder/scripts/build_dataset.py \
  --simplify-tolerance 3.5
```

### `build_dataset.py` 支持的命令行覆盖参数

- `--config`
- `--dataset-root`
- `--train-root`
- `--val-root`
- `--output-root`
- `--empty-annotation-keep-ratio`
- `--empty-annotation-seed`
- `--max-samples-per-split`
- `--simplify-tolerance`

其他构建参数请直接修改 `build.yaml`。

## 可视化脚本使用方法

### 最常用命令

直接使用默认配置：

```bash
python data_builder/scripts/visualize_dataset.py
```

临时切换配置文件：

```bash
python data_builder/scripts/visualize_dataset.py \
  --config /path/to/another_visualize.yaml
```

只看少量样本：

```bash
python data_builder/scripts/visualize_dataset.py \
  --max-samples-per-split 50
```

让点更醒目一些：

```bash
python data_builder/scripts/visualize_dataset.py \
  --point-radius 6 \
  --line-width 4
```

显示每个点的序号：

```bash
python data_builder/scripts/visualize_dataset.py \
  --show-point-index true
```

调整左右间隔和标题栏高度：

```bash
python data_builder/scripts/visualize_dataset.py \
  --panel-gap 24 \
  --panel-title-height 36
```

### `visualize_dataset.py` 支持的命令行覆盖参数

- `--config`
- `--dataset-root`
- `--output-root`
- `--max-samples-per-split`
- `--line-width`
- `--point-radius`
- `--point-outline-width`
- `--panel-gap`
- `--panel-title-height`
- `--show-point-index`

其中 `--show-point-index` 支持这些布尔写法：

- `true` / `false`
- `1` / `0`
- `yes` / `no`
- `on` / `off`

## 可视化输出说明

输出目录结构会和数据集里的 `img_train/`、`img_val/` 保持一致，例如：

```text
vis_compare/
├── img_train/
│   ├── sample_a/
│   │   ├── r0_c0_p01.png
│   │   ├── r0_c1_p02.png
│   │   └── ...
│   └── ...
├── img_val/
│   └── ...
```

每张输出图默认是左右并排：

- 左侧：`raw patch`
- 右侧：`patch + jsonl labels`

右侧会把 `assistant.content` 里的折线画到 patch 图像上，并为每个点画出清晰可见的圆点。

## build.yaml 参数说明

当前 `data_builder/configs/build.yaml` 里的字段含义如下：

- `dataset_root`: 统一数据根目录。若设置后，默认读取 `<dataset_root>/<split>/...`
- `train_root`: 单独指定 train 根目录。设置后优先于 `dataset_root/train`
- `val_root`: 单独指定 val 根目录。设置后优先于 `dataset_root/val`
- `output_root`: 输出数据集目录
- `splits`: 要导出的 split 列表，通常为 `["train", "val"]`
- `image_dir_relpath`: 每个样本目录下图像子目录名
- `lane_relpath`: 每个样本目录下 `Lane.geojson` 的相对路径
- `image_glob`: 图像匹配模式
- `mask_suffix`: review mask 文件后缀
- `exclude_image_stem_suffixes`: 需要排除的派生图后缀，默认会排除 `_ground`、`_lane`、`_pose`
- `patch_size`: patch 边长
- `mask_threshold`: mask 二值化阈值
- `min_mask_ratio`: patch 的最小 mask 比例阈值
- `min_mask_pixels`: patch 的最小 mask 像素阈值
- `drop_empty_annotations`: 是否启用空标注 patch 丢弃策略
- `empty_annotation_keep_ratio`: 空标注 patch 的随机保留比例
- `empty_annotation_seed`: 空标注抽样随机种子
- `save_black_background_from_mask`: 是否把 mask 外像素置黑
- `max_samples_per_split`: 每个 split 最多处理多少个样本目录，`0` 表示全部
- `simplify_tolerance`: Douglas–Peucker 简化容差，单位是 patch 像素
- `system_prompt`: 写入每条样本 `messages[0].content`
- `user_prompt`: 写入每条样本 `messages[1].content`

### 构建时的路径优先级

- 如果 `train_root` 已设置，`train` split 优先使用它
- 如果 `val_root` 已设置，`val` split 优先使用它
- 只有在某个 split 没有单独根目录时，才会回退到 `dataset_root/<split>`

### 空样本保留逻辑

- 有标注 patch：全部保留
- 没标注 patch：如果 `drop_empty_annotations: true`，则按 `empty_annotation_keep_ratio` 随机保留
- 如果 `drop_empty_annotations: false`，则空标注 patch 也全部保留

### 折线简化逻辑

为了降低 `assistant.content` 的 token 长度，构建流程会在写出 patch 局部坐标前，对每条线应用 Douglas–Peucker algorithm。

- `simplify_tolerance` 越大，保留点越少
- `simplify_tolerance` 越小，形状保留越细
- 设为 `0` 表示关闭简化

### patch 切窗逻辑

为了避免原始图宽高不能被 `patch_size` 整除时，边缘导出出小尺寸 patch，当前切窗逻辑使用固定尺寸滑窗：

- 常规区域按 `patch_size` 步长切分
- 最后一行或最后一列如果不够一个完整 patch，会把窗口向回滑到最后一个完整位置
- 如果原图本身某一边小于 `patch_size`，导出时会在右侧或下侧补黑边

因此最终写出的 patch 图片尺寸会统一为 `patch_size x patch_size`。

## visualize.yaml 参数说明

当前 `data_builder/configs/visualize.yaml` 里的字段含义如下：

- `dataset_root`: 已生成数据集目录，内部应包含 `train.jsonl`、`val.jsonl` 和 `img_*`
- `output_root`: 可视化输出目录
- `splits`: 要可视化的 split 列表
- `max_samples_per_split`: 每个 split 最多输出多少条，`0` 表示全部
- `line_width`: 叠加折线宽度
- `point_radius`: 每个点的半径
- `point_outline_width`: 点轮廓宽度
- `panel_gap`: 左右对比图之间的间距
- `panel_title_height`: 顶部标题栏高度，设为 `0` 可关闭标题栏
- `show_point_index`: 是否在点旁边绘制点序号
