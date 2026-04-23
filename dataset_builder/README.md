# dataset_builder

这个目录只保留一条最小主链：先构建 family manifest，再直接导出指定框大小的 road centerline 数据集。不再切 fixed16，不再构建 stageb，也不再输出 intersection 相关目标。

当前默认会同时导出两套数据：

1. line 数据集：用于当前的 road centerline 重建任务。
2. segmentation 数据集：用于语义分割，真值 mask 直接由道路中心线 GeoJSON 栅格化得到。

## line 数据集输出格式

导出的每条样本都是 ShareGPT 风格，结构如下：

```json
{
  "id": "sample_image_0000",
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
            "points": [[256, 200], [256, 300]]
          }
        ]
      }
    }
  ],
  "images": ["images/train/sample_image_0000.png"]
}
```

说明：

1. user prompt 里的 box 坐标总是当前导出图像的整图范围，采用右开上界写法，比如 512x512 图像对应 [0,0,512,512]。
2. assistant 只输出 lines，不包含 start_type、end_type、polygon 或其他冗余字段。
3. assistant content 直接是对象，不再把 JSON 再包成字符串，因此不会出现大量反斜杠转义。
4. category 统一为 road_centerline。

## 适用数据

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

默认处理逻辑：

1. 扫描 patch_tif 下所有原始 tif。
2. 自动忽略 *_edit_poly.tif 本身。
3. 自动为每个 tif 查找同名 review mask。
4. 从 Lane.geojson 读取车道中心线。
5. 按指定框大小切图，并用 review mask 过滤候选框。

## 当前目录结构

```text
dataset_builder/
├── README.md
├── scripts/
│   ├── build_dataset.py
│   ├── build_family_manifest.py
│   ├── export_patch_dataset.py
│   └── visualize_label_comparison.py
├── prompts/
│   └── box_prompt.txt
└── road_builder/
  ├── __init__.py
  ├── geometry_utils.py
  ├── io_utils.py
  ├── patch_export.py
  ├── source_data.py
  └── tile_windows.py
```

## 脚本职责

1. [scripts/build_family_manifest.py](scripts/build_family_manifest.py)

- 扫描 train 和 val 样本目录。
- 读取 GeoTIFF 元信息和 review mask。
- 生成候选 crop box，并输出 family_manifest.jsonl。

2. [scripts/export_patch_dataset.py](scripts/export_patch_dataset.py)

- 读取 family manifest。
- 从 Lane.geojson 解析 road centerline。
- 裁剪图像并同时生成 line 数据集与 segmentation 数据集。

3. [scripts/build_dataset.py](scripts/build_dataset.py)

- 用默认配置串联前两步。
- 启动前会检查 prompt 模板占位符是否合法。

## 依赖

至少需要：

1. numpy
2. pillow
3. rasterio
4. pyproj

## 一键运行

编辑 [scripts/build_dataset.py](scripts/build_dataset.py) 顶部的 CONFIG，至少填写 dataset_root，或者分别填写 train_root 和 val_root，然后执行：

```bash
python dataset_builder/scripts/build_dataset.py
```

它会依次执行：

1. build_family_manifest.py
2. export_patch_dataset.py

默认输出目录结构：

```text
dataset_test/
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

命名整理规则：

1. 后续直接训练会用到的主入口，只保留各数据集根目录下的 train.jsonl 和 val.jsonl。
2. images 和 masks 按任务归档到各自数据集目录中。
3. manifest、meta、summary 这类辅助文件统一收进 artifacts。
4. 不再额外保留 dataset_info.json、family_manifest.summary.json、export_summary.json 这类重复产物。

## 关键参数

只需要改 [scripts/build_dataset.py](scripts/build_dataset.py) 顶部的 CONFIG。

常用参数：

1. dataset_root: 总数据目录，内部包含 train 和 val。
2. train_root 和 val_root: 当 train 和 val 分开存放时单独指定。
3. output_root: 总输出目录。
4. box_size_px: 导出框大小，也是最终图像的目标宽高。
5. overlap_px: 相邻框重叠像素。
6. keep_margin_px: keep_box 收缩边距，用来约束监督归属。
7. box_min_mask_ratio 和 box_min_mask_pixels: review mask 过滤阈值。
8. band_indices: 读取 tif 的波段索引。
9. empty_box_drop_ratio: 空样本丢弃比例。
10. box_user_prompt_file: prompt 模板路径。

## 完整参数说明

下面按 [scripts/build_dataset.py](scripts/build_dataset.py) 里的 BuildConfig 顺序说明每个参数的作用。

### 数据路径参数

1. dataset_root

- 统一的数据根目录。
- 默认假设目录结构是 dataset_root/train 和 dataset_root/val。
- 当 train 和 val 本来就在同一棵目录下时，通常只改这个参数就够了。

2. train_root

- 单独指定 train 目录。
- 适合 train 和 val 不在同一个根目录下的情况。
- 如果填写了它，train 路径就不会再从 dataset_root/train 推导。

3. val_root

- 单独指定 val 目录。
- 适合 train 和 val 分开存放的情况。
- 如果填写了它，val 路径就不会再从 dataset_root/val 推导。

### 输出参数

4. output_root

- 总输出目录。
- 里面会生成 data_line、data_seg、artifacts 三个子目录。
- 如果想区分不同实验结果，优先改这个参数。

5. splits

- 要处理的 split 列表，默认是 train 和 val。
- 调试时可以只保留 train，或者只保留 val。
- 例如改成 ["train"] 可以只构建训练集数据。

### manifest 构建参数

6. image_relpath

- 强制每个样本只处理一个固定影像相对路径。
- 适合你只想验证某张 tif，而不想扫描整个 patch_tif 目录的情况。
- 一旦填写，脚本会优先使用它，而不是 image_dir_relpath 和 image_glob 的扫描逻辑。

7. mask_relpath

- 在使用 image_relpath 时，为这张影像单独指定 mask 路径。
- 适合 mask 命名不满足默认后缀规则时使用。

8. image_dir_relpath

- 每个样本内部原始影像目录的相对路径。
- 默认是 patch_tif。
- 如果你的 tif 不放在 patch_tif 下，就需要改它。

9. image_glob

- 在 image_dir_relpath 目录下扫描原始影像时使用的匹配规则。
- 默认是 *.tif。
- 如果影像是 .tiff，或者你只想挑出一部分文件，可以改这个参数。

10. mask_suffix

- 根据原图文件名自动推导 review mask 文件名时使用的后缀。
- 默认是 _edit_poly.tif。
- 例如 0.tif 会自动匹配 0_edit_poly.tif。

11. lane_relpath

- 每个样本目录中 lane 标注文件的相对路径。
- 默认是 label_check_crop/Lane.geojson。

12. mask_threshold

- review mask 二值化阈值。
- 高于该阈值的像素会被视为有效区域。
- 如果 mask 不是标准 0/255，而是其他灰度分布，可以调整它。

13. box_size_px

- 切出来的框大小，也是最终导出图像的宽高。
- 默认是 512。
- 同时也决定 prompt 里的框坐标范围，比如 [0,0,512,512]。

14. overlap_px

- 相邻框之间的重叠像素。
- 值越大，框之间共享内容越多，样本数通常也会更多。
- 如果你想减少边界截断问题，可以适当调大。

15. keep_margin_px

- 从 crop box 向内收缩得到 keep box 的边距。
- 最终线段的归属判定更依赖 keep box，而不是整张 crop。
- 这个值越大，边缘区域越不容易归到当前框。

16. review_crop_pad_px

- 当启用 search_within_review_bbox 时，对 review 区域最小外接框额外扩张的像素。
- 用于避免搜索区域裁得太紧。

17. box_min_mask_ratio

- 候选框最少需要满足的 mask 覆盖比例。
- 默认是 0.02。
- 如果比例太低，该框会被过滤掉，除非 box_min_mask_pixels 条件满足。

18. box_min_mask_pixels

- 候选框最少需要满足的 mask 像素数。
- 默认是 256。
- 它和 box_min_mask_ratio 是并列关系，满足其中一个即可保留。

19. box_max_per_sample

- 每个样本最多保留多少个框。
- 默认 0，表示不限制。
- 如果单个样本能切出太多框，想控量时可以设成一个正整数。

20. search_within_review_bbox

- 是否只在 review mask 的最小外接框附近搜索候选框。
- 默认关闭。
- 打开后通常更高效，但如果 review mask 不完整，可能漏掉边缘框。

21. fallback_to_all_if_empty

- 如果按 mask 条件筛选后一个框都没有，是否退回保留全部候选框。
- 默认关闭。
- 调试数据时比较有用，可以避免某个样本被完全跳过。

22. max_samples_per_split

- 每个 split 最多扫描多少个样本目录。
- 默认 0，表示不限制。
- 调试时建议先改成 1 到 5，可以显著缩短迭代时间。

### 最终导出参数

23. band_indices

- 读取 tif 时选取的波段索引。
- 默认是 [1, 2, 3]。
- 这里使用的是 1-based 索引，不是 0-based。

24. box_resample_step_px

- 线段重采样步长，也就是折线点之间的采样距离。
- 默认是 4.0。
- 值越小，导出的 points 越密；值越大，points 越稀。
- 如果设成 0 或负数，当前导出链里等价于关闭这一步重采样。

25. box_boundary_tol_px

- 判断线段是否贴近框边界时使用的容差。
- 默认是 2.5。
- 主要影响边界附近的几何裁剪和方向判定。

26. max_families_per_split

- 每个 split 最多导出多少个 family。
- 默认 0，表示不限制。
- 它发生在 manifest 已经构建完成之后，适合控制最终导出体量。

27. empty_box_drop_ratio

- 空样本丢弃比例。
- 默认是 0.95，表示大约丢掉 95% 没有目标线的框。
- 如果你想保留更多负样本，就调小它。

28. empty_box_seed

- 空样本随机下采样的随机种子。
- 默认是 42。
- 想复现实验时保持不变，想换一批空样本时可以改它。

29. box_user_prompt_file

- user prompt 模板文件路径。
- 默认指向 [prompts/box_prompt.txt](prompts/box_prompt.txt)。
- 模板里必须保留这四个占位符：{box_x_min}、{box_y_min}、{box_x_max}、{box_y_max}。

### 最常改的一组参数

如果只是日常调参，通常优先改这些：

1. dataset_root 或 train_root / val_root
2. output_root
3. box_size_px
4. overlap_px
5. box_min_mask_ratio
6. box_min_mask_pixels
7. band_indices
8. box_resample_step_px
9. empty_box_drop_ratio
10. max_samples_per_split

## 单步运行

### 1. 构建 manifest

```bash
python dataset_builder/scripts/build_family_manifest.py \
  --dataset-root /path/to/rc_dataset \
  --output-manifest /path/to/output/family_manifest.jsonl \
  --tile-size-px 512
```

### 2. 导出 line + segmentation 数据集

```bash
python dataset_builder/scripts/export_patch_dataset.py \
  --family-manifest /path/to/output/family_manifest.jsonl \
  --output-root /path/to/output/dataset_test
```

导出后会生成：

1. data_line/train.jsonl 和 data_line/val.jsonl
2. data_seg/train.jsonl 和 data_seg/val.jsonl
3. artifacts/family_manifest.jsonl、artifacts/meta_train.jsonl、artifacts/meta_val.jsonl、artifacts/build_summary.json

## segmentation 数据集格式

语义分割数据集目录如下：

```text
data_seg/
├── train.jsonl
├── val.jsonl
├── images/
│   └── 000001.png
└── masks/
    └── 000001.png
```

其中：

1. images/ 下是 512 patch 原图。
2. masks/ 下是对应的单通道 PNG mask。
3. mask 中道路中心线像素值为 255，背景为 0。

train.jsonl / val.jsonl 每一行格式如下：

```json
{
  "id": "000001",
  "image": "images/000001.png",
  "mask": "masks/000001.png"
}
```

## prompt 模板

默认模板文件是 [prompts/box_prompt.txt](prompts/box_prompt.txt)。

模板必须保留这四个占位符：

1. {box_x_min}
2. {box_y_min}
3. {box_x_max}
4. {box_y_max}

默认模板内容：

```text
<image>
Task: reconstruct all visible road centerlines inside box [{box_x_min},{box_y_min},{box_x_max},{box_y_max}].
Return strict JSON only with top-level key lines.
```

## artifacts 元信息文件

artifacts/meta_train.jsonl 和 artifacts/meta_val.jsonl 会额外保存：

1. source_image_path
2. source_mask_path
3. source_lane_path
4. crop_box 和 keep_box
5. target_box
6. prompt_text
7. target_lines

这些文件只用于排查数据，不是训练主输入。

更具体地说：

1. data_line/train.jsonl / data_line/val.jsonl 是给 line 重建任务训练用的主输入，只包含 messages 和 images。
2. artifacts/meta_train.jsonl / artifacts/meta_val.jsonl 是和主数据一一对应的辅助定位文件。
3. meta 文件额外保存了 source_image_path、source_lane_path、family_id、patch_id、crop_box、keep_box 这些可视化和排查必须用到的信息。

为什么不能只用 data_line/train.jsonl 画图：

1. train.jsonl 里只有 512 patch 图像路径和监督线 points。
2. 它不知道这张 patch 是从哪张大图裁出来的，也不知道原始 Lane.geojson 在哪里。
3. 它也没有 crop_box、keep_box、family_id、source_image_path 这些字段，所以没法稳定回到原始数据上下文。

为什么要保留 artifacts/meta_train.jsonl：

1. 可视化时需要用它把 data_line/train.jsonl 的这一条样本，和原始大图、原始 Lane 标注、patch 裁剪框对应起来。
2. 只有靠 artifacts/meta_train.jsonl，才能把“模型训练看到的这张 512 图”和“它来自哪张原始大图的哪个位置”关联起来。
3. 同一张大图切出来的多个 patch，要想分到同一个目录下，也依赖 meta 里的 family_id。

## 可视化对比脚本

如果想直接检查“data_line/train.jsonl 里的最终训练监督”和“原始 Lane.geojson 标注”在每个 512 patch 上的差异，可以使用 [scripts/visualize_label_comparison.py](scripts/visualize_label_comparison.py)。

如果你前面使用的是 [scripts/build_dataset.py](scripts/build_dataset.py) 的默认输出目录，那么这个脚本已经带了默认参数，可以直接运行：

```bash
python dataset_builder/scripts/visualize_label_comparison.py
```

默认会读取：

1. dataset/dataset_test/data_line/train.jsonl
2. dataset/dataset_test/artifacts/meta_train.jsonl
3. dataset/dataset_test/artifacts/family_manifest.jsonl

默认会输出到：

1. dataset/dataset_test/visualize_train

脚本会对每个 patch 输出一张三栏对比图：

1. 512 patch 原图 + 原始 Lane 标注
2. 512 patch 原图 + train.jsonl 里的监督线段
3. 512 patch 原图 + 两者叠加对比

其中：

1. 训练监督直接读取 data_line/train.jsonl / data_line/val.jsonl 里的 assistant 标注内容。
2. 原始 Lane 标注会按对应 patch 的 crop_box 裁到同一张 512 图上。
3. 同一张大图切出来的所有 512 patch 对比图，会被放进同一个 family 子目录下。

关于 family_manifest.jsonl：

1. 它不是画线必须的数据文件。
2. 当前脚本即使没有 family_manifest.jsonl，也可以仅依赖 data_line/train.jsonl + artifacts/meta_train.jsonl 完成绘图。
3. 如果提供了 family_manifest.jsonl，主要作用是保持 family 的遍历顺序和构建阶段一致，排查时更稳定。

示例：

```bash
python dataset_builder/scripts/visualize_label_comparison.py \
  --family-manifest /path/to/output/artifacts/family_manifest.jsonl \
  --label-files /path/to/output/data_line/train.jsonl \
  --meta-jsonl /path/to/output/artifacts/meta_train.jsonl \
  --output-dir /path/to/output/visualize_train \
  --max-families 20 \
  --draw-keep-boxes
```

常用参数：

1. --label-files: 要可视化的输出标签文件，可传 line 数据集 train/val.jsonl，也可传模型输出 json/jsonl。
2. --meta-jsonl: 对应的 artifacts/meta jsonl，用来提供 crop_box、source_image_path、source_lane_path 等定位信息。
3. --family-id: 只渲染指定 family，适合排查单张大图。
4. --splits: 只保留指定 split。
5. --max-families: 最多导出多少个 family。
6. --draw-crop-boxes: 在 patch 图上画出 crop 边界。
7. --draw-keep-boxes: 在 patch 图上画出 keep box。
8. --max-side-px: 限制可视化输出的最长边，避免横向拼图过大。

输出目录下会同时生成：

1. 每个 family 一个子目录。
2. 子目录里保存该大图对应的所有 512 patch 对比图。
3. summary.json，记录 family 目录、patch 输出路径和线段数量统计。
