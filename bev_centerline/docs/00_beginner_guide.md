# 新手从零开始教程

这份教程假设你完全从零开始，只知道当前工作区是：

```text
/Users/tzy/PT/whu
```

当前已经有：

```text
/Users/tzy/PT/whu/dinov3
/Users/tzy/PT/whu/Qwen3.5_2B
/Users/tzy/PT/whu/dataset_builder
/Users/tzy/PT/whu/bev_centerline
```

你最终要得到一个可以推理的模型：

```text
bev_centerline/outputs/stage3/latest.pt
```

推理时 stdout 只输出 JSON。

## 第 1 步：进入项目目录

```bash
cd /Users/tzy/PT/whu/bev_centerline
```

确认你在正确目录：

```bash
pwd
```

应该看到：

```text
/Users/tzy/PT/whu/bev_centerline
```

## 第 2 步：创建 Python 环境

如果你还没有虚拟环境：

```bash
python -m venv .venv
```

激活环境：

```bash
source .venv/bin/activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

如果你的机器需要 CUDA 版 PyTorch，请先按 PyTorch 官网命令安装对应版本，再执行上面的 `pip install -r requirements.txt`。

检查 Python 能不能找到项目包：

```bash
PYTHONPATH=src python -c "import centerline_mm; print(centerline_mm.__version__)"
```

应该输出：

```text
0.1.0
```

## 第 3 步：准备原始数据给 dataset_builder

dataset_builder 需要的原始样本目录大致是：

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

说明：

- `0.tif`、`1.tif` 是原始图像。
- `0_edit_poly.tif`、`1_edit_poly.tif` 是人工 review mask。
- `Lane.geojson` 是道路中心线标注。
- 如果 train 和 val 放在同一个根目录，dataset_builder 会自己按 family manifest 中的 split 处理。
- 如果 train 和 val 已经分开，就分别传 `--train-root` 和 `--val-root`。

假设你的原始数据在：

```text
/path/to/rc_dataset
```

## 第 4 步：用 dataset_builder 生成训练数据

从工作区根目录运行：

```bash
cd /Users/tzy/PT/whu
python dataset_builder/scripts/build_dataset.py \
  --dataset-root /path/to/rc_dataset \
  --output-root /Users/tzy/PT/whu/bev_centerline/data/dataset_test
```

如果 train 和 val 分开：

```bash
cd /Users/tzy/PT/whu
python dataset_builder/scripts/build_dataset.py \
  --train-root /path/to/train \
  --val-root /path/to/val \
  --output-root /Users/tzy/PT/whu/bev_centerline/data/dataset_test
```

生成完成后，必须看到这些文件：

```text
bev_centerline/data/dataset_test/
├── data_line/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── images/
├── data_seg/
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── images/
│   └── masks/
└── artifacts/
    ├── family_manifest.jsonl
    ├── meta_train.jsonl
    ├── meta_val.jsonl
    └── build_summary.json
```

本项目默认配置已经指向这些路径：

- 阶段一：`data/dataset_test/data_seg/train.jsonl`
- 阶段二：`data/dataset_test/data_line/train.jsonl`
- 阶段三：`data/dataset_test/data_line/train.jsonl`

## 第 5 步：准备 DINOv3 权重

本项目默认使用 DINOv3 ViT-L/16：

```yaml
dinov3_arch: dinov3_vitl16
```

请把 DINOv3 官方下载到的 ViT-L/16 LVD-1689M 权重放到：

```text
bev_centerline/weights/dinov3/dinov3_vitl16_pretrain_lvd1689m.pth
```

完整路径是：

```text
/Users/tzy/PT/whu/bev_centerline/weights/dinov3/dinov3_vitl16_pretrain_lvd1689m.pth
```

创建目录：

```bash
cd /Users/tzy/PT/whu/bev_centerline
mkdir -p weights/dinov3
```

放好后检查：

```bash
ls -lh weights/dinov3/
```

如果你使用别的 DINOv3 ViT 权重，比如 ViT-B/16，需要同时修改三个配置：

```yaml
model:
  dinov3_arch: dinov3_vitb16
  dinov3_weights: weights/dinov3/你的权重文件.pth
  feature_dim: 768
```

阶段二和阶段三里的 `dino_feature_dim` 也要改成对应维度。

## 第 6 步：确认 Qwen3.5 模型目录

默认配置读取：

```text
/Users/tzy/PT/whu/Qwen3.5_2B
```

它里面至少应该有：

```text
config.json
tokenizer.json
preprocessor_config.json
model.safetensors-00001-of-00001.safetensors
```

检查：

```bash
ls /Users/tzy/PT/whu/Qwen3.5_2B
```

如果你的 Qwen3.5 模型放在别处，修改：

```text
bev_centerline/configs/stage2_alignment.yaml
bev_centerline/configs/stage3_json_lora.yaml
bev_centerline/configs/infer.yaml
```

里面的：

```yaml
paths:
  qwen_path: ../Qwen3.5_2B
```

## 第 7 步：一键体检

回到项目目录：

```bash
cd /Users/tzy/PT/whu/bev_centerline
```

执行：

```bash
./scripts/check_setup.sh
```

这个脚本会检查：

- DINOv3 代码目录是否存在。
- Qwen3.5 模型目录是否存在。
- DINOv3 权重文件是否存在。
- `data_seg/train.jsonl` 和 `data_seg/val.jsonl` 是否能读。
- segmentation 图片和 mask 是否存在。
- `data_line/train.jsonl` 是否是 dataset_builder 的 ShareGPT 格式。
- line 样本能否被转成阶段二坐标文本和阶段三严格 JSON。

如果看到 `[FAIL]`，先修掉对应路径或文件，再训练。

## 第 8 步：训练阶段一

阶段一训练 DINOv3 外部任务视觉分支和一个临时分割头。

```bash
./scripts/train_stage1.sh
```

训练完成后应该有：

```text
outputs/stage1/latest.pt
outputs/stage1/best.pt
```

`best.pt` 会给阶段二使用。

## 第 9 步：评估阶段一

```bash
./scripts/eval_stage1.sh
```

输出：

```text
outputs/stage1/eval/
├── metrics.json
└── visualizations/
    ├── 00000.png
    └── ...
```

`metrics.json` 里有：

- IoU
- Dice
- Precision
- Recall

可视化图是 Input、GT、Pred、Overlay 四栏。

## 第 10 步：训练阶段二

阶段二做坐标级语义对齐。默认冻结 Qwen3.5 语言主体和原生视觉塔，只训练 DINOv3 少量尾部 adapter、任务视觉投影、二维位置编码和重采样。

```bash
./scripts/train_stage2.sh
```

训练完成后应该有：

```text
outputs/stage2/latest.pt
```

## 第 11 步：训练阶段三

阶段三在阶段二基础上训练最终 JSON 输出。默认对 Qwen3.5 开启 LoRA。

```bash
./scripts/train_stage3.sh
```

训练完成后应该有：

```text
outputs/stage3/latest.pt
```

这个文件用于最终推理。

## 第 12 步：推理

找一张 512x512 或可 resize 的 BEV 图片，例如：

```text
data/dataset_test/data_line/images/000001.png
```

执行：

```bash
./scripts/infer.sh --image data/dataset_test/data_line/images/000001.png
```

stdout 只会输出 JSON，例如：

```json
{"role":"assistant","content":{"lines":[{"category":"road_centerline","points":[[56,420],[88,396]]}]}}
```

如果模型输出无法解析，推理入口会返回：

```json
{"role":"assistant","content":{"lines":[]}}
```

## 第 13 步：常见问题

### 找不到 DINOv3 权重

报错类似：

```text
FileNotFoundError: ... weights/dinov3/dinov3_vitl16_pretrain_lvd1689m.pth
```

解决：

1. 创建 `weights/dinov3`。
2. 把官方权重放进去。
3. 或修改三个训练配置里的 `model.dinov3_weights`。

### feature_dim 不匹配

如果你把 `dinov3_arch` 从 `dinov3_vitl16` 改成别的模型，必须同步改维度：

```text
dinov3_vits16      384
dinov3_vits16plus 384
dinov3_vitb16      768
dinov3_vitl16      1024
```

阶段一改 `feature_dim`，阶段二和阶段三改 `dino_feature_dim`。

### dataset_builder 生成的是 top-level lines，为什么阶段三训练 role/content

dataset_builder 的 assistant content 是：

```json
{"lines":[...]}
```

本项目数据层会自动规范化成最终要求：

```json
{"role":"assistant","content":{"lines":[...]}}
```

所以不需要手动转换。
