# 项目总览

本项目目标是从黑底 BEV 道路结构图恢复每条有效通行路径的道路中心线，并在最终推理时只输出合法 JSON。

输入：

- 单张 RGB 或可转 RGB 的图片。
- 训练和推理默认 resize 到 `512x512`。
- 图像局部坐标系范围是 `0..512`。

最终输出：

```json
{"role":"assistant","content":{"lines":[{"category":"road_centerline","points":[[56,420],[88,396]]}]}}
```

无中心线时：

```json
{"role":"assistant","content":{"lines":[]}}
```

项目目录：

```text
bev_centerline/
  configs/                 配置文件
  src/centerline_mm/
    data/                  数据集与图像变换
    models/                Qwen、DINOv3、投影、注入、分割头
    train/                 三阶段训练入口
    eval/                  第一阶段分割评估
    infer/                 最终 JSON 推理
    utils/                 配置、checkpoint、JSON 规范化
  scripts/                 便捷运行脚本
  docs/                    文档
```

默认依赖本地路径：

- `../dinov3`
- `../qwen3.5_2B`

相对路径以 `bev_centerline/configs/*.yaml` 的位置解析，因此从哪个目录启动都不会改变路径含义。

