from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from .config import load_yaml, resolve_optional_path
from .io_utils import ensure_dir


@dataclass
class VisualizeConfig:
    dataset_root: Path
    output_root: Path
    splits: list[str]
    max_samples_per_split: int
    line_width: int
    point_radius: int
    point_outline_width: int
    panel_gap: int
    panel_title_height: int
    show_point_index: bool


def build_visualize_config(config_path: str | Path, overrides: dict | None = None) -> VisualizeConfig:
    cfg = load_yaml(config_path)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                cfg[key] = value

    config_dir = Path(cfg["_config_dir"]).resolve()
    dataset_root = resolve_optional_path(config_dir, cfg.get("dataset_root"))
    output_root = resolve_optional_path(config_dir, cfg.get("output_root"))
    if dataset_root is None:
        raise ValueError("Please set dataset_root in the visualize config file.")

    return VisualizeConfig(
        dataset_root=dataset_root,
        output_root=output_root or (config_dir.parent / "vis_compare").resolve(),
        splits=[str(x) for x in cfg.get("splits", ["train", "val"])],
        max_samples_per_split=int(cfg.get("max_samples_per_split", 0)),
        line_width=max(1, int(cfg.get("line_width", 3))),
        point_radius=max(1, int(cfg.get("point_radius", 4))),
        point_outline_width=max(1, int(cfg.get("point_outline_width", 2))),
        panel_gap=max(0, int(cfg.get("panel_gap", 16))),
        panel_title_height=max(0, int(cfg.get("panel_title_height", 28))),
        show_point_index=bool(cfg.get("show_point_index", False)),
    )


def iter_jsonl_rows(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL row at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be a dict at {path}:{line_number}")
            yield row


def assistant_payload(row: dict) -> list[dict]:
    messages = row.get("messages", [])
    if not isinstance(messages, list):
        return []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).strip() != "assistant":
            continue
        content = message.get("content", "[]")
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                sample_id = row.get("id", "unknown")
                raise ValueError(f"Assistant content is not valid JSON for sample {sample_id}") from exc
            return parsed if isinstance(parsed, list) else []
    return []


def sample_image_path(row: dict, dataset_root: Path) -> tuple[Path, Path]:
    images = row.get("images", [])
    if not isinstance(images, list) or not images:
        sample_id = row.get("id", "unknown")
        raise ValueError(f"Missing images field for sample {sample_id}")
    rel = Path(str(images[0]).replace("\\", "/"))
    return (dataset_root / rel).resolve(), rel


def color_for_index(index: int) -> tuple[int, int, int]:
    palette = [
        (245, 99, 99),
        (66, 165, 245),
        (102, 187, 106),
        (255, 202, 40),
        (171, 71, 188),
        (38, 166, 154),
        (255, 112, 67),
        (124, 179, 66),
    ]
    return palette[index % len(palette)]


def draw_point(draw: ImageDraw.ImageDraw, x: int, y: int, radius: int, outline_width: int, color: tuple[int, int, int]) -> None:
    bounds = (x - radius, y - radius, x + radius, y + radius)
    draw.ellipse(bounds, fill=(255, 255, 255), outline=color, width=outline_width)


def draw_annotations(image: Image.Image, lines: list[dict], cfg: VisualizeConfig) -> Image.Image:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    for line_index, line in enumerate(lines):
        points = line.get("points", [])
        if not isinstance(points, list) or len(points) < 2:
            continue
        xy = [(int(pt[0]), int(pt[1])) for pt in points if isinstance(pt, list) and len(pt) >= 2]
        if len(xy) < 2:
            continue
        color = color_for_index(line_index)
        draw.line(xy, fill=color, width=cfg.line_width, joint="curve")
        for point_index, (x, y) in enumerate(xy):
            draw_point(draw, x, y, cfg.point_radius, cfg.point_outline_width, color)
            if cfg.show_point_index:
                draw.text((x + cfg.point_radius + 2, y - cfg.point_radius - 2), str(point_index), fill=color, font=font)
    return overlay


def add_panel_title(image: Image.Image, title: str, title_height: int) -> Image.Image:
    if title_height <= 0:
        return image
    canvas = Image.new("RGB", (image.width, image.height + title_height), color=(250, 250, 250))
    canvas.paste(image, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, max(0, (title_height - 10) // 2)), title, fill=(40, 40, 40), font=font)
    return canvas


def compose_compare(raw_image: Image.Image, overlay_image: Image.Image, cfg: VisualizeConfig) -> Image.Image:
    raw_panel = add_panel_title(raw_image.convert("RGB"), "raw patch", cfg.panel_title_height)
    overlay_panel = add_panel_title(overlay_image.convert("RGB"), "patch + jsonl labels", cfg.panel_title_height)
    width = raw_panel.width + cfg.panel_gap + overlay_panel.width
    height = max(raw_panel.height, overlay_panel.height)
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    canvas.paste(raw_panel, (0, 0))
    canvas.paste(overlay_panel, (raw_panel.width + cfg.panel_gap, 0))
    return canvas


def render_sample(row: dict, cfg: VisualizeConfig) -> tuple[Image.Image, Path]:
    image_path, rel_path = sample_image_path(row, cfg.dataset_root)
    if not image_path.is_file():
        sample_id = row.get("id", "unknown")
        raise FileNotFoundError(f"Image file not found for sample {sample_id}: {image_path}")

    lines = assistant_payload(row)
    with Image.open(image_path) as raw_image:
        raw_rgb = raw_image.convert("RGB")
    overlay = draw_annotations(raw_rgb, lines, cfg)
    compare = compose_compare(raw_rgb, overlay, cfg)
    return compare, rel_path


def visualize_split(split: str, cfg: VisualizeConfig) -> int:
    jsonl_path = cfg.dataset_root / f"{split}.jsonl"
    if not jsonl_path.is_file():
        return 0
    count = 0
    for row in iter_jsonl_rows(jsonl_path):
        if cfg.max_samples_per_split > 0 and count >= cfg.max_samples_per_split:
            break
        compare, rel_path = render_sample(row, cfg)
        out_path = cfg.output_root / rel_path
        ensure_dir(out_path.parent)
        compare.save(out_path)
        compare.close()
        count += 1
    return count


def run(config_path: str | Path, overrides: dict | None = None) -> None:
    cfg = build_visualize_config(config_path, overrides=overrides)
    ensure_dir(cfg.output_root)
    for split in cfg.splits:
        count = visualize_split(split, cfg)
        print(f"[data_builder.visualize] split={split} rows={count} output={cfg.output_root / f'img_{split}'}", flush=True)
