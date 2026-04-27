from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from .config import load_yaml, resolve_optional_path
from .io_utils import ensure_dir
from .rendering import AnnotationStyle, draw_annotations


@dataclass
class VisualizeConfig:
    dataset_root: Path
    output_root: Path
    label_image_dirname: str
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
        label_image_dirname=str(cfg.get("label_image_dirname", "img_label")).strip() or "img_label",
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
    conversations = row.get("conversations", [])
    if isinstance(conversations, list):
        for message in conversations:
            if not isinstance(message, dict):
                continue
            if str(message.get("from", "")).strip() != "assistant":
                continue
            content = message.get("value", "[]")
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError as exc:
                    sample_id = row.get("id", "unknown")
                    raise ValueError(f"Assistant content is not valid JSON for sample {sample_id}") from exc
                return parsed if isinstance(parsed, list) else []

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


def add_panel_title(image: Image.Image, title: str, title_height: int) -> Image.Image:
    if title_height <= 0:
        return image
    canvas = Image.new("RGB", (image.width, image.height + title_height), color=(250, 250, 250))
    canvas.paste(image, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, max(0, (title_height - 10) // 2)), title, fill=(40, 40, 40), font=font)
    return canvas


def label_image_path(rel_path: Path, cfg: VisualizeConfig) -> Path | None:
    parts = rel_path.parts
    if len(parts) < 2:
        return None
    split_part = parts[0]
    if split_part == "img_train":
        split = "train"
    elif split_part == "img_val":
        split = "val"
    else:
        return None
    return (cfg.dataset_root / cfg.label_image_dirname / split / Path(*parts[1:])).resolve()


def compose_compare(raw_image: Image.Image, label_image: Image.Image | None, overlay_image: Image.Image, cfg: VisualizeConfig) -> Image.Image:
    panels = [add_panel_title(raw_image.convert("RGB"), "raw patch", cfg.panel_title_height)]
    if label_image is not None:
        panels.append(add_panel_title(label_image.convert("RGB"), "original centerline labels", cfg.panel_title_height))
    panels.append(add_panel_title(overlay_image.convert("RGB"), "patch + jsonl labels", cfg.panel_title_height))
    width = sum(panel.width for panel in panels) + cfg.panel_gap * max(0, len(panels) - 1)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    x_offset = 0
    for index, panel in enumerate(panels):
        canvas.paste(panel, (x_offset, 0))
        x_offset += panel.width
        if index < len(panels) - 1:
            x_offset += cfg.panel_gap
    return canvas


def render_sample(row: dict, cfg: VisualizeConfig) -> tuple[Image.Image, Path]:
    image_path, rel_path = sample_image_path(row, cfg.dataset_root)
    if not image_path.is_file():
        sample_id = row.get("id", "unknown")
        raise FileNotFoundError(f"Image file not found for sample {sample_id}: {image_path}")

    lines = assistant_payload(row)
    with Image.open(image_path) as raw_image:
        raw_rgb = raw_image.convert("RGB")
    style = AnnotationStyle(
        line_width=cfg.line_width,
        point_radius=cfg.point_radius,
        point_outline_width=cfg.point_outline_width,
        show_point_index=cfg.show_point_index,
        fixed_line_color=(255, 255, 255),
    )
    overlay = draw_annotations(raw_rgb, lines, style)
    label_path = label_image_path(rel_path, cfg)
    label_image = None
    if label_path is not None and label_path.is_file():
        with Image.open(label_path) as source:
            label_image = source.convert("RGB")
    compare = compose_compare(raw_rgb, label_image, overlay, cfg)
    if label_image is not None:
        label_image.close()
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
