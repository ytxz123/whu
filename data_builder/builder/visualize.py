from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from .config import load_yaml, resolve_optional_path
from .io_utils import ensure_dir
from .rendering import AnnotationStyle, draw_annotations, render_label_image


@dataclass
class VisualizeConfig:
    dataset_root: Path
    output_root: Path
    label_image_dirname: str
    jsonl_sources: list[str]
    max_samples_per_split: int
    line_width: int
    point_radius: int
    point_outline_width: int
    panel_gap: int
    panel_title_height: int
    show_point_index: bool


def normalize_string_list(value, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        items = [value]
    else:
        items = list(value)
    out = [str(item).strip() for item in items if str(item).strip()]
    return out or list(default)


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
        jsonl_sources=normalize_string_list(cfg.get("jsonl_sources", cfg.get("splits", ["train", "val"])), ["train", "val"]),
        max_samples_per_split=int(cfg.get("max_samples_per_split", 0)),
        line_width=max(1, int(cfg.get("line_width", 3))),
        point_radius=max(1, int(cfg.get("point_radius", 4))),
        point_outline_width=max(1, int(cfg.get("point_outline_width", 2))),
        panel_gap=max(0, int(cfg.get("panel_gap", 16))),
        panel_title_height=max(0, int(cfg.get("panel_title_height", 28))),
        show_point_index=bool(cfg.get("show_point_index", False)),
    )


def iter_jsonl_rows(path: Path) -> Iterable[dict]:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            yield data
            return
        if isinstance(data, list):
            for row in data:
                if not isinstance(row, dict):
                    raise ValueError(f"JSON row must be a dict at {path}")
                yield row
            return
        raise ValueError(f"JSON file must contain a dict or list of dicts: {path}")

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


def parse_annotation_text(text: str, sample_id: str, field_name: str) -> list[dict]:
    cleaned = re.sub(r"<think>.*?</think>\s*", "", str(text), flags=re.DOTALL).strip()
    candidates = [cleaned]
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start >= 0 and end >= start:
        extracted = cleaned[start : end + 1].strip()
        if extracted and extracted not in candidates:
            candidates.append(extracted)
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed
    raise ValueError(f"Cannot parse annotation JSON from {field_name} for sample {sample_id}")


def response_payload(row: dict) -> list[dict]:
    sample_id = str(row.get("id", "unknown"))
    response = row.get("response")
    if isinstance(response, str) and response.strip():
        return parse_annotation_text(response, sample_id, "response")

    conversations = row.get("conversations", [])
    if isinstance(conversations, list):
        for message in conversations:
            if not isinstance(message, dict):
                continue
            if str(message.get("from", "")).strip() != "assistant":
                continue
            content = message.get("value", "[]")
            if isinstance(content, str):
                return parse_annotation_text(content, sample_id, "conversations.assistant.value")

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
            return parse_annotation_text(content, sample_id, "messages.assistant.content")
    return []


def labels_payload(row: dict) -> list[dict]:
    sample_id = str(row.get("id", "unknown"))
    labels = row.get("labels")
    if isinstance(labels, str) and labels.strip():
        return parse_annotation_text(labels, sample_id, "labels")
    return []


def sample_image_path(row: dict, dataset_root: Path) -> tuple[Path, Path]:
    images = row.get("images", [])
    if not isinstance(images, list) or not images:
        sample_id = row.get("id", "unknown")
        raise ValueError(f"Missing images field for sample {sample_id}")
    first = images[0]
    if isinstance(first, dict):
        path_value = first.get("path", "")
    else:
        path_value = first
    rel = Path(str(path_value).replace("\\", "/"))
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

    lines = response_payload(row)
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
    else:
        label_lines = labels_payload(row)
        if label_lines:
            label_image = render_label_image(
                raw_rgb.width,
                raw_rgb.height,
                label_lines,
                AnnotationStyle(line_width=2, draw_points=False, fixed_line_color=(255, 255, 255)),
            )
    compare = compose_compare(raw_rgb, label_image, overlay, cfg)
    if label_image is not None:
        label_image.close()
    return compare, rel_path


def resolve_jsonl_path(source: str, dataset_root: Path) -> Path:
    source = str(source).strip()
    if source.endswith(".json") or source.endswith(".jsonl"):
        return (dataset_root / source).resolve()
    return (dataset_root / f"{source}.jsonl").resolve()


def visualize_source(source: str, cfg: VisualizeConfig) -> int:
    jsonl_path = resolve_jsonl_path(source, cfg.dataset_root)
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
    for source in cfg.jsonl_sources:
        count = visualize_source(source, cfg)
        print(f"[data_builder.visualize] source={source} rows={count} output={cfg.output_root}", flush=True)
