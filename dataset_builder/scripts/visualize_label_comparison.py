"""Visualize train.jsonl labels against raw lane labels on each 512 patch image."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image, ImageColor, ImageDraw, ImageFont

DATASET_BUILDER_ROOT = Path(__file__).resolve().parents[1]
if str(DATASET_BUILDER_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_BUILDER_ROOT))

from road_builder.io_utils import ensure_dir, load_jsonl, log_event, log_warning, require_existing_path, write_json
from road_builder.patch_export import build_patch_segments_global, build_patch_target_lines
from road_builder.source_data import load_sample_global_features, read_raster_metadata, read_rgb_geotiff


RAW_LANE_COLOR = "#d4af37"
OUTPUT_LANE_COLOR = "#ff5a5f"
CROP_BOX_COLOR = "#f5f5f5"
KEEP_BOX_COLOR = "#ffb400"


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "dataset_test"
DEFAULT_LINE_DATASET_ROOT = DEFAULT_OUTPUT_ROOT / "data_line"
DEFAULT_ARTIFACTS_ROOT = DEFAULT_OUTPUT_ROOT / "artifacts"


@dataclass(frozen=True)
class VisualizeConfig:
    label_files: List[str] = field(default_factory=lambda: [str(DEFAULT_LINE_DATASET_ROOT / "train.jsonl")])
    meta_jsonl: List[str] = field(default_factory=lambda: [str(DEFAULT_ARTIFACTS_ROOT / "meta_train.jsonl")])
    family_manifest: str = str(DEFAULT_ARTIFACTS_ROOT / "family_manifest.jsonl")
    output_dir: str = str(DEFAULT_OUTPUT_ROOT / "visualize_train")
    label_name: str = "Output Labels"
    family_id: List[str] = field(default_factory=list)
    splits: List[str] = field(default_factory=lambda: ["train"])
    image_dir_relpath: str = "patch_tif"
    lane_relpath: str = "label_check_crop/Lane.geojson"
    max_families: int = 20
    max_side_px: int = 1536
    line_width: int = 3
    box_width: int = 2
    band_indices: List[int] = field(default_factory=lambda: [1, 2, 3])
    draw_crop_boxes: bool = False
    draw_keep_boxes: bool = True


CONFIG = VisualizeConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize raw lane labels and output labels on each exported 512 patch image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--label-files",
        "--train-jsonl",
        type=str,
        nargs="+",
        default=list(CONFIG.label_files),
        help="output label files to visualize; supports jsonl and json, including train/val exports and model outputs",
    )
    parser.add_argument(
        "--meta-jsonl",
        type=str,
        nargs="+",
        default=list(CONFIG.meta_jsonl),
        help="metadata jsonl files aligned 1:1 with the label rows",
    )
    parser.add_argument(
        "--family-manifest",
        type=str,
        default=str(CONFIG.family_manifest),
        help="optional family manifest for stable family ordering; missing file will be ignored",
    )
    parser.add_argument("--output-dir", type=str, default=str(CONFIG.output_dir), help="visualization output directory")
    parser.add_argument("--label-name", type=str, default=str(CONFIG.label_name), help="display name for the third panel")
    parser.add_argument("--family-id", type=str, nargs="*", default=list(CONFIG.family_id), help="only render selected family ids")
    parser.add_argument("--splits", type=str, nargs="*", default=list(CONFIG.splits), help="only keep selected splits")
    parser.add_argument("--image-dir-relpath", type=str, default=str(CONFIG.image_dir_relpath), help="relative image directory under each sample root")
    parser.add_argument("--lane-relpath", type=str, default=str(CONFIG.lane_relpath), help="relative Lane geojson path under each sample root")
    parser.add_argument("--max-families", type=int, default=int(CONFIG.max_families))
    parser.add_argument("--max-side-px", type=int, default=int(CONFIG.max_side_px))
    parser.add_argument("--line-width", type=int, default=int(CONFIG.line_width))
    parser.add_argument("--box-width", type=int, default=int(CONFIG.box_width))
    parser.add_argument("--band-indices", type=int, nargs="+", default=list(CONFIG.band_indices))
    parser.add_argument("--draw-crop-boxes", action="store_true", default=bool(CONFIG.draw_crop_boxes))
    parser.add_argument("--draw-keep-boxes", action="store_true", default=bool(CONFIG.draw_keep_boxes))
    return parser.parse_args()


def load_optional_family_manifest(path_text: str) -> List[str]:
    if not str(path_text).strip():
        return []
    manifest_path = Path(path_text).expanduser().resolve()
    if not manifest_path.is_file():
        log_warning("Visualize", f"family manifest not found, continue without it: {manifest_path}")
        return []
    rows = load_jsonl(manifest_path)
    return [str(row.get("family_id", "")).strip() for row in rows if str(row.get("family_id", "")).strip()]


def _normalize_rows_payload(payload: Any) -> List[Dict]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("rows", "data", "samples", "predictions", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        if any(key in payload for key in ("id", "sample_id", "messages", "lines", "images")):
            return [payload]
    raise ValueError("Unsupported label file format. Expected jsonl rows or a json list/object.")


def load_label_rows(paths: Sequence[str]) -> List[Dict]:
    rows: List[Dict] = []
    for path_text in paths:
        file_path = Path(path_text)
        try:
            label_path = require_existing_path(file_path, kind="file")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"label file not found: {Path(path_text).expanduser().resolve()} ; pass --label-files explicitly"
            ) from exc
        label_dir = label_path.parent
        if label_path.suffix.lower() == ".jsonl":
            loaded_rows = load_jsonl(label_path)
        else:
            with label_path.open("r", encoding="utf-8-sig") as handle:
                loaded_rows = _normalize_rows_payload(json.load(handle))
        for row in loaded_rows:
            copied = dict(row)
            copied["_label_dir"] = str(label_dir)
            copied["_label_path"] = str(label_path)
            rows.append(copied)
    return rows


def load_meta_rows(paths: Sequence[str]) -> Dict[str, Dict]:
    rows_by_id: Dict[str, Dict] = {}
    for path_text in paths:
        try:
            meta_path = require_existing_path(Path(path_text), kind="file")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"meta jsonl not found: {Path(path_text).expanduser().resolve()} ; this file is required to locate source images and crop boxes, or pass --meta-jsonl explicitly"
            ) from exc
        for row in load_jsonl(meta_path):
            sample_id = str(row.get("id", "")).strip()
            if sample_id:
                rows_by_id[sample_id] = row
    return rows_by_id


def parse_assistant_payload(row: Dict) -> Dict:
    messages = row.get("messages", [])
    for message in reversed(messages):
        if str(message.get("role", "")).strip().lower() != "assistant":
            continue
        content = message.get("content", {})
        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            text = content.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
    return {}


def _extract_output_payload(row: Dict) -> Dict:
    direct_keys = ("lines", "prediction", "result", "output", "response")
    for key in direct_keys:
        value = row.get(key)
        if isinstance(value, dict):
            return value
        if key == "lines" and isinstance(value, list):
            return {"lines": value}
        if isinstance(value, str):
            text = value.strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
            if key == "lines" and isinstance(parsed, list):
                return {"lines": parsed}
    return parse_assistant_payload(row)


def normalize_output_lines(row: Dict) -> List[Dict]:
    payload = _extract_output_payload(row)
    raw_lines = payload.get("lines", [])
    if not isinstance(raw_lines, list):
        return []
    output: List[Dict] = []
    for raw_line in raw_lines:
        if not isinstance(raw_line, dict):
            continue
        points = raw_line.get("points", [])
        if not isinstance(points, list) or len(points) < 2:
            continue
        normalized_points: List[List[float]] = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            normalized_points.append([float(point[0]), float(point[1])])
        if len(normalized_points) < 2:
            continue
        output.append({"category": str(raw_line.get("category", "road_centerline")), "points": normalized_points})
    return output


def resolve_row_sample_id(row: Dict) -> str:
    for key in ("id", "sample_id"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def resolve_patch_image_path(label_row: Dict) -> Path | None:
    label_dir = Path(str(label_row.get("_label_dir", ""))).resolve()
    images = label_row.get("images", [])
    if not isinstance(images, list) or not images:
        return None
    image_rel_path = str(images[0]).strip()
    if not image_rel_path:
        return None
    candidate = (label_dir / image_rel_path).resolve()
    return candidate if candidate.is_file() else None


def build_patch_contexts(label_rows: Sequence[Dict], meta_rows_by_id: Dict[str, Dict], allowed_splits: set[str]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for label_row in label_rows:
        sample_id = resolve_row_sample_id(label_row)
        if not sample_id:
            continue
        meta_row = meta_rows_by_id.get(sample_id)
        if meta_row is None:
            log_warning("Visualize", f"skip sample_id={sample_id} because meta row is missing")
            continue
        split = str(meta_row.get("split", "")).strip()
        if allowed_splits and split not in allowed_splits:
            continue
        family_id = str(meta_row.get("family_id", "")).strip()
        if not family_id:
            log_warning("Visualize", f"skip sample_id={sample_id} because family_id is missing")
            continue
        grouped[family_id].append(
            {
                "sample_id": sample_id,
                "split": split,
                "patch_id": int(meta_row.get("patch_id", 0)),
                "label_row": label_row,
                "meta_row": meta_row,
                "output_lines": normalize_output_lines(label_row),
            }
        )
    for family_rows in grouped.values():
        family_rows.sort(key=lambda item: (int(item.get("patch_id", 0)), str(item.get("sample_id", ""))))
    return dict(grouped)


def build_ordered_family_ids(
    *,
    explicit_family_ids: Sequence[str],
    ordered_manifest_ids: Sequence[str],
    grouped_patch_rows: Dict[str, List[Dict]],
    max_families: int,
) -> List[str]:
    if explicit_family_ids:
        selected = [str(family_id).strip() for family_id in explicit_family_ids if str(family_id).strip()]
    elif ordered_manifest_ids:
        selected = [family_id for family_id in ordered_manifest_ids if family_id in grouped_patch_rows]
    else:
        selected = sorted(grouped_patch_rows.keys())

    output: List[str] = []
    for family_id in selected:
        if family_id not in grouped_patch_rows:
            continue
        output.append(family_id)
        if int(max_families) > 0 and len(output) >= int(max_families):
            break
    return output


def hex_to_rgba(color_text: str, alpha: int) -> Tuple[int, int, int, int]:
    red, green, blue = ImageColor.getrgb(color_text)
    return red, green, blue, int(alpha)


def scale_points(points: Sequence[Sequence[float]], scale: float) -> List[Tuple[float, float]]:
    return [(float(point[0]) * scale, float(point[1]) * scale) for point in points]


def scale_box(box: Dict[str, float], scale: float) -> Tuple[float, float, float, float]:
    return (
        float(box.get("x_min", 0.0)) * scale,
        float(box.get("y_min", 0.0)) * scale,
        float(box.get("x_max", 0.0)) * scale,
        float(box.get("y_max", 0.0)) * scale,
    )


def add_header(image: Image.Image, title: str, lines: Sequence[str]) -> Image.Image:
    font = ImageFont.load_default()
    header_height = 32 + 14 * max(1, len(lines))
    canvas = Image.new("RGB", (image.width, image.height + header_height), "#101214")
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill="#f5f5f5", font=font)
    for index, line in enumerate(lines):
        draw.text((10, 24 + index * 14), str(line), fill="#d0d7de", font=font)
    return canvas


def build_preview_image(image: Image.Image, max_side_px: int) -> Tuple[Image.Image, float]:
    if int(max_side_px) <= 0:
        return image.convert("RGB"), 1.0
    max_side = max(image.size)
    if max_side <= int(max_side_px):
        return image.convert("RGB"), 1.0
    scale = float(max_side_px) / float(max_side)
    new_width = max(1, int(round(image.width * scale)))
    new_height = max(1, int(round(image.height * scale)))
    resampling = getattr(Image, "Resampling", Image)
    resized = image.resize((new_width, new_height), resampling.BILINEAR)
    return resized.convert("RGB"), scale


def draw_local_lines(panel: Image.Image, lines: Sequence[Dict], scale: float, line_width: int, color_text: str) -> None:
    draw = ImageDraw.Draw(panel, "RGBA")
    for line in lines:
        points = line.get("points", [])
        if len(points) < 2:
            continue
        draw.line(scale_points(points, scale), fill=hex_to_rgba(color_text, 210), width=max(1, int(line_width)), joint="curve")


def localize_box(box: Dict, crop_box: Dict) -> Dict[str, float]:
    offset_x = float(crop_box.get("x_min", 0.0))
    offset_y = float(crop_box.get("y_min", 0.0))
    return {
        "x_min": float(box.get("x_min", 0.0)) - offset_x,
        "y_min": float(box.get("y_min", 0.0)) - offset_y,
        "x_max": float(box.get("x_max", 0.0)) - offset_x,
        "y_max": float(box.get("y_max", 0.0)) - offset_y,
    }


def draw_box(panel: Image.Image, box: Dict[str, float], scale: float, box_width: int, color_text: str) -> None:
    draw = ImageDraw.Draw(panel, "RGBA")
    draw.rectangle(scale_box(box, scale), outline=hex_to_rgba(color_text, 180), width=max(1, int(box_width)))


def make_side_by_side(panels: Sequence[Image.Image], gap_px: int = 12) -> Image.Image:
    width = sum(panel.width for panel in panels) + max(0, len(panels) - 1) * int(gap_px)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), "#0b0d0f")
    offset_x = 0
    for panel in panels:
        canvas.paste(panel, (offset_x, 0))
        offset_x += panel.width + int(gap_px)
    return canvas


def load_patch_image(label_row: Dict, meta_row: Dict, band_indices: Sequence[int]) -> Tuple[Image.Image, str]:
    patch_image_path = resolve_patch_image_path(label_row)
    if patch_image_path is not None:
        with Image.open(patch_image_path) as handle:
            return handle.convert("RGB"), str(patch_image_path)

    source_image_path = require_existing_path(Path(str(meta_row.get("source_image_path", ""))), kind="file")
    image_hwc, _ = read_rgb_geotiff(source_image_path, band_indices=tuple(int(index) for index in band_indices))
    crop_box = meta_row.get("crop_box", {})
    crop = image_hwc[
        int(crop_box.get("y_min", 0)):int(crop_box.get("y_max", 0)),
        int(crop_box.get("x_min", 0)):int(crop_box.get("x_max", 0)),
        :,
    ]
    return Image.fromarray(crop).convert("RGB"), str(source_image_path)


def build_raw_patch_lines(global_features: Sequence[Dict], meta_row: Dict) -> List[Dict]:
    crop_box = meta_row.get("crop_box", {})
    segments_global = build_patch_segments_global(
        global_features=global_features,
        rect_global=(
            float(crop_box.get("x_min", 0.0)),
            float(crop_box.get("y_min", 0.0)),
            float(crop_box.get("x_max", 0.0)),
            float(crop_box.get("y_max", 0.0)),
        ),
        resample_step_px=0.0,
        boundary_tol_px=0.0,
    )
    return build_patch_target_lines(segments_global=segments_global, patch={"crop_box": crop_box}, quantize=False)


def resolve_source_lane_path(meta_row: Dict, image_dir_relpath: str, lane_relpath: str) -> Path:
    source_image_path = require_existing_path(Path(str(meta_row.get("source_image_path", ""))), kind="file")
    sample_dir = source_image_path.parent
    image_dir_parts = [part for part in Path(str(image_dir_relpath)).parts if part not in (".", "")]
    for _ in image_dir_parts:
        sample_dir = sample_dir.parent
    lane_path = (sample_dir / str(lane_relpath)).resolve()
    return require_existing_path(lane_path, kind="file")


def summarize_family(family_rows: Sequence[Dict]) -> Dict[str, int]:
    patch_count = len(family_rows)
    output_line_count = sum(len(item.get("output_lines", [])) for item in family_rows)
    non_empty_patch_count = sum(1 for item in family_rows if item.get("output_lines", []))
    return {
        "patch_count": int(patch_count),
        "non_empty_patch_count": int(non_empty_patch_count),
        "output_line_count": int(output_line_count),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    ordered_manifest_ids = load_optional_family_manifest(str(args.family_manifest))
    label_rows = load_label_rows(args.label_files)
    meta_rows_by_id = load_meta_rows(args.meta_jsonl)
    allowed_splits = {str(split).strip() for split in args.splits if str(split).strip()}
    grouped_patch_rows = build_patch_contexts(label_rows, meta_rows_by_id, allowed_splits)
    selected_family_ids = build_ordered_family_ids(
        explicit_family_ids=args.family_id,
        ordered_manifest_ids=ordered_manifest_ids,
        grouped_patch_rows=grouped_patch_rows,
        max_families=int(args.max_families),
    )
    if not selected_family_ids:
        raise RuntimeError("No families matched the requested filters.")

    summary_rows: List[Dict] = []
    log_event("Visualize", f"start family_count={len(selected_family_ids)} output_dir={output_dir}")
    for family_index, family_id in enumerate(selected_family_ids, start=1):
        family_rows = grouped_patch_rows.get(family_id, [])
        if not family_rows:
            continue
        first_meta = family_rows[0]["meta_row"]
        source_image_path = require_existing_path(Path(str(first_meta.get("source_image_path", ""))), kind="file")
        source_lane_path = resolve_source_lane_path(first_meta, image_dir_relpath=str(args.image_dir_relpath), lane_relpath=str(args.lane_relpath))
        raster_meta = read_raster_metadata(source_image_path)
        global_features = load_sample_global_features(lane_path=source_lane_path, raster_meta=raster_meta)

        family_output_dir = output_dir / family_id
        ensure_dir(family_output_dir)
        family_counts = summarize_family(family_rows)
        family_patch_summaries: List[Dict] = []

        for patch_row in family_rows:
            meta_row = patch_row["meta_row"]
            patch_image, patch_image_source = load_patch_image(patch_row["label_row"], meta_row, args.band_indices)
            preview_base, scale = build_preview_image(patch_image, max_side_px=int(args.max_side_px))

            raw_lines = build_raw_patch_lines(global_features, meta_row)
            output_lines = list(patch_row.get("output_lines", []))
            crop_box = meta_row.get("crop_box", {})
            keep_box = localize_box(meta_row.get("keep_box", {}), crop_box)
            local_crop_box = {
                "x_min": 0.0,
                "y_min": 0.0,
                "x_max": float(preview_base.width) / float(scale),
                "y_max": float(preview_base.height) / float(scale),
            }

            image_panel = preview_base.copy().convert("RGBA")
            raw_panel = preview_base.copy().convert("RGBA")
            output_panel = preview_base.copy().convert("RGBA")
            draw_local_lines(raw_panel, raw_lines, scale=scale, line_width=int(args.line_width), color_text=RAW_LANE_COLOR)
            draw_local_lines(output_panel, output_lines, scale=scale, line_width=int(args.line_width), color_text=OUTPUT_LANE_COLOR)

            if args.draw_crop_boxes:
                draw_box(raw_panel, local_crop_box, scale=scale, box_width=int(args.box_width), color_text=CROP_BOX_COLOR)
                draw_box(output_panel, local_crop_box, scale=scale, box_width=int(args.box_width), color_text=CROP_BOX_COLOR)
            if args.draw_keep_boxes:
                draw_box(raw_panel, keep_box, scale=scale, box_width=int(args.box_width), color_text=KEEP_BOX_COLOR)
                draw_box(output_panel, keep_box, scale=scale, box_width=int(args.box_width), color_text=KEEP_BOX_COLOR)

            image_title = add_header(
                image_panel.convert("RGB"),
                title="Original 512 Patch",
                lines=[
                    f"sample_id={patch_row['sample_id']}",
                    f"patch_id={patch_row['patch_id']} split={patch_row['split']}",
                ],
            )

            raw_title = add_header(
                raw_panel.convert("RGB"),
                title="Original Patch + Raw Lane Labels",
                lines=[
                    f"raw_lines={len(raw_lines)} lane_file={source_lane_path.name}",
                    f"patch_image={Path(patch_image_source).name}",
                ],
            )
            output_title = add_header(
                output_panel.convert("RGB"),
                title=f"Original Patch + {str(args.label_name)}",
                lines=[
                    f"output_lines={len(output_lines)} file={Path(str(patch_row['label_row'].get('_label_path', ''))).name}",
                    f"source_image={Path(str(first_meta.get('source_image_path', ''))).name}",
                ],
            )
            comparison = make_side_by_side([image_title, raw_title, output_title])
            output_path = family_output_dir / f"{patch_row['sample_id']}.png"
            comparison.save(output_path)
            patch_image.close()

            family_patch_summaries.append(
                {
                    "sample_id": patch_row["sample_id"],
                    "patch_id": int(patch_row["patch_id"]),
                    "split": patch_row["split"],
                    "patch_image_path": patch_image_source,
                    "label_file": str(patch_row["label_row"].get("_label_path", "")),
                    "output_image": str(output_path),
                    "raw_line_count": int(len(raw_lines)),
                    "output_line_count": int(len(output_lines)),
                }
            )

        summary_rows.append(
            {
                "family_id": family_id,
                "split": str(first_meta.get("split", "")),
                "source_image_path": str(source_image_path),
                "source_lane_path": str(source_lane_path),
                "output_dir": str(family_output_dir),
                "patch_count": family_counts["patch_count"],
                "non_empty_patch_count": family_counts["non_empty_patch_count"],
                "output_line_count": family_counts["output_line_count"],
                "patches": family_patch_summaries,
            }
        )
        log_event(
            "Visualize",
            f"family_progress={family_index}/{len(selected_family_ids)} family_id={family_id} patch_count={len(family_patch_summaries)}",
        )

    write_json(
        output_dir / "summary.json",
        {
            "label_files": [str(Path(path).resolve()) for path in args.label_files],
            "meta_jsonl": [str(Path(path).resolve()) for path in args.meta_jsonl],
            "family_manifest": str(Path(args.family_manifest).resolve()) if str(args.family_manifest).strip() else "",
            "lane_relpath": str(args.lane_relpath),
            "image_dir_relpath": str(args.image_dir_relpath),
            "output_dir": str(output_dir),
            "family_count": int(len(summary_rows)),
            "families": summary_rows,
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "family_count": len(summary_rows)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()