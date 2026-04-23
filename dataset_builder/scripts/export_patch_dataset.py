"""从 RC family manifest 导出 line 数据集和 segmentation 数据集。"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw

DATASET_BUILDER_ROOT = Path(__file__).resolve().parents[1]
if str(DATASET_BUILDER_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_BUILDER_ROOT))

from road_builder.io_utils import ensure_dir, format_progress, load_jsonl, log_error, log_event, log_warning, require_existing_path, resolve_optional_text, validate_ratio, write_json, write_jsonl
from road_builder.patch_export import BOX_PROMPT_TEMPLATE, build_patch_segments_global, build_patch_target_lines, format_box_prompt, make_box_record
from road_builder.source_data import load_family_raster_and_mask, load_sample_global_features


LINE_DATASET_DIRNAME = "data_line"
SEG_DATASET_DIRNAME = "data_seg"
ARTIFACTS_DIRNAME = "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export road centerline line/segmentation datasets from family manifest.")
    parser.add_argument("--family-manifest", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    parser.add_argument("--band-indices", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--resample-step-px", type=float, default=4.0)
    parser.add_argument("--boundary-tol-px", type=float, default=2.5)
    parser.add_argument("--seg-line-width-px", type=int, default=3)
    parser.add_argument("--max-families-per-split", type=int, default=0)
    parser.add_argument("--empty-patch-drop-ratio", type=float, default=0.95)
    parser.add_argument("--empty-patch-seed", type=int, default=42)
    parser.add_argument("--user-prompt", type=str, default="")
    parser.add_argument("--user-prompt-file", type=str, default="")
    return parser.parse_args()


def build_patch_meta_row(
    *,
    sample_id: str,
    split: str,
    family: Dict,
    patch: Dict,
    source_patch_name: str,
    line_image_rel_path: str,
    seg_image_rel_path: str,
    seg_mask_rel_path: str,
    target_lines: Sequence[Dict],
    segmentation_lines: Sequence[Dict],
    prompt_text: str,
    resample_step_px: float,
    box_size: List[int],
) -> Dict:
    return {
        "id": sample_id,
        "split": split,
        "family_id": family["family_id"],
        "source_sample_id": family.get("source_sample_id", ""),
        "source_image": family.get("source_image", ""),
        "source_image_path": family.get("source_image_path", ""),
        "source_mask_path": family.get("source_mask_path", ""),
        "source_lane_path": family.get("source_lane_path", ""),
        "source_patch_name": source_patch_name,
        "patch_id": int(patch["patch_id"]),
        "row": int(patch["row"]),
        "col": int(patch["col"]),
        "scan_index": int(patch["patch_id"]),
        "line_image": line_image_rel_path,
        "seg_image": seg_image_rel_path,
        "seg_mask": seg_mask_rel_path,
        "image_size": list(box_size),
        "crop_box": patch["crop_box"],
        "keep_box": patch["keep_box"],
        "target_box": {"x_min": 0, "y_min": 0, "x_max": int(box_size[0]), "y_max": int(box_size[1])},
        "mask_ratio": float(patch.get("mask_ratio", 0.0)),
        "mask_pixels": int(patch.get("mask_pixels", 0)),
        "num_target_lines": len(target_lines),
        "num_seg_lines": len(segmentation_lines),
        "resample_step_px": float(resample_step_px),
        "prompt_text": prompt_text,
        "target_lines": list(target_lines),
    }


def make_segmentation_record(*, sample_id: str, image_rel_path: str, mask_rel_path: str) -> Dict:
    return {
        "id": str(sample_id),
        "image": str(image_rel_path).replace("\\", "/"),
        "mask": str(mask_rel_path).replace("\\", "/"),
    }


def build_segmentation_lines(*, patch: Dict, global_features: Sequence[Dict]) -> List[Dict]:
    crop_box = patch["crop_box"]
    segments_global = build_patch_segments_global(
        global_features=global_features,
        rect_global=(float(crop_box["x_min"]), float(crop_box["y_min"]), float(crop_box["x_max"]), float(crop_box["y_max"])),
        resample_step_px=0.0,
        boundary_tol_px=0.0,
    )
    return build_patch_target_lines(segments_global=segments_global, patch=patch, quantize=False)


def render_segmentation_mask(*, segmentation_lines: Sequence[Dict], image_size: Tuple[int, int], line_width_px: int) -> Image.Image:
    mask_image = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask_image)
    for line in segmentation_lines:
        points = line.get("points", [])
        if not isinstance(points, list) or len(points) < 2:
            continue
        draw.line([(float(point[0]), float(point[1])) for point in points], fill=255, width=max(1, int(line_width_px)), joint="curve")
    return mask_image


def downsample_empty_records(rows: Sequence[Dict], meta_rows: Sequence[Dict], drop_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict], Dict[str, int | float]]:
    safe_ratio = validate_ratio("empty patch drop ratio", float(drop_ratio))
    paired = list(zip(rows, meta_rows))
    non_empty = [pair for pair in paired if int(pair[1].get("num_target_lines", 0)) > 0]
    empty = [pair for pair in paired if int(pair[1].get("num_target_lines", 0)) <= 0]
    keep_empty = int(round(len(empty) * (1.0 - safe_ratio)))
    keep_empty = max(0, min(len(empty), keep_empty))
    rng = random.Random(int(seed))
    chosen_empty = empty if keep_empty >= len(empty) else rng.sample(empty, keep_empty)
    keep_ids = {id(pair) for pair in non_empty}
    keep_ids.update(id(pair) for pair in chosen_empty)
    kept_rows: List[Dict] = []
    kept_meta: List[Dict] = []
    for pair in paired:
        if id(pair) not in keep_ids:
            continue
        kept_rows.append(pair[0])
        kept_meta.append(pair[1])
    summary = {
        "generated_total": int(len(paired)),
        "generated_non_empty": int(len(non_empty)),
        "generated_empty": int(len(empty)),
        "kept_total": int(len(kept_rows)),
        "kept_non_empty": int(len(non_empty)),
        "kept_empty": int(len(kept_rows) - len(non_empty)),
        "drop_ratio": float(safe_ratio),
    }
    return kept_rows, kept_meta, summary


def filter_rows_by_sample_ids(rows: Sequence[Dict], sample_ids: set[str]) -> List[Dict]:
    return [row for row in rows if str(row.get("id", "")) in sample_ids]


def cleanup_unused_outputs(saved_outputs: Sequence[Dict], kept_sample_ids: set[str]) -> None:
    for output in saved_outputs:
        if str(output.get("sample_id", "")) in kept_sample_ids:
            continue
        for key in ("line_image_path", "seg_image_path", "seg_mask_path"):
            path = output.get(key)
            if isinstance(path, Path) and path.exists():
                path.unlink()


def export_split(
    *,
    split: str,
    families: Sequence[Dict],
    line_dataset_root: Path,
    seg_dataset_root: Path,
    artifacts_root: Path,
    band_indices: Sequence[int],
    mask_threshold: int,
    resample_step_px: float,
    boundary_tol_px: float,
    seg_line_width_px: int,
    max_families_per_split: int,
    empty_patch_drop_ratio: float,
    empty_patch_seed: int,
    user_prompt_template: str,
    sample_index_start: int,
) -> Tuple[Dict[str, object], int]:
    line_rows: List[Dict] = []
    seg_rows: List[Dict] = []
    meta_rows: List[Dict] = []
    saved_outputs: List[Dict] = []
    family_count = 0
    sample_index = int(sample_index_start)
    split_families = [family for family in families if str(family.get("split", "")) == split]
    if int(max_families_per_split) > 0:
        split_families = split_families[: int(max_families_per_split)]
    log_event("PatchExport", f"split={split} family_count={len(split_families)}")
    for index, family in enumerate(split_families, start=1):
        if index == 1 or index == len(split_families) or index % 10 == 0:
            log_event("PatchExport", f"split={split} family_progress={format_progress(index, len(split_families))} family_id={family.get('family_id', '')}")
        try:
            raw_image_hwc, raster_meta, _ = load_family_raster_and_mask(family, band_indices=[int(value) for value in band_indices], mask_threshold=int(mask_threshold))
            global_features = load_sample_global_features(
                lane_path=Path(str(family.get("source_lane_path", ""))),
                raster_meta=raster_meta,
            )
            if not global_features:
                log_warning("PatchExport", f"split={split} family_id={family.get('family_id', '')} has no valid GeoJSON features")
            patches = sorted(list(family.get("patches", [])), key=lambda item: int(item["patch_id"]))
            for patch in patches:
                crop_box = patch["crop_box"]
                keep_box = patch["keep_box"]
                patch_image = Image.fromarray(raw_image_hwc[int(crop_box["y_min"]):int(crop_box["y_max"]), int(crop_box["x_min"]):int(crop_box["x_max"]), :])
                segments_global = build_patch_segments_global(
                    global_features=global_features,
                    rect_global=(float(keep_box["x_min"]), float(keep_box["y_min"]), float(keep_box["x_max"]), float(keep_box["y_max"])),
                    resample_step_px=float(resample_step_px),
                    boundary_tol_px=float(boundary_tol_px),
                )
                target_lines = build_patch_target_lines(segments_global=segments_global, patch=patch, quantize=True)
                segmentation_lines = build_segmentation_lines(patch=patch, global_features=global_features)

                patch_id = int(patch["patch_id"])
                sample_id = f"{sample_index:06d}"
                sample_index += 1
                source_patch_name = f"{family['family_id']}_{patch_id:04d}"
                image_name = f"{sample_id}.png"
                line_image_rel = Path("images") / image_name
                seg_image_rel = Path("images") / image_name
                seg_mask_rel = Path("masks") / image_name
                line_image_path = line_dataset_root / line_image_rel
                seg_image_path = seg_dataset_root / seg_image_rel
                seg_mask_path = seg_dataset_root / seg_mask_rel
                ensure_dir(line_image_path.parent)
                ensure_dir(seg_image_path.parent)
                ensure_dir(seg_mask_path.parent)
                patch_image.save(line_image_path)
                patch_image.save(seg_image_path)
                image_width, image_height = patch_image.size
                mask_image = render_segmentation_mask(segmentation_lines=segmentation_lines, image_size=(int(image_width), int(image_height)), line_width_px=int(seg_line_width_px))
                mask_image.save(seg_mask_path)
                patch_image.close()
                mask_image.close()

                prompt_text = format_box_prompt(
                    {
                        "box_x_min": 0,
                        "box_y_min": 0,
                        "box_x_max": int(image_width),
                        "box_y_max": int(image_height),
                    },
                    prompt_template=user_prompt_template,
                )

                line_rows.append(make_box_record(sample_id=sample_id, image_rel_path=line_image_rel.as_posix(), target_lines=target_lines, prompt_text=prompt_text))
                seg_rows.append(make_segmentation_record(sample_id=sample_id, image_rel_path=seg_image_rel.as_posix(), mask_rel_path=seg_mask_rel.as_posix()))
                meta_rows.append(
                    build_patch_meta_row(
                        sample_id=sample_id,
                        split=split,
                        family=family,
                        patch=patch,
                        source_patch_name=source_patch_name,
                        line_image_rel_path=line_image_rel.as_posix(),
                        seg_image_rel_path=seg_image_rel.as_posix(),
                        seg_mask_rel_path=seg_mask_rel.as_posix(),
                        target_lines=target_lines,
                        segmentation_lines=segmentation_lines,
                        prompt_text=prompt_text,
                        resample_step_px=float(resample_step_px),
                        box_size=[int(image_width), int(image_height)],
                    )
                )
                saved_outputs.append(
                    {
                        "sample_id": sample_id,
                        "line_image_path": line_image_path,
                        "seg_image_path": seg_image_path,
                        "seg_mask_path": seg_mask_path,
                    }
                )
        except Exception as exc:
            log_error("PatchExport", f"split={split} family_id={family.get('family_id', '')} failed: {exc}")
            raise
        family_count += 1

    kept_line_rows, kept_meta, filter_summary = downsample_empty_records(line_rows, meta_rows, drop_ratio=float(empty_patch_drop_ratio), seed=int(empty_patch_seed))
    kept_sample_ids = {str(row.get("id", "")) for row in kept_line_rows}
    kept_seg_rows = filter_rows_by_sample_ids(seg_rows, kept_sample_ids)
    cleanup_unused_outputs(saved_outputs, kept_sample_ids)

    count_line = write_jsonl(line_dataset_root / f"{split}.jsonl", kept_line_rows)
    count_seg = write_jsonl(seg_dataset_root / f"{split}.jsonl", kept_seg_rows)
    count_meta = write_jsonl(artifacts_root / f"meta_{split}.jsonl", kept_meta)
    log_event("PatchExport", f"split={split} done line_kept={count_line} seg_kept={count_seg} empty_kept={filter_summary['kept_empty']}")
    return {
        "families": int(family_count),
        "line_samples": int(count_line),
        "seg_samples": int(count_seg),
        "meta_samples": int(count_meta),
        "empty_patch_filter": filter_summary,
    }, sample_index


def main() -> None:
    args = parse_args()
    validate_ratio("--empty-patch-drop-ratio", float(args.empty_patch_drop_ratio))
    manifest_path = require_existing_path(Path(args.family_manifest), kind="file")
    families = load_jsonl(manifest_path)
    if not families:
        raise RuntimeError(f"Family manifest is empty: {manifest_path}")

    output_root = Path(args.output_root).resolve()
    line_dataset_root = output_root / LINE_DATASET_DIRNAME
    seg_dataset_root = output_root / SEG_DATASET_DIRNAME
    artifacts_root = output_root / ARTIFACTS_DIRNAME
    ensure_dir(line_dataset_root)
    ensure_dir(seg_dataset_root)
    ensure_dir(artifacts_root)
    user_prompt_text = resolve_optional_text(inline_text=str(args.user_prompt), file_path=str(args.user_prompt_file), fallback=BOX_PROMPT_TEMPLATE)

    log_event("PatchExport", f"start manifest={manifest_path} output_root={output_root} family_count={len(families)}")
    summary: Dict[str, object] = {
        "source_family_manifest": str(manifest_path),
        "output_root": str(output_root),
        "line_dataset_root": str(line_dataset_root),
        "seg_dataset_root": str(seg_dataset_root),
        "artifacts_root": str(artifacts_root),
        "task": "road_centerline_line_and_segmentation",
        "category": "road_centerline",
        "user_prompt_template": user_prompt_text,
        "band_indices": [int(index) for index in args.band_indices],
        "seg_line_width_px": int(args.seg_line_width_px),
        "splits": {},
    }

    sample_index = 1
    for split in [str(item) for item in args.splits]:
        split_summary, sample_index = export_split(
            split=split,
            families=families,
            line_dataset_root=line_dataset_root,
            seg_dataset_root=seg_dataset_root,
            artifacts_root=artifacts_root,
            band_indices=[int(index) for index in args.band_indices],
            mask_threshold=int(args.mask_threshold),
            resample_step_px=float(args.resample_step_px),
            boundary_tol_px=float(args.boundary_tol_px),
            seg_line_width_px=int(args.seg_line_width_px),
            max_families_per_split=int(args.max_families_per_split),
            empty_patch_drop_ratio=float(args.empty_patch_drop_ratio),
            empty_patch_seed=int(args.empty_patch_seed),
            user_prompt_template=user_prompt_text,
            sample_index_start=sample_index,
        )
        summary["splits"][split] = split_summary
        log_event("PatchExport", f"split={split} summary line={split_summary['line_samples']} seg={split_summary['seg_samples']}")

    write_json(artifacts_root / "build_summary.json", summary)
    log_event("PatchExport", f"done summary={artifacts_root / 'build_summary.json'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
