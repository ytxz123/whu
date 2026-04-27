from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import BuildPaths, build_paths, load_yaml
from .geo import load_lane_lines, read_mask, read_raster_rgb
from .io_utils import ensure_dir, sanitize_name, write_jsonl
from .rendering import AnnotationStyle, render_label_image
from .tiling import PatchWindow, clip_polyline_to_rect, generate_patch_windows, localize_and_quantize


@dataclass
class RuntimeConfig:
    dataset_root: Path | None
    train_root: Path | None
    val_root: Path | None
    output_root: Path
    splits: list[str]
    image_dir_relpath: str
    lane_relpath: str
    image_glob: str
    mask_suffix: str
    exclude_image_stem_suffixes: tuple[str, ...]
    label_image_dirname: str
    patch_size: int
    mask_threshold: int
    min_mask_ratio: float
    min_mask_pixels: int
    drop_empty_annotations: bool
    empty_annotation_keep_ratio: float
    empty_annotation_seed: int
    save_black_background_from_mask: bool
    max_samples_per_split: int
    simplify_tolerance: float
    system_prompt: str
    user_prompt: str


def build_runtime(cfg: dict) -> tuple[RuntimeConfig, BuildPaths]:
    paths = build_paths(cfg)
    return (
        RuntimeConfig(
            dataset_root=paths.dataset_root,
            train_root=paths.train_root,
            val_root=paths.val_root,
            output_root=paths.output_root,
            splits=[str(x) for x in cfg.get("splits", ["train", "val"])],
            image_dir_relpath=str(cfg.get("image_dir_relpath", "patch_tif")),
            lane_relpath=str(cfg.get("lane_relpath", "label_check_crop/Lane.geojson")),
            image_glob=str(cfg.get("image_glob", "*.tif")),
            mask_suffix=str(cfg.get("mask_suffix", "_edit_poly.tif")),
            exclude_image_stem_suffixes=tuple(
                str(x).strip() for x in cfg.get("exclude_image_stem_suffixes", ["_ground", "_lane", "_pose"]) if str(x).strip()
            ),
            label_image_dirname=str(cfg.get("label_image_dirname", "img_label")).strip() or "img_label",
            patch_size=int(cfg.get("patch_size", 512)),
            mask_threshold=int(cfg.get("mask_threshold", 127)),
            min_mask_ratio=float(cfg.get("min_mask_ratio", 0.02)),
            min_mask_pixels=int(cfg.get("min_mask_pixels", 64)),
            drop_empty_annotations=bool(cfg.get("drop_empty_annotations", True)),
            empty_annotation_keep_ratio=float(cfg.get("empty_annotation_keep_ratio", 0.10)),
            empty_annotation_seed=int(cfg.get("empty_annotation_seed", 42)),
            save_black_background_from_mask=bool(cfg.get("save_black_background_from_mask", True)),
            max_samples_per_split=int(cfg.get("max_samples_per_split", 0)),
            simplify_tolerance=float(cfg.get("simplify_tolerance", 2.0)),
            system_prompt=str(cfg.get("system_prompt", "")).strip(),
            user_prompt=str(cfg.get("user_prompt", "")).strip(),
        ),
        paths,
    )


def resolve_split_root(split: str, paths: BuildPaths) -> Path | None:
    if split == "train" and paths.train_root is not None:
        return paths.train_root
    if split == "val" and paths.val_root is not None:
        return paths.val_root
    if paths.dataset_root is None:
        return None
    return (paths.dataset_root / split).resolve()


def family_name(sample_dir: Path, image_path: Path, image_count: int) -> str:
    base = sanitize_name(sample_dir.name)
    return base if image_count <= 1 else sanitize_name(f"{sample_dir.name}_{image_path.stem}")


def should_use_source_image(image_path: Path, mask_suffix: str, exclude_stem_suffixes: tuple[str, ...]) -> bool:
    if image_path.name.endswith(mask_suffix):
        return False
    stem = image_path.stem
    return not any(stem.endswith(suffix) for suffix in exclude_stem_suffixes)


def image_pairs(
    sample_dir: Path,
    image_dir_relpath: str,
    image_glob: str,
    mask_suffix: str,
    exclude_stem_suffixes: tuple[str, ...],
) -> list[tuple[Path, Path | None]]:
    image_dir = sample_dir / image_dir_relpath
    if not image_dir.is_dir():
        return []
    pairs = []
    for image_path in sorted(image_dir.glob(image_glob)):
        if not image_path.is_file():
            continue
        if not should_use_source_image(image_path, mask_suffix, exclude_stem_suffixes):
            continue
        mask_path = image_path.with_name(f"{image_path.stem}{mask_suffix}")
        pairs.append((image_path, mask_path if mask_path.is_file() else None))
    return pairs


def patch_mask_stats(mask, window: PatchWindow) -> tuple[float, int]:
    if mask is None:
        return 0.0, 0
    crop = crop_array_to_window(mask, window, fill_value=0)
    return float(crop.mean()), int(crop.sum())


def crop_array_to_window(array, window: PatchWindow, fill_value: int = 0):
    target_height = max(1, window.y1 - window.y0)
    target_width = max(1, window.x1 - window.x0)
    y0 = max(0, int(window.y0))
    x0 = max(0, int(window.x0))
    y1 = min(int(window.y1), int(array.shape[0]))
    x1 = min(int(window.x1), int(array.shape[1]))
    crop = array[y0:y1, x0:x1]
    if crop.shape[0] == target_height and crop.shape[1] == target_width:
        return crop.copy()

    if crop.ndim == 2:
        out = np.full((target_height, target_width), fill_value, dtype=array.dtype)
        out[: crop.shape[0], : crop.shape[1]] = crop
        return out

    out = np.full((target_height, target_width, crop.shape[2]), fill_value, dtype=array.dtype)
    out[: crop.shape[0], : crop.shape[1], :] = crop
    return out


def should_keep_patch(mask_ratio: float, mask_pixels: int, lines: list[dict], cfg: RuntimeConfig) -> bool:
    if mask_pixels >= cfg.min_mask_pixels or mask_ratio >= cfg.min_mask_ratio:
        return True
    return bool(lines)


def make_record(sample_id: str, system_prompt: str, user_prompt: str, assistant_payload: list[dict], image_rel: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=False, separators=(",", ":"))},
        ],
        "images": [image_rel.replace("\\", "/")],
    }


def keep_empty_annotation(cfg: RuntimeConfig, rng: random.Random) -> bool:
    if not cfg.drop_empty_annotations:
        return True
    ratio = min(max(float(cfg.empty_annotation_keep_ratio), 0.0), 1.0)
    return rng.random() < ratio


def export_split(split: str, split_root: Path, cfg: RuntimeConfig) -> int:
    out_dir = ensure_dir(cfg.output_root)
    img_root = ensure_dir(out_dir / f"img_{split}")
    label_split_root = ensure_dir(out_dir / cfg.label_image_dirname / split)
    rows: list[dict] = []
    rng = random.Random(int(cfg.empty_annotation_seed) + (0 if split == "train" else 100003))
    label_style = AnnotationStyle(line_width=2, draw_points=False, fixed_line_color=(255, 255, 255))
    sample_dirs = [p for p in sorted(split_root.iterdir()) if p.is_dir()]
    if cfg.max_samples_per_split > 0:
        sample_dirs = sample_dirs[: cfg.max_samples_per_split]

    for sample_dir in sample_dirs:
        pairs = image_pairs(
            sample_dir,
            cfg.image_dir_relpath,
            cfg.image_glob,
            cfg.mask_suffix,
            cfg.exclude_image_stem_suffixes,
        )
        pair_count = len(pairs)
        for image_path, mask_path in pairs:
            image, raster_meta = read_raster_rgb(image_path)
            mask = read_mask(mask_path, cfg.mask_threshold) if mask_path is not None else None
            if mask is not None and cfg.save_black_background_from_mask:
                image = image.copy()
                image[mask <= 0] = 0
            lines_global = load_lane_lines(sample_dir / cfg.lane_relpath, raster_meta)
            family_id = family_name(sample_dir, image_path, pair_count)
            family_dir = ensure_dir(img_root / family_id)
            label_family_dir = ensure_dir(label_split_root / family_id)
            windows = generate_patch_windows(raster_meta.width, raster_meta.height, cfg.patch_size)
            patch_number = 0
            for window in windows:
                rect = (float(window.x0), float(window.y0), float(window.x1), float(window.y1))
                clipped = []
                for line in lines_global:
                    clipped.extend(clip_polyline_to_rect(line, rect))
                label_lines = localize_and_quantize(clipped, window, 0.0)
                local_lines = localize_and_quantize(clipped, window, cfg.simplify_tolerance)
                mask_ratio, mask_pixels = patch_mask_stats(mask, window)
                if not should_keep_patch(mask_ratio, mask_pixels, local_lines, cfg):
                    continue
                if not local_lines and not keep_empty_annotation(cfg, rng):
                    continue

                patch_number += 1
                patch_name = f"r{window.row}_c{window.col}_p{patch_number:02d}.png"
                patch_path = family_dir / patch_name
                label_patch_path = label_family_dir / patch_name
                patch_array = crop_array_to_window(image, window, fill_value=0)
                patch = Image.fromarray(patch_array)
                patch.save(patch_path)
                patch.close()
                label_patch = render_label_image(patch_array.shape[1], patch_array.shape[0], label_lines, label_style)
                label_patch.save(label_patch_path)
                label_patch.close()

                sample_id = f"{family_id}_r{window.row}_c{window.col}_p{patch_number:02d}"
                image_rel = f"img_{split}/{family_id}/{patch_name}"
                rows.append(make_record(sample_id, cfg.system_prompt, cfg.user_prompt, local_lines, image_rel))

    return write_jsonl(cfg.output_root / f"{split}.jsonl", rows)


def run(config_path: str | Path, overrides: dict | None = None) -> None:
    cfg_dict = load_yaml(config_path)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                cfg_dict[key] = value
    cfg, paths = build_runtime(cfg_dict)
    if not cfg.dataset_root and not cfg.train_root and not cfg.val_root:
        raise ValueError("Please set dataset_root or train_root/val_root in the config file.")
    ensure_dir(cfg.output_root)
    for split in cfg.splits:
        split_root = resolve_split_root(split, paths)
        if split_root is None or not split_root.is_dir():
            continue
        count = export_split(split, split_root, cfg)
        print(f"[data_builder] split={split} rows={count} output={cfg.output_root / f'{split}.jsonl'}", flush=True)
