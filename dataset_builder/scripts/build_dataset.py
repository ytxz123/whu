"""带默认参数的一键构建入口。

用途：
- 串联执行 manifest 构建和 line/segmentation 数据集导出
- 把常用参数集中到顶部配置区，便于直接修改
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from string import Formatter
from typing import List


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_BUILDER_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
PROMPTS_DIR = DATASET_BUILDER_ROOT / "prompts"

BOX_PROMPT_REQUIRED_FIELDS = {
    "box_x_min",
    "box_y_min",
    "box_x_max",
    "box_y_max",
}


@dataclass
class BuildConfig:
    # 数据根目录。若 train / val 分开，可以改成分别填写 train_root / val_root。
    dataset_root: str = "/home/tzy/Desktop/test"
    train_root: str = ""
    val_root: str = ""

    # 主输出目录。
    output_root: str = str(PROJECT_ROOT / "dataset" / "dataset_test")

    # 通用 split。
    splits: List[str] = field(default_factory=lambda: ["train", "val"])

    # manifest 参数。
    image_relpath: str = ""
    mask_relpath: str = ""
    image_dir_relpath: str = "patch_tif"
    image_glob: str = "*.tif"
    mask_suffix: str = "_edit_poly.tif"
    lane_relpath: str = "label_check_crop/Lane.geojson"
    mask_threshold: int = 127
    box_size_px: int = 512
    overlap_px: int = 0
    keep_margin_px: int = 0
    review_crop_pad_px: int = 0
    box_min_mask_ratio: float = 0.05
    box_min_mask_pixels: int = 256
    box_max_per_sample: int = 0
    search_within_review_bbox: bool = False
    fallback_to_all_if_empty: bool = False
    max_samples_per_split: int = 0

    # box 数据集导出参数。
    band_indices: List[int] = field(default_factory=lambda: [1, 2, 3])
    box_resample_step_px: float = 12.0
    box_boundary_tol_px: float = 2.5
    max_families_per_split: int = 0
    empty_box_drop_ratio: float = 0.85
    empty_box_seed: int = 42
    box_user_prompt_file: str = str(PROMPTS_DIR / "box_prompt.txt")


CONFIG = BuildConfig()


def log(message: str) -> None:
    print(f"[run-build-all] {message}", flush=True)


def load_template_text(path_text: str, name: str) -> str:
    path = Path(str(path_text)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} template file does not exist: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{name} template file is empty: {path}")
    return text


def extract_template_fields(template_text: str) -> set[str]:
    fields: set[str] = set()
    for _, field_name, _, _ in Formatter().parse(template_text):
        if field_name is None:
            continue
        normalized = str(field_name).strip()
        if not normalized:
            continue
        if any(token in normalized for token in (".", "[", "]")):
            raise ValueError(f"Unsupported template placeholder syntax: {normalized}")
        fields.add(normalized)
    return fields


def validate_template_fields(*, template_name: str, template_text: str, required_fields: set[str], allow_fields: set[str] | None = None) -> None:
    found_fields = extract_template_fields(template_text)
    if allow_fields is None:
        allow_fields = set(required_fields)
    extra_fields = sorted(found_fields - allow_fields)
    missing_fields = sorted(required_fields - found_fields)
    if extra_fields:
        raise ValueError(f"{template_name} template contains unknown placeholders: {extra_fields}")
    if missing_fields:
        raise ValueError(f"{template_name} template is missing required placeholders: {missing_fields}")


def validate_prompt_templates(config: BuildConfig) -> None:
    box_text = load_template_text(config.box_user_prompt_file, "box-dataset") if config.box_user_prompt_file else ""
    if box_text:
        validate_template_fields(template_name="box-dataset", template_text=box_text, required_fields=BOX_PROMPT_REQUIRED_FIELDS, allow_fields=BOX_PROMPT_REQUIRED_FIELDS)
    log("template validation passed")


def run_command(command: List[str], step_name: str) -> None:
    log(f"start step={step_name}")
    log("command=" + " ".join(command))
    try:
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Step failed: {step_name}") from exc
    log(f"done step={step_name}")


def main() -> None:
    python_executable = sys.executable
    config = CONFIG

    if config.dataset_root == "/path/to/rc_dataset" and not config.train_root and not config.val_root:
        raise ValueError("Please edit CONFIG.dataset_root or CONFIG.train_root / CONFIG.val_root before running.")

    validate_prompt_templates(config)

    output_root = Path(config.output_root).resolve()
    artifacts_root = output_root / "artifacts"
    line_dataset_root = output_root / "data_line"
    seg_dataset_root = output_root / "data_seg"
    manifest_path = artifacts_root / "family_manifest.jsonl"

    log(f"project_root={PROJECT_ROOT}")
    log(f"output_root={output_root}")
    log(f"manifest_path={manifest_path}")
    log(f"artifacts_root={artifacts_root}")
    log(f"line_dataset_root={line_dataset_root}")
    log(f"seg_dataset_root={seg_dataset_root}")

    manifest_cmd = [
        python_executable,
        str(SCRIPT_DIR / "build_family_manifest.py"),
        "--output-manifest",
        str(manifest_path),
        "--splits",
        *config.splits,
        "--image-dir-relpath",
        config.image_dir_relpath,
        "--image-glob",
        config.image_glob,
        "--mask-suffix",
        config.mask_suffix,
        "--lane-relpath",
        config.lane_relpath,
        "--mask-threshold",
        str(config.mask_threshold),
        "--tile-size-px",
        str(config.box_size_px),
        "--overlap-px",
        str(config.overlap_px),
        "--keep-margin-px",
        str(config.keep_margin_px),
        "--review-crop-pad-px",
        str(config.review_crop_pad_px),
        "--tile-min-mask-ratio",
        str(config.box_min_mask_ratio),
        "--tile-min-mask-pixels",
        str(config.box_min_mask_pixels),
        "--tile-max-per-sample",
        str(config.box_max_per_sample),
        "--max-samples-per-split",
        str(config.max_samples_per_split),
    ]
    if config.image_relpath:
        manifest_cmd.extend(["--image-relpath", config.image_relpath])
    if config.mask_relpath:
        manifest_cmd.extend(["--mask-relpath", config.mask_relpath])
    if config.train_root:
        manifest_cmd.extend(["--train-root", config.train_root])
    if config.val_root:
        manifest_cmd.extend(["--val-root", config.val_root])
    if (not config.train_root) and (not config.val_root):
        manifest_cmd.extend(["--dataset-root", config.dataset_root])
    if config.search_within_review_bbox:
        manifest_cmd.append("--search-within-review-bbox")
    if config.fallback_to_all_if_empty:
        manifest_cmd.append("--fallback-to-all-if-empty")

    box_dataset_cmd = [
        python_executable,
        str(SCRIPT_DIR / "export_patch_dataset.py"),
        "--family-manifest",
        str(manifest_path),
        "--output-root",
        str(output_root),
        "--splits",
        *config.splits,
        "--band-indices",
        *[str(index) for index in config.band_indices],
        "--mask-threshold",
        str(config.mask_threshold),
        "--resample-step-px",
        str(config.box_resample_step_px),
        "--boundary-tol-px",
        str(config.box_boundary_tol_px),
        "--max-families-per-split",
        str(config.max_families_per_split),
        "--empty-patch-drop-ratio",
        str(config.empty_box_drop_ratio),
        "--empty-patch-seed",
        str(config.empty_box_seed),
    ]
    if config.box_user_prompt_file:
        box_dataset_cmd.extend(["--user-prompt-file", config.box_user_prompt_file])

    run_command(manifest_cmd, "build-manifest")
    run_command(box_dataset_cmd, "export-road-centerline-datasets")

    log("all steps completed")


if __name__ == "__main__":
    main()