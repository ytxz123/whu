from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from centerline_mm.data.sharegpt import extract_assistant_payload, extract_image_path, extract_prompt
from centerline_mm.utils.config import get_by_path, load_yaml, resolve_paths
from centerline_mm.utils.geometry_text import to_coordinate_text
from centerline_mm.utils.json_format import dumps_strict_centerline_json

STAGE3_PROMPT = (
    "You are given a 512x512 black-background BEV road-structure image. "
    "Recover every valid road centerline. Output only strict JSON with keys "
    "role, content, lines, category, points. Coordinates must be integers in [0,512]."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def ok(text: str) -> None:
    print(f"[OK] {text}")


def warn(text: str) -> None:
    print(f"[WARN] {text}")


def fail(text: str) -> None:
    print(f"[FAIL] {text}")


def check_file(path: Path, label: str, required: bool = True) -> bool:
    if path.is_file():
        ok(f"{label}: {path}")
        return True
    if required:
        fail(f"{label} not found: {path}")
    else:
        warn(f"{label} not found: {path}")
    return False


def check_dir(path: Path, label: str, required: bool = True) -> bool:
    if path.is_dir():
        ok(f"{label}: {path}")
        return True
    if required:
        fail(f"{label} not found: {path}")
    else:
        warn(f"{label} not found: {path}")
    return False


def check_jsonl(path: Path, kind: str, max_rows: int = 3) -> bool:
    if not check_file(path, f"{kind} manifest"):
        return False
    rows = read_jsonl(path)
    if not rows:
        fail(f"{kind} manifest is empty: {path}")
        return False
    ok(f"{kind} manifest rows: {len(rows)}")
    root = path.parent
    good = True
    for index, row in enumerate(rows[:max_rows], 1):
        try:
            if kind == "seg":
                image = root / str(row["image"])
                mask = root / str(row["mask"])
                good &= check_file(image if image.is_absolute() else image.resolve(), f"seg sample {index} image")
                good &= check_file(mask if mask.is_absolute() else mask.resolve(), f"seg sample {index} mask")
            else:
                image_path = Path(extract_image_path(row, root))
                payload: Any = extract_assistant_payload(row)
                prompt = extract_prompt(row, STAGE3_PROMPT)
                strict = dumps_strict_centerline_json(payload)
                coord = to_coordinate_text(payload)
                good &= check_file(image_path, f"line sample {index} image")
                ok(f"line sample {index} prompt chars={len(prompt)} strict_json_chars={len(strict)} coord='{coord[:80]}'")
        except Exception as exc:
            fail(f"{kind} sample {index} invalid: {exc}")
            good = False
    return good


def check_qwen_model_dir(path: Path) -> bool:
    good = True
    good &= check_file(path / "config.json", "Qwen config")
    good &= check_file(path / "tokenizer.json", "Qwen tokenizer")
    good &= check_file(path / "preprocessor_config.json", "Qwen processor config")
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        ok(f"Qwen safetensors index: {index_path}")
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            shard_names = sorted(set(str(v) for v in payload.get("weight_map", {}).values()))
            if not shard_names:
                fail(f"Qwen safetensors index has no weight_map entries: {index_path}")
                return False
            for shard in shard_names:
                good &= check_file(path / shard, f"Qwen weight shard {shard}")
        except Exception as exc:
            fail(f"Cannot parse Qwen safetensors index {index_path}: {exc}")
            good = False
    else:
        shards = sorted(path.glob("*.safetensors"))
        if shards:
            ok(f"Qwen safetensors shards found: {len(shards)}")
        else:
            fail(f"No Qwen safetensors index or shard files found in: {path}")
            good = False
    return good


def main() -> None:
    parser = argparse.ArgumentParser(description="Check bev_centerline paths, weights, and dataset_builder manifests.")
    parser.add_argument("--stage1-config", default="configs/stage1_segmentation.yaml")
    parser.add_argument("--stage2-config", default="configs/stage2_alignment.yaml")
    parser.add_argument("--stage3-config", default="configs/stage3_json_lora.yaml")
    args = parser.parse_args()

    cfg1 = load_yaml(args.stage1_config)
    cfg2 = load_yaml(args.stage2_config)
    cfg3 = load_yaml(args.stage3_config)
    paths = resolve_paths(cfg1)

    all_good = True
    all_good &= check_dir(paths.project_root, "project_root")
    all_good &= check_dir(paths.dinov3_repo, "DINOv3 repo")
    all_good &= check_dir(paths.qwen_path, "Qwen3.5 model dir")
    all_good &= check_qwen_model_dir(paths.qwen_path)
    all_good &= check_file(paths.dinov3_repo / "hubconf.py", "DINOv3 hubconf.py")

    dino_weights = Path(str(get_by_path(cfg1, "model.dinov3_weights", "")))
    if dino_weights and str(dino_weights) not in ("", "None", "null"):
        if not dino_weights.is_absolute():
            dino_weights = paths.project_root / dino_weights
        all_good &= check_file(dino_weights, "DINOv3 backbone weights")

    seg_train = paths.project_root / get_by_path(cfg1, "data.train_manifest")
    seg_val = paths.project_root / get_by_path(cfg1, "data.val_manifest")
    line_train2 = paths.project_root / get_by_path(cfg2, "data.train_manifest")
    line_train3 = paths.project_root / get_by_path(cfg3, "data.train_manifest")
    all_good &= check_jsonl(seg_train, "seg")
    all_good &= check_jsonl(seg_val, "seg")
    all_good &= check_jsonl(line_train2, "line")
    if line_train3 != line_train2:
        all_good &= check_jsonl(line_train3, "line")

    if all_good:
        ok("Setup check passed.")
    else:
        raise SystemExit("Setup check failed. Fix the [FAIL] items above.")


if __name__ == "__main__":
    main()
