from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BuildPaths:
    project_root: Path
    dataset_root: Path | None
    train_root: Path | None
    val_root: Path | None
    output_root: Path


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    data["_config_path"] = str(path)
    data["_config_dir"] = str(path.parent)
    return data


def resolve_optional_path(config_dir: Path, value: str | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    return path


def build_paths(cfg: dict[str, Any]) -> BuildPaths:
    config_dir = Path(cfg["_config_dir"]).resolve()
    project_root = config_dir.parent
    return BuildPaths(
        project_root=project_root,
        dataset_root=resolve_optional_path(config_dir, cfg.get("dataset_root")),
        train_root=resolve_optional_path(config_dir, cfg.get("train_root")),
        val_root=resolve_optional_path(config_dir, cfg.get("val_root")),
        output_root=resolve_optional_path(config_dir, cfg.get("output_root")) or (project_root / "data_0427").resolve(),
    )

