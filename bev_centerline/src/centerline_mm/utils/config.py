from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data["_config_path"] = str(path.resolve())
    data["_config_dir"] = str(path.resolve().parent)
    return data


def get_by_path(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


@dataclass
class Paths:
    project_root: Path
    qwen_path: Path
    dinov3_repo: Path
    output_dir: Path


def resolve_paths(cfg: dict[str, Any]) -> Paths:
    config_dir = Path(cfg.get("_config_dir", ".")).resolve()
    raw_root = Path(cfg.get("project_root", config_dir.parent))
    project_root = raw_root if raw_root.is_absolute() else (config_dir / raw_root).resolve()

    def resolve(value: str | None, fallback: str) -> Path:
        raw = value or fallback
        path = Path(raw)
        if not path.is_absolute():
            path = (project_root / path).resolve()
        return path

    return Paths(
        project_root=project_root,
        qwen_path=resolve(get_by_path(cfg, "paths.qwen_path"), "../Qwen3.5_2B"),
        dinov3_repo=resolve(get_by_path(cfg, "paths.dinov3_repo"), "../dinov3"),
        output_dir=resolve(get_by_path(cfg, "paths.output_dir"), "outputs"),
    )
