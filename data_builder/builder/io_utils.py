from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def sanitize_name(text: str) -> str:
    chars = []
    for ch in str(text):
        chars.append(ch if ch.isalnum() or ch in ("_", "-") else "_")
    return "".join(chars).strip("_") or "sample"

