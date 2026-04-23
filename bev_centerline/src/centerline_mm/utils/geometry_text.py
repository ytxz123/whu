from __future__ import annotations

from typing import Any

from .json_format import clamp_int, normalize_centerline_json


def to_coordinate_text(value: Any) -> str:
    obj = normalize_centerline_json(value)
    parts = []
    for line in obj["content"]["lines"]:
        pts = [f"{clamp_int(x)},{clamp_int(y)}" for x, y in line["points"]]
        if len(pts) >= 2:
            parts.append("LINE " + " ".join(pts))
    return "\n".join(parts) if parts else "NO_LINES"

