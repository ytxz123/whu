from __future__ import annotations

import json
from typing import Any


def clamp_int(value: Any, low: int = 0, high: int = 512) -> int:
    try:
        ivalue = int(round(float(value)))
    except (TypeError, ValueError):
        ivalue = low
    return max(low, min(high, ivalue))


def normalize_centerline_json(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)

    lines_in = []
    if isinstance(value, dict):
        if "content" in value and isinstance(value["content"], dict):
            lines_in = value["content"].get("lines", [])
        else:
            lines_in = value.get("lines", [])

    lines_out = []
    for line in lines_in or []:
        pts = line.get("points", []) if isinstance(line, dict) else []
        clean_pts = []
        for pt in pts:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                clean_pts.append([clamp_int(pt[0]), clamp_int(pt[1])])
        if len(clean_pts) >= 2:
            lines_out.append({"category": "road_centerline", "points": clean_pts})

    return {"role": "assistant", "content": {"lines": lines_out}}


def dumps_strict_centerline_json(value: Any) -> str:
    normalized = normalize_centerline_json(value)
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def empty_result() -> str:
    return dumps_strict_centerline_json({"lines": []})

