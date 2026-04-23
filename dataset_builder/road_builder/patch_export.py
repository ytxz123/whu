"""RC 指定框大小数据集导出 helper。"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .geometry_utils import canonicalize_line_direction, clamp_points_float_rect, clip_polyline_to_rect, dedup_points, point_boundary_side, resample_polyline_keep_tail, simplify_for_json, sort_lines
from .io_utils import make_sharegpt_record


ROAD_CENTERLINE_CATEGORY = "road_centerline"

BOX_PROMPT_TEMPLATE = """<image>
Task: reconstruct all visible road centerlines inside box [{box_x_min},{box_y_min},{box_x_max},{box_y_max}].
Return strict JSON only with top-level key lines."""


def format_box_prompt(prompt_fields: Dict[str, int], prompt_template: str = "") -> str:
    template = str(prompt_template).strip() or BOX_PROMPT_TEMPLATE
    try:
        return template.format(**prompt_fields)
    except KeyError as exc:
        missing_key = str(exc).strip("'\"")
        raise ValueError(f"Box prompt template contains an unknown placeholder: {missing_key}") from exc


def _line_piece_cut_flags_after_clip(source_points: np.ndarray, clipped_points: np.ndarray) -> Tuple[bool, bool]:
    source = np.asarray(source_points, dtype=np.float32)
    clipped = np.asarray(clipped_points, dtype=np.float32)
    if source.ndim != 2 or clipped.ndim != 2 or source.shape[0] == 0 or clipped.shape[0] == 0:
        return False, False
    tol = 1e-3
    return (not np.allclose(clipped[0], source[0], atol=tol), not np.allclose(clipped[-1], source[-1], atol=tol))


def build_patch_segments_global(global_features: Sequence[Dict], rect_global: Tuple[float, float, float, float], resample_step_px: float, boundary_tol_px: float) -> List[Dict]:
    output: List[Dict] = []
    for feature in global_features:
        points_global = np.asarray(feature.get("points_global", []), dtype=np.float32)
        for clipped_piece in clip_polyline_to_rect(points_global, rect_global):
            piece = np.asarray(clipped_piece, dtype=np.float32)
            if piece.ndim != 2 or piece.shape[0] < 2:
                continue
            cut_start, cut_end = _line_piece_cut_flags_after_clip(points_global, piece)
            if float(resample_step_px) > 0.0:
                piece = resample_polyline_keep_tail(piece, step_px=float(resample_step_px))
            if piece.ndim != 2 or piece.shape[0] < 2:
                continue
            start_side = point_boundary_side(piece[0], rect_global, tol_px=float(boundary_tol_px))
            end_side = point_boundary_side(piece[-1], rect_global, tol_px=float(boundary_tol_px))
            start_type = "cut" if cut_start or start_side is not None else "start"
            end_type = "cut" if cut_end or end_side is not None else "end"
            piece, start_type, end_type = canonicalize_line_direction(piece, start_type=start_type, end_type=end_type)
            piece = dedup_points(piece)
            if piece.ndim != 2 or piece.shape[0] < 2:
                continue
            output.append({"category": ROAD_CENTERLINE_CATEGORY, "points_global": piece.astype(np.float32)})
    return sort_lines(output)


def build_patch_target_lines(segments_global: Sequence[Dict], patch: Dict, quantize: bool = True) -> List[Dict]:
    crop_box = patch["crop_box"]
    patch_width = float(crop_box["x_max"] - crop_box["x_min"])
    patch_height = float(crop_box["y_max"] - crop_box["y_min"])
    patch_size = max(1, int(round(max(patch_width, patch_height))))
    offset = np.asarray([float(crop_box["x_min"]), float(crop_box["y_min"])], dtype=np.float32)[None, :]
    output: List[Dict] = []
    for segment in segments_global:
        local = np.asarray(segment.get("points_global", []), dtype=np.float32) - offset
        local = clamp_points_float_rect(local, patch_width=patch_width, patch_height=patch_height)
        local = dedup_points(local)
        if local.ndim != 2 or local.shape[0] < 2:
            continue
        points = simplify_for_json(local, patch_size=patch_size) if quantize else [[float(x), float(y)] for x, y in local.tolist()]
        output.append({"category": ROAD_CENTERLINE_CATEGORY, "points": points})
    return sort_lines(output)


def make_box_record(*, sample_id: str, image_rel_path: str, target_lines: Sequence[Dict], prompt_text: str) -> Dict:
    return make_sharegpt_record(
        sample_id=sample_id,
        image_rel_path=image_rel_path,
        user_text=prompt_text,
        assistant_payload={"lines": list(target_lines)},
    )
