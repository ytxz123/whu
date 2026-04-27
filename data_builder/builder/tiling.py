from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PatchWindow:
    patch_index: int
    row: int
    col: int
    x0: int
    y0: int
    x1: int
    y1: int


def generate_window_starts(total_size: int, window_size: int) -> list[int]:
    total_size = max(0, int(total_size))
    window_size = max(1, int(window_size))
    if total_size == 0:
        return []
    if total_size <= window_size:
        return [0]
    starts = list(range(0, total_size - window_size + 1, window_size))
    last_start = total_size - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def generate_patch_windows(width: int, height: int, patch_size: int) -> list[PatchWindow]:
    patch_size = max(1, int(patch_size))
    windows: list[PatchWindow] = []
    patch_index = 0
    y_starts = generate_window_starts(height, patch_size)
    x_starts = generate_window_starts(width, patch_size)
    for row_index, y0 in enumerate(y_starts):
        y1 = y0 + patch_size
        for col_index, x0 in enumerate(x_starts):
            x1 = x0 + patch_size
            windows.append(PatchWindow(patch_index, row_index, col_index, x0, y0, x1, y1))
            patch_index += 1
    return windows


def clip_segment_to_rect(p0: np.ndarray, p1: np.ndarray, rect: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray] | None:
    x_min, y_min, x_max, y_max = rect
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    p_vals = [-dx, dx, -dy, dy]
    q_vals = [float(p0[0] - x_min), float(x_max - p0[0]), float(p0[1] - y_min), float(y_max - p0[1])]
    low = 0.0
    high = 1.0
    for p_val, q_val in zip(p_vals, q_vals):
        if abs(p_val) < 1e-8:
            if q_val < 0:
                return None
            continue
        ratio = q_val / p_val
        if p_val < 0:
            if ratio > high:
                return None
            low = max(low, ratio)
        else:
            if ratio < low:
                return None
            high = min(high, ratio)
    start = np.asarray([p0[0] + low * dx, p0[1] + low * dy], dtype=np.float32)
    end = np.asarray([p0[0] + high * dx, p0[1] + high * dy], dtype=np.float32)
    return start, end


def dedup_points(points: Iterable[np.ndarray], eps: float = 1e-3) -> np.ndarray:
    arr = np.asarray(list(points), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    out = [arr[0]]
    for idx in range(1, arr.shape[0]):
        if float(np.linalg.norm(arr[idx] - out[-1])) > eps:
            out.append(arr[idx])
    return np.asarray(out, dtype=np.float32)


def point_to_segment_distances(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    seg_len_sq = float(np.dot(segment, segment))
    if seg_len_sq <= 1e-12:
        return np.linalg.norm(points - start[None, :], axis=1)
    rel = points - start[None, :]
    t = np.clip((rel @ segment) / seg_len_sq, 0.0, 1.0)
    projection = start[None, :] + t[:, None] * segment[None, :]
    return np.linalg.norm(points - projection, axis=1)


def simplify_polyline(points: np.ndarray, epsilon: float) -> np.ndarray:
    points = dedup_points(np.asarray(points, dtype=np.float32))
    if points.shape[0] < 3 or epsilon <= 0.0:
        return points

    keep = np.zeros(points.shape[0], dtype=bool)
    keep[0] = True
    keep[-1] = True
    stack: list[tuple[int, int]] = [(0, points.shape[0] - 1)]

    while stack:
        start_idx, end_idx = stack.pop()
        if end_idx - start_idx < 2:
            continue
        middle = points[start_idx + 1 : end_idx]
        distances = point_to_segment_distances(middle, points[start_idx], points[end_idx])
        max_offset = int(np.argmax(distances))
        max_distance = float(distances[max_offset])
        if max_distance > epsilon:
            split_idx = start_idx + 1 + max_offset
            keep[split_idx] = True
            stack.append((start_idx, split_idx))
            stack.append((split_idx, end_idx))

    return points[keep]


def clip_polyline_to_rect(points: np.ndarray, rect: tuple[float, float, float, float]) -> list[np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 2:
        return []
    pieces: list[np.ndarray] = []
    current: list[np.ndarray] = []
    for idx in range(points.shape[0] - 1):
        clipped = clip_segment_to_rect(points[idx], points[idx + 1], rect)
        if clipped is None:
            if len(current) >= 2:
                pieces.append(dedup_points(current))
            current = []
            continue
        start, end = clipped
        if not current:
            current = [start, end]
        elif float(np.linalg.norm(current[-1] - start)) <= 1e-3:
            current.append(end)
        else:
            if len(current) >= 2:
                pieces.append(dedup_points(current))
            current = [start, end]
        x, y = float(points[idx + 1][0]), float(points[idx + 1][1])
        x_min, y_min, x_max, y_max = rect
        inside = x_min <= x <= x_max and y_min <= y <= y_max
        if not inside:
            if len(current) >= 2:
                pieces.append(dedup_points(current))
            current = []
    if len(current) >= 2:
        pieces.append(dedup_points(current))
    return [piece for piece in pieces if piece.shape[0] >= 2]


def localize_and_quantize(lines_global: list[np.ndarray], window: PatchWindow, simplify_tolerance: float = 0.0) -> list[dict]:
    out = []
    width = max(1, window.x1 - window.x0)
    height = max(1, window.y1 - window.y0)
    for line in lines_global:
        local = line - np.asarray([[float(window.x0), float(window.y0)]], dtype=np.float32)
        local[:, 0] = np.clip(local[:, 0], 0.0, float(width - 1))
        local[:, 1] = np.clip(local[:, 1], 0.0, float(height - 1))
        local = simplify_polyline(local, float(simplify_tolerance))
        rounded = np.rint(local).astype(np.int32)
        rounded = dedup_points(rounded.astype(np.float32)).astype(np.int32)
        if rounded.shape[0] >= 2:
            out.append({"points": [[int(x), int(y)] for x, y in rounded.tolist()]})
    return out
