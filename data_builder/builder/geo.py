from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import CRS, Transformer
from rasterio import open as rasterio_open
from rasterio.transform import Affine

from .io_utils import read_json


@dataclass(frozen=True)
class RasterMeta:
    width: int
    height: int
    crs: str
    transform: tuple[float, float, float, float, float, float]

    @property
    def affine(self) -> Affine:
        return Affine(*self.transform)


def read_raster_rgb(path: Path) -> tuple[np.ndarray, RasterMeta]:
    with rasterio_open(path) as ds:
        channels = [ds.read(index) for index in (1, 2, 3)]
        meta = RasterMeta(
            width=int(ds.width),
            height=int(ds.height),
            crs=str(ds.crs) if ds.crs is not None else "",
            transform=tuple(list(ds.transform)[:6]),
        )
    image = np.stack(channels, axis=-1)
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image, meta


def read_mask(path: Path, threshold: int) -> np.ndarray:
    with rasterio_open(path) as ds:
        mask = ds.read(1)
    return (mask > int(threshold)).astype(np.uint8)


def detect_geojson_crs(geojson_dict: dict[str, Any]) -> str:
    crs_value = geojson_dict.get("crs")
    if not isinstance(crs_value, dict):
        return "OGC:CRS84"
    props = crs_value.get("properties")
    if not isinstance(props, dict):
        return "OGC:CRS84"
    name = str(props.get("name", "")).strip()
    return name or "OGC:CRS84"


def dedup_points(points: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    if points.ndim != 2 or points.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    out = [points[0]]
    for idx in range(1, points.shape[0]):
        if float(np.linalg.norm(points[idx] - out[-1])) > eps:
            out.append(points[idx])
    return np.asarray(out, dtype=np.float32)


def world_to_pixel(points_world: np.ndarray, affine: Affine) -> np.ndarray:
    inverse = ~affine
    out = []
    for x, y in points_world.tolist():
        px, py = inverse * (float(x), float(y))
        out.append([float(px), float(py)])
    return np.asarray(out, dtype=np.float32)


def load_lane_lines(lane_path: Path, raster_meta: RasterMeta) -> list[np.ndarray]:
    if not lane_path.is_file():
        return []
    geojson_dict = read_json(lane_path)
    source_crs = detect_geojson_crs(geojson_dict)
    transformer = Transformer.from_crs(CRS.from_user_input(source_crs), CRS.from_user_input(raster_meta.crs), always_xy=True)
    lines: list[np.ndarray] = []
    for feature in geojson_dict.get("features", []):
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry", {})
        if str(geometry.get("type", "")).strip().lower() != "linestring":
            continue
        coords = np.asarray(geometry.get("coordinates", []), dtype=np.float64)
        if coords.ndim != 2 or coords.shape[0] < 2 or coords.shape[1] < 2:
            continue
        xs, ys = transformer.transform(coords[:, 0], coords[:, 1])
        world = np.stack([xs, ys], axis=1).astype(np.float32)
        pixels = dedup_points(world_to_pixel(world, raster_meta.affine))
        if pixels.shape[0] >= 2:
            lines.append(pixels)
    return lines

