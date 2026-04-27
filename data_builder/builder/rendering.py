from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class AnnotationStyle:
    line_width: int = 3
    point_radius: int = 4
    point_outline_width: int = 2
    show_point_index: bool = False
    draw_points: bool = True
    fixed_line_color: tuple[int, int, int] | None = None


def color_for_index(index: int) -> tuple[int, int, int]:
    palette = [
        (245, 99, 99),
        (66, 165, 245),
        (102, 187, 106),
        (255, 202, 40),
        (171, 71, 188),
        (38, 166, 154),
        (255, 112, 67),
        (124, 179, 66),
    ]
    return palette[index % len(palette)]


def draw_point(draw: ImageDraw.ImageDraw, x: int, y: int, radius: int, outline_width: int, color: tuple[int, int, int]) -> None:
    bounds = (x - radius, y - radius, x + radius, y + radius)
    draw.ellipse(bounds, fill=(255, 255, 255), outline=color, width=outline_width)


def draw_annotations(image: Image.Image, lines: list[dict], style: AnnotationStyle) -> Image.Image:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    for line_index, line in enumerate(lines):
        points = line.get("points", [])
        if not isinstance(points, list) or len(points) < 2:
            continue
        xy = [(int(pt[0]), int(pt[1])) for pt in points if isinstance(pt, list) and len(pt) >= 2]
        if len(xy) < 2:
            continue
        color = style.fixed_line_color if style.fixed_line_color is not None else color_for_index(line_index)
        draw.line(xy, fill=color, width=style.line_width, joint="curve")
        if style.draw_points:
            for point_index, (x, y) in enumerate(xy):
                draw_point(draw, x, y, style.point_radius, style.point_outline_width, color)
                if style.show_point_index:
                    draw.text((x + style.point_radius + 2, y - style.point_radius - 2), str(point_index), fill=color, font=font)
    return overlay


def render_label_image(width: int, height: int, lines: list[dict], style: AnnotationStyle, background_color: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    canvas = Image.new("RGB", (max(1, int(width)), max(1, int(height))), color=background_color)
    return draw_annotations(canvas, lines, style)
