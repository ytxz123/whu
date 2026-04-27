from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "data_builder" / "configs" / "visualize.yaml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_config_path(config_arg: str | None) -> Path:
    if config_arg is None or not str(config_arg).strip():
        return DEFAULT_CONFIG_PATH.resolve()
    return Path(config_arg).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize dataset JSONL annotations on patch images.")
    parser.add_argument(
        "--config",
        default=None,
        help=f"Optional config path. If omitted, automatically uses {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--line-width", type=int, default=None)
    parser.add_argument("--point-radius", type=int, default=None)
    parser.add_argument("--point-outline-width", type=int, default=None)
    parser.add_argument("--panel-gap", type=int, default=None)
    parser.add_argument("--panel-title-height", type=int, default=None)
    parser.add_argument("--show-point-index", default=None)
    return parser.parse_args()


def parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    overrides = {
        "dataset_root": args.dataset_root,
        "output_root": args.output_root,
        "max_samples_per_split": args.max_samples_per_split,
        "line_width": args.line_width,
        "point_radius": args.point_radius,
        "point_outline_width": args.point_outline_width,
        "panel_gap": args.panel_gap,
        "panel_title_height": args.panel_title_height,
        "show_point_index": parse_bool(args.show_point_index),
    }
    from data_builder.builder.visualize import run

    run(config_path, overrides=overrides)


if __name__ == "__main__":
    main()
