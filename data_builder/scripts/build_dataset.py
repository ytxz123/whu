from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "data_builder" / "configs" / "build.yaml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_config_path(config_arg: str | None) -> Path:
    if config_arg is None or not str(config_arg).strip():
        return DEFAULT_CONFIG_PATH.resolve()
    return Path(config_arg).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a minimal patch-level line dataset from RC raw data.")
    parser.add_argument(
        "--config",
        default=None,
        help=f"Optional config path. If omitted, automatically uses {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--train-root", default=None)
    parser.add_argument("--val-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--empty-annotation-keep-ratio", type=float, default=None)
    parser.add_argument("--empty-annotation-seed", type=int, default=None)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--simplify-tolerance", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    overrides = {
        key: getattr(args, key)
        for key in (
            "dataset_root",
            "train_root",
            "val_root",
            "output_root",
            "empty_annotation_keep_ratio",
            "empty_annotation_seed",
            "max_samples_per_split",
            "simplify_tolerance",
        )
    }
    from data_builder.builder.build import run

    run(config_path, overrides=overrides)


if __name__ == "__main__":
    main()
