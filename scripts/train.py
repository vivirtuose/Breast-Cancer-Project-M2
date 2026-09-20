"""Train a segmentation model from a YAML config, with optional key=value overrides."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from cbis_ddsm.training.config import load_config, parse_overrides
    from cbis_ddsm.training.trainer import train

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/unet_smoke.yaml"))
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="key=value")
    args = parser.parse_args()

    if not args.config.is_file():
        parser.error(f"missing config file: {args.config}")

    try:
        overrides = parse_overrides(args.overrides)
        cfg = load_config(args.config, overrides)
    except ValueError as error:
        parser.error(str(error))

    result = train(cfg)
    print(
        f"wrote run to {result.run_dir}: best_epoch={result.best_epoch} "
        f"best_val_dice={result.best_val_dice:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
