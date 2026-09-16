"""Overfit a fixed, unaugmented batch of patches: a gate against misaligned image/mask pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def _parse_overrides(pairs: list[str]) -> dict[str, object]:
    """Parse `--set key=value` pairs; each value is decoded with `yaml.safe_load`."""
    overrides: dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--set expects key=value, got: {pair}")
        key, _, value = pair.partition("=")
        overrides[key] = yaml.safe_load(value)
    return overrides


def main() -> int:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from cbis_ddsm.training.config import load_config
    from cbis_ddsm.training.trainer import overfit_batch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/unet_smoke.yaml"))
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--target-dice", type=float, default=0.95)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="key=value")
    args = parser.parse_args()

    if not args.config.is_file():
        parser.error(f"missing config file: {args.config}")

    try:
        overrides = _parse_overrides(args.overrides)
        cfg = load_config(args.config, overrides)
    except ValueError as error:
        parser.error(str(error))

    dice = overfit_batch(cfg, n=args.n, steps=args.steps)
    print(f"overfit gate: dice={dice:.4f} (target={args.target_dice:.4f}, n={args.n}, steps={args.steps})")
    if dice < args.target_dice:
        print(f"overfit gate failed: dice={dice:.4f} < target={args.target_dice:.4f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
