"""Patch inference: score a trained run's checkpoint, writing metrics, a summary and figures."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from cbis_ddsm.inference.predict import evaluate_patches

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--patch-root", type=Path, required=True)
    parser.add_argument("--partition", default="val")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-figures", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    checkpoint = args.run_dir / "checkpoint_best.pt"
    if not checkpoint.is_file():
        parser.error(f"missing checkpoint: {checkpoint}")
    if not args.patch_root.is_dir():
        parser.error(f"missing patch root: {args.patch_root}")

    summary_path = evaluate_patches(
        args.run_dir,
        args.patch_root,
        partition=args.partition,
        threshold=args.threshold,
        limit=args.limit,
        n_figures=args.n_figures,
        device=args.device,
    )
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
