"""Build patient-grouped stratified train/val/test partitions (split_v1.json)."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from cbis_ddsm.data.manifest import read_manifest_csv
    from cbis_ddsm.data.splits import build_split, patient_labels, sha256_file, write_split_json
    from cbis_ddsm.data.targets import group_images, read_excluded_identities, select_masses

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("data/interim/manifest.csv"))
    parser.add_argument("--exclusions", type=Path, default=Path("data/interim/exclusions.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/splits/split_v1.json"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    raw = args.raw_dir.resolve()
    output = args.output.resolve()
    # Validate destinations before any work: raw data is immutable, and a missing audit is a clear user error.
    if output.is_relative_to(raw):
        parser.error(f"--output must be outside the raw directory {raw}")
    if not args.exclusions.is_file():
        parser.error(
            f"missing exclusions file: {args.exclusions}; run scripts/audit_data.py first to produce it"
        )

    try:
        rows = read_manifest_csv(args.manifest, raw)
        masses = select_masses(rows)
        excluded = read_excluded_identities(args.exclusions)
        records, _dropped = group_images(masses, excluded)
        manifest_sha256 = sha256_file(args.manifest)
        split = build_split(
            records, n_folds=args.n_folds, val_fold=args.val_fold, seed=args.seed, manifest_sha256=manifest_sha256,
        )
        write_split_json(split, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    labels = patient_labels(records)
    partitions = split["partitions"]

    def malignant_share(patients: list[str]) -> float:
        return sum(labels.get(patient, 0) for patient in patients) / len(patients) if patients else 0.0

    print(
        f"wrote split to {args.output}: "
        f"train={len(partitions['train'])} val={len(partitions['val'])} test={len(partitions['test'])} patients; "
        f"malignant share train={malignant_share(partitions['train']):.3f} val={malignant_share(partitions['val']):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
