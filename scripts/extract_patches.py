"""Extract native-resolution lesion/background patches and their index (patches_v1)."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def main() -> int:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import numpy as np

    from cbis_ddsm.data.manifest import read_manifest_csv
    from cbis_ddsm.data.patches import extract_image_patches, write_patch_index
    from cbis_ddsm.data.splits import partition_of, read_split_json
    from cbis_ddsm.data.targets import group_images, read_excluded_identities, select_masses

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("data/interim/manifest.csv"))
    parser.add_argument("--exclusions", type=Path, default=Path("data/interim/exclusions.csv"))
    parser.add_argument("--split", type=Path, default=Path("data/splits/split_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/patches_v1"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--background-per-image", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-patients", type=int, default=None)
    parser.add_argument("--partitions", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    args = parser.parse_args()

    raw = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    # Validate destinations before any work: raw data is immutable.
    if output_dir.is_relative_to(raw):
        parser.error(f"--output-dir must be outside the raw directory {raw}")

    try:
        rows = read_manifest_csv(args.manifest, raw)
        masses = select_masses(rows)
        excluded = read_excluded_identities(args.exclusions)
        records, _dropped = group_images(masses, excluded)
        split = read_split_json(args.split)
        partition_map = partition_of(split)

        requested = set(args.partitions)
        patients_by_partition: dict[str, set[str]] = {}
        for patient_id, partition in partition_map.items():
            if partition in requested:
                patients_by_partition.setdefault(partition, set()).add(patient_id)
        allowed_patients: set[str] = set()
        for partition, patients in patients_by_partition.items():
            ordered = sorted(patients)
            if args.limit_patients is not None:
                ordered = ordered[: args.limit_patients]
            allowed_patients.update(ordered)

        selected = sorted(
            (record for record in records if partition_map.get(record.patient_id) in requested and record.patient_id in allowed_patients),
            key=lambda record: record.image_key,
        )

        images_dir = output_dir / "images"
        masks_dir = output_dir / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(args.seed)
        all_records = []
        shortfall_total = 0
        for record in selected:
            patch_records, shortfall = extract_image_patches(
                record, partition=partition_map[record.patient_id], size=args.size,
                background_per_image=args.background_per_image, rng=rng,
                images_dir=images_dir, masks_dir=masks_dir,
            )
            all_records.extend(patch_records)
            shortfall_total += shortfall

        write_patch_index(all_records, output_dir / "index.csv")
    except (OSError, ValueError) as error:
        parser.error(str(error))

    counts: Counter[tuple[str, str]] = Counter((record.partition, record.kind) for record in all_records)
    summary = ", ".join(f"{partition}/{kind}={count}" for (partition, kind), count in sorted(counts.items()))
    print(f"wrote {len(all_records)} patches to {output_dir}: {summary}")
    if shortfall_total:
        print(f"warning: {shortfall_total} requested background windows were not found (fewer background patches than requested)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
