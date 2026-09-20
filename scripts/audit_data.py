"""Audit a CBIS-DDSM manifest and write one exclusion reason per rejected row."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from cbis_ddsm.data.audit import audit_manifest, write_exclusions_csv
    from cbis_ddsm.data.manifest import read_manifest_csv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("data/interim/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/interim/exclusions.csv"))
    parser.add_argument(
        "--lesion-type",
        choices=("mass", "calc", "all"),
        default="all",
        help="restrict the audit to one lesion type before auditing (default: all)",
    )
    args = parser.parse_args()
    if args.output.resolve() == args.manifest.resolve():
        parser.error("--output must not overwrite the manifest it audits")
    try:
        # Manifest paths are raw-relative, so they are rebased on the raw directory given here.
        rows = read_manifest_csv(args.manifest, args.raw_dir)
        if args.lesion_type != "all":
            rows = tuple(row for row in rows if row.lesion_type == args.lesion_type)
        result = audit_manifest(rows)
        write_exclusions_csv(result.exclusions, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"wrote {len(result.exclusions)} exclusions; retained {len(result.included_rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
