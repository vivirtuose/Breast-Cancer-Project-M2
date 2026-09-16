"""Build the deterministic CBIS-DDSM lesion manifest."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from cbis_ddsm.data.manifest import (
        ManifestResolutionError,
        build_manifest,
        write_acquisition_exclusions_csv,
        write_manifest_csv,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/interim/manifest.csv"))
    parser.add_argument("--acquisition-exclusions", type=Path, default=Path("data/interim/acquisition_exclusions.csv"))
    args = parser.parse_args()
    raw = args.raw_dir.resolve()
    outputs = (args.output.resolve(), args.acquisition_exclusions.resolve())
    # Validate destinations before any work: raw data is immutable and each output needs its own file.
    if any(output.is_relative_to(raw) for output in outputs):
        parser.error(f"outputs must be written outside the raw directory {raw}")
    if outputs[0] == outputs[1]:
        parser.error("--output and --acquisition-exclusions must be different files")
    try:
        result = build_manifest(raw)
        write_manifest_csv(result, args.output)
        write_acquisition_exclusions_csv(result.acquisition_exclusions, args.acquisition_exclusions)
    except (ManifestResolutionError, OSError, ValueError) as error:
        parser.error(str(error))
    print(
        f"wrote {len(result)} manifest rows to {args.output}; "
        f"{len(result.acquisition_exclusions)} acquisition exclusions to {args.acquisition_exclusions}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
