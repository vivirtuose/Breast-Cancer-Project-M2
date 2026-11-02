"""Kaggle smoke entry point: chain every pipeline stage on a small subset, end to end.

Runs, in order: capture_environment, build_manifest, audit (masses only), make_splits,
extract_patches, overfit_batch (recorded, never fatal), train, predict. Each stage is invoked
as a subprocess of the current interpreter (`sys.executable`), exactly as a human would type it;
its stdout/stderr are captured to `<work-dir>/logs/<stage>.log`, and a one-line summary is
printed. A `smoke_report.json` and a `smoke_run.zip` are written under `<work-dir>` when the run
finishes. Exit code 0 requires capture_environment, build_manifest, audit, make_splits,
extract_patches, train, and predict to all succeed; the overfit gate may fail without affecting
the exit code.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
DEFAULT_TRAIN_CONFIG = ROOT / "configs" / "unet_smoke.yaml"
MAX_ZIP_FILE_BYTES = 200 * 1024 * 1024

# Stages required for a successful smoke run (exit code 0); "overfit" is deliberately excluded.
REQUIRED_STAGE_NAMES = (
    "capture_environment", "build_manifest", "audit", "make_splits",
    "extract_patches", "train", "predict",
)


@dataclass
class StageOutcome:
    name: str
    status: str  # "ok" | "failed" | "skipped"
    seconds: float
    detail: str
    stdout: str = ""

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "status": self.status, "seconds": round(self.seconds, 3), "detail": self.detail}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _summarize(stdout: str, stderr: str, pattern: str | None) -> str:
    """The first stdout line matching `pattern`, else the last non-empty line of the log."""
    if pattern is not None:
        match = re.search(pattern, stdout, re.MULTILINE)
        if match:
            return match.group(0).strip()
    for text in (stdout, stderr):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            return lines[-1]
    return ""


def _run_stage(name: str, command: list[str], log_path: Path, pattern: str | None = None) -> StageOutcome:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    elapsed = time.monotonic() - start
    log_path.write_text(
        "$ " + " ".join(str(part) for part in command) + "\n\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}\n",
        encoding="utf-8",
    )
    status = "ok" if completed.returncode == 0 else "failed"
    detail = _summarize(completed.stdout, completed.stderr, pattern)
    print(f"[{name}] {status} ({elapsed:.1f}s): {detail}")
    return StageOutcome(name, status, elapsed, detail, stdout=completed.stdout)


def _skipped(name: str, reason: str) -> StageOutcome:
    print(f"[{name}] skipped: {reason}")
    return StageOutcome(name, "skipped", 0.0, reason)


def _parse_run_dir(stdout: str) -> Path | None:
    match = re.search(r"^wrote run to (?P<run_dir>.+?): best_epoch=\d+", stdout, re.MULTILINE)
    return Path(match.group("run_dir")) if match else None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    patients = parser.add_mutually_exclusive_group()
    patients.add_argument("--limit-patients", type=int, default=30)
    patients.add_argument(
        "--all-patients", action="store_true",
        help="extract every patient of the requested partitions (a full training run, not a smoke run)",
    )
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument(
        "--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG,
        help="training config driving the overfit gate and the training stage",
    )
    # `--epochs` and `--encoder-weights` default to None so a full-training config keeps its own
    # values; they are only forwarded as overrides when explicitly asked for on the command line.
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--overfit-steps", type=int, default=300)
    parser.add_argument("--overfit-n", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--encoder-weights", choices=("imagenet", "none"), default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-audit-if-present", action="store_true")
    parser.add_argument(
        "--resume-run", type=Path, default=None,
        help="continue training inside this existing run directory instead of starting a new one",
    )
    return parser


def _collect_zip_entries(run_dir: Path | None, env_path: Path, split_path: Path, patch_index_path: Path) -> tuple[list[tuple[Path, str]], list[str]]:
    """Files to store in the zip (path, archive name), and archive names skipped for size."""
    entries: list[tuple[Path, str]] = []
    skipped: list[str] = []
    if run_dir is not None and run_dir.is_dir():
        for path in sorted(run_dir.rglob("*")):
            if not path.is_file():
                continue
            arcname = f"{run_dir.name}/{path.relative_to(run_dir).as_posix()}"
            if path.stat().st_size > MAX_ZIP_FILE_BYTES:
                skipped.append(arcname)
                continue
            entries.append((path, arcname))
    for path, arcname in (
        (env_path, "env.yaml"),
        (split_path, "splits/split_v1.json"),
        (patch_index_path, "processed/patches_v1/index.csv"),
    ):
        if path.is_file():
            if path.stat().st_size > MAX_ZIP_FILE_BYTES:
                skipped.append(arcname)
                continue
            entries.append((path, arcname))
    return entries, skipped


def _write_zip(zip_path: Path, report: dict, entries: list[tuple[Path, str]]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    report_bytes = json.dumps(report, indent=2).encode("utf-8")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("smoke_report.json", report_bytes)
        for path, arcname in entries:
            archive.write(path, arcname)


def run_smoke(args: argparse.Namespace) -> tuple[int, dict]:
    """Run every stage into `args.work_dir`; return `(exit_code, report_dict)`."""
    raw_dir = args.raw_dir.resolve()
    work_dir = args.work_dir.resolve()
    if work_dir.is_relative_to(raw_dir):
        raise ValueError(f"--work-dir must be outside the raw directory {raw_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = work_dir / "logs"
    env_path = work_dir / "env.yaml"
    manifest_path = work_dir / "interim" / "manifest.csv"
    acquisition_path = work_dir / "interim" / "acquisition_exclusions.csv"
    exclusions_path = work_dir / "interim" / "exclusions.csv"
    split_path = work_dir / "splits" / "split_v1.json"
    patches_dir = work_dir / "processed" / "patches_v1"
    patches_index_path = patches_dir / "index.csv"
    experiments_dir = work_dir / "experiments"

    train_config = args.train_config.resolve()
    if not train_config.is_file():
        raise ValueError(f"missing training config: {train_config}")
    limit_patients = None if args.all_patients else args.limit_patients
    py = sys.executable

    shared_train_overrides: list[str] = []
    if args.encoder_weights is not None:
        shared_train_overrides += [
            "--set", f"encoder_weights={'null' if args.encoder_weights == 'none' else args.encoder_weights}"
        ]

    started_at = _utc_now()
    outcomes: list[StageOutcome] = []
    aborted = False
    run_dir: Path | None = None

    def run_or_skip(
        name: str, command: list[str], log_name: str, *, output_for_skip: Path | None = None, pattern: str | None = None,
    ) -> StageOutcome:
        nonlocal aborted
        if aborted:
            outcome = _skipped(name, "earlier required stage failed")
        elif output_for_skip is not None and args.skip_audit_if_present and output_for_skip.exists():
            outcome = _skipped(name, f"output already present: {output_for_skip}")
        else:
            outcome = _run_stage(name, command, logs_dir / log_name, pattern=pattern)
        outcomes.append(outcome)
        if outcome.status == "failed":
            aborted = True
        return outcome

    run_or_skip(
        "capture_environment",
        [py, str(SCRIPTS_DIR / "capture_environment.py"), "--record-only", "--output", str(env_path)],
        "capture_environment.log",
    )

    run_or_skip(
        "build_manifest",
        [
            py, str(SCRIPTS_DIR / "build_manifest.py"), "--raw-dir", str(raw_dir),
            "--output", str(manifest_path), "--acquisition-exclusions", str(acquisition_path),
        ],
        "build_manifest.log",
        output_for_skip=manifest_path,
        pattern=r"^wrote \d+ manifest rows.*$",
    )

    run_or_skip(
        "audit",
        [
            py, str(SCRIPTS_DIR / "audit_data.py"), "--raw-dir", str(raw_dir), "--manifest", str(manifest_path),
            "--output", str(exclusions_path), "--lesion-type", "mass",
        ],
        "audit.log",
        output_for_skip=exclusions_path,
        pattern=r"^wrote \d+ exclusions.*$",
    )

    run_or_skip(
        "make_splits",
        [
            py, str(SCRIPTS_DIR / "make_splits.py"), "--raw-dir", str(raw_dir), "--manifest", str(manifest_path),
            "--exclusions", str(exclusions_path), "--output", str(split_path),
            "--n-folds", str(args.n_folds), "--seed", str(args.seed),
        ],
        "make_splits.log",
        output_for_skip=split_path,
        pattern=r"^wrote split to.*$",
    )

    extract_command = [
        py, str(SCRIPTS_DIR / "extract_patches.py"), "--raw-dir", str(raw_dir), "--manifest", str(manifest_path),
        "--exclusions", str(exclusions_path), "--split", str(split_path), "--output-dir", str(patches_dir),
        "--size", str(args.size), "--seed", str(args.seed),
    ]
    if limit_patients is not None:
        extract_command += ["--limit-patients", str(limit_patients)]
    extract_command += ["--partitions", "train", "val"]
    run_or_skip(
        "extract_patches",
        extract_command,
        "extract_patches.log",
        output_for_skip=patches_index_path,
        pattern=r"^wrote \d+ patches.*$",
    )

    # The overfit gate is recorded but never blocks the rest of the smoke run.
    if aborted:
        outcomes.append(_skipped("overfit", "earlier required stage failed"))
    else:
        outcomes.append(_run_stage(
            "overfit",
            [
                py, str(SCRIPTS_DIR / "overfit_batch.py"), "--config", str(train_config),
                "--n", str(args.overfit_n), "--steps", str(args.overfit_steps),
                "--set", f"patch_root={patches_dir}", "--set", f"device={args.device}",
                *shared_train_overrides,
            ],
            logs_dir / "overfit.log",
            pattern=r"^overfit gate: dice=.*$",
        ))

    train_command = [
        py, str(SCRIPTS_DIR / "train.py"), "--config", str(train_config),
        "--set", f"patch_root={patches_dir}", "--set", f"output_root={experiments_dir}",
        "--set", f"device={args.device}", "--set", f"seed={args.seed}",
        *shared_train_overrides,
    ]
    if args.epochs is not None:
        train_command += ["--set", f"epochs={args.epochs}"]
    if args.resume_run is not None:
        train_command += ["--set", f"resume_from={args.resume_run}"]
    elif train_config == DEFAULT_TRAIN_CONFIG.resolve():
        # Only the smoke config gets its run renamed; a full-training config keeps its own run_name.
        train_command += ["--set", "run_name=smoke"]
    train_outcome = run_or_skip(
        "train",
        train_command,
        "train.log",
        pattern=r"^wrote run to .*: best_epoch=\d+ best_val_dice=[0-9.]+.*$",
    )
    if train_outcome.status == "ok":
        run_dir = _parse_run_dir(train_outcome.stdout)
        if run_dir is None:
            train_outcome.status = "failed"
            train_outcome.detail = f"{train_outcome.detail} (could not parse the run directory from stdout)"
            aborted = True

    if aborted or run_dir is None:
        outcomes.append(_skipped("predict", "earlier required stage failed"))
    else:
        run_or_skip(
            "predict",
            [
                py, str(SCRIPTS_DIR / "predict.py"), "--run-dir", str(run_dir), "--patch-root", str(patches_dir),
                "--partition", "val", "--device", args.device,
            ],
            "predict.log",
            pattern=r"^wrote .*_summary\.json$",
        )

    finished_at = _utc_now()
    stage_status = {outcome.name: outcome.status for outcome in outcomes}
    success = all(stage_status.get(name) == "ok" for name in REQUIRED_STAGE_NAMES)

    report: dict[str, object] = {
        "stages": [outcome.to_dict() for outcome in outcomes],
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "work_dir": str(work_dir),
        "arguments": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
    }

    zip_path = work_dir / "smoke_run.zip"
    entries, skipped_zip_files = _collect_zip_entries(run_dir, env_path, split_path, patches_index_path)
    report["zip"] = {"path": str(zip_path), "skipped_files": skipped_zip_files}
    _write_zip(zip_path, report, entries)

    report_path = work_dir / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return (0 if success else 1), report


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.raw_dir.is_dir():
        parser.error(f"missing raw directory: {args.raw_dir}")
    try:
        exit_code, _report = run_smoke(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
