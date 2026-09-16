"""Capture an auditable, machine-readable environment snapshot.

The emitted document is JSON-formatted YAML 1.2, deliberately avoiding a
runtime dependency on PyYAML before the project environment is installed.
"""

from __future__ import annotations

import argparse
import csv
import importlib
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from typing import Any, Callable


REQUIRED_PYTHON = "3.12"
REQUIRED_PACKAGES = (
    "torch",
    "numpy",
    "pandas",
    "opencv-python-headless",
    "albumentations",
    "scipy",
    "scikit-image",
    "PyYAML",
    "pytest",
    "tqdm",
)
CUDA_EXPRESSION = "torch.cuda.is_available() == True"
DEFAULT_LOCK_PATH = Path("uv.lock")
ALLOWED_UNLOCKED_TOOLING = frozenset({"pip", "setuptools", "wheel"})
LOCAL_PROJECT_DISTRIBUTIONS = frozenset({"cbis-ddsm-breast-cancer"})


def parse_nvidia_smi(output: str) -> dict[str, Any]:
    """Parse the first row of the explicit nvidia-smi CSV query."""
    rows = [row for row in csv.reader(line for line in output.splitlines() if line.strip())]
    if not rows or len(rows[0]) != 4:
        raise ValueError("nvidia-smi did not return name, VRAM, driver, and compute capability")
    name, vram, driver, capability = (value.strip() for value in rows[0])
    try:
        vram_mib = int(vram.removesuffix(" MiB").strip())
    except ValueError as error:
        raise ValueError(f"nvidia-smi returned an invalid VRAM value: {vram!r}") from error
    return {
        "detected": True,
        "name": name or None,
        "vram_mib": vram_mib,
        "driver_version": driver or None,
        "compute_capability": capability or None,
        "blockers": [],
    }


def probe_nvidia_smi(
    runner: Callable[[list[str]], Any] | None = None,
) -> dict[str, Any]:
    """Query NVIDIA hardware; a failure is recorded rather than raised."""
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    if runner is None:
        runner = lambda args: subprocess.run(args, capture_output=True, text=True, check=True)
    try:
        completed = runner(command)
        output = getattr(completed, "stdout", completed)
        return parse_nvidia_smi(str(output))
    except FileNotFoundError as error:
        message = f"nvidia-smi unavailable: {error}"
    except subprocess.CalledProcessError as error:
        message = f"nvidia-smi failed: exit status {error.returncode}"
    except (TypeError, ValueError) as error:
        message = f"nvidia-smi output could not be parsed: {error}"
    return {
        "detected": False,
        "name": None,
        "vram_mib": None,
        "driver_version": None,
        "compute_capability": None,
        "blockers": [message],
    }


def collect_system_info() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def probe_disk(path: str) -> tuple[int, str]:
    usage = shutil.disk_usage(path)
    return usage.free, path


def import_torch() -> Any:
    return importlib.import_module("torch")


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_distribution_name(name: str) -> str:
    """Apply the distribution-name normalization used by package metadata."""
    return "-".join(part for part in name.lower().replace("_", "-").replace(".", "-").split("-") if part)


REQUIRED_DIRECT_PACKAGES = frozenset(
    normalize_distribution_name(name) for name in REQUIRED_PACKAGES
)


def probe_installed_distributions() -> dict[str, str]:
    """Return the installed distribution inventory with normalized names."""
    distributions: dict[str, str] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name") or distribution.name
        if name:
            distributions[normalize_distribution_name(name)] = distribution.version
    return dict(sorted(distributions.items()))


def parse_uv_lock(lock_data: str | bytes) -> dict[str, str]:
    """Return registry distributions and their expected versions from uv.lock."""
    if isinstance(lock_data, bytes):
        lock_data = lock_data.decode("utf-8")
    parsed = tomllib.loads(lock_data)
    packages: dict[str, str] = {}
    for entry in parsed.get("package", []):
        source = entry.get("source", {})
        if not isinstance(source, dict) or "registry" not in source:
            continue
        name = normalize_distribution_name(str(entry["name"]))
        version = str(entry["version"])
        previous = packages.setdefault(name, version)
        if previous != version:
            raise ValueError(f"lock contains multiple versions for {name}: {previous} and {version}")
    return dict(sorted(packages.items()))


def load_lock(
    lock_data: str | bytes | None = None,
    lock_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Load the lock from injected data or the repository-relative default path."""
    repository_root = Path(__file__).resolve().parents[1]
    if lock_path is None:
        resolved_path = repository_root / DEFAULT_LOCK_PATH
        display_path = DEFAULT_LOCK_PATH.as_posix()
    else:
        resolved_path = Path(lock_path)
        try:
            display_path = resolved_path.resolve().relative_to(repository_root).as_posix()
        except ValueError:
            display_path = str(lock_path)
    try:
        if lock_data is None:
            lock_data = resolved_path.read_bytes()
        packages = parse_uv_lock(lock_data)
        blockers: list[str] = []
        if not packages:
            blockers.append("Lock contains no registry packages.")
        return {
            "path": display_path,
            "status": "verified",
            "locked_package_count": len(packages),
            "blockers": blockers,
        }, packages
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, KeyError, TypeError, ValueError) as error:
        blocker = f"Lock unavailable: {type(error).__name__}: {error}"
        return {
            "path": display_path,
            "status": "unavailable",
            "locked_package_count": 0,
            "blockers": [blocker],
        }, {}


def capture_snapshot(
    *,
    captured_at: str | None = None,
    python_executable: str | None = None,
    python_version: str | None = None,
    torch_importer: Callable[[], Any] = import_torch,
    gpu_probe: Callable[[], dict[str, Any]] = probe_nvidia_smi,
    disk_probe: Callable[[str], tuple[int, str]] = probe_disk,
    disk_path: str | None = None,
    system_info: dict[str, str] | None = None,
    lock_data: str | bytes | None = None,
    lock_path: str | Path | None = None,
    installed_distributions: Callable[[], dict[str, str]] = probe_installed_distributions,
) -> dict[str, Any]:
    """Collect a snapshot with injectable probes for deterministic tests."""
    captured_at = captured_at or utc_now()
    python_executable = python_executable or sys.executable
    python_version = python_version or platform.python_version()
    disk_path = disk_path or Path.cwd().anchor
    system_info = system_info or collect_system_info()
    blockers: list[str] = []

    matches_required = ".".join(python_version.split(".")[:2]) == REQUIRED_PYTHON
    if not matches_required:
        blockers.append(f"Python {REQUIRED_PYTHON} required; found {python_version}.")

    lock, locked_packages = load_lock(lock_data=lock_data, lock_path=lock_path)
    blockers.extend(lock["blockers"])
    missing_direct_from_lock = sorted(REQUIRED_DIRECT_PACKAGES - locked_packages.keys())
    for distribution in missing_direct_from_lock:
        blockers.append(
            f"Lock contract mismatch: required direct package missing from lock: {distribution}."
        )
    try:
        installed = {
            normalize_distribution_name(name): str(version)
            for name, version in installed_distributions().items()
        }
    except Exception as error:
        installed = {}
        blockers.append(
            f"Installed distribution inventory unavailable: {type(error).__name__}: {error}"
        )

    packages: dict[str, dict[str, str | bool | None]] = {}
    installed_locked_count = 0
    absent_platform_optional_count = 0
    missing_direct_count = len(missing_direct_from_lock)
    mismatched_count = 0
    for distribution, expected_version in locked_packages.items():
        required_direct = distribution in REQUIRED_DIRECT_PACKAGES
        installed_version = installed.get(distribution)
        if installed_version is not None:
            matches_lock = installed_version == expected_version
            packages[distribution] = {
                "distribution": distribution,
                "expected_version": expected_version,
                "installed_version": installed_version,
                "status": "installed",
                "matches_lock": matches_lock,
                "required_direct": required_direct,
            }
            if not matches_lock:
                mismatched_count += 1
                blockers.append(
                    f"Package version mismatch: {distribution} expected {expected_version}, "
                    f"installed {installed_version}."
                )
            else:
                installed_locked_count += 1
        else:
            packages[distribution] = {
                "distribution": distribution,
                "expected_version": expected_version,
                "installed_version": None,
                "status": "not_installed",
                "matches_lock": False if required_direct else None,
                "required_direct": required_direct,
            }
            if required_direct:
                missing_direct_count += 1
                blockers.append(f"Package not installed: {distribution}")
            else:
                absent_platform_optional_count += 1

    local_projects = {
        distribution: version
        for distribution, version in installed.items()
        if distribution in LOCAL_PROJECT_DISTRIBUTIONS
    }
    unlocked_installed = {
        distribution: version
        for distribution, version in installed.items()
        if distribution not in locked_packages
        and distribution not in ALLOWED_UNLOCKED_TOOLING
        and distribution not in LOCAL_PROJECT_DISTRIBUTIONS
        and distribution not in REQUIRED_DIRECT_PACKAGES
    }
    for distribution, version in unlocked_installed.items():
        blockers.append(f"Unexpected unlocked package installed: {distribution} {version}.")
    lock.update(
        {
            "installed_locked_count": installed_locked_count,
            "absent_platform_optional_count": absent_platform_optional_count,
            "missing_direct_count": missing_direct_count,
            "mismatched_count": mismatched_count,
            "unlocked_unexpected_count": len(unlocked_installed),
        }
    )

    gpu = gpu_probe()
    blockers.extend(gpu["blockers"])
    if not gpu["detected"] and not gpu["blockers"]:
        blockers.append("GPU capture failed: no NVIDIA GPU information was returned.")

    try:
        torch = torch_importer()
        cuda_result = torch.cuda.is_available() == True
        if not cuda_result:
            blockers.append("CUDA gate evaluated false: torch.cuda.is_available() returned False.")
    except ModuleNotFoundError as error:
        cuda_result = None
        if error.name == "torch":
            blockers.append("CUDA gate unavailable: torch is not installed.")
        else:
            blockers.append(f"CUDA gate could not be evaluated: {type(error).__name__}: {error}")
    except Exception as error:
        cuda_result = None
        blockers.append(f"CUDA gate could not be evaluated: {type(error).__name__}: {error}")

    free_bytes, observed_disk_path = disk_probe(disk_path)
    disk = {
        "path": observed_disk_path,
        "free_bytes": int(free_bytes),
        "free_gib": int(free_bytes) / (1024 ** 3),
    }
    passed = (
        cuda_result is True
        and matches_required
        and lock["status"] == "verified"
        and not missing_direct_from_lock
        and all(
            item["matches_lock"] is True
            for item in packages.values()
            if item["required_direct"] or item["status"] == "installed"
        )
        and gpu["detected"]
        and not unlocked_installed
        and not blockers
    )
    return {
        "schema_version": 1,
        "capture": {
            "status": "complete" if passed else "blocked",
            "captured_at_utc": captured_at,
            "script": "scripts/capture_environment.py",
        },
        "python": {
            "required": REQUIRED_PYTHON,
            "executable": python_executable,
            "version": python_version,
            "matches_required": matches_required,
        },
        "system": system_info,
        "gpu": {
            "detected": gpu["detected"],
            "name": gpu["name"],
            "vram_mib": gpu["vram_mib"],
            "driver_version": gpu["driver_version"],
            "compute_capability": gpu["compute_capability"],
        },
        "disk": disk,
        "lock": lock,
        "packages": packages,
        "local_projects": local_projects,
        "unlocked_installed": unlocked_installed,
        "cuda_gate": {
            "expression": CUDA_EXPRESSION,
            "result": cuda_result,
            "passed": passed,
            "blockers": blockers,
        },
    }


def write_snapshot(snapshot: dict[str, Any], output_path: Path) -> None:
    """Write JSON, which is a valid YAML 1.2 document, in stable key order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = Path(__file__).resolve().parents[1] / "configs" / "env.yaml"
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="exit successfully even when the CUDA readiness gate is blocked",
    )
    args = parser.parse_args(argv)
    snapshot = capture_snapshot()
    write_snapshot(snapshot, args.output)
    return 0 if snapshot["cuda_gate"]["passed"] or args.allow_blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())
