"""Focused tests for the reproducibility environment capture."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "capture_environment.py"
SPEC = importlib.util.spec_from_file_location("capture_environment", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
capture_environment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = capture_environment
SPEC.loader.exec_module(capture_environment)

LOCK_NAMES = tuple(
    "pyyaml" if name == "PyYAML" else name
    for name in capture_environment.REQUIRED_PACKAGES
)
LOCK_DATA = "version = 1\n" + "".join(
    f"""
[[package]]
name = "{name}"
version = "1.0.0"
source = {{ registry = "https://pypi.org/simple" }}
"""
    for name in LOCK_NAMES
)
WINDOWS_LOCK_DATA = LOCK_DATA + """
[[package]]
name = "linux-only-runtime"
version = "9.9.9"
source = { registry = "https://pypi.org/simple" }
"""


class FakeTorch:
    class cuda:
        @staticmethod
        def is_available() -> bool:
            return True


class FalseTorch:
    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False


class CaptureEnvironmentTests(unittest.TestCase):
    def make_snapshot(self, **overrides):
        versions = {package: "1.0.0" for package in LOCK_NAMES}
        defaults = {
            "captured_at": "2026-09-15T00:00:00Z",
            "python_executable": "C:/Python312/python.exe",
            "python_version": "3.12.14",
            "installed_distributions": lambda: versions,
            "torch_importer": lambda: FakeTorch(),
            "gpu_probe": lambda: {
                "detected": True,
                "name": "NVIDIA Test GPU",
                "vram_mib": 2048,
                "driver_version": "537.58",
                "compute_capability": "6.1",
                "blockers": [],
            },
            "disk_probe": lambda path: (10_737_418_240, path),
            "disk_path": "C:/",
            "system_info": {"platform": "Windows-11", "system": "Windows"},
            "lock_data": LOCK_DATA,
        }
        defaults.update(overrides)
        return capture_environment.capture_snapshot(**defaults)

    def test_exact_match_for_full_injected_lock_passes_package_and_gate_checks(self):
        snapshot = self.make_snapshot()

        self.assertEqual(snapshot["schema_version"], 1)
        self.assertEqual(snapshot["capture"]["status"], "complete")
        self.assertEqual(snapshot["cuda_gate"]["expression"], "torch.cuda.is_available() == True")
        self.assertIs(snapshot["cuda_gate"]["result"], True)
        self.assertTrue(snapshot["cuda_gate"]["passed"])
        self.assertEqual(snapshot["cuda_gate"]["blockers"], [])
        self.assertEqual(snapshot["lock"]["status"], "verified")
        self.assertEqual(snapshot["lock"]["locked_package_count"], len(LOCK_NAMES))
        self.assertEqual(snapshot["lock"]["missing_direct_count"], 0)
        self.assertEqual(list(snapshot["packages"]), sorted(LOCK_NAMES))

    def test_one_version_drift_blocks_gate_with_expected_and_installed_versions(self):
        snapshot = self.make_snapshot(
            installed_distributions=lambda: {
                name: "0.9.0" if name == "torch" else "1.0.0" for name in LOCK_NAMES
            }
        )

        self.assertFalse(snapshot["cuda_gate"]["passed"])
        self.assertEqual(snapshot["packages"]["torch"]["expected_version"], "1.0.0")
        self.assertEqual(snapshot["packages"]["torch"]["installed_version"], "0.9.0")
        self.assertFalse(snapshot["packages"]["torch"]["matches_lock"])
        self.assertIn(
            "Package version mismatch: torch expected 1.0.0, installed 0.9.0.",
            snapshot["cuda_gate"]["blockers"],
        )

    def test_editable_project_is_recorded_without_being_treated_as_unlocked_dependency(self):
        snapshot = self.make_snapshot(
            installed_distributions=lambda: {
                **{name: "1.0.0" for name in LOCK_NAMES},
                "cbis-ddsm-breast-cancer": "0.1.0",
            }
        )

        self.assertTrue(snapshot["cuda_gate"]["passed"])
        self.assertEqual(snapshot["local_projects"], {"cbis-ddsm-breast-cancer": "0.1.0"})
        self.assertEqual(snapshot["unlocked_installed"], {})

    def test_unavailable_lock_is_a_hard_blocker(self):
        snapshot = self.make_snapshot(lock_data="this is not valid TOML = [")

        self.assertEqual(snapshot["lock"]["status"], "unavailable")
        self.assertEqual(snapshot["lock"]["locked_package_count"], 0)
        self.assertFalse(snapshot["cuda_gate"]["passed"])
        self.assertTrue(snapshot["lock"]["blockers"])

    def test_windows_passes_with_absent_linux_only_transitive_lock_record(self):
        snapshot = self.make_snapshot(lock_data=WINDOWS_LOCK_DATA)

        self.assertTrue(snapshot["cuda_gate"]["passed"])
        self.assertEqual(snapshot["packages"]["linux-only-runtime"], {
            "distribution": "linux-only-runtime",
            "expected_version": "9.9.9",
            "installed_version": None,
            "status": "not_installed",
            "matches_lock": None,
            "required_direct": False,
        })
        self.assertEqual(snapshot["lock"]["absent_platform_optional_count"], 1)

    def test_missing_direct_dependency_blocks_gate_with_concrete_blocker(self):
        snapshot = self.make_snapshot(
            installed_distributions=lambda: {
                name: "1.0.0" for name in LOCK_NAMES if name != "numpy"
            }
        )

        self.assertFalse(snapshot["cuda_gate"]["passed"])
        self.assertTrue(snapshot["packages"]["numpy"]["required_direct"])
        self.assertFalse(snapshot["packages"]["numpy"]["matches_lock"])
        self.assertIn("Package not installed: numpy", snapshot["cuda_gate"]["blockers"])

    def test_missing_torch_is_null_gate_result_with_concrete_blocker(self):
        def missing_torch():
            raise ModuleNotFoundError("No module named 'torch'", name="torch")

        snapshot = self.make_snapshot(
            installed_distributions=lambda: {
                name: "1.0.0" for name in LOCK_NAMES if name != "torch"
            },
            torch_importer=missing_torch,
        )

        self.assertIsNone(snapshot["cuda_gate"]["result"])
        self.assertFalse(snapshot["cuda_gate"]["passed"])
        self.assertIn("Package not installed: torch", snapshot["cuda_gate"]["blockers"])
        self.assertIn("CUDA gate unavailable: torch is not installed.", snapshot["cuda_gate"]["blockers"])

    def test_evaluated_cuda_false_remains_false_and_blocks_capture(self):
        snapshot = self.make_snapshot(torch_importer=lambda: FalseTorch())

        self.assertIs(snapshot["cuda_gate"]["result"], False)
        self.assertFalse(snapshot["cuda_gate"]["passed"])
        self.assertIn("CUDA gate evaluated false: torch.cuda.is_available() returned False.", snapshot["cuda_gate"]["blockers"])
        self.assertEqual(snapshot["capture"]["status"], "blocked")

    def test_torch_import_failure_from_internal_dependency_is_not_missing_torch(self):
        def broken_torch_import():
            raise ModuleNotFoundError("No module named 'broken_dep'", name="broken_dep")

        snapshot = self.make_snapshot(torch_importer=broken_torch_import)

        self.assertIsNone(snapshot["cuda_gate"]["result"])
        self.assertFalse(snapshot["cuda_gate"]["passed"])
        self.assertIn(
            "CUDA gate could not be evaluated: ModuleNotFoundError: No module named 'broken_dep'",
            snapshot["cuda_gate"]["blockers"],
        )
        self.assertNotIn(
            "CUDA gate unavailable: torch is not installed.",
            snapshot["cuda_gate"]["blockers"],
        )

    def test_package_capture_keeps_missing_distributions_explicit(self):
        snapshot = self.make_snapshot(installed_distributions=lambda: {"numpy": "2.1.0"})

        self.assertEqual(snapshot["packages"]["numpy"], {
            "distribution": "numpy",
            "expected_version": "1.0.0",
            "installed_version": "2.1.0",
            "status": "installed",
            "matches_lock": False,
            "required_direct": True,
        })
        self.assertEqual(snapshot["packages"]["pyyaml"], {
            "distribution": "pyyaml",
            "expected_version": "1.0.0",
            "installed_version": None,
            "status": "not_installed",
            "matches_lock": False,
            "required_direct": True,
        })

    def test_nvidia_csv_parsing_and_probe_failure_are_explicit(self):
        parsed = capture_environment.parse_nvidia_smi(
            "NVIDIA GeForce MX350, 2048, 537.58, 6.1\n"
        )
        self.assertEqual(parsed["name"], "NVIDIA GeForce MX350")
        self.assertEqual(parsed["vram_mib"], 2048)
        self.assertEqual(parsed["driver_version"], "537.58")
        self.assertEqual(parsed["compute_capability"], "6.1")

        failed = capture_environment.probe_nvidia_smi(
            runner=lambda command: (_ for _ in ()).throw(FileNotFoundError("nvidia-smi"))
        )
        self.assertFalse(failed["detected"])
        self.assertIsNone(failed["vram_mib"])
        self.assertIn("nvidia-smi unavailable", failed["blockers"][0])


def test_record_only_mode_exits_zero_and_marks_snapshot(tmp_path, monkeypatch):
    blocked = {"cuda_gate": {"passed": False}, "capture": {"status": "blocked"}}
    monkeypatch.setattr(capture_environment, "capture_snapshot", lambda **_: dict(blocked))
    output = tmp_path / "env.yaml"
    assert capture_environment.main(["--record-only", "--output", str(output)]) == 0
    snapshot = json.loads(output.read_text(encoding="utf-8"))
    assert snapshot["capture"]["mode"] == "record-only"


if __name__ == "__main__":
    unittest.main()
