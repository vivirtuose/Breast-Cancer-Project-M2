"""Regression checks for the Python/CUDA dependency contract."""

from pathlib import Path
import tomllib


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIRECT_DISTRIBUTIONS = {
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "opencv-python-headless",
    "albumentations",
    "scipy",
    "scikit-image",
    "pyyaml",
    "pytest",
    "tqdm",
    "segmentation-models-pytorch",
    "scikit-learn",
}


def read_toml(filename: str) -> dict:
    with (REPOSITORY_ROOT / filename).open("rb") as file:
        return tomllib.load(file)


def project_direct_pins(project: dict) -> dict[str, str]:
    pins = {}
    for requirement in project["project"]["dependencies"]:
        name, separator, version = requirement.partition("==")
        assert separator == "==" and requirement.count("==") == 1 and version
        assert requirement == f"{name}=={version}"
        pins[name.lower()] = version
    assert len(pins) == len(project["project"]["dependencies"])
    return pins


def packages_by_name(lock: dict) -> dict[str, dict]:
    return {package["name"].lower(): package for package in lock["package"]}


def test_project_is_python_312_only_with_canonical_exact_direct_pins() -> None:
    project = read_toml("pyproject.toml")

    assert project["project"]["requires-python"] == ">=3.12,<3.13"
    direct_pins = project_direct_pins(project)
    assert set(direct_pins) == CANONICAL_DIRECT_DISTRIBUTIONS
    assert len(direct_pins) == len(CANONICAL_DIRECT_DISTRIBUTIONS)


def test_project_is_installable_from_the_src_layout_for_kaggle() -> None:
    project = read_toml("pyproject.toml")

    assert project["build-system"] == {
        "requires": ["setuptools==84.0.0"],
        "build-backend": "setuptools.build_meta",
    }
    assert project["tool"]["setuptools"]["package-dir"] == {"": "src"}
    assert project["tool"]["setuptools"]["packages"]["find"] == {"where": ["src"]}


def test_torch_uses_the_explicit_official_cu121_index_only() -> None:
    project = read_toml("pyproject.toml")

    direct_pins = project_direct_pins(project)
    assert direct_pins["torch"] == "2.5.1+cu121"
    assert direct_pins["torchvision"] == "0.20.1+cu121"
    assert project["tool"]["uv"]["index"] == [
        {
            "name": "pytorch-cu121",
            "url": "https://download.pytorch.org/whl/cu121",
            "explicit": True,
        }
    ]
    assert project["tool"]["uv"]["sources"] == {
        "torch": {"index": "pytorch-cu121"},
        "torchvision": {"index": "pytorch-cu121"},
    }


def test_lock_root_metadata_repeats_every_direct_pin_identically() -> None:
    project = read_toml("pyproject.toml")
    lock = read_toml("uv.lock")
    root_package = packages_by_name(lock)["cbis-ddsm-breast-cancer"]

    assert root_package["source"] == {"editable": "."}
    locked_requirements = {
        requirement["name"].lower(): requirement["specifier"]
        for requirement in root_package["metadata"]["requires-dist"]
    }
    assert locked_requirements == {
        name: f"=={version}"
        for name, version in project_direct_pins(project).items()
    }
    torch_requirement = next(
        requirement
        for requirement in root_package["metadata"]["requires-dist"]
        if requirement["name"].lower() == "torch"
    )
    assert torch_requirement["index"] == "https://download.pytorch.org/whl/cu121"


def test_lock_is_python_312_only_and_resolves_every_direct_pin() -> None:
    project = read_toml("pyproject.toml")
    lock = read_toml("uv.lock")
    locked_packages = packages_by_name(lock)

    # uv canonicalizes the project's >=3.12,<3.13 constraint to this equivalent form.
    assert lock["requires-python"] == "==3.12.*"
    for name, version in project_direct_pins(project).items():
        assert locked_packages[name]["version"] == version
    assert locked_packages["torch"]["source"] == {
        "registry": "https://download.pytorch.org/whl/cu121"
    }


def test_lock_contains_hashed_registry_artifacts_for_transitive_closure() -> None:
    lock = read_toml("uv.lock")
    locked_packages = lock["package"]
    registry_packages = [
        package for package in locked_packages if "registry" in package["source"]
    ]

    assert len(locked_packages) > len(CANONICAL_DIRECT_DISTRIBUTIONS)
    assert registry_packages
    for package in registry_packages:
        assert isinstance(package["version"], str) and package["version"]
        assert package["source"] == {"registry": package["source"]["registry"]}
        distributions = [*package.get("wheels", [])]
        if source_distribution := package.get("sdist"):
            distributions.append(source_distribution)
        assert distributions, package["name"]
        for distribution in distributions:
            assert distribution["url"]
            assert distribution["hash"].startswith("sha256:")
