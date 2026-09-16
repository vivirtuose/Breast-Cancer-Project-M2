"""Patient-grouped stratified train/val/test partitions."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .targets import ImageRecord


def patient_labels(records: Iterable[ImageRecord]) -> dict[str, int]:
    """1 if any train-split image of the patient has a malignant lesion, else 0."""
    labels: dict[str, int] = {}
    for record in records:
        if record.split != "train":
            continue
        malignant = 1 if record.has_malignant else 0
        labels[record.patient_id] = max(labels.get(record.patient_id, 0), malignant)
    return labels


def assign_folds(labels: Mapping[str, int], n_folds: int = 5, seed: int = 0) -> dict[str, int]:
    """Deterministic stratified fold assignment, one sample per patient (grouping is exact)."""
    patients = sorted(labels)
    if not patients:
        raise ValueError("assign_folds requires at least one patient")
    class_counts = Counter(labels[patient] for patient in patients)
    smallest_class = min(class_counts.values())
    if n_folds > smallest_class:
        raise ValueError(
            f"n_folds={n_folds} exceeds the number of patients in the smallest label class "
            f"({smallest_class}); reduce --n-folds or provide more patients"
        )
    features = np.zeros((len(patients), 1))
    targets = np.array([labels[patient] for patient in patients])
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds: dict[str, int] = {}
    for fold_index, (_train_index, val_index) in enumerate(splitter.split(features, targets)):
        for index in val_index:
            folds[patients[index]] = fold_index
    return folds


def build_split(
    records: Iterable[ImageRecord], *, n_folds: int = 5, val_fold: int = 0, seed: int = 0, manifest_sha256: str,
) -> dict:
    """Assign every train patient to a stratified fold; val_fold is validation, the rest is train."""
    materialized = tuple(records)
    if not 0 <= val_fold < n_folds:
        raise ValueError(f"val_fold={val_fold} must be in range [0, {n_folds})")
    train_patients = {record.patient_id for record in materialized if record.split == "train"}
    test_patients = {record.patient_id for record in materialized if record.split == "test"}
    overlap = train_patients & test_patients
    if overlap:
        raise ValueError(f"patients appear in both official train and test splits: {sorted(overlap)}")
    labels = patient_labels(materialized)
    folds = assign_folds(labels, n_folds=n_folds, seed=seed)
    return {
        "version": 1,
        "seed": seed,
        "n_folds": n_folds,
        "val_fold": val_fold,
        "manifest_sha256": manifest_sha256,
        "folds": folds,
        "partitions": {
            "train": sorted(patient for patient, fold in folds.items() if fold != val_fold),
            "val": sorted(patient for patient, fold in folds.items() if fold == val_fold),
            "test": sorted(test_patients),
        },
    }


def write_split_json(split: Mapping, path: Path | str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(split, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def read_split_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def partition_of(split: Mapping) -> dict[str, str]:
    """patient_id -> "train" | "val" | "test"."""
    return {
        patient_id: partition
        for partition, patients in split["partitions"].items()
        for patient_id in patients
    }


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()
