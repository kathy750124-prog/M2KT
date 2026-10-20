"""Regression checks for split integrity and held-out report isolation."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classroom_m2kt_data import (  # noqa: E402
    DEFAULT_DATASET_DIR,
    load_inference_examples,
    load_training_examples,
)


@pytest.mark.skipif(not DEFAULT_DATASET_DIR.is_dir(), reason="Classroom AV dataset absent")
def test_inference_never_reads_reference_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    original_open = Path.open

    def reject_labels(path: Path, *args: object, **kwargs: object):
        if path.suffix in {".jsonl", ".txt"} or path.name == "annotation.json":
            raise AssertionError(f"Inference opened a reference label: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_labels)
    train = load_inference_examples(split="train")
    val = load_inference_examples(split="val")
    test = load_inference_examples(split="test")
    assert len(train) == 56 and len(val) == 10 and len(test) == 10
    assert all(set(row) == {"course_id", "split", "clip_paths"}
               for row in train + val + test)


@pytest.mark.skipif(not DEFAULT_DATASET_DIR.is_dir(), reason="Classroom AV dataset absent")
def test_training_rejects_report_assigned_to_another_course(tmp_path: Path) -> None:
    for name in ("metadata.json", "provenance.json", "train.jsonl"):
        shutil.copyfile(DEFAULT_DATASET_DIR / name, tmp_path / name)
    train_path = tmp_path / "train.jsonl"
    lines = train_path.read_text(encoding="utf-8").splitlines()
    lines[0], lines[1] = lines[1], lines[0]
    train_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["files_sha256"]["train.jsonl"] = hashlib.sha256(train_path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="AV/report pair|Report target"):
        load_training_examples(tmp_path)


def test_training_does_not_accept_test_labels() -> None:
    with pytest.raises(ValueError, match="Held-out test"):
        load_training_examples(split="test")
