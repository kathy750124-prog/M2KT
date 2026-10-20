"""Read the existing classroom AV dataset for an M2KT feature adapter.

The report is a training target only. Inference uses ``metadata.json`` and
``provenance.json`` and never opens any split JSONL or reference report. This
keeps held-out labels out of feature extraction and report generation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_DATASET_DIR = (
    Path(__file__).resolve().parent / "data" / "classroom" / "omni_dataset_v3"
)
SPLITS = ("train", "val", "test")
EXPECTED_SPLIT_COUNTS = {"train": 56, "val": 10, "test": 10}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _dataset_dir(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _split(value: str) -> str:
    if value not in SPLITS:
        raise ValueError(f"Unknown classroom split: {value!r}")
    return value


def validate_dataset_identity(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    *,
    check_clips: bool = True,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Verify the fixed 56/10/10 split and its audio/video clip provenance.

    Only metadata, provenance and clip bytes are read. Set ``check_clips=False``
    only when a separate process has already checked every clip checksum.
    """

    folder = _dataset_dir(dataset_dir)
    metadata = _json_object(folder / "metadata.json")
    provenance_path = folder / "provenance.json"
    if metadata.get("audio_in_video_required") is not True:
        raise ValueError("Classroom dataset must require audio inside each video")
    if metadata.get("split_counts") != EXPECTED_SPLIT_COUNTS:
        raise ValueError("Expected the original 56/10/10 course split")
    expected_ids = metadata.get("split_course_ids")
    if not isinstance(expected_ids, dict) or set(expected_ids) != set(SPLITS):
        raise ValueError("Metadata lacks the exact three course ID lists")
    hashes = metadata.get("files_sha256")
    if (not isinstance(hashes, dict) or
            hashes.get("provenance.json") != _sha256_file(provenance_path)):
        raise ValueError("Dataset provenance SHA-256 mismatch")
    provenance = _json_object(provenance_path)
    if set(provenance) != set(SPLITS):
        raise ValueError("Provenance must contain exactly train, val and test")

    seen_courses: set[str] = set()
    seen_clips: set[Path] = set()
    selected_clip_count = 0
    for split in SPLITS:
        rows = provenance[split]
        ids = expected_ids[split]
        if (not isinstance(rows, list) or not isinstance(ids, list) or
                len(rows) != EXPECTED_SPLIT_COUNTS[split] or
                len(ids) != EXPECTED_SPLIT_COUNTS[split]):
            raise ValueError(f"Incorrect {split} course count")
        observed_ids = []
        for row in rows:
            if not isinstance(row, dict) or row.get("split") != split:
                raise ValueError(f"Invalid {split} provenance record")
            course_id = row.get("course_id")
            if (not isinstance(course_id, str) or not course_id.isdigit() or
                    course_id in seen_courses):
                raise ValueError(f"Invalid or repeated course ID: {course_id!r}")
            seen_courses.add(course_id)
            observed_ids.append(course_id)
            clips = row.get("selected_clips")
            if not isinstance(clips, list) or not clips:
                raise ValueError(f"No clips for course {course_id}")
            if len(clips) > metadata.get("max_clips_per_course", 0):
                raise ValueError(f"Too many clips for course {course_id}")
            for index, clip in enumerate(clips):
                if not isinstance(clip, dict) or clip.get("clip_index") != index:
                    raise ValueError(f"Invalid clip ordering for course {course_id}")
                raw_path = clip.get("clip_path")
                expected_hash = clip.get("clip_sha256")
                if (not isinstance(raw_path, str) or
                        not isinstance(expected_hash, str) or
                        len(expected_hash) != 64):
                    raise ValueError(f"Invalid clip provenance for course {course_id}")
                path = Path(raw_path)
                if (not path.is_absolute() or path.suffix.lower() != ".mp4" or
                        path in seen_clips or path.parent.name != course_id):
                    raise ValueError(f"Clip path reused or misassigned: {raw_path}")
                seen_clips.add(path)
                if check_clips:
                    if not path.is_file() or path.stat().st_size == 0:
                        raise FileNotFoundError(f"Missing AV clip: {path}")
                    if _sha256_file(path) != expected_hash:
                        raise ValueError(f"AV clip SHA-256 mismatch: {path}")
                selected_clip_count += 1
        if observed_ids != ids:
            raise ValueError(f"{split} course IDs differ from metadata")
    if metadata.get("selected_clip_count") != selected_clip_count:
        raise ValueError("Selected clip count differs from metadata")
    return metadata, provenance


def load_training_examples(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    split: str = "train",
) -> list[dict[str, Any]]:
    """Load supervised examples from train or val; never load test labels."""

    if _split(split) == "test":
        raise ValueError("Held-out test reports cannot be loaded for training")
    folder = _dataset_dir(dataset_dir)
    metadata, provenance = validate_dataset_identity(folder)
    filename = f"{split}.jsonl"
    path = folder / filename
    hashes = metadata["files_sha256"]
    if hashes.get(filename) != _sha256_file(path):
        raise ValueError(f"Dataset {filename} SHA-256 mismatch")
    lines = path.read_text(encoding="utf-8").splitlines()
    records = provenance[split]
    if len(lines) != len(records):
        raise ValueError(f"{split} JSONL count differs from provenance")

    examples = []
    for line_number, (line, record) in enumerate(zip(lines, records), 1):
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Invalid {filename} row {line_number}")
        messages = row.get("messages")
        paths = [clip["clip_path"] for clip in record["selected_clips"]]
        if (not isinstance(messages, list) or len(messages) != 3 or
                any(not isinstance(message, dict) for message in messages) or
                [message.get("role") for message in messages] !=
                ["system", "user", "assistant"] or
                any(not isinstance(message.get("content"), str) for message in messages) or
                row.get("videos") != paths or
                messages[1]["content"].count("<video>") != len(paths)):
            raise ValueError(f"Invalid AV/report pair in {filename}:{line_number}")
        report = messages[2]["content"]
        if (not report.strip() or
                hashlib.sha256(report.encode("utf-8")).hexdigest() !=
                record.get("report_sha256") or
                len(report) != record.get("report_characters")):
            raise ValueError(f"Report target differs from provenance in {filename}:{line_number}")
        examples.append({
            "course_id": record["course_id"],
            "split": split,
            "clip_paths": paths,
            "report": report,
        })
    return examples


def load_inference_examples(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    split: str = "val",
) -> list[dict[str, Any]]:
    """Load train, val or test clip paths without reading reference reports."""

    _split(split)
    _, provenance = validate_dataset_identity(dataset_dir)
    return [
        {
            "course_id": row["course_id"],
            "split": split,
            "clip_paths": [clip["clip_path"] for clip in row["selected_clips"]],
        }
        for row in provenance[split]
    ]
