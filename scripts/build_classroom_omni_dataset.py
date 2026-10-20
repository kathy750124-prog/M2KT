"""Build ms-swift Qwen2.5-Omni SFT JSONL from validated classroom AV clips.

Each course produces one example.  The assistant target is the *entire* report
from annotation.json; this script does not shorten, paraphrase, or tokenize it.
For video audio to reach Qwen2.5-Omni, run ms-swift with
``USE_AUDIO_IN_VIDEO=true`` at both training and inference time.
"""

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
EXPECTED_COUNTS = {"train": 56, "val": 10, "test": 10}

SYSTEM_PROMPT = (
    "你是一名课堂观察报告撰写助手。根据提供的课堂视频和其中的声音撰写中文课堂观察报告。"
    "只把能够从所给片段核实的内容当作事实；片段没有覆盖的教学过程应谨慎表述。"
)
USER_PROMPT = (
    "以下是同一门课的{count}段课堂音视频片段。请依据片段中可见、可听的信息，"
    "撰写完整中文课堂评价报告，以“综合评价：”起首，按课堂观察、分析与建议自然分段。"
    "直接输出报告正文，不要输出分析过程或 Markdown 代码块。\n{videos}"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_text(value, pretty=False):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      indent=2 if pretty else None,
                      separators=None if pretty else (",", ":")) + "\n"


def require_file(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError("{} must be a nonempty absolute path".format(name))
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("{} must be absolute: {}".format(name, value))
    path = path.resolve(strict=True)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("{} is empty or not a regular file: {}".format(name, path))
    return path


def load_annotation(path):
    annotation_bytes = path.read_bytes()
    annotation = json.loads(annotation_bytes.decode("utf-8"))
    if not isinstance(annotation, dict) or set(annotation) != set(SPLITS):
        raise ValueError("Annotation must have train, val, test splits only")
    by_id = {}
    split_order = {}
    for split in SPLITS:
        rows = annotation[split]
        if not isinstance(rows, list):
            raise ValueError("Invalid annotation split: " + split)
        split_order[split] = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Invalid annotation row in " + split)
            course_id = str(row.get("id", ""))
            report = row.get("report")
            if (not course_id.isdigit() or course_id in by_id or
                    not isinstance(report, str) or not report.strip()):
                raise ValueError("Duplicate, invalid, or empty course: " + course_id)
            by_id[course_id] = {"split": split, "report": report}
            split_order[split].append(course_id)
    return by_id, split_order, hashlib.sha256(annotation_bytes).hexdigest()


def load_manifest(path, annotation, annotation_path, annotation_hash, reports_root):
    manifest_bytes = path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    rows = manifest.get("clips") if isinstance(manifest, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must contain a nonempty clips list")
    config = manifest.get("config")
    if isinstance(config, dict):
        if (config.get("annotation_sha256") is not None and
                config["annotation_sha256"] != annotation_hash):
            raise ValueError("Manifest was built from another annotation version")
        if (config.get("annotation_path") is not None and
                Path(config["annotation_path"]).resolve() != annotation_path):
            raise ValueError("Manifest annotation path mismatch")
        if (config.get("reports_root") is not None and
                Path(config["reports_root"]).resolve() != reports_root):
            raise ValueError("Manifest report root mismatch")
    grouped = defaultdict(list)
    seen_clip_paths = set()
    seen_clip_indices = set()
    expected_report_dir = reports_root
    for row_number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError("Invalid clip row {}".format(row_number))
        course_id = str(row.get("course_id", ""))
        if course_id not in annotation:
            raise ValueError("Unknown course in manifest: " + course_id)
        expected = annotation[course_id]
        if row.get("split") != expected["split"]:
            raise ValueError("Split mismatch for course " + course_id)
        report_path = require_file(row.get("report_path"), "report_path")
        wanted_path = expected_report_dir / (course_id + ".txt")
        if report_path != wanted_path.resolve(strict=True):
            raise ValueError("Report path mismatch for course " + course_id)
        report_hash = sha256_file(report_path)
        if row.get("report_sha256") != report_hash:
            raise ValueError("Report SHA-256 mismatch for course " + course_id)
        report_file_text = report_path.read_text(encoding="utf-8")
        report = expected["report"]
        if report_file_text not in (report, report + "\n", report + "\r\n"):
            raise ValueError("Source TXT differs from annotation report: " + course_id)
        if (row.get("annotation_report_sha256") is not None and
                row["annotation_report_sha256"] != sha256_text(report)):
            raise ValueError("Annotation report SHA-256 mismatch: " + course_id)

        clip_path = require_file(row.get("clip_path"), "clip_path")
        source_video = require_file(row.get("source_video"), "source_video")
        if clip_path.suffix.lower() != ".mp4" or source_video.suffix.lower() != ".mp4":
            raise ValueError("Expected MP4 clip and source video: " + course_id)
        if source_video.parent.name != course_id:
            raise ValueError("Source video belongs to another course: " + course_id)
        source_stat = source_video.stat()
        if (row.get("source_bytes") is not None and
                row["source_bytes"] != source_stat.st_size):
            raise ValueError("Source video size changed: " + str(source_video))
        if (row.get("source_mtime_ns") is not None and
                row["source_mtime_ns"] != source_stat.st_mtime_ns):
            raise ValueError("Source video modification time changed: " + str(source_video))
        if str(clip_path) in seen_clip_paths:
            raise ValueError("Duplicate clip path: " + str(clip_path))
        seen_clip_paths.add(str(clip_path))
        if row.get("clip_sha256") != sha256_file(clip_path):
            raise ValueError("Clip SHA-256 mismatch: " + str(clip_path))
        clip_index = row.get("clip_index")
        if isinstance(clip_index, bool) or not isinstance(clip_index, int) or clip_index < 0:
            raise ValueError("Invalid clip_index for course " + course_id)
        if (course_id, clip_index) in seen_clip_indices:
            raise ValueError("Duplicate clip_index for course " + course_id)
        seen_clip_indices.add((course_id, clip_index))
        try:
            start = float(row["start_seconds"])
            end = float(row["end_seconds"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("Invalid clip times for course " + course_id)
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError("Invalid clip interval for course " + course_id)
        if (row.get("source_duration_seconds") is not None and
                end > float(row["source_duration_seconds"]) + 0.5):
            raise ValueError("Clip interval exceeds source video: " + course_id)
        if (row.get("duration_seconds") is not None and
                abs((end - start) - float(row["duration_seconds"])) > 0.5):
            raise ValueError("Clip duration disagrees with interval: " + course_id)
        grouped[course_id].append({
            "clip_index": clip_index,
            "clip_path": str(clip_path),
            "clip_sha256": row["clip_sha256"],
            "source_video": str(source_video),
            "start_seconds": start,
            "end_seconds": end,
            "session": row.get("session"),
            "view": row.get("view"),
        })
    for course_id in grouped:
        grouped[course_id].sort(key=lambda row: row["clip_index"])
    return grouped, hashlib.sha256(manifest_bytes).hexdigest()


def make_dataset(annotation, split_order, clips, reports_root,
                 max_clips, allow_subset):
    if not allow_subset and set(clips) != set(annotation):
        missing = sorted(set(annotation) - set(clips))
        raise ValueError("Manifest does not cover all 76 courses: " + ", ".join(missing))
    examples = {split: [] for split in SPLITS}
    provenance = {split: [] for split in SPLITS}
    for split in SPLITS:
        for course_id in split_order[split]:
            if course_id not in clips:
                continue
            all_clips = clips[course_id]
            selected = all_clips[:max_clips]
            video_lines = "\n".join(
                "片段{}：<video>".format(index)
                for index in range(1, len(selected) + 1))
            report = annotation[course_id]["report"]
            user_content = USER_PROMPT.format(count=len(selected), videos=video_lines)
            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": report},
                ],
                "videos": [row["clip_path"] for row in selected],
            }
            if user_content.count("<video>") != len(example["videos"]):
                raise AssertionError("Video token/path count mismatch")
            examples[split].append(example)
            provenance[split].append({
                "course_id": course_id,
                "split": split,
                "report_path": str(reports_root / (course_id + ".txt")),
                "report_sha256": sha256_text(report),
                "report_characters": len(report),
                "available_clip_count": len(all_clips),
                "selected_clips": selected,
                "omitted_clip_count": len(all_clips) - len(selected),
            })
    if not allow_subset:
        actual = {split: len(examples[split]) for split in SPLITS}
        if actual != EXPECTED_COUNTS:
            raise ValueError("Expected 56/10/10 course split, got " + str(actual))
    return examples, provenance


def write_outputs(output_dir, examples, provenance, manifest, manifest_hash,
                  annotation, annotation_hash, max_clips):
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError("Output directory already exists: " + str(output_dir))
    with tempfile.TemporaryDirectory(prefix="." + output_dir.name + "-",
                                     dir=str(output_dir.parent)) as temporary:
        staging = Path(temporary)
        file_hashes = {}
        for split in SPLITS:
            file = staging / (split + ".jsonl")
            file.write_text("".join(json_text(row) for row in examples[split]),
                            encoding="utf-8")
            file_hashes[file.name] = sha256_file(file)
        (staging / "provenance.json").write_text(
            json_text(provenance, pretty=True), encoding="utf-8")
        file_hashes["provenance.json"] = sha256_file(staging / "provenance.json")
        all_records = [row for split in SPLITS for row in provenance[split]]
        metadata = {
            "format": "ms-swift-messages-video-jsonl",
            "model_family": "Qwen2.5-Omni",
            "audio_in_video_required": True,
            "training_environment": "USE_AUDIO_IN_VIDEO=true",
            "source_manifest": str(manifest),
            "source_manifest_sha256": manifest_hash,
            "source_annotation": str(annotation),
            "source_annotation_sha256": annotation_hash,
            "split_counts": {split: len(examples[split]) for split in SPLITS},
            "split_course_ids": {
                split: [row["course_id"] for row in provenance[split]]
                for split in SPLITS
            },
            "max_clips_per_course": max_clips,
            "selected_clip_count": sum(len(row["selected_clips"]) for row in all_records),
            "omitted_clip_count": sum(row["omitted_clip_count"] for row in all_records),
            "target_report_truncation": False,
            "target_token_length_checked": False,
            "minimum_report_characters": min(row["report_characters"] for row in all_records),
            "maximum_report_characters": max(row["report_characters"] for row in all_records),
            "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
            "user_prompt_template_sha256": sha256_text(USER_PROMPT),
            "files_sha256": file_hashes,
            "limitation": (
                "A short AV clip does not establish all claims in its full source report. "
                "Validate factual grounding separately from report-style similarity."
            ),
        }
        (staging / "metadata.json").write_text(
            json_text(metadata, pretty=True), encoding="utf-8")
        os.replace(str(staging), str(output_dir))
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="AV clip manifest.json from prepare_classroom_av.py")
    parser.add_argument("--annotation", type=Path,
                        default=ROOT / "data/classroom/annotation.json")
    parser.add_argument("--reports-root", type=Path,
                        help="Report TXT directory (default: beside annotation)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-clips-per-course", type=int, default=1,
                        help="First N chronological clips/course (default: 1)")
    parser.add_argument("--allow-subset", action="store_true",
                        help="Permit a partial manifest for a CPU smoke test")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_clips_per_course < 1:
        raise ValueError("--max-clips-per-course must be positive")
    manifest = args.manifest.expanduser().resolve(strict=True)
    annotation_path = args.annotation.expanduser().resolve(strict=True)
    reports_root = (args.reports_root or annotation_path.parent / "reports")
    reports_root = reports_root.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    annotation, split_order, annotation_hash = load_annotation(annotation_path)
    clips, manifest_hash = load_manifest(
        manifest, annotation, annotation_path, annotation_hash, reports_root)
    examples, provenance = make_dataset(
        annotation, split_order, clips, reports_root,
        args.max_clips_per_course,
        args.allow_subset)
    metadata = write_outputs(output_dir, examples, provenance, manifest,
                             manifest_hash, annotation_path, annotation_hash,
                             args.max_clips_per_course)
    print(json_text({"output_dir": str(output_dir),
                     "split_counts": metadata["split_counts"],
                     "selected_clip_count": metadata["selected_clip_count"],
                     "omitted_clip_count": metadata["omitted_clip_count"]}).strip())


if __name__ == "__main__":
    main()
