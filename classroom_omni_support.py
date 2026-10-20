"""Shared Omni model, classroom clip, and output helpers for the M2KT adapter."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from build_classroom_omni_dataset import SYSTEM_PROMPT, USER_PROMPT  # noqa: E402


MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
DEFAULT_MODEL = ROOT / "models_pretrained" / "Qwen2.5-Omni-3B"
DEFAULT_DATASET_DIR = ROOT / "data" / "classroom" / "omni_dataset_v3"
DEFAULT_ADAPTER = ROOT / "results" / "classroom_omni" / "lora_v3_10ep" / "checkpoint-63"
SPLITS = ("train", "val", "test")


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json_object(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: {}".format(path))
    return value


def source_clips(args: argparse.Namespace) -> tuple[list[dict], dict]:
    """Resolve exact clips, without loading an assistant target/report."""
    if args.clip:
        if args.dataset_dir:
            raise ValueError("--dataset-dir is only used with --course-id")
        paths = [resolve_path(value) for value in args.clip]
        if len(set(paths)) != len(paths):
            raise ValueError("Repeated --clip path")
        return ([{"clip_path": str(path)} for path in paths],
                {"source": "direct_clips", "course_id": None, "split": None})

    if not args.course_id.isdigit():
        raise ValueError("--course-id must contain digits only")
    dataset_dir = resolve_path(args.dataset_dir or DEFAULT_DATASET_DIR)
    metadata_path = dataset_dir / "metadata.json"
    provenance_path = dataset_dir / "provenance.json"
    metadata = load_json_object(metadata_path)
    if metadata.get("audio_in_video_required") is not True:
        raise ValueError("Dataset metadata does not require audio in video")
    if metadata.get("system_prompt_sha256") != sha256_text(SYSTEM_PROMPT):
        raise ValueError("Dataset system prompt differs from the inference prompt")
    if metadata.get("user_prompt_template_sha256") != sha256_text(USER_PROMPT):
        raise ValueError("Dataset user prompt differs from the inference prompt")
    provenance_hash = sha256_file(provenance_path)
    recorded_hash = metadata.get("files_sha256", {}).get("provenance.json")
    if recorded_hash != provenance_hash:
        raise ValueError("Dataset provenance SHA-256 mismatch")
    provenance = load_json_object(provenance_path)
    matches = []
    for split in SPLITS:
        rows = provenance.get(split)
        if not isinstance(rows, list):
            raise ValueError("Invalid provenance split: " + split)
        for row in rows:
            if row.get("course_id") == args.course_id:
                matches.append((split, row))
    if len(matches) != 1:
        raise ValueError("Expected one provenance record for course {}, found {}".format(
            args.course_id, len(matches)))
    split, record = matches[0]
    if record.get("split") != split:
        raise ValueError("Provenance split mismatch for " + args.course_id)
    clips = record.get("selected_clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("No selected clips for course " + args.course_id)
    for row in clips:
        if not isinstance(row, dict) or not row.get("clip_path"):
            raise ValueError("Invalid selected clip for course " + args.course_id)
    return clips, {
        "source": "dataset_provenance",
        "course_id": args.course_id,
        "split": split,
        "dataset_dir": str(dataset_dir),
        "dataset_metadata_path": str(metadata_path),
        "dataset_metadata_sha256": sha256_file(metadata_path),
        "dataset_provenance_path": str(provenance_path),
        "dataset_provenance_sha256": provenance_hash,
        "source_manifest": metadata.get("source_manifest"),
        "source_manifest_sha256": metadata.get("source_manifest_sha256"),
        "available_clip_count": record.get("available_clip_count"),
        "omitted_clip_count": record.get("omitted_clip_count"),
    }


def validate_av_clip(row: dict) -> dict:
    path = resolve_path(row["clip_path"])
    if path.suffix.lower() != ".mp4" or not path.is_file() or path.stat().st_size == 0:
        raise ValueError("Clip must be a nonempty MP4 file: " + str(path))
    clip_hash = sha256_file(path)
    if row.get("clip_sha256") and row["clip_sha256"] != clip_hash:
        raise ValueError("Clip SHA-256 differs from provenance: " + str(path))
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required to verify MP4 video and audio streams") from exc
    try:
        with av.open(str(path)) as media:
            if not media.streams.video or not media.streams.audio:
                raise ValueError("MP4 must contain both video and audio: " + str(path))
            has_video_frame = False
            has_audio_frame = False
            for frame in media.decode(video=0, audio=0):
                has_video_frame |= isinstance(frame, av.VideoFrame)
                has_audio_frame |= isinstance(frame, av.AudioFrame)
                if has_video_frame and has_audio_frame:
                    break
            if not has_video_frame or not has_audio_frame:
                raise ValueError("MP4 has no decodable video/audio frames: " + str(path))
            duration = (float(media.duration) / av.time_base
                        if media.duration is not None else None)
    except av.FFmpegError as exc:
        raise ValueError("Cannot decode MP4: {} ({})".format(path, exc)) from exc
    return {
        "clip_path": str(path),
        "clip_sha256": clip_hash,
        "clip_size_bytes": path.stat().st_size,
        "audio_and_video_decoded": True,
        "duration_seconds": duration,
        **{key: row.get(key) for key in (
            "clip_index", "source_video", "start_seconds", "end_seconds",
            "session", "view") if key in row},
    }


def make_messages(clip_count: int) -> list[dict]:
    videos = "\n".join("片段{}：<video>".format(index)
                       for index in range(1, clip_count + 1))
    user_prompt = USER_PROMPT.format(count=clip_count, videos=videos)
    if user_prompt.count("<video>") != clip_count:
        raise AssertionError("Video placeholder count mismatch")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def validate_model_and_adapter(args: argparse.Namespace) -> tuple[Path, Path | None]:
    model_path = resolve_path(args.model)
    if not (model_path.is_dir() and (model_path / "config.json").is_file()):
        raise FileNotFoundError("Local Qwen2.5-Omni model missing: " + str(model_path))
    config = load_json_object(model_path / "config.json")
    if config.get("model_type") != "qwen2_5_omni":
        raise ValueError("Expected a Qwen2.5-Omni model: " + str(model_path))
    adapter = resolve_path(args.adapter) if args.adapter else None
    if adapter is not None:
        if not (adapter.is_dir() and (adapter / "adapter_config.json").is_file()):
            raise FileNotFoundError("LoRA adapter_config.json missing: " + str(adapter))
        if not any((adapter / name).is_file() for name in
                   ("adapter_model.safetensors", "adapter_model.bin")):
            raise FileNotFoundError("LoRA adapter weights missing: " + str(adapter))
    return model_path, adapter


def set_omni_environment(args: argparse.Namespace) -> None:
    """These variables must be set before importing ms-swift model templates."""
    os.environ["USE_AUDIO_IN_VIDEO"] = "true"
    os.environ["ENABLE_AUDIO_OUTPUT"] = "0"
    os.environ["FPS_MAX_FRAMES"] = str(args.fps_max_frames)
    os.environ["VIDEO_MAX_PIXELS"] = str(args.video_max_pixels)
    os.environ["VIDEO_MIN_PIXELS"] = str(args.video_max_pixels)
    # The MP4 audio decoder invokes ffmpeg by name. Running a venv Python
    # directly does not otherwise put the venv's bundled ffmpeg on PATH.
    python_bin = str(Path(sys.executable).parent)
    os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to decode audio from MP4 clips")


def package_versions() -> dict[str, str | None]:
    versions = {}
    for package in ("torch", "transformers", "ms-swift"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(path.parent),
            prefix="." + path.name + "-", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def output_path(args: argparse.Namespace, clips: list[dict], source: dict) -> Path:
    if args.output:
        destination = resolve_path(args.output)
    else:
        label = source["course_id"]
        if label is None:
            stem = Path(clips[0]["clip_path"]).stem
            suffix = sha256_text("\n".join(row["clip_path"] for row in clips))[:8]
            label = stem + "_" + suffix
        destination = resolve_path(args.output_dir) / (label + ".txt")
    if destination.suffix.lower() != ".txt":
        raise ValueError("--output must have a .txt suffix")
    if not args.dry_run and not args.overwrite and (
            destination.exists() or destination.with_suffix(".json").exists()):
        raise FileExistsError("Prediction already exists; pass --overwrite: " + str(destination))
    return destination
