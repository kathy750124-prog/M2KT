"""Prepare short, synchronized classroom video/audio clips for Omni training.

The course-level split and target report come from ``annotation.json``.  Only
the selected seconds of one recording view are decoded for each clip; source
MP4s and earlier frame-based experiments are never modified.  PyAV provides
the decoder and encoder, so an ffmpeg command-line executable is unnecessary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SESSION_RE = re.compile(r"(?:^|_)s(\d+)(?:_|$)", re.IGNORECASE)
VIEW_RE = re.compile(r"(?:^|_)view(\d+)(?:_|$)", re.IGNORECASE)
SPLITS = ("train", "val", "test")
MANIFEST_VERSION = 1
AUDIO_RATE = 16000


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix="." + path.name + "-", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_courses(annotation_path: Path, reports_root: Path) -> list[dict]:
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    if set(annotation) != set(SPLITS):
        raise ValueError("annotation must contain train, val and test splits")
    courses = []
    seen = set()
    for split in SPLITS:
        for item in annotation[split]:
            course_id = str(item["id"])
            if not course_id.isdigit() or course_id in seen:
                raise ValueError("invalid or repeated course ID: " + course_id)
            seen.add(course_id)
            report_path = (reports_root / (course_id + ".txt")).resolve()
            report_bytes = report_path.read_bytes()
            if report_bytes.decode("utf-8-sig").rstrip("\r\n") != item["report"]:
                raise ValueError("source TXT and annotation report differ: " + course_id)
            courses.append({
                "course_id": course_id,
                "split": split,
                "report_path": str(report_path),
                "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
                "annotation_report_sha256": hashlib.sha256(
                    item["report"].encode("utf-8")).hexdigest(),
            })
    return courses


def probe_source(path: Path, session: int, view: int) -> dict | None:
    try:
        with av.open(str(path)) as source:
            if len(source.streams.video) != 1 or len(source.streams.audio) != 1:
                return None
            duration = float(source.duration or 0) / av.time_base
            if duration <= 0:
                return None
    except av.FFmpegError:
        return None
    stat = path.stat()
    return {
        "path": path.resolve(), "session": session, "view": view,
        "duration": duration, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
    }


def select_sources(course_dir: Path, priority: list[int], seconds: float) -> list[dict]:
    if not course_dir.is_dir():
        raise FileNotFoundError(course_dir)
    groups = defaultdict(list)
    for path in sorted(course_dir.glob("*.mp4")):
        session_match = SESSION_RE.search(path.stem)
        view_match = VIEW_RE.search(path.stem)
        if session_match is None or view_match is None:
            continue
        session, view = int(session_match.group(1)), int(view_match.group(1))
        info = probe_source(path, session, view)
        if info is not None and info["duration"] >= seconds + 1:
            groups[session].append(info)
    if not groups:
        raise ValueError("no audio+video MP4 long enough in " + str(course_dir))
    rank = {view: index for index, view in enumerate(priority)}
    return [
        min(groups[session], key=lambda row: (
            rank.get(row["view"], len(rank)), row["view"], row["path"].name))
        for session in sorted(groups)
    ]


def output_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    scale = min(1.0, max_side / max(width, height))
    out_width = max(2, int(width * scale) // 2 * 2)
    out_height = max(2, int(height * scale) // 2 * 2)
    return out_width, out_height


def read_audio_segment(source_path: Path, start: float, seconds: float) -> tuple[np.ndarray, float]:
    sample_count = round(seconds * AUDIO_RATE)
    samples = np.zeros(sample_count, dtype=np.float32)
    with av.open(str(source_path)) as source:
        stream = source.streams.audio[0]
        source.seek(int(start / stream.time_base), stream=stream, backward=True)
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=AUDIO_RATE)
        chunks = []
        first_time = None
        accumulated = 0
        for original in source.decode(stream):
            if original.pts is None:
                continue
            for frame in resampler.resample(original):
                if frame.pts is None:
                    raise ValueError("resampled audio frame lacks timestamp")
                if first_time is None:
                    first_time = float(frame.pts * frame.time_base)
                values = frame.to_ndarray().reshape(-1)
                chunks.append(values)
                accumulated += len(values)
            if first_time is not None:
                offset = round((start - first_time) * AUDIO_RATE)
                if accumulated >= sample_count + max(0, offset):
                    break
    if first_time is None or not chunks:
        raise ValueError("no audio decoded from selected interval")
    # These recordings contain small irregularities in AAC packet PTS.  Decoded
    # AAC frames are continuous in sample order, so anchor at the first decoded
    # timestamp and use sample counts thereafter instead of leaving false gaps.
    decoded = np.concatenate(chunks)
    offset = round((start - first_time) * AUDIO_RATE)
    source_begin = max(0, offset)
    destination_begin = max(0, -offset)
    length = min(len(decoded) - source_begin, sample_count - destination_begin)
    if length > 0:
        samples[destination_begin:destination_begin + length] = decoded[
            source_begin:source_begin + length]
    coverage = max(0.0, length) / sample_count
    if coverage < 0.90:
        raise ValueError("audio covers only {:.1%} of selected interval".format(coverage))
    return samples, coverage


def encode_clip(source_path: Path, destination: Path, start: float,
                seconds: float, fps: int, max_side: int) -> tuple[int, float, float]:
    """Decode a short source interval and encode 2 fps video plus continuous audio."""
    audio, coverage = read_audio_segment(source_path, start, seconds)
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    frame_total = round(seconds * fps)
    if not math.isclose(frame_total / fps, seconds, abs_tol=1e-6):
        raise ValueError("seconds must contain a whole number of output frames")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("." + destination.stem + "-" + next(
        tempfile._get_candidate_names()) + destination.suffix)
    try:
        with av.open(str(source_path)) as source, av.open(str(temporary), "w", format="mp4") as output:
            video_in = source.streams.video[0]
            width, height = output_size(
                video_in.codec_context.width, video_in.codec_context.height,
                max_side)
            video_out = output.add_stream("libx264", rate=fps)
            video_out.width = width
            video_out.height = height
            video_out.pix_fmt = "yuv420p"
            video_out.options = {"preset": "ultrafast", "crf": "27"}
            video_out.codec_context.thread_count = 2
            audio_out = output.add_stream("aac", rate=AUDIO_RATE)
            audio_out.layout = "mono"
            audio_out.bit_rate = 64000

            source.seek(int(start / video_in.time_base), stream=video_in, backward=True)
            frame_index = 0
            for frame in source.decode(video_in):
                if frame.pts is None:
                    continue
                timestamp = float(frame.pts * frame.time_base)
                if timestamp + 1e-5 < start + frame_index / fps:
                    continue
                while frame_index < frame_total and timestamp + 1e-5 >= start + frame_index / fps:
                    prepared = frame.reformat(width=width, height=height, format="yuv420p")
                    prepared.pts = frame_index
                    prepared.time_base = Fraction(1, fps)
                    for packet in video_out.encode(prepared):
                        output.mux(packet)
                    frame_index += 1
                if frame_index >= frame_total:
                    break
            if frame_index != frame_total:
                raise ValueError("only {} of {} video frames sampled".format(
                    frame_index, frame_total))
            for packet in video_out.encode(None):
                output.mux(packet)

            # AAC accepts planar float frames; packet PTS starts at zero in the
            # same timeline as the video frames.  The encoder adds its own
            # priming samples, which MP4 records as an edit list.
            for offset in range(0, len(audio), 1024):
                values = audio[offset:offset + 1024]
                frame = av.AudioFrame.from_ndarray(
                    values.reshape(1, -1), format="fltp", layout="mono")
                frame.sample_rate = AUDIO_RATE
                frame.pts = offset
                frame.time_base = Fraction(1, AUDIO_RATE)
                for packet in audio_out.encode(frame):
                    output.mux(packet)
            for packet in audio_out.encode(None):
                output.mux(packet)
        with av.open(str(temporary)) as check:
            if len(check.streams.video) != 1 or len(check.streams.audio) != 1:
                raise ValueError("encoded clip lacks a video or audio stream")
            duration = float(check.duration or 0) / av.time_base
            if not seconds - 0.25 <= duration <= seconds + 0.25:
                raise ValueError("unexpected encoded duration: {:.3f}".format(duration))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return frame_total, rms, coverage


def load_or_create_manifest(path: Path, config: dict) -> dict:
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != MANIFEST_VERSION or payload.get("config") != config:
            raise ValueError("existing manifest uses different settings; choose a new output directory")
        if not isinstance(payload.get("clips"), list):
            raise ValueError("invalid existing clip manifest")
        return payload
    return {"version": MANIFEST_VERSION, "config": config, "clips": []}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", default="data/classroom/annotation.json")
    parser.add_argument("--videos-root", default="data/classroom/videos")
    parser.add_argument("--reports-root", default="data/classroom/reports")
    parser.add_argument("--output-dir", default="data/classroom/av_clips_v2")
    parser.add_argument("--clip-count", type=int, default=2,
                        help="clips per course (default: 2)")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--max-side", type=int, default=960)
    parser.add_argument("--view-priority", default="7,5,1,3",
                        help="preferred view IDs, first available per session")
    parser.add_argument("--start-fraction", type=float, default=0.25,
                        help="fractional position within each source video")
    parser.add_argument("--ids", default=None,
                        help="comma-separated course IDs; omit for all courses")
    parser.add_argument("--split", choices=("all",) + SPLITS, default="all")
    parser.add_argument("--limit", type=int, default=None,
                        help="maximum courses for a small smoke run")
    args = parser.parse_args()
    if args.clip_count < 1 or args.seconds <= 0 or args.fps < 1 or args.max_side < 64:
        parser.error("clip-count, seconds, fps and max-side must be positive")
    if not 0 <= args.start_fraction < 1:
        parser.error("start-fraction must be in [0, 1)")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if not math.isclose(args.seconds * args.fps, round(args.seconds * args.fps), abs_tol=1e-6):
        parser.error("seconds * fps must be an integer")
    try:
        priority = [int(value.strip()) for value in args.view_priority.split(",")]
    except ValueError:
        parser.error("view-priority must be comma-separated view numbers")
    if not priority or len(set(priority)) != len(priority):
        parser.error("view-priority must contain distinct view numbers")

    annotation_path = resolve_path(args.annotation)
    videos_root = resolve_path(args.videos_root)
    reports_root = resolve_path(args.reports_root)
    output_dir = resolve_path(args.output_dir)
    courses = load_courses(annotation_path, reports_root)
    if args.split != "all":
        courses = [row for row in courses if row["split"] == args.split]
    if args.ids:
        ids = {value.strip() for value in args.ids.split(",") if value.strip()}
        courses = [row for row in courses if row["course_id"] in ids]
        missing = ids - {row["course_id"] for row in courses}
        if missing:
            parser.error("IDs absent from selected split: " + ", ".join(sorted(missing)))
    if args.limit is not None:
        courses = courses[:args.limit]
    if not courses:
        parser.error("no courses selected")
    config = {
        "annotation_path": str(annotation_path),
        "annotation_sha256": sha256_file(annotation_path),
        "videos_root": str(videos_root), "reports_root": str(reports_root),
        "clip_count": args.clip_count, "seconds": args.seconds, "fps": args.fps,
        "max_side": args.max_side, "view_priority": priority,
        "start_fraction": args.start_fraction, "audio_rate": AUDIO_RATE,
    }
    manifest_path = output_dir / "manifest.json"
    manifest = load_or_create_manifest(manifest_path, config)
    existing = {(row["course_id"], row["clip_index"]): row for row in manifest["clips"]}

    for course in courses:
        course_id = course["course_id"]
        sources = select_sources(videos_root / course_id, priority, args.seconds)
        repeat_count = math.ceil(args.clip_count / len(sources))
        for index in range(args.clip_count):
            source = sources[index % len(sources)]
            repeat = index // len(sources)
            fraction = args.start_fraction + repeat * (1 - args.start_fraction) / repeat_count
            start = min(source["duration"] - args.seconds - 0.5,
                        source["duration"] * fraction)
            start = max(0.0, round(start, 3))
            end = round(start + args.seconds, 3)
            path = output_dir / course_id / (
                "{:02d}_s{}_v{}.mp4".format(index, source["session"], source["view"]))
            expected = {
                **course, "clip_index": index, "clip_path": str(path),
                "source_video": str(source["path"]),
                "source_bytes": source["size"],
                "source_mtime_ns": source["mtime_ns"],
                "source_duration_seconds": round(source["duration"], 3),
                "session": source["session"], "view": source["view"],
                "start_seconds": start, "end_seconds": end,
            }
            old = existing.get((course_id, index))
            if old is not None:
                if any(old.get(key) != value for key, value in expected.items()):
                    raise ValueError("existing clip provenance changed: " + str(path))
                if not path.is_file() or sha256_file(path) != old.get("clip_sha256"):
                    raise ValueError("existing clip is missing or changed: " + str(path))
                print("reused", course_id, index, path, flush=True)
                continue
            if path.exists():
                raise FileExistsError("untracked output clip exists: " + str(path))
            frame_count, rms, coverage = encode_clip(
                source["path"], path, start, args.seconds, args.fps, args.max_side)
            record = {
                **expected, "clip_sha256": sha256_file(path),
                "duration_seconds": args.seconds,
                "output_frames": frame_count,
                "audio_rms": round(rms, 8),
                "audio_coverage": round(coverage, 6),
            }
            manifest["clips"].append(record)
            existing[(course_id, index)] = record
            manifest["clips"].sort(key=lambda row: (row["course_id"], row["clip_index"]))
            write_json_atomic(manifest_path, manifest)
            print("created", course_id, index, path, flush=True)
    print("manifest", manifest_path, "clips", len(manifest["clips"]), flush=True)


if __name__ == "__main__":
    main()
