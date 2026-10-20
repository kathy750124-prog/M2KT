"""Cache Omni teacher-forcing targets for the classroom M2KT adapter.

Only train.jsonl and val.jsonl are opened.  The test report file is never
read.  Run with ``M2KT/.venv-omni27/bin/python`` for the pinned ms-swift
environment.  Each cache file retains the exact audio/video token positions
and multimodal rotary positions produced by the Qwen2.5-Omni train template.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent
VERSION = 1
EXPECTED = {"train": 56, "val": 10}


def resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def load_rows(dataset: Path) -> tuple[dict, dict[str, list[dict]], dict[str, str]]:
    metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("audio_in_video_required") is not True:
        raise ValueError("Dataset must require audio inside each video")
    if metadata.get("split_counts", {}) != {"train": 56, "val": 10, "test": 10}:
        raise ValueError("Expected the retained 56/10/10 course split")
    rows: dict[str, list[dict]] = {}
    hashes: dict[str, str] = {}
    ids_seen: set[str] = set()
    paths_seen: set[str] = set()
    for split, count in EXPECTED.items():
        path = dataset / f"{split}.jsonl"
        hashes[split] = sha256_file(path)
        if hashes[split] != metadata.get("files_sha256", {}).get(path.name):
            raise ValueError(f"Dataset {split} hash differs from metadata")
        ids = metadata.get("split_course_ids", {}).get(split)
        if not isinstance(ids, list) or len(ids) != count:
            raise ValueError(f"Missing {split} course IDs")
        split_rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line_number > count:
                    raise ValueError(f"Too many {split} records")
                row = json.loads(line)
                course_id = str(ids[line_number - 1])
                if not course_id.isdigit() or course_id in ids_seen:
                    raise ValueError(f"Invalid or repeated course ID: {course_id}")
                ids_seen.add(course_id)
                messages = row.get("messages")
                videos = row.get("videos")
                if (not isinstance(messages, list) or len(messages) != 3 or
                        [item.get("role") for item in messages] !=
                        ["system", "user", "assistant"] or
                        any(not isinstance(item.get("content"), str) for item in messages) or
                        not messages[2]["content"].strip() or
                        not isinstance(videos, list) or len(videos) != 2 or
                        messages[1]["content"].count("<video>") != len(videos)):
                    raise ValueError(f"Malformed {split} sample: {course_id}")
                for video in videos:
                    clip = Path(video)
                    if (not isinstance(video, str) or not clip.is_absolute() or
                            clip.parent.name != course_id or clip.suffix.lower() != ".mp4" or
                            not clip.is_file() or video in paths_seen):
                        raise ValueError(f"Missing, repeated, or misplaced clip: {video}")
                    paths_seen.add(video)
                split_rows.append(row)
        if len(split_rows) != count:
            raise ValueError(f"Expected {count} {split} records, found {len(split_rows)}")
        rows[split] = split_rows
    if ids_seen & set(metadata.get("split_course_ids", {}).get("test", [])):
        raise ValueError("Train/val course overlaps metadata test IDs")
    return metadata, rows, hashes


def make_rope_calculator(thinker_config, merge_size: int):
    """Use the installed Transformers rope implementation without model weights."""
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniPreTrainedModelForConditionalGeneration as RopeMethods,
    )

    class RopeOnly:
        get_llm_pos_ids_for_vision = RopeMethods.get_llm_pos_ids_for_vision
        get_chunked_index = RopeMethods.get_chunked_index
        get_rope_index = RopeMethods.get_rope_index

    result = RopeOnly()
    result.config = thinker_config
    result.spatial_merge_size = merge_size
    return result


def safe_load(path: Path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def encode_one(template, rope, row: dict, course_id: str, split: str,
               source_hash: str, dataset_hash: str, model_hash: str,
               dataset_metadata_hash: str, dataset_provenance_hash: str,
               settings: dict) -> dict:
    import torch

    encoded = template.encode(row)
    if not isinstance(encoded, dict):
        raise ValueError(f"Template returned multiple encodings for {course_id}")
    input_ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long).flatten()
    labels = torch.as_tensor(encoded["labels"], dtype=torch.long).flatten()
    if input_ids.ndim != 1 or input_ids.numel() != labels.numel():
        raise ValueError(f"Token/label length mismatch for {course_id}")
    if input_ids.numel() > settings["max_length"]:
        raise ValueError(f"Course {course_id} exceeds max_length")
    supervised = labels.ne(-100)
    if not supervised.any() or supervised.sum().item() < 20:
        raise ValueError(f"Course {course_id} has no meaningful supervised report tokens")
    if supervised[0]:
        raise ValueError(f"Course {course_id} supervises the prompt")
    if torch.any(labels[supervised] != input_ids[supervised]):
        raise ValueError(f"Course {course_id} labels do not match input_ids")

    config = template.config.thinker_config
    audio_positions = input_ids.eq(config.audio_token_index).nonzero().flatten()
    video_positions = input_ids.eq(config.video_token_index).nonzero().flatten()
    if not audio_positions.numel() or not video_positions.numel():
        raise ValueError(f"Course {course_id} has no expanded audio or video tokens")
    if supervised[audio_positions].any() or supervised[video_positions].any():
        raise ValueError(f"Course {course_id} supervises AV placeholders")
    video_grid = torch.as_tensor(encoded["video_grid_thw"], dtype=torch.long).cpu()
    seconds = torch.as_tensor(encoded["video_second_per_grid"], dtype=torch.float32).cpu()
    feature_mask = torch.as_tensor(encoded["feature_attention_mask"]).cpu()
    if (video_grid.shape != (len(row["videos"]), 3) or
            seconds.numel() != len(row["videos"]) or
            feature_mask.ndim != 2 or feature_mask.shape[0] != len(row["videos"])):
        raise ValueError(f"Course {course_id} AV metadata is inconsistent")
    audio_feature_lengths = feature_mask.sum(dim=1).to(torch.long)
    audio_token_lengths = ((audio_feature_lengths - 1) // 2 + 1 - 2) // 2 + 1
    expected_audio = int(audio_token_lengths.sum().item())
    merge_size = template.processor.image_processor.merge_size
    expected_video = int((video_grid.prod(dim=1) // (merge_size**2)).sum().item())
    if (audio_positions.numel() != expected_audio or
            video_positions.numel() != expected_video):
        raise ValueError(f"Course {course_id} AV token count does not match processor grids")

    attention_mask = torch.ones_like(input_ids)
    position_ids, rope_deltas = rope.get_rope_index(
        input_ids=input_ids.unsqueeze(0), image_grid_thw=None,
        video_grid_thw=video_grid, attention_mask=attention_mask.unsqueeze(0),
        use_audio_in_video=True, audio_seqlens=audio_feature_lengths,
        second_per_grids=seconds,
    )
    if position_ids.shape != (3, 1, input_ids.numel()):
        raise ValueError(f"Course {course_id} multimodal position shape is wrong")

    return {
        "version": VERSION,
        "course_id": course_id,
        "split": split,
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "position_ids": position_ids.cpu().long(),
        "rope_deltas": rope_deltas.cpu().long(),
        "audio_positions": audio_positions,
        "video_positions": video_positions,
        "audio_feature_lengths": audio_feature_lengths,
        "audio_token_lengths": audio_token_lengths,
        "video_grid_thw": video_grid,
        "video_second_per_grid": seconds,
        "clip_paths": row["videos"],
        "clip_sha256": [sha256_file(Path(path)) for path in row["videos"]],
        "source_sha256": source_hash,
        "dataset_file_sha256": dataset_hash,
        "dataset_metadata_sha256": dataset_metadata_hash,
        "dataset_provenance_sha256": dataset_provenance_hash,
        "report_sha256": hashlib.sha256(
            row["messages"][2]["content"].encode("utf-8")).hexdigest(),
        "model_config_sha256": model_hash,
        "template_settings": settings,
        "supervised_token_count": int(supervised.sum().item()),
    }


def write_atomic(path: Path, value: dict) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                      dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="data/classroom/omni_dataset_v3")
    parser.add_argument("--model", default="models_pretrained/Qwen2.5-Omni-3B")
    parser.add_argument("--output-dir", default="results/classroom_m2kt_adapter/targets")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--fps-max-frames", type=int, default=12)
    parser.add_argument("--video-max-pixels", type=int, default=50176)
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N examples across train then val (smoke runs)")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.max_length < 1024 or args.fps_max_frames < 1 or
            args.video_max_pixels < 1024 or args.limit < 0):
        raise ValueError("Invalid preprocessing parameter")
    dataset, model, output = map(resolve, (args.dataset_dir, args.model, args.output_dir))
    settings = {
        "max_length": args.max_length,
        "fps_max_frames": args.fps_max_frames,
        "video_max_pixels": args.video_max_pixels,
        "video_min_pixels": args.video_max_pixels,
        "use_audio_in_video": True,
        "enable_audio_output": False,
        "template_mode": "train",
        "truncation_strategy": "raise",
    }
    model_config = model / "config.json"
    if not model_config.is_file():
        raise FileNotFoundError(model_config)
    model_hash = sha256_file(model_config)
    metadata, rows, dataset_hashes = load_rows(dataset)
    metadata_hash = sha256_file(dataset / "metadata.json")
    provenance_hash = sha256_file(dataset / "provenance.json")
    if provenance_hash != metadata.get("files_sha256", {}).get("provenance.json"):
        raise ValueError("Dataset provenance hash differs from metadata")
    summary = {"dataset": str(dataset), "model": str(model),
               "output": str(output), "counts": EXPECTED,
               "settings": settings, "dry_run": args.dry_run}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if args.dry_run:
        return

    # These environment values must be established before importing ms-swift.
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": "",
        "USE_AUDIO_IN_VIDEO": "true",
        "ENABLE_AUDIO_OUTPUT": "0",
        "VIDEO_MAX_PIXELS": str(args.video_max_pixels),
        "VIDEO_MIN_PIXELS": str(args.video_max_pixels),
        "FPS_MAX_FRAMES": str(args.fps_max_frames),
        "TOKENIZERS_PARALLELISM": "false",
    })
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    from swift import get_processor, get_template

    processor = get_processor(str(model), download_model=False)
    template = get_template(processor, max_length=args.max_length,
                            truncation_strategy="raise")
    template.set_mode("train")
    rope = make_rope_calculator(template.config.thinker_config,
                                template.processor.image_processor.merge_size)
    completed = skipped = 0
    for split in ("train", "val"):
        ids = metadata["split_course_ids"][split]
        for course_id, row in zip(ids, rows[split]):
            if args.limit and completed + skipped >= args.limit:
                break
            clips = [{"path": path, "sha256": sha256_file(Path(path))}
                     for path in row["videos"]]
            source_hash = sha256_json({
                "version": VERSION,
                "course_id": course_id,
                "split": split,
                "row": row,
                "clips": clips,
                "dataset_file_sha256": dataset_hashes[split],
                "dataset_metadata_sha256": metadata_hash,
                "dataset_provenance_sha256": provenance_hash,
                "model_config_sha256": model_hash,
                "template_settings": settings,
            })
            path = output / split / f"{course_id}.pt"
            if path.exists():
                prior = safe_load(path)
                if (prior.get("version") == VERSION and
                        prior.get("source_sha256") == source_hash and
                        prior.get("course_id") == course_id and
                        prior.get("split") == split):
                    skipped += 1
                    print(f"skip {split}/{course_id}", flush=True)
                    continue
                raise ValueError(f"Stale target cache; move or remove it: {path}")
            value = encode_one(template, rope, row, course_id, split,
                               source_hash, dataset_hashes[split], model_hash,
                               metadata_hash, provenance_hash, settings)
            if value["clip_sha256"] != [clip["sha256"] for clip in clips]:
                raise ValueError(f"Clip changed during processing: {course_id}")
            write_atomic(path, value)
            completed += 1
            print(f"saved {split}/{course_id}: {len(value['input_ids'])} tokens, "
                  f"{value['supervised_token_count']} supervised", flush=True)
    print(json.dumps({"saved": completed, "skipped": skipped,
                      "output": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
