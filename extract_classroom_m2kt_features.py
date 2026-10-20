"""Cache Qwen2.5-Omni audio/video encoder tokens for the classroom M2KT adapter.

The only dataset inputs are metadata.json, provenance.json and the selected MP4
clips. Reference reports, split JSONL files and assistant messages are never
opened. The retained LoRA changes the Thinker text layers, not these encoders,
so this cache deliberately uses the local *base* Omni model without an adapter.

Examples::

    M2KT/.venv-omni27/bin/python M2KT/extract_classroom_m2kt_features.py --dry-run
    M2KT/.venv-omni27/bin/python M2KT/extract_classroom_m2kt_features.py --split train
    M2KT/.venv-omni27/bin/python M2KT/extract_classroom_m2kt_features.py --course-id 39170

Each ``{split}/{course_id}.pt`` is written atomically. Repeated runs verify its
identity and tensor layout before skipping it. Use ``--overwrite`` to rebuild.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

from classroom_m2kt_data import (
    DEFAULT_DATASET_DIR,
    SPLITS,
    validate_dataset_identity,
)
from classroom_omni_support import (
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    USER_PROMPT,
    make_messages,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "results" / "classroom_m2kt_adapter" / "features"
SCHEMA_VERSION = 1
FEATURE_WIDTH = 2048


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def set_media_environment(args: argparse.Namespace) -> None:
    # ms-swift reads these when its Omni template is first constructed.
    os.environ["USE_AUDIO_IN_VIDEO"] = "true"
    os.environ["ENABLE_AUDIO_OUTPUT"] = "0"
    os.environ["FPS_MAX_FRAMES"] = str(args.fps_max_frames)
    os.environ["VIDEO_MAX_PIXELS"] = str(args.video_max_pixels)
    os.environ["VIDEO_MIN_PIXELS"] = str(args.video_max_pixels)
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to decode MP4 audio")


def extraction_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "fps_max_frames": args.fps_max_frames,
        "video_max_pixels": args.video_max_pixels,
        "use_audio_in_video": True,
        "enable_audio_output": False,
        "quantize_4bit": args.quantize_4bit,
        "attn_impl": args.attn_impl,
    }


def expected_identity(
    *,
    args: argparse.Namespace,
    row: dict[str, Any],
    metadata_path: Path,
    provenance_path: Path,
    model_path: Path,
) -> dict[str, Any]:
    # Do not copy provenance's report_path, report text or report hashes into
    # the cache. A clip's hash is checked independently before any model work.
    clips = row["selected_clips"]
    return {
        "schema_version": SCHEMA_VERSION,
        "course_id": row["course_id"],
        "split": row["split"],
        "encoder": "Qwen2.5-Omni-3B base audio/video encoders; no LoRA",
        "model_path": str(model_path),
        "model_config_sha256": sha256_file(model_path / "config.json"),
        "adapter_config_sha256": None,
        "dataset_metadata_sha256": sha256_file(metadata_path),
        "dataset_provenance_sha256": sha256_file(provenance_path),
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "user_prompt_template_sha256": sha256_text(USER_PROMPT),
        "clip_sha256": [clip["clip_sha256"] for clip in clips],
        "clip_paths": [clip["clip_path"] for clip in clips],
        "extraction_settings": extraction_settings(args),
    }


def _validate_tensor(name: str, value: Any, shape: tuple[int | None, ...], dtype: torch.dtype) -> None:
    if not isinstance(value, torch.Tensor) or value.dtype != dtype or value.ndim != len(shape):
        raise ValueError(f"Invalid {name} tensor type, dtype or rank")
    for actual, wanted in zip(value.shape, shape):
        if wanted is not None and actual != wanted:
            raise ValueError(f"Invalid {name} shape: {tuple(value.shape)}")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be stored on CPU")


def validate_cache(cache: dict[str, Any], identity: dict[str, Any]) -> None:
    """Validate a loaded cache before treating it as reusable."""
    if not isinstance(cache, dict):
        raise ValueError("Feature cache is not a dictionary")
    for key, expected in identity.items():
        if cache.get(key) != expected:
            raise ValueError(f"Feature cache identity differs at {key}")
    if any("report" in key.lower() or "assistant" in key.lower() for key in cache):
        raise ValueError("Feature cache contains a forbidden report/assistant field")

    ids = cache.get("prompt_input_ids")
    _validate_tensor("prompt_input_ids", ids, (None,), torch.int64)
    if ids.numel() == 0:
        raise ValueError("Empty prompt_input_ids")
    audio = cache.get("audio_features")
    video = cache.get("video_features")
    _validate_tensor("audio_features", audio, (None, FEATURE_WIDTH), torch.float16)
    _validate_tensor("video_features", video, (None, FEATURE_WIDTH), torch.float16)
    if audio.shape[0] == 0 or video.shape[0] == 0:
        raise ValueError("Missing audio or video encoder features")
    if not torch.isfinite(audio).all() or not torch.isfinite(video).all():
        raise ValueError("Nonfinite audio/video features")

    n_clips = len(identity["clip_sha256"])
    for modality, features in (("audio", audio), ("video", video)):
        positions = cache.get(f"{modality}_positions")
        clip_index = cache.get(f"{modality}_clip_index")
        _validate_tensor(f"{modality}_positions", positions, (features.shape[0],), torch.int64)
        _validate_tensor(f"{modality}_clip_index", clip_index, (features.shape[0],), torch.int64)
        if not bool(((positions >= 0) & (positions < ids.numel())).all()):
            raise ValueError(f"{modality} positions outside prompt")
        if not bool((positions[1:] > positions[:-1]).all()):
            raise ValueError(f"{modality} positions are not strictly ordered")
        if not bool(((clip_index >= 0) & (clip_index < n_clips)).all()):
            raise ValueError(f"{modality} clip IDs are outside the course")
        if not bool((clip_index[1:] >= clip_index[:-1]).all()):
            raise ValueError(f"{modality} clip IDs are not ordered")
        token_id = cache.get(f"{modality}_token_id")
        if not isinstance(token_id, int) or not bool((ids[positions] == token_id).all()):
            raise ValueError(f"{modality} positions do not match prompt tokens")
        if int((ids == token_id).sum()) != features.shape[0]:
            raise ValueError(f"{modality} feature count differs from prompt tokens")

    if bool((cache["audio_positions"][:, None] == cache["video_positions"]).any()):
        raise ValueError("Audio/video prompt positions overlap")
    _validate_tensor("video_grid_thw", cache.get("video_grid_thw"), (n_clips, 3), torch.int64)
    _validate_tensor("video_second_per_grid", cache.get("video_second_per_grid"), (n_clips,), torch.float32)
    _validate_tensor("audio_feature_lengths", cache.get("audio_feature_lengths"), (n_clips,), torch.int64)
    if not bool((cache["video_grid_thw"] > 0).all()):
        raise ValueError("Invalid video grid")
    if not bool((cache["video_second_per_grid"] > 0).all()):
        raise ValueError("Invalid video temporal spacing")
    if not bool((cache["audio_feature_lengths"] > 0).all()):
        raise ValueError("Invalid audio lengths")


def atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("Omni encoder has no parameters") from exc


def encode_course(engine: Any, row: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    """Extract aligned, unpooled AV encoder tokens from one inference prompt."""
    from swift import InferRequest

    clips = row["selected_clips"]
    messages = make_messages(len(clips))
    if any(message.get("role") == "assistant" for message in messages):
        raise AssertionError("An assistant/report target entered feature extraction")
    request = InferRequest(messages=messages, videos=identity["clip_paths"])
    encoded = engine.template.encode(request)
    if isinstance(encoded, list) or "labels" in encoded:
        raise ValueError("Expected a single inference prompt without labels")

    prompt_ids = torch.tensor(encoded["input_ids"], dtype=torch.int64)
    thinker = engine.model.thinker
    config = thinker.config
    audio_token_id = int(getattr(config, "audio_token_id", getattr(config, "audio_token_index", -1)))
    video_token_id = int(getattr(config, "video_token_id", getattr(config, "video_token_index", -1)))
    if audio_token_id < 0 or video_token_id < 0 or audio_token_id == video_token_id:
        raise RuntimeError("Unable to determine Omni modality token IDs")
    audio_positions = torch.where(prompt_ids == audio_token_id)[0]
    video_positions = torch.where(prompt_ids == video_token_id)[0]
    if audio_positions.numel() == 0 or video_positions.numel() == 0:
        raise ValueError("Prompt is missing audio or video tokens; check USE_AUDIO_IN_VIDEO")

    input_features = encoded["input_features"]
    feature_mask = encoded["feature_attention_mask"]
    pixel_values_videos = encoded["pixel_values_videos"]
    video_grid_thw = encoded["video_grid_thw"].to(torch.int64).cpu()
    video_second_per_grid = encoded["video_second_per_grid"].to(torch.float32).cpu()
    n_clips = len(clips)
    if (input_features.shape[0] != n_clips or feature_mask.shape[0] != n_clips or
            video_grid_thw.shape != (n_clips, 3) or
            video_second_per_grid.shape != (n_clips,)):
        raise ValueError("Encoded audio/video inputs do not match selected clips")

    audio_device = _module_device(thinker.audio_tower)
    video_device = _module_device(thinker.visual)
    if audio_device.type == "meta" or video_device.type == "meta":
        raise RuntimeError("Omni encoders were not materialized on a device")
    audio_features = []
    video_features = []
    audio_clip_index = []
    video_clip_index = []
    audio_feature_lengths = feature_mask.sum(dim=1).to(torch.int64).cpu()
    patch_offset = 0
    with torch.inference_mode():
        for clip_index in range(n_clips):
            mask = feature_mask[clip_index:clip_index + 1].to(audio_device)
            values = input_features[clip_index:clip_index + 1].to(audio_device)
            audio_output = thinker.get_audio_features(
                values, feature_attention_mask=mask, return_dict=True
            ).last_hidden_state
            if audio_output.ndim != 2 or audio_output.shape[-1] != FEATURE_WIDTH:
                raise ValueError("Unexpected Omni audio encoder output shape")
            audio_features.append(audio_output.detach().to(device="cpu", dtype=torch.float16))
            audio_clip_index.append(torch.full((audio_output.shape[0],), clip_index, dtype=torch.int64))
            del audio_output, values, mask

            grid = video_grid_thw[clip_index:clip_index + 1]
            patch_count = int(grid.prod().item())
            pixels = pixel_values_videos[patch_offset:patch_offset + patch_count]
            if pixels.shape[0] != patch_count:
                raise ValueError("Video patch count differs from its temporal grid")
            patch_offset += patch_count
            video_output = thinker.get_video_features(
                pixels.to(video_device), grid.to(video_device), return_dict=True
            ).pooler_output
            if video_output.ndim != 2 or video_output.shape[-1] != FEATURE_WIDTH:
                raise ValueError("Unexpected Omni video encoder output shape")
            video_features.append(video_output.detach().to(device="cpu", dtype=torch.float16))
            video_clip_index.append(torch.full((video_output.shape[0],), clip_index, dtype=torch.int64))
            del video_output, pixels

    if patch_offset != pixel_values_videos.shape[0]:
        raise ValueError("Unassigned video patches remain after extraction")
    audio_features = torch.cat(audio_features, dim=0)
    video_features = torch.cat(video_features, dim=0)
    if audio_features.shape[0] != audio_positions.numel():
        raise ValueError("Audio encoder token count differs from prompt placeholders")
    if video_features.shape[0] != video_positions.numel():
        raise ValueError("Video encoder token count differs from prompt placeholders")

    result = {
        **identity,
        "prompt_input_ids": prompt_ids,
        "audio_token_id": audio_token_id,
        "video_token_id": video_token_id,
        "audio_features": audio_features,
        "video_features": video_features,
        "audio_positions": audio_positions,
        "video_positions": video_positions,
        "audio_clip_index": torch.cat(audio_clip_index),
        "video_clip_index": torch.cat(video_clip_index),
        "video_grid_thw": video_grid_thw,
        "video_second_per_grid": video_second_per_grid,
        "audio_feature_lengths": audio_feature_lengths,
    }
    validate_cache(result, identity)
    return result


def load_engine(model_path: Path, args: argparse.Namespace) -> Any:
    from swift import TransformersEngine

    quantization_config = None
    if args.quantize_4bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    engine = TransformersEngine(
        str(model_path),
        torch_dtype=torch.bfloat16,
        attn_impl=args.attn_impl,
        device_map="auto",
        max_batch_size=1,
        quantization_config=quantization_config,
    )
    if not hasattr(engine.model, "thinker"):
        raise RuntimeError("Loaded model does not expose the Qwen2.5-Omni Thinker")
    engine.model.eval()
    return engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="all")
    parser.add_argument("--course-id", help="Extract one course from its original split")
    parser.add_argument("--limit", type=int, help="Limit selected courses for a pilot run")
    parser.add_argument("--fps-max-frames", type=int, default=12)
    parser.add_argument("--video-max-pixels", type=int, default=50176)
    parser.add_argument("--attn-impl", default="sdpa")
    parser.add_argument("--no-quantize-4bit", action="store_false", dest="quantize_4bit")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Verify data and plan without loading Omni or writing")
    parser.set_defaults(quantize_4bit=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.fps_max_frames < 1 or args.video_max_pixels < 1024:
        raise ValueError("Invalid video sampling limits")
    if args.course_id is not None and not args.course_id.isdigit():
        raise ValueError("--course-id must contain digits only")
    dataset_dir = resolve(args.dataset_dir)
    model_path = resolve(args.model)
    output_dir = resolve(args.output_dir)
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Local Omni model config missing: {model_path}")
    model_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if model_config.get("model_type") != "qwen2_5_omni":
        raise ValueError("Expected the local Qwen2.5-Omni model")
    if model_config.get("thinker_config", {}).get("text_config", {}).get("hidden_size") != FEATURE_WIDTH:
        raise ValueError("Omni Thinker width is not 2048")

    metadata_path = dataset_dir / "metadata.json"
    provenance_path = dataset_dir / "provenance.json"
    metadata, provenance = validate_dataset_identity(dataset_dir, check_clips=True)
    if (metadata.get("system_prompt_sha256") != sha256_text(SYSTEM_PROMPT) or
            metadata.get("user_prompt_template_sha256") != sha256_text(USER_PROMPT)):
        raise ValueError("Dataset prompt hashes differ from current inference prompts")
    splits = SPLITS if args.split == "all" else (args.split,)
    chosen = [(split, row) for split in splits for row in provenance[split]
              if args.course_id is None or row["course_id"] == args.course_id]
    if args.course_id and len(chosen) != 1:
        raise ValueError("Course ID is absent from the requested split")
    if args.limit is not None:
        chosen = chosen[:args.limit]
    if not chosen:
        raise ValueError("No courses selected")

    work = []
    skipped = 0
    for split, row in chosen:
        identity = expected_identity(
            args=args,
            row=row,
            metadata_path=metadata_path,
            provenance_path=provenance_path,
            model_path=model_path,
        )
        path = output_dir / split / f"{row['course_id']}.pt"
        if path.exists() and not args.overwrite:
            try:
                validate_cache(torch.load(path, map_location="cpu", weights_only=True), identity)
            except Exception as exc:
                raise ValueError(f"Existing feature cache is invalid: {path}; use --overwrite") from exc
            skipped += 1
            continue
        work.append((row, identity, path))
    print(f"courses={len(chosen)} cached={skipped} pending={len(work)} output={output_dir}", flush=True)
    if args.dry_run:
        print("dry-run: provenance, clip hashes, prompts and cache identities verified; no model loaded", flush=True)
        return
    if not work:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; AV encoder features were not extracted")
    set_media_environment(args)
    engine = load_engine(model_path, args)
    for index, (row, identity, path) in enumerate(work, 1):
        result = encode_course(engine, row, identity)
        atomic_torch_save(result, path)
        # Read the atomic result back to catch serialization errors before it
        # is considered complete by a resumed run.
        validate_cache(torch.load(path, map_location="cpu", weights_only=True), identity)
        print(
            f"[{index}/{len(work)}] {row['split']}/{row['course_id']} "
            f"audio={result['audio_features'].shape[0]} "
            f"video={result['video_features'].shape[0]} {path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
