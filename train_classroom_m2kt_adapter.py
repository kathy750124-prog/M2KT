"""Train a small classroom M2KT memory path on frozen Omni AV features.

The original 56/10/10 course split is immutable.  Only train reports contribute
gradients; validation reports select the checkpoint; test reports are never read.
Qwen2.5-Omni and its retained LoRA are frozen.  Cached audio/video tower outputs
are inserted at their exact multimodal token positions, enhanced by the small
M2KT memory adapter, and passed through the frozen Thinker for report loss.

Prepare the feature and target caches first.  Run ``--dry-run`` to validate
their provenance and alignment without loading the GPU model.  ``--smoke``
uses one training and one validation course for a single optimizer update.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
EXPECTED_COUNTS = {"train": 56, "val": 10, "test": 10}
FEATURE_WIDTH = 2048
ADAPTER_CONFIG = {
    "hidden_size": FEATURE_WIDTH,
    "d_model": 256,
    "num_memory_slots": 16,
    "num_heads": 4,
    "max_clips": 8,
    "dropout": 0.1,
}


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"Invalid SHA-256 for {name}")
    return value


def atomic_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=".writing-", delete=False
    ) as stream:
        temp_path = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    os.replace(temp_path, path)


def atomic_checkpoint(path: Path, payload: dict) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=".writing-", delete=False
    ) as stream:
        temp_path = Path(stream.name)
        try:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    os.replace(temp_path, path)


def _load_tensor_dict(path: Path) -> dict:
    import torch

    if not path.is_file():
        raise FileNotFoundError(path)
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a tensor dictionary: {path}")
    return value


def _long_vector(record: dict, key: str, path: Path):
    import torch

    value = record.get(key)
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.dtype != torch.long:
        raise ValueError(f"{path}: {key} must be a one-dimensional int64 tensor")
    return value


def _feature_matrix(record: dict, key: str, count: int, path: Path):
    import torch

    value = record.get(key)
    if (not isinstance(value, torch.Tensor) or value.ndim != 2 or
            value.shape != (count, FEATURE_WIDTH) or
            value.dtype not in (torch.float16, torch.bfloat16, torch.float32) or
            not torch.isfinite(value).all()):
        raise ValueError(f"{path}: {key} must be finite [{count}, {FEATURE_WIDTH}]")
    return value


def _check_positions(positions, length: int, name: str, path: Path) -> None:
    import torch

    if (positions.numel() < 1 or torch.any(positions < 0) or
            torch.any(positions >= length) or
            (positions.numel() > 1 and not torch.all(positions[1:] > positions[:-1]))):
        raise ValueError(f"{path}: invalid or unordered {name}")


def validate_pair(
    feature: dict,
    target: dict,
    *,
    course_id: str,
    split: str,
    feature_path: Path,
    target_path: Path,
    provenance: dict,
    dataset_hashes: dict,
    model_config_sha256: str,
    token_ids: dict[str, int],
) -> None:
    """Reject swapped labels, stale features, and multimodal alignment errors."""
    import torch

    if feature.get("schema_version") != 1 or target.get("version") != 1:
        raise ValueError(f"Unsupported cache schema for course {course_id}")
    for name, row, path in (("feature", feature, feature_path), ("target", target, target_path)):
        if row.get("course_id") != course_id or row.get("split") != split:
            raise ValueError(f"{path}: {name} course/split differs from dataset")
        if row.get("model_config_sha256") != model_config_sha256:
            raise ValueError(f"{path}: base model config fingerprint differs")
        for key in ("dataset_metadata_sha256", "dataset_provenance_sha256"):
            if row.get(key) != dataset_hashes[key]:
                raise ValueError(f"{path}: {key} differs from dataset")
        expected_clips = [clip["clip_sha256"] for clip in provenance["selected_clips"]]
        if row.get("clip_sha256") != expected_clips:
            raise ValueError(f"{path}: clip hashes differ from course provenance")

    if target.get("dataset_file_sha256") != dataset_hashes[f"{split}.jsonl"]:
        raise ValueError(f"{target_path}: supervised dataset file hash differs")
    if target.get("report_sha256") != provenance.get("report_sha256"):
        raise ValueError(f"{target_path}: report target hash differs from provenance")

    prompt = _long_vector(feature, "prompt_input_ids", feature_path)
    ids = _long_vector(target, "input_ids", target_path)
    labels = _long_vector(target, "labels", target_path)
    attention = _long_vector(target, "attention_mask", target_path)
    if (prompt.numel() < 1 or ids.numel() <= prompt.numel() or
            labels.shape != ids.shape or attention.shape != ids.shape or
            not torch.equal(ids[:prompt.numel()], prompt)):
        raise ValueError(f"{target_path}: target does not have the exact feature prompt prefix")
    if not torch.all(attention == 1):
        raise ValueError(f"{target_path}: cached single-course inputs must be unpadded")
    supervised = labels != -100
    if (not supervised.any() or supervised[:prompt.numel()].any() or
            not torch.equal(labels[supervised], ids[supervised])):
        raise ValueError(f"{target_path}: labels must supervise only the report suffix")
    position_ids = target.get("position_ids")
    if (not isinstance(position_ids, torch.Tensor) or
            position_ids.dtype != torch.long or
            position_ids.shape != (3, 1, ids.numel())):
        raise ValueError(f"{target_path}: expected multimodal position_ids [3,1,S]")

    audio_pos = _long_vector(feature, "audio_positions", feature_path)
    video_pos = _long_vector(feature, "video_positions", feature_path)
    _check_positions(audio_pos, prompt.numel(), "audio_positions", feature_path)
    _check_positions(video_pos, prompt.numel(), "video_positions", feature_path)
    if not torch.equal(audio_pos, _long_vector(target, "audio_positions", target_path)):
        raise ValueError(f"{course_id}: audio token positions differ between caches")
    if not torch.equal(video_pos, _long_vector(target, "video_positions", target_path)):
        raise ValueError(f"{course_id}: video token positions differ between caches")
    if (not torch.equal(torch.where(ids == token_ids["audio"])[0], audio_pos) or
            not torch.equal(torch.where(ids == token_ids["video"])[0], video_pos)):
        raise ValueError(f"{course_id}: modality positions differ from Omni token IDs")
    if torch.isin(audio_pos, video_pos).any():
        raise ValueError(f"{course_id}: an AV position was assigned twice")
    _feature_matrix(feature, "audio_features", audio_pos.numel(), feature_path)
    _feature_matrix(feature, "video_features", video_pos.numel(), feature_path)

    clip_count = len(provenance["selected_clips"])
    if clip_count > ADAPTER_CONFIG["max_clips"]:
        raise ValueError(f"{course_id}: adapter has too few clip embeddings")
    clip_ids = (ids == token_ids["vision_start"]).long().cumsum(0) - 1
    for modality, positions in (("audio", audio_pos), ("video", video_pos)):
        expected = clip_ids[positions]
        cached = _long_vector(feature, f"{modality}_clip_index", feature_path)
        if (cached.shape != expected.shape or not torch.equal(cached, expected) or
                torch.any(expected < 0) or torch.any(expected >= clip_count)):
            raise ValueError(f"{course_id}: {modality} clip indexes differ from Omni prompt")
    for key, expected_rows in (("video_grid_thw", clip_count),
                               ("video_second_per_grid", clip_count),
                               ("audio_feature_lengths", clip_count)):
        value = target.get(key)
        if not isinstance(value, torch.Tensor) or value.shape[0] != expected_rows:
            raise ValueError(f"{target_path}: malformed {key}")


def validate_and_load(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    """Read only train/val targets; the test JSONL is never opened."""
    from classroom_m2kt_data import validate_dataset_identity

    dataset_dir = resolve(args.dataset_dir)
    features_dir = resolve(args.features_dir)
    targets_dir = resolve(args.targets_dir)
    model_dir = resolve(args.model)
    lora_dir = resolve(args.lora)

    metadata, provenance = validate_dataset_identity(dataset_dir, check_clips=False)
    if metadata.get("split_counts") != EXPECTED_COUNTS:
        raise ValueError("The original 56/10/10 course split is required")
    model_config_path = model_dir / "config.json"
    lora_config_path = lora_dir / "adapter_config.json"
    lora_weights_path = lora_dir / "adapter_model.safetensors"
    for path in (model_config_path, lora_config_path, lora_weights_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    if model_config.get("model_type") != "qwen2_5_omni":
        raise ValueError("Expected a local Qwen2.5-Omni model")
    thinker_config = model_config.get("thinker_config", {})
    text_config = thinker_config.get("text_config", {})
    if text_config.get("hidden_size") != FEATURE_WIDTH:
        raise ValueError("Omni Thinker hidden size differs from cached feature width")
    token_ids = {
        "audio": thinker_config.get("audio_token_index"),
        "video": thinker_config.get("video_token_index"),
        "vision_start": thinker_config.get("vision_start_token_id"),
    }
    if any(not isinstance(value, int) for value in token_ids.values()):
        raise ValueError("Omni model config lacks modality token IDs")
    model_hash = sha256_file(model_config_path)
    dataset_hashes = {
        "dataset_metadata_sha256": sha256_file(dataset_dir / "metadata.json"),
        "dataset_provenance_sha256": sha256_file(dataset_dir / "provenance.json"),
    }
    for split in ("train", "val"):
        path = dataset_dir / f"{split}.jsonl"
        actual = sha256_file(path)
        if actual != metadata.get("files_sha256", {}).get(path.name):
            raise ValueError(f"{path}: changed since dataset build")
        dataset_hashes[path.name] = actual

    samples: dict[str, list[dict]] = {"train": [], "val": []}
    cache_hashes: dict[str, dict[str, dict[str, str]]] = {"features": {}, "targets": {}}
    for split in ("train", "val"):
        all_ids = metadata["split_course_ids"][split]
        selected_ids = all_ids[:1] if args.smoke else all_ids
        cache_hashes["features"][split] = {}
        cache_hashes["targets"][split] = {}
        if [row["course_id"] for row in provenance[split]] != all_ids:
            raise ValueError(f"{split}: provenance and metadata course order differ")
        rows_by_id = {row["course_id"]: row for row in provenance[split]}
        for course_id in selected_ids:
            feature_path = features_dir / split / f"{course_id}.pt"
            target_path = targets_dir / split / f"{course_id}.pt"
            feature = _load_tensor_dict(feature_path)
            target = _load_tensor_dict(target_path)
            validate_pair(
                feature, target, course_id=course_id, split=split,
                feature_path=feature_path, target_path=target_path,
                provenance=rows_by_id[course_id], dataset_hashes=dataset_hashes,
                model_config_sha256=model_hash, token_ids=token_ids,
            )
            cache_hashes["features"][split][course_id] = sha256_file(feature_path)
            cache_hashes["targets"][split][course_id] = sha256_file(target_path)
            samples[split].append({
                "course_id": course_id,
                "feature": feature,
                "target": target,
            })

    fingerprints = {
        "model_dir": str(model_dir),
        "model_config_sha256": model_hash,
        "lora_dir": str(lora_dir),
        "lora_adapter_config_sha256": sha256_file(lora_config_path),
        "lora_adapter_weights_sha256": sha256_file(lora_weights_path),
        "dataset_dir": str(dataset_dir),
        **dataset_hashes,
        "features_dir": str(features_dir),
        "targets_dir": str(targets_dir),
        "source_cache_sha256": cache_hashes,
        "train_ids": metadata["split_course_ids"]["train"],
        "val_ids": metadata["split_course_ids"]["val"],
        "used_train_ids": [sample["course_id"] for sample in samples["train"]],
        "used_val_ids": [sample["course_id"] for sample in samples["val"]],
        "held_out_test_count": EXPECTED_COUNTS["test"],
    }
    return samples, fingerprints, token_ids


def _prepare_inputs(sample: dict, thinker, adapter, token_ids: dict[str, int]):
    import torch

    feature = sample["feature"]
    target = sample["target"]
    embedding_device = thinker.get_input_embeddings().weight.device
    ids = target["input_ids"].unsqueeze(0).to(embedding_device)
    with torch.no_grad():
        inputs_embeds = thinker.get_input_embeddings()(ids).clone()
    audio_pos = feature["audio_positions"].to(embedding_device)
    video_pos = feature["video_positions"].to(embedding_device)
    inputs_embeds[0, audio_pos] = feature["audio_features"].to(
        device=embedding_device, dtype=inputs_embeds.dtype
    )
    inputs_embeds[0, video_pos] = feature["video_features"].to(
        device=embedding_device, dtype=inputs_embeds.dtype
    )

    audio_mask = ids == token_ids["audio"]
    video_mask = ids == token_ids["video"]
    av_mask = audio_mask | video_mask
    clip_index = (ids == token_ids["vision_start"]).long().cumsum(dim=1) - 1
    clip_index = torch.where(av_mask, clip_index, -1)
    if not torch.any(av_mask):
        raise ValueError(f"{sample['course_id']}: no AV evidence in training input")
    enhanced = adapter(
        inputs_embeds, av_mask, audio_mask=audio_mask,
        video_mask=video_mask, clip_index=clip_index,
    )
    return {
        "input_ids": ids,
        "inputs_embeds": enhanced,
        "attention_mask": target["attention_mask"].unsqueeze(0).to(embedding_device),
        "position_ids": target["position_ids"].to(embedding_device),
        "labels": target["labels"].unsqueeze(0).to(embedding_device),
        "use_cache": False,
    }


def _loss_for_sample(sample: dict, thinker, adapter, token_ids: dict[str, int]):
    import torch

    kwargs = _prepare_inputs(sample, thinker, adapter, token_ids)
    result = thinker(**kwargs)
    if result.loss is None or not bool(torch.isfinite(result.loss).item()):
        raise FloatingPointError(f"{sample['course_id']}: non-finite or missing report loss")
    supervised_tokens = int((sample["target"]["labels"][1:] != -100).sum().item())
    if supervised_tokens < 1:
        raise ValueError(f"{sample['course_id']}: no predicted report tokens")
    return result.loss, supervised_tokens


def _evaluate(samples: list[dict], thinker, adapter, token_ids: dict[str, int]) -> float:
    import torch

    adapter.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for sample in samples:
            loss, count = _loss_for_sample(sample, thinker, adapter, token_ids)
            total_loss += float(loss.item()) * count
            total_tokens += count
    if total_tokens < 1:
        raise ValueError("Validation has no supervised report tokens")
    return total_loss / total_tokens


def _load_frozen_thinker(args: argparse.Namespace):
    import torch

    # The cached AV tokens already contain the media encoding.  Swift still
    # constructs the Omni model and template, so keep its text-only output and
    # media settings identical to the retained classroom LoRA run.
    os.environ["ENABLE_AUDIO_OUTPUT"] = "0"
    os.environ["USE_AUDIO_IN_VIDEO"] = "true"
    os.environ["FPS_MAX_FRAMES"] = "12"
    os.environ["VIDEO_MAX_PIXELS"] = "50176"
    os.environ["VIDEO_MIN_PIXELS"] = "50176"
    from peft import PeftModel
    from swift import TransformersEngine
    from transformers import BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen Omni Thinker training run")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    engine = TransformersEngine(
        str(resolve(args.model)), adapters=[str(resolve(args.lora))],
        torch_dtype=torch.bfloat16, attn_impl=args.attn_impl,
        device_map="auto", max_batch_size=1,
        quantization_config=quantization,
    )
    model = engine.model
    if not isinstance(model, PeftModel):
        raise TypeError("Expected ms-swift to load the retained LoRA as a PeftModel")
    core = model.get_base_model()
    thinker = getattr(core, "thinker", None)
    if thinker is None:
        raise TypeError("Loaded Omni base model does not expose a Thinker")
    model.eval()
    model.requires_grad_(False)
    thinker.config.use_cache = False
    thinker.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    return engine, thinker


def train(args: argparse.Namespace, samples: dict, fingerprints: dict,
          token_ids: dict[str, int]) -> dict:
    import torch
    from torch import nn
    from classroom_m2kt_adapter import ClassroomM2KTAdapter

    class IdentityAdapter(nn.Module):
        def forward(self, hidden_states, _av_mask, **_kwargs):
            return hidden_states

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    engine, thinker = _load_frozen_thinker(args)
    embedding_device = thinker.get_input_embeddings().weight.device
    if embedding_device.type != "cuda":
        raise RuntimeError("Omni Thinker embeddings were offloaded from GPU")
    adapter = ClassroomM2KTAdapter(**ADAPTER_CONFIG).to(embedding_device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if any(parameter.requires_grad for parameter in engine.model.parameters()):
        raise AssertionError("Omni base or LoRA parameters are trainable")
    if not all(parameter.requires_grad for parameter in adapter.parameters()):
        raise AssertionError("An adapter parameter was unexpectedly frozen")

    output_dir = resolve(args.output_dir)
    if args.smoke and args.output_dir == "results/classroom_m2kt_adapter/train_v1":
        output_dir = resolve("results/classroom_m2kt_adapter/smoke_train")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Training output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_val_loss = _evaluate(
        samples["val"], thinker, IdentityAdapter(), token_ids
    )
    print(json.dumps({"baseline_val_loss": baseline_val_loss}), flush=True)

    best_val_loss = math.inf
    best_epoch = -1
    history: list[dict] = []
    optimizer_steps = 0
    for epoch in range(1, (1 if args.smoke else args.epochs) + 1):
        adapter.train()
        shuffled = list(samples["train"])
        random.Random(args.seed + epoch).shuffle(shuffled)
        if args.smoke:
            shuffled = shuffled[:1]
        epoch_loss = 0.0
        epoch_tokens = 0
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(shuffled), args.gradient_accumulation):
            chunk = shuffled[start:start + args.gradient_accumulation]
            for sample in chunk:
                loss, count = _loss_for_sample(sample, thinker, adapter, token_ids)
                (loss / len(chunk)).backward()
                epoch_loss += float(loss.detach().item()) * count
                epoch_tokens += count
            grad_norm = torch.nn.utils.clip_grad_norm_(
                adapter.parameters(), args.max_grad_norm, error_if_nonfinite=True
            )
            if not bool(torch.isfinite(grad_norm).item()):
                raise FloatingPointError("Non-finite adapter gradient norm")
            if float(grad_norm.item()) == 0:
                raise FloatingPointError("Adapter received no gradient from report loss")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        if epoch_tokens < 1:
            raise ValueError("Training epoch has no supervised report tokens")
        train_loss = epoch_loss / epoch_tokens
        val_loss = _evaluate(samples["val"], thinker, adapter, token_ids)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "optimizer_steps": optimizer_steps,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        atomic_json(output_dir / "history.json", history)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint_metadata = {
                "schema_version": 1,
                "method": "frozen_omni_lora_plus_trainable_m2kt_memory",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "baseline_val_loss": baseline_val_loss,
                "best_train_loss": train_loss,
                "optimizer_steps": optimizer_steps,
                "seed": args.seed,
                "smoke": args.smoke,
                "dataset_split_counts": EXPECTED_COUNTS,
                "adapter_module_sha256": sha256_file(ROOT / "classroom_m2kt_adapter.py"),
                **fingerprints,
            }
            payload = {
                "adapter_state_dict": {
                    name: value.detach().cpu().clone()
                    for name, value in adapter.state_dict().items()
                },
                "adapter_config": dict(ADAPTER_CONFIG),
                "metadata": checkpoint_metadata,
            }
            atomic_checkpoint(output_dir / "best_adapter.pt", payload)
            atomic_json(output_dir / "best_metadata.json", checkpoint_metadata)
        if args.smoke:
            break

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "baseline_val_loss": baseline_val_loss,
        "improves_val_loss": best_val_loss < baseline_val_loss,
        "epochs_completed": len(history),
        "optimizer_steps": optimizer_steps,
        "checkpoint": str(output_dir / "best_adapter.pt"),
        "checkpoint_sha256": sha256_file(output_dir / "best_adapter.pt"),
        "train_course_count": len(samples["train"]),
        "val_course_count": len(samples["val"]),
        "smoke": args.smoke,
    }
    atomic_json(output_dir / "summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="data/classroom/omni_dataset_v3")
    parser.add_argument("--features-dir", default="results/classroom_m2kt_adapter/features")
    parser.add_argument("--targets-dir", default="results/classroom_m2kt_adapter/targets")
    parser.add_argument("--model", default="models_pretrained/Qwen2.5-Omni-3B")
    parser.add_argument("--lora", default="results/classroom_omni/lora_v3_10ep/checkpoint-63")
    parser.add_argument("--output-dir", default="results/classroom_m2kt_adapter/train_v1")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn-impl", default="sdpa")
    parser.add_argument("--smoke", action="store_true", help="One train/val pair, one update")
    parser.add_argument("--dry-run", action="store_true", help="Validate caches without GPU")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if (args.epochs < 1 or args.learning_rate <= 0 or args.weight_decay < 0 or
            args.gradient_accumulation < 1 or args.max_grad_norm <= 0):
        raise ValueError("Invalid training hyperparameters")
    samples, fingerprints, token_ids = validate_and_load(args)
    print(json.dumps({
        "cache_validation": "passed",
        "train_count": len(samples["train"]),
        "val_count": len(samples["val"]),
        "held_out_test_count": EXPECTED_COUNTS["test"],
        "dry_run": args.dry_run,
        "smoke": args.smoke,
    }, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    print(json.dumps(train(args, samples, fingerprints, token_ids), ensure_ascii=False))


if __name__ == "__main__":
    main()
