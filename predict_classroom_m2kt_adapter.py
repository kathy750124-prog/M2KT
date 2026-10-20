"""Generate a classroom report with the retained Omni LoRA and M2KT memory adapter.

The adapter enhances audio/video token embeddings immediately before the Omni
Thinker text network.  Its only input is the current course's MP4 clips.  In
particular, validation/test reference reports are never loaded for inference.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from classroom_m2kt_adapter import ClassroomM2KTAdapter
from classroom_omni_support import (
    DEFAULT_ADAPTER,
    DEFAULT_MODEL,
    MODEL_ID,
    atomic_write,
    make_messages,
    output_path,
    package_versions,
    resolve_path,
    set_omni_environment,
    sha256_file,
    sha256_text,
    source_clips,
    validate_av_clip,
    validate_model_and_adapter,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_M2KT_CHECKPOINT = (
    ROOT / "results" / "classroom_m2kt_adapter" / "train_v1" / "best_adapter.pt"
)
DEFAULT_OUTPUT_DIR = ROOT / "results" / "classroom_m2kt_adapter" / "predictions"
SECTION_HEADING_RE = re.compile(
    r"(^|[。！？；\n])\s*(?:综合评价|总体评价|可见优点|优点|问题与改进建议|问题与建议|问题与局限|存在问题|改进建议|改进措施|总体结论|结论)\s*[：:]\s*",
    flags=re.MULTILINE,
)
LIST_MARKER_RE = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.、])\s*")


def normalize_single_paragraph(report: str) -> str:
    """Keep the generated observations while removing section layout."""
    report = LIST_MARKER_RE.sub("", report.strip())
    report = SECTION_HEADING_RE.sub(lambda match: match.group(1), report)
    return re.sub(r"\s+", " ", report).strip()


def _core_model(model):
    """Return the Omni model inside the retained PEFT wrapper."""
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    if not hasattr(core, "thinker") and hasattr(core, "base_model"):
        core = core.base_model
    if not hasattr(core, "thinker"):
        raise TypeError("Loaded model has no Qwen2.5-Omni Thinker")
    return core


class MemoryInjection:
    """Apply the trained memory adapter to live Omni audio/video embeddings.

    The first hook captures input token IDs.  The second runs after the Omni
    audio/video encoders have scattered their outputs into ``inputs_embeds``.
    The adapter therefore sees the same token features as in cached training.
    """

    def __init__(self, thinker, adapter: ClassroomM2KTAdapter):
        self.thinker = thinker
        self.adapter = adapter
        self._input_ids = None
        self.enhanced_calls = 0
        self._handles = []

    def __enter__(self):
        self._handles = [
            self.thinker.register_forward_pre_hook(self._capture_ids, with_kwargs=True),
            self.thinker.model.register_forward_pre_hook(self._enhance, with_kwargs=True),
        ]
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._input_ids = None

    def _capture_ids(self, _module, args, kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        self._input_ids = ids
        return args, kwargs

    def _enhance(self, _module, args, kwargs):
        import torch

        if self.enhanced_calls:
            return args, kwargs
        hidden = kwargs.get("inputs_embeds")
        ids = self._input_ids
        if hidden is None or ids is None or ids.shape != hidden.shape[:2]:
            return args, kwargs
        config = self.thinker.config
        audio_mask = ids.eq(config.audio_token_id)
        video_mask = ids.eq(config.video_token_id)
        av_mask = audio_mask | video_mask
        if not torch.any(av_mask):
            return args, kwargs

        clip_index = ids.eq(config.vision_start_token_id).long().cumsum(dim=1) - 1
        clip_index = clip_index.masked_fill(~av_mask, -1)
        if torch.any(clip_index[av_mask] < 0):
            raise ValueError("An audio/video token has no preceding clip marker")
        kwargs["inputs_embeds"] = self.adapter(
            hidden, av_mask, audio_mask, video_mask, clip_index
        )
        self.enhanced_calls += 1
        return args, kwargs


def _load_memory_checkpoint(path: Path, model_path: Path, lora_path: Path):
    import torch

    if not path.is_file():
        raise FileNotFoundError("Trained M2KT adapter checkpoint missing: " + str(path))
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("adapter_config"), dict):
        raise ValueError("Invalid classroom M2KT adapter checkpoint")
    if not isinstance(checkpoint.get("adapter_state_dict"), dict):
        raise ValueError("Checkpoint has no adapter_state_dict")
    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Invalid checkpoint metadata")
    expected = {
        "model_config_sha256": sha256_file(model_path / "config.json"),
        "lora_adapter_config_sha256": sha256_file(lora_path / "adapter_config.json"),
    }
    weight_file = next(
        (lora_path / name for name in ("adapter_model.safetensors", "adapter_model.bin")
         if (lora_path / name).is_file()),
        None,
    )
    if weight_file is None:
        raise FileNotFoundError("LoRA adapter weights missing: " + str(lora_path))
    expected["lora_adapter_weights_sha256"] = sha256_file(weight_file)
    for key, digest in expected.items():
        if metadata.get(key) != digest:
            raise ValueError("Checkpoint source fingerprint differs: " + key)

    adapter = ClassroomM2KTAdapter(**checkpoint["adapter_config"])
    adapter.load_state_dict(checkpoint["adapter_state_dict"], strict=True)
    adapter.eval()
    return adapter, metadata


def predict(args: argparse.Namespace) -> Path | None:
    import torch

    if (args.max_new_tokens < 64 or args.fps_max_frames < 1 or
            args.video_max_pixels < 1024 or args.repetition_penalty < 1.0):
        raise ValueError("Invalid generation or video setting")
    source_rows, source = source_clips(args)
    clips = [validate_av_clip(row) for row in source_rows]
    if len(clips) > 8:
        raise ValueError("The classroom M2KT adapter supports at most eight clips")
    model_path, lora_path = validate_model_and_adapter(args)
    if lora_path is None:
        raise ValueError("Scheme 2 requires the retained Omni LoRA adapter")
    memory_path = resolve_path(args.m2kt_checkpoint)
    destination = output_path(args, clips, source)
    messages = make_messages(len(clips))
    if args.dry_run:
        print(json.dumps({
            "course_id": source["course_id"], "split": source["split"],
            "clips": [row["clip_path"] for row in clips],
            "model": str(model_path), "lora": str(lora_path),
            "m2kt_checkpoint": str(memory_path), "output": str(destination),
            "report_style": args.report_style,
            "user_prompt_sha256": sha256_text(messages[1]["content"]),
        }, ensure_ascii=False, indent=2))
        return None

    memory, train_metadata = _load_memory_checkpoint(memory_path, model_path, lora_path)
    set_omni_environment(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; classroom M2KT inference was not run")
    from swift import InferRequest, RequestConfig, TransformersEngine

    quantization_config = None
    if args.quantize_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    engine = TransformersEngine(
        str(model_path), adapters=[str(lora_path)], torch_dtype=torch.bfloat16,
        attn_impl=args.attn_impl, device_map="auto", max_batch_size=1,
        quantization_config=quantization_config,
    )
    thinker = _core_model(engine.model).thinker
    memory.to(next(thinker.model.parameters()).device)
    request = InferRequest(messages=messages, videos=[row["clip_path"] for row in clips])
    config = RequestConfig(
        max_tokens=args.max_new_tokens,
        temperature=0,
        repetition_penalty=(None if args.repetition_penalty == 1.0
                            else args.repetition_penalty),
    )
    with torch.inference_mode(), MemoryInjection(thinker, memory) as injection:
        responses = engine.infer([request], config)
    if injection.enhanced_calls != 1:
        raise RuntimeError("Expected one M2KT-enhanced prompt pass; got " +
                           str(injection.enhanced_calls))
    if len(responses) != 1 or not responses[0].choices:
        raise RuntimeError("Omni model returned no response")
    choice = responses[0].choices[0]
    report = choice.message.content
    if not isinstance(report, str) or not report.strip():
        raise RuntimeError("Omni model returned an empty report")
    raw_report = report.strip()
    report = (normalize_single_paragraph(raw_report)
              if args.report_style == "single-paragraph" else raw_report)
    if not report:
        raise RuntimeError("Report became empty after formatting")
    record = {
        "schema_version": 1,
        "kind": "qwen2_5_omni_m2kt_classroom_report",
        "model_id": MODEL_ID,
        "model_path": str(model_path),
        "lora_path": str(lora_path),
        "m2kt_checkpoint": str(memory_path),
        "m2kt_checkpoint_sha256": sha256_file(memory_path),
        "m2kt_best_val_loss": train_metadata.get("best_val_loss"),
        "source": source,
        "clips": clips,
        "system_prompt_sha256": sha256_text(messages[0]["content"]),
        "user_prompt_sha256": sha256_text(messages[1]["content"]),
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0,
            "repetition_penalty": args.repetition_penalty,
            "report_style": args.report_style,
            "formatting_applied": report != raw_report,
            "raw_report_sha256": sha256_text(raw_report),
            "finish_reason": getattr(choice, "finish_reason", None),
            "quantize_4bit": args.quantize_4bit,
            "audio_in_video": True,
            "fps_max_frames": args.fps_max_frames,
            "video_max_pixels": args.video_max_pixels,
            "attn_impl": args.attn_impl,
            "enhanced_prompt_passes": injection.enhanced_calls,
        },
        "report": report,
        "software": package_versions(),
        "caution": "Draft based on selected clips; review the full lesson before formal use.",
    }
    atomic_write(destination.with_suffix(".json"),
                 json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    atomic_write(destination, report + "\n")
    print("Saved " + str(destination), flush=True)
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--course-id", help="Course ID from dataset provenance")
    source.add_argument("--clip", action="append", help="MP4 clip; repeat for each segment")
    parser.add_argument("--dataset-dir", help="Dataset directory for --course-id")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER),
                        help="Retained Omni LoRA checkpoint")
    parser.add_argument("--m2kt-checkpoint", default=str(DEFAULT_M2KT_CHECKPOINT))
    parser.add_argument("--fps-max-frames", type=int, default=12)
    parser.add_argument("--video-max-pixels", type=int, default=50176)
    parser.add_argument("--attn-impl", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--report-style", choices=("single-paragraph", "training"),
                        default="single-paragraph",
                        help="Output one paragraph by default; training returns the raw report")
    parser.add_argument("--quantize-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output", help="Explicit report TXT path")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    predict(parse_args())
