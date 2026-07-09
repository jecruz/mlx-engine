#!/usr/bin/env python3
"""Benchmark LFM2.5-VL text-only generated-token cache reuse."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL_PATH = Path(
    "/Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/"
    "lmstudio-community/LFM2.5-VL-1.6B-MLX-8bit"
)
MODEL_PATH_ENV = "MLX_ENGINE_LFM25_VL_MODEL_PATH"


class CapturePromptProgressReporter:
    """Capture prompt-cache progress events emitted by mlx-engine."""

    def __init__(self) -> None:
        """Initialize an empty prompt-progress event list."""
        self.events: list[dict[str, Any]] = []

    def begin(
        self,
        is_draft: bool,
        cached_tokens: int,
        total_prompt_tokens: int,
        prefill_tokens_processed: int,
    ) -> bool:
        """Record the initial prompt-progress event."""
        self.events.append(
            {
                "type": "begin",
                "is_draft": is_draft,
                "cached_tokens": cached_tokens,
                "total_prompt_tokens": total_prompt_tokens,
                "prefill_tokens_processed": prefill_tokens_processed,
            }
        )
        return True

    def update(self, is_draft: bool, prefill_tokens_processed: int) -> bool:
        """Record an intermediate prompt-progress update."""
        self.events.append(
            {
                "type": "update",
                "is_draft": is_draft,
                "prefill_tokens_processed": prefill_tokens_processed,
            }
        )
        return True

    def finish(
        self,
        is_draft: bool,
        prefill_tokens_processed: int | None = None,
    ) -> bool:
        """Record the final prompt-progress event."""
        self.events.append(
            {
                "type": "finish",
                "is_draft": is_draft,
                "prefill_tokens_processed": prefill_tokens_processed,
            }
        )
        return True


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=(
            "Local LFM2.5-VL model path. Defaults to "
            f"${MODEL_PATH_ENV}, then {DEFAULT_MODEL_PATH}."
        ),
    )
    parser.add_argument("--samples", type=int, default=2, help="Independent samples.")
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=20_000,
        help="max_kv_size passed to load_model.",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=512,
        help="prefill_step_size passed to load_model.",
    )
    parser.add_argument("--story-tokens", type=int, default=512)
    parser.add_argument("--followup-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def resolve_model_path(configured_path: Path | None = None) -> Path:
    """Resolve a local LFM2.5-VL model path without prompting or downloading."""
    candidates: list[Path] = []
    if configured_path is not None:
        candidates.append(configured_path.expanduser())
    env_path = os.environ.get(MODEL_PATH_ENV)
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(DEFAULT_MODEL_PATH)

    pointer_path = Path("~/.lmstudio-home-pointer").expanduser().resolve()
    if pointer_path.exists():
        lmstudio_home = Path(pointer_path.read_text().strip())
        candidates.extend(
            [
                lmstudio_home
                / "models"
                / "lmstudio-community"
                / "LFM2.5-VL-1.6B-MLX-8bit",
                lmstudio_home
                / "models"
                / "lmstudio-community"
                / "LFM2.5-VL-1.6B-MLX-4bit",
            ]
        )

    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("*.safetensors")):
            return candidate.resolve()
    raise FileNotFoundError(
        "No local LFM2.5-VL MLX model found. Pass --model or set "
        f"{MODEL_PATH_ENV}; this benchmark never prompts or downloads."
    )


def first_event(events: list[dict[str, Any]], event_type: str) -> dict[str, Any] | None:
    """Return the first prompt-progress event with the requested type."""
    return next((event for event in events if event.get("type") == event_type), None)


def numeric_summary(values: list[float]) -> dict[str, float | None]:
    """Return min/max/avg for numeric sample values."""
    if not values:
        return {"min": None, "max": None, "avg": None}
    return {"min": min(values), "max": max(values), "avg": mean(values)}


def ratio_summary(
    numerators: list[float],
    denominators: list[float],
) -> dict[str, float | None]:
    """Return min/max/avg for paired numerator/denominator ratios."""
    ratios = [
        numerator / denominator
        for numerator, denominator in zip(numerators, denominators, strict=False)
        if denominator > 0
    ]
    return numeric_summary(ratios)


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize repeated two-turn cache benchmark samples."""
    followups = [sample["followup"] for sample in samples]
    first_turns = [sample["first_turn"] for sample in samples]
    row_errors = sum(1 for sample in samples if sample.get("error"))
    followup_cached = [
        float(row["cached_tokens"])
        for row in followups
        if isinstance(row.get("cached_tokens"), int | float)
    ]
    followup_total = [
        float(row["total_prompt_tokens"])
        for row in followups
        if isinstance(row.get("total_prompt_tokens"), int | float)
    ]
    followup_prefill = [
        float(row["prefill_tokens_processed"])
        for row in followups
        if isinstance(row.get("prefill_tokens_processed"), int | float)
    ]
    return {
        "sample_count": len(samples),
        "row_errors": row_errors,
        "all_followups_cached": bool(samples)
        and row_errors == 0
        and all(row.get("cached_tokens", 0) > 0 for row in followups),
        "all_followups_small_prefill": bool(samples)
        and row_errors == 0
        and all(
            row.get("prefill_tokens_processed", 10**9) < row.get("total_prompt_tokens", 0)
            for row in followups
        ),
        "all_outputs_preserve_name": bool(samples)
        and row_errors == 0
        and all("silas" in row.get("output_text", "").lower() for row in followups),
        "first_turn_ttft_s": numeric_summary(
            [row["ttft_s"] for row in first_turns if row.get("ttft_s") is not None]
        ),
        "followup_ttft_s": numeric_summary(
            [row["ttft_s"] for row in followups if row.get("ttft_s") is not None]
        ),
        "followup_total_s": numeric_summary(
            [row["total_s"] for row in followups if row.get("total_s") is not None]
        ),
        "followup_cached_tokens": numeric_summary(followup_cached),
        "followup_total_prompt_tokens": numeric_summary(followup_total),
        "followup_prefill_tokens_processed": numeric_summary(followup_prefill),
        "followup_cache_reuse_ratio": ratio_summary(
            followup_cached,
            followup_total,
        ),
        "followup_prefill_ratio": ratio_summary(
            followup_prefill,
            followup_total,
        ),
    }


def generation_prompt() -> str:
    """Return the first-turn story prompt for the retained workload."""
    return (
        "<|im_start|>user\n"
        "Tell me a 500-word story about a traveler named Silas. "
        "Keep the name Silas important.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def followup_prompt(story_prompt: str, generated_text: str) -> str:
    """Return the second-turn prompt that should reuse generated-token cache."""
    return (
        story_prompt
        + generated_text
        + "<|im_end|>\n"
        + "<|im_start|>user\n"
        + "What was the main character's name? Answer with only the name.<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )


def run_generation(
    *,
    model_kit: Any,
    prompt: str,
    max_tokens: int,
    request_id: str,
) -> dict[str, Any]:
    """Run one generation request and return text, timing, and cache metrics."""
    from mlx_engine.generate import create_generator, tokenize

    prompt_tokens = tokenize(model_kit, prompt)
    reporter = CapturePromptProgressReporter()
    started = time.perf_counter()
    first_token = None
    token_count = 0
    text_parts: list[str] = []

    for result in create_generator(
        model_kit=model_kit,
        prompt_tokens=prompt_tokens,
        seed=0,
        temp=0.0,
        max_tokens=max_tokens,
        prompt_progress_reporter=reporter,
        request_id=request_id,
    ):
        tokens = getattr(result, "tokens", []) or []
        if tokens and first_token is None:
            first_token = time.perf_counter()
        token_count += len(tokens)
        delta_text = getattr(result, "new_text", None)
        if delta_text is None:
            delta_text = getattr(result, "text", "")
        text_parts.append(delta_text)

    finished = time.perf_counter()
    begin_event = first_event(reporter.events, "begin") or {}
    finish_event = first_event(reporter.events, "finish") or {}
    return {
        "prompt_tokens": len(prompt_tokens),
        "cached_tokens": begin_event.get("cached_tokens"),
        "total_prompt_tokens": begin_event.get("total_prompt_tokens"),
        "prefill_tokens_processed": finish_event.get("prefill_tokens_processed"),
        "completion_tokens": token_count,
        "ttft_s": (first_token - started) if first_token is not None else None,
        "total_s": finished - started,
        "output_text": "".join(text_parts),
    }


def run_sample(
    *,
    model_path: Path,
    sample_index: int,
    max_kv_size: int,
    prefill_step_size: int,
    story_tokens: int,
    followup_tokens: int,
) -> dict[str, Any]:
    """Run one independent two-turn generated-token cache sample."""
    from mlx_engine.generate import load_model, unload

    model_kit = load_model(
        model_path=model_path,
        max_kv_size=max_kv_size,
        prefill_step_size=prefill_step_size,
    )
    try:
        story_prompt = generation_prompt()
        first_turn = run_generation(
            model_kit=model_kit,
            prompt=story_prompt,
            max_tokens=story_tokens,
            request_id=f"lfm25-text-cache-s{sample_index}-story",
        )
        followup = run_generation(
            model_kit=model_kit,
            prompt=followup_prompt(story_prompt, first_turn["output_text"]),
            max_tokens=followup_tokens,
            request_id=f"lfm25-text-cache-s{sample_index}-followup",
        )
        return {
            "sample_index": sample_index,
            "first_turn": first_turn,
            "followup": followup,
            "error": None,
        }
    except Exception as exc:
        return {
            "sample_index": sample_index,
            "first_turn": {},
            "followup": {},
            "error": repr(exc),
        }
    finally:
        unload(model_kit)


def main() -> int:
    """Run the benchmark and write a JSON report."""
    args = parse_args()
    model_path = resolve_model_path(args.model)
    samples = [
        run_sample(
            model_path=model_path,
            sample_index=index,
            max_kv_size=args.max_kv_size,
            prefill_step_size=args.prefill_step_size,
            story_tokens=args.story_tokens,
            followup_tokens=args.followup_tokens,
        )
        for index in range(1, args.samples + 1)
    ]
    report = {
        "model_path": str(model_path),
        "config": {
            "samples": args.samples,
            "max_kv_size": args.max_kv_size,
            "prefill_step_size": args.prefill_step_size,
            "story_tokens": args.story_tokens,
            "followup_tokens": args.followup_tokens,
        },
        "summary": summarize_samples(samples),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"Wrote LFM2.5 text-cache benchmark to {args.output}")
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["row_errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
