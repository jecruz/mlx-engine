"""Tests for the LFM2.5 text-cache benchmark script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "lfm25_text_cache_bench.py"
)


def _load_bench_module():
    spec = importlib.util.spec_from_file_location(
        "lfm25_text_cache_bench",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import sanity
        raise RuntimeError("Failed to load lfm25_text_cache_bench module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BENCH = _load_bench_module()


def test_summarize_samples_reports_followup_cache_reuse():
    """Summary should classify repeated follow-up cache reuse and fidelity."""
    payload = BENCH.summarize_samples(
        [
            {
                "first_turn": {"ttft_s": 0.8},
                "followup": {
                    "cached_tokens": 542,
                    "total_prompt_tokens": 565,
                    "prefill_tokens_processed": 23,
                    "ttft_s": 0.03,
                    "total_s": 0.05,
                    "output_text": "Silas.",
                },
                "error": None,
            },
            {
                "first_turn": {"ttft_s": 0.7},
                "followup": {
                    "cached_tokens": 540,
                    "total_prompt_tokens": 563,
                    "prefill_tokens_processed": 23,
                    "ttft_s": 0.04,
                    "total_s": 0.06,
                    "output_text": "The name was Silas.",
                },
                "error": None,
            },
        ]
    )

    assert payload["sample_count"] == 2
    assert payload["row_errors"] == 0
    assert payload["all_followups_cached"] is True
    assert payload["all_followups_small_prefill"] is True
    assert payload["all_outputs_preserve_name"] is True
    assert payload["followup_cached_tokens"]["min"] == 540.0
    assert payload["followup_cached_tokens"]["max"] == 542.0
    assert payload["followup_prefill_tokens_processed"]["avg"] == 23.0


def test_summarize_samples_flags_missing_followup_cache():
    """Summary should reject samples that do not reuse follow-up cache."""
    payload = BENCH.summarize_samples(
        [
            {
                "first_turn": {"ttft_s": 0.8},
                "followup": {
                    "cached_tokens": 0,
                    "total_prompt_tokens": 565,
                    "prefill_tokens_processed": 565,
                    "ttft_s": 0.5,
                    "total_s": 0.6,
                    "output_text": "Silas.",
                },
                "error": None,
            }
        ]
    )

    assert payload["sample_count"] == 1
    assert payload["all_followups_cached"] is False
    assert payload["all_followups_small_prefill"] is False
