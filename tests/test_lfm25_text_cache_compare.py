"""Tests for the LFM2.5 text-cache comparison gate."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "lfm25_text_cache_compare.py"
)


def _load_compare_module():
    spec = importlib.util.spec_from_file_location(
        "lfm25_text_cache_compare",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import sanity
        raise RuntimeError("Failed to load lfm25_text_cache_compare module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COMPARE = _load_compare_module()


def _report(
    *,
    cached_tokens: float = 542.0,
    total_prompt_tokens: float = 565.0,
    prefill_tokens: float = 23.0,
    row_errors: int = 0,
    all_followups_cached: bool = True,
    all_followups_small_prefill: bool = True,
    all_outputs_preserve_name: bool = True,
    include_ratios: bool = True,
) -> dict:
    summary = {
        "sample_count": 2,
        "row_errors": row_errors,
        "all_followups_cached": all_followups_cached,
        "all_followups_small_prefill": all_followups_small_prefill,
        "all_outputs_preserve_name": all_outputs_preserve_name,
        "followup_cached_tokens": {"avg": cached_tokens},
        "followup_total_prompt_tokens": {"avg": total_prompt_tokens},
        "followup_prefill_tokens_processed": {"avg": prefill_tokens},
        "followup_ttft_s": {"avg": 0.02},
        "followup_total_s": {"avg": 0.03},
    }
    if include_ratios:
        summary["followup_cache_reuse_ratio"] = {
            "avg": cached_tokens / total_prompt_tokens
        }
        summary["followup_prefill_ratio"] = {"avg": prefill_tokens / total_prompt_tokens}
    return {"summary": summary}


def test_compare_reports_computes_missing_baseline_ratios():
    """Old baseline reports without ratio fields should remain comparable."""
    result = COMPARE.compare_reports(
        _report(include_ratios=False),
        _report(include_ratios=True),
        COMPARE.Thresholds(
            max_cache_ratio_regression=0.01,
            max_prefill_ratio_regression=0.01,
            max_ttft_regression_pct=None,
            max_total_regression_pct=None,
        ),
    )

    assert result["status"] == "pass"
    assert result["metrics"]["followup_cache_reuse_ratio_avg"]["baseline"] == 542.0 / 565.0
    assert result["metrics"]["followup_cache_reuse_ratio_avg"]["candidate"] == 542.0 / 565.0
    assert all(check["status"] == "pass" for check in result["checks"])


def test_compare_reports_fails_candidate_cache_regression():
    """A candidate that loses cache reuse should fail the comparison gate."""
    result = COMPARE.compare_reports(
        _report(),
        _report(cached_tokens=500.0, prefill_tokens=65.0),
        COMPARE.Thresholds(
            max_cache_ratio_regression=0.01,
            max_prefill_ratio_regression=0.01,
            max_ttft_regression_pct=None,
            max_total_regression_pct=None,
        ),
    )

    failed = {check["name"] for check in result["checks"] if check["status"] == "fail"}
    assert result["status"] == "fail"
    assert "cache_reuse_ratio_regression" in failed
    assert "prefill_ratio_regression" in failed


def test_main_writes_comparison_report(tmp_path, monkeypatch):
    """CLI entrypoint should write JSON and return success for passing reports."""
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "compare.json"
    baseline.write_text(json.dumps(_report(include_ratios=False)))
    candidate.write_text(json.dumps(_report()))
    monkeypatch.setattr(
        "sys.argv",
        [
            "lfm25_text_cache_compare.py",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--output",
            str(output),
        ],
    )

    assert COMPARE.main() == 0
    payload = json.loads(output.read_text())
    assert payload["status"] == "pass"
    assert payload["baseline"] == str(baseline.resolve())
    assert payload["candidate"] == str(candidate.resolve())
