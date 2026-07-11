"""Tests for the LFM2.5 text-cache Markdown reporter."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "lfm25_text_cache_report.py"
)


def _load_report_module():
    spec = importlib.util.spec_from_file_location(
        "lfm25_text_cache_report",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import sanity
        raise RuntimeError("Failed to load lfm25_text_cache_report module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REPORT = _load_report_module()


def _benchmark_fixture():
    return {
        "model_path": "/models/LFM2.5",
        "config": {
            "prefill_step_size": 512,
            "story_tokens": 512,
            "followup_tokens": 64,
        },
        "summary": {
            "sample_count": 1,
            "row_errors": 0,
            "all_followups_cached": True,
            "all_followups_small_prefill": True,
            "all_outputs_preserve_name": True,
            "first_turn_ttft_s": {"avg": 0.1},
            "followup_ttft_s": {"avg": 0.02},
            "followup_total_s": {"avg": 0.03},
            "followup_cached_tokens": {"avg": 542},
            "followup_total_prompt_tokens": {"avg": 565},
            "followup_prefill_tokens_processed": {"avg": 23},
            "followup_cache_reuse_ratio": {"avg": 0.959292},
            "followup_prefill_ratio": {"avg": 0.040708},
        },
        "samples": [
            {
                "sample_index": 1,
                "followup": {
                    "cached_tokens": 542,
                    "prefill_tokens_processed": 23,
                    "ttft_s": 0.02,
                    "output_text": "Silas.",
                },
                "error": None,
            }
        ],
    }


def _comparison_fixture():
    return {
        "status": "pass",
        "baseline": "/tmp/baseline.json",
        "candidate": "/tmp/candidate.json",
        "checks": [
            {
                "name": "candidate_row_errors",
                "status": "pass",
                "value": 0,
                "threshold": 0,
            }
        ],
        "metrics": {
            "followup_ttft_s_avg": {
                "baseline": 0.03,
                "candidate": 0.02,
                "delta": -0.01,
                "delta_pct": -33.3333,
            }
        },
    }


def test_render_benchmark_includes_summary_and_sample_rows():
    """Benchmark Markdown should expose cache, latency, fidelity, and sample evidence."""
    markdown = "\n".join(
        REPORT.render_benchmark(Path("bench.json"), _benchmark_fixture())
    )

    assert "## Benchmark `bench.json`" in markdown
    assert "- Row errors: `0`" in markdown
    assert "| Follow-up cache reuse ratio | 0.959292 |" in markdown
    assert "| 1 | 542 | 23 | 0.020000 | Silas. |  |" in markdown


def test_render_comparison_includes_checks_and_metric_deltas():
    """Comparison Markdown should expose pass/fail checks and baseline deltas."""
    markdown = "\n".join(
        REPORT.render_comparison(Path("compare.json"), _comparison_fixture())
    )

    assert "## Comparison `compare.json`" in markdown
    assert "- Status: `pass`" in markdown
    assert "| candidate_row_errors | pass | 0 | 0 |" in markdown
    assert (
        "| followup_ttft_s_avg | 0.030000 | 0.020000 | -0.010000 | -33.333300 |"
        in markdown
    )


def test_main_writes_combined_markdown(tmp_path, monkeypatch):
    """CLI entrypoint should combine benchmark and comparison JSON inputs."""
    benchmark = tmp_path / "bench.json"
    comparison = tmp_path / "compare.json"
    output = tmp_path / "report.md"
    benchmark.write_text(json.dumps(_benchmark_fixture()))
    comparison.write_text(json.dumps(_comparison_fixture()))
    monkeypatch.setattr(
        "sys.argv",
        [
            "lfm25_text_cache_report.py",
            "--benchmark",
            str(benchmark),
            "--comparison",
            str(comparison),
            "--output",
            str(output),
            "--title",
            "M63 Report",
        ],
    )

    assert REPORT.main() == 0
    assert output.read_text().startswith("# M63 Report")
