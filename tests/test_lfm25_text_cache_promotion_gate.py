"""Tests for the LFM2.5 text-cache promotion gate."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "lfm25_text_cache_promotion_gate.py"
)


def _load_gate_module():
    spec = importlib.util.spec_from_file_location(
        "lfm25_text_cache_promotion_gate",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import sanity
        raise RuntimeError("Failed to load lfm25_text_cache_promotion_gate module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GATE = _load_gate_module()


def _benchmark():
    return {
        "summary": {
            "sample_count": 3,
            "row_errors": 0,
            "all_followups_cached": True,
            "all_followups_small_prefill": True,
            "all_outputs_preserve_name": True,
        }
    }


def _comparison():
    return {"status": "pass"}


def _preflight(*, ready: bool):
    return {"ready_for_live_validation": ready}


def test_gate_fails_without_live_lmstudio_validation(tmp_path):
    """Promotion should fail closed when live LM Studio evidence is absent."""
    report_path = tmp_path / "report.md"
    report_path.write_text("# evidence\n")

    report = GATE.build_gate_report(
        benchmark_path=Path("bench.json"),
        benchmark=_benchmark(),
        comparison_path=Path("compare.json"),
        comparison=_comparison(),
        readable_report_path=report_path,
        preflight_path=Path("preflight.json"),
        preflight=_preflight(ready=False),
        live_validation_path=None,
        live_validation=None,
        min_samples=2,
    )

    assert report["status"] == "fail"
    assert report["promotion_status"] == "NO_PROMOTION"
    checks = {check["name"]: check["status"] for check in report["checks"]}
    assert checks["lmstudio_preflight_ready"] == "fail"
    assert checks["live_lmstudio_validation_passed"] == "fail"


def test_gate_passes_when_all_required_evidence_passes(tmp_path):
    """Promotion gate should pass only when every required evidence item passes."""
    report_path = tmp_path / "report.md"
    report_path.write_text("# evidence\n")

    report = GATE.build_gate_report(
        benchmark_path=Path("bench.json"),
        benchmark=_benchmark(),
        comparison_path=Path("compare.json"),
        comparison=_comparison(),
        readable_report_path=report_path,
        preflight_path=Path("preflight.json"),
        preflight=_preflight(ready=True),
        live_validation_path=Path("live.json"),
        live_validation={"status": "pass"},
        min_samples=2,
    )

    assert report["status"] == "pass"
    assert report["promotion_status"] == "PROMOTION_READY"


def test_main_writes_failure_report_and_returns_nonzero(tmp_path, monkeypatch):
    """CLI should write JSON and return non-zero for blocked promotion evidence."""
    benchmark = tmp_path / "benchmark.json"
    comparison = tmp_path / "comparison.json"
    preflight = tmp_path / "preflight.json"
    markdown = tmp_path / "report.md"
    output = tmp_path / "gate.json"
    benchmark.write_text(json.dumps(_benchmark()))
    comparison.write_text(json.dumps(_comparison()))
    preflight.write_text(json.dumps(_preflight(ready=False)))
    markdown.write_text("# evidence\n")
    monkeypatch.setattr(
        "sys.argv",
        [
            "lfm25_text_cache_promotion_gate.py",
            "--benchmark",
            str(benchmark),
            "--comparison",
            str(comparison),
            "--readable-report",
            str(markdown),
            "--preflight",
            str(preflight),
            "--output",
            str(output),
        ],
    )

    assert GATE.main() == 1
    payload = json.loads(output.read_text())
    assert payload["promotion_status"] == "NO_PROMOTION"
