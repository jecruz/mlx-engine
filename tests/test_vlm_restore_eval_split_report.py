"""Tests for the VLM restore eval split report script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "vlm_restore_eval_split_report.py"
)


def _load_report_module():
    spec = importlib.util.spec_from_file_location(
        "vlm_restore_eval_split_report",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import sanity
        raise RuntimeError("Failed to load vlm_restore_eval_split_report module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPORT = _load_report_module()


def _write_shared_bench_report(path: Path, *, eval_barrier_ms: float, eval_ms: float):
    event = {
        "event": "vlm_cache_restore_detail",
        "cached_tokens": 7373,
        "records": 2,
        "record_count_by_kind": {
            "kv_delta": 1,
            "rotating_delta": 0,
            "state_checkpoint": 1,
        },
        "record_bytes_by_kind": {
            "kv_delta": 90600551,
            "rotating_delta": 0,
            "state_checkpoint": 82940,
        },
        "load_chunks_ms": 0.5,
        "assemble_ms": 0.01,
        "eval_collect_ms": 0.05,
        "eval_barrier_ms": eval_barrier_ms,
        "eval_ms": eval_ms,
        "touch_ms": 0.1,
        "duration_ms": 5.5,
        "eval_target_count": 22,
        "eval_target_count_by_kind": {
            "kv_delta": 12,
            "rotating_delta": 0,
            "state_checkpoint": 10,
        },
        "materialized_bytes": 90681344,
        "materialized_bytes_by_kind": {
            "kv_delta": 90599424,
            "rotating_delta": 0,
            "state_checkpoint": 81920,
        },
    }
    payload = {
        "results": [
            {
                "runner_processes": [
                    {
                        "stderr": (
                            "[batched_timing][WARNING]: "
                            "MLX_ENGINE_BATCHED_TIMING "
                            f"{json.dumps(event)}\n"
                        )
                    }
                ]
            }
        ],
        "row_audit": [
            {
                "prompt_id": "image_long_toucan",
                "run_index": 1,
                "cached_tokens": 0,
                "output_preview": "A toucan.",
                "error": None,
            },
            {
                "prompt_id": "image_long_toucan",
                "run_index": 2,
                "cached_tokens": 7373,
                "output_preview": "A toucan.",
                "error": None,
            },
        ],
    }
    path.write_text(json.dumps(payload))


def test_build_report_extracts_restore_eval_split(tmp_path):
    """Reports should summarize barrier share and row evidence."""
    report_path = tmp_path / "shared-bench.json"
    _write_shared_bench_report(report_path, eval_barrier_ms=4.9, eval_ms=5.0)

    payload = REPORT.build_report([report_path])

    assert payload["sample_count"] == 1
    assert payload["missing_timing_reports"] == []
    assert payload["barrier_dominated"] is True
    assert payload["aggregate"]["row_errors"] == 0
    assert payload["aggregate"]["eval_collect_ms"]["min"] == 0.05
    assert payload["aggregate"]["eval_barrier_ms"]["max"] == 4.9
    assert payload["aggregate"]["dominant_materialized_kind"] == "kv_delta"
    assert payload["aggregate"]["eval_target_count_by_kind"] == {
        "kv_delta": 12,
        "rotating_delta": 0,
        "state_checkpoint": 10,
    }
    assert payload["aggregate"]["materialized_bytes_by_kind"] == {
        "kv_delta": 90599424,
        "rotating_delta": 0,
        "state_checkpoint": 81920,
    }
    assert payload["aggregate"]["record_bytes_by_kind"] == {
        "kv_delta": 90600551,
        "rotating_delta": 0,
        "state_checkpoint": 82940,
    }
    sample = payload["samples"][0]
    assert sample["barrier_share_of_eval_ms"] == 0.9800000000000001
    assert sample["eval_target_count_by_kind"]["kv_delta"] == 12
    assert sample["materialized_bytes_by_kind"]["kv_delta"] == 90599424
    assert sample["row_audit"]["cached_tokens"] == [0, 7373]
    assert sample["row_audit"]["output_previews"] == ["A toucan.", "A toucan."]


def test_build_report_flags_missing_timing_report(tmp_path):
    """Reports without timing events should be identified explicitly."""
    report_path = tmp_path / "no-timing.json"
    report_path.write_text(json.dumps({"results": [], "row_audit": []}))

    payload = REPORT.build_report([report_path])

    assert payload["sample_count"] == 0
    assert payload["barrier_dominated"] is False
    assert payload["missing_timing_reports"] == [str(report_path)]


def test_build_report_respects_barrier_threshold(tmp_path):
    """Low barrier-share samples should not be classified as barrier dominated."""
    report_path = tmp_path / "shared-bench.json"
    _write_shared_bench_report(report_path, eval_barrier_ms=2.0, eval_ms=5.0)

    payload = REPORT.build_report([report_path], barrier_share_threshold=0.95)

    assert payload["sample_count"] == 1
    assert payload["barrier_dominated"] is False
