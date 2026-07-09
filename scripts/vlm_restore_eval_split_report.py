#!/usr/bin/env python3
"""Summarize VLM restore eval timing split events from shared-bench reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

TIMING_MARKER = "MLX_ENGINE_BATCHED_TIMING "
RESTORE_DETAIL_EVENT = "vlm_cache_restore_detail"
DEFAULT_BARRIER_SHARE_THRESHOLD = 0.95


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reports",
        nargs="+",
        type=Path,
        help="shared_bench.py JSON report paths to summarize",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path",
    )
    parser.add_argument(
        "--barrier-share-threshold",
        type=float,
        default=DEFAULT_BARRIER_SHARE_THRESHOLD,
        help=(
            "Barrier-share threshold for classifying samples as barrier dominated "
            f"({DEFAULT_BARRIER_SHARE_THRESHOLD})"
        ),
    )
    return parser.parse_args()


def _extract_timing_events(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return restore-detail timing events captured in runner stderr."""
    events = []
    for result in report.get("results", []):
        for process in result.get("runner_processes", []):
            stderr = process.get("stderr") or ""
            for line in stderr.splitlines():
                if TIMING_MARKER not in line:
                    continue
                _, payload = line.split(TIMING_MARKER, 1)
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if event.get("event") == RESTORE_DETAIL_EVENT:
                    events.append(event)
    return events


def _row_audit_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Return row-level success, cache, and output evidence from one report."""
    rows = report.get("row_audit", [])
    errors = [row for row in rows if row.get("error")]
    return {
        "rows": len(rows),
        "row_errors": len(errors),
        "cached_tokens": [row.get("cached_tokens") for row in rows],
        "output_previews": [row.get("output_preview") for row in rows],
        "errors": [
            {
                "prompt_id": row.get("prompt_id"),
                "run_index": row.get("run_index"),
                "error": row.get("error"),
            }
            for row in errors
        ],
    }


def _summarize_sample(
    *,
    report_path: Path,
    report: dict[str, Any],
    event: dict[str, Any],
) -> dict[str, Any]:
    """Convert one restore-detail event into a compact sample record."""
    eval_ms = float(event.get("eval_ms") or 0.0)
    eval_barrier_ms = float(event.get("eval_barrier_ms") or 0.0)
    barrier_share = eval_barrier_ms / eval_ms if eval_ms > 0 else None
    return {
        "report": str(report_path),
        "cached_tokens": event.get("cached_tokens"),
        "records": event.get("records"),
        "record_count_by_kind": event.get("record_count_by_kind"),
        "load_chunks_ms": event.get("load_chunks_ms"),
        "assemble_ms": event.get("assemble_ms"),
        "eval_collect_ms": event.get("eval_collect_ms"),
        "eval_barrier_ms": event.get("eval_barrier_ms"),
        "eval_ms": event.get("eval_ms"),
        "touch_ms": event.get("touch_ms"),
        "duration_ms": event.get("duration_ms"),
        "eval_target_count": event.get("eval_target_count"),
        "materialized_bytes": event.get("materialized_bytes"),
        "barrier_share_of_eval_ms": barrier_share,
        "row_audit": _row_audit_summary(report),
    }


def _numeric_values(samples: list[dict[str, Any]], key: str) -> list[float]:
    """Return numeric sample values for one key."""
    values = []
    for sample in samples:
        value = sample.get(key)
        if isinstance(value, int | float):
            values.append(float(value))
    return values


def _range_summary(samples: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    """Return min/max/avg for a numeric sample key."""
    values = _numeric_values(samples, key)
    if not values:
        return {"min": None, "max": None, "avg": None}
    return {
        "min": min(values),
        "max": max(values),
        "avg": mean(values),
    }


def build_report(
    report_paths: list[Path],
    *,
    barrier_share_threshold: float = DEFAULT_BARRIER_SHARE_THRESHOLD,
) -> dict[str, Any]:
    """Build a restore eval split summary from one or more benchmark reports."""
    samples = []
    missing_timing_reports = []
    for report_path in report_paths:
        report = json.loads(report_path.read_text())
        events = _extract_timing_events(report)
        if not events:
            missing_timing_reports.append(str(report_path))
            continue
        samples.extend(
            _summarize_sample(report_path=report_path, report=report, event=event)
            for event in events
        )

    barrier_shares = _numeric_values(samples, "barrier_share_of_eval_ms")
    return {
        "reports": [str(path) for path in report_paths],
        "sample_count": len(samples),
        "missing_timing_reports": missing_timing_reports,
        "barrier_share_threshold": barrier_share_threshold,
        "barrier_dominated": bool(barrier_shares)
        and all(value >= barrier_share_threshold for value in barrier_shares),
        "aggregate": {
            "eval_collect_ms": _range_summary(samples, "eval_collect_ms"),
            "eval_barrier_ms": _range_summary(samples, "eval_barrier_ms"),
            "eval_ms": _range_summary(samples, "eval_ms"),
            "barrier_share_of_eval_ms": _range_summary(
                samples,
                "barrier_share_of_eval_ms",
            ),
            "row_errors": sum(
                sample["row_audit"]["row_errors"] for sample in samples
            ),
        },
        "samples": samples,
    }


def main() -> int:
    """Write or print a restore eval split summary report."""
    args = _parse_args()
    payload = build_report(
        args.reports,
        barrier_share_threshold=args.barrier_share_threshold,
    )
    output = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
        print(f"Wrote VLM restore eval split report to {args.output}")
    else:
        print(output)

    print(f"sample_count={payload['sample_count']}")
    print(f"barrier_dominated={str(payload['barrier_dominated']).lower()}")
    print(f"row_errors={payload['aggregate']['row_errors']}")
    return 0 if payload["sample_count"] and not payload["missing_timing_reports"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
