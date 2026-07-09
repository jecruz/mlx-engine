#!/usr/bin/env python3
"""Gate LFM2.5 text-cache promotion on benchmark, report, and LM Studio evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PASS_STRINGS = {"pass", "passed", "success", "succeeded", "ok"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--readable-report", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--live-validation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-samples", type=int, default=2)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file from disk."""
    return json.loads(path.read_text())


def nested_value(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    """Return a nested dictionary value or None if the path is absent."""
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def check_result(
    name: str,
    passed: bool,
    *,
    value: Any,
    required: Any,
    evidence: str,
) -> dict[str, Any]:
    """Build one gate check result."""
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "value": value,
        "required": required,
        "evidence": evidence,
    }


def live_validation_passed(payload: dict[str, Any] | None) -> bool:
    """Return true when a live validation payload has a recognized pass status."""
    if payload is None:
        return False
    for key_path in (
        ("status",),
        ("validation_status",),
        ("live_lm_studio_validation",),
        ("decision", "status"),
    ):
        value = nested_value(payload, key_path)
        if isinstance(value, str) and value.lower() in PASS_STRINGS:
            return True
    return bool(payload.get("passed") is True or payload.get("success") is True)


def build_gate_report(
    *,
    benchmark_path: Path,
    benchmark: dict[str, Any],
    comparison_path: Path,
    comparison: dict[str, Any],
    readable_report_path: Path,
    preflight_path: Path,
    preflight: dict[str, Any],
    live_validation_path: Path | None,
    live_validation: dict[str, Any] | None,
    min_samples: int,
) -> dict[str, Any]:
    """Build a fail-closed promotion gate report."""
    benchmark_summary = benchmark.get("summary", {})
    if not isinstance(benchmark_summary, dict):
        benchmark_summary = {}

    readable_report_exists = readable_report_path.exists()
    readable_report_bytes = (
        readable_report_path.stat().st_size if readable_report_exists else 0
    )
    ready_for_live_validation = preflight.get("ready_for_live_validation")
    live_passed = live_validation_passed(live_validation)
    checks = [
        check_result(
            "benchmark_min_samples",
            isinstance(benchmark_summary.get("sample_count"), int)
            and benchmark_summary["sample_count"] >= min_samples,
            value=benchmark_summary.get("sample_count"),
            required=f">= {min_samples}",
            evidence=str(benchmark_path),
        ),
        check_result(
            "benchmark_row_errors",
            benchmark_summary.get("row_errors") == 0,
            value=benchmark_summary.get("row_errors"),
            required=0,
            evidence=str(benchmark_path),
        ),
        check_result(
            "benchmark_followups_cached",
            benchmark_summary.get("all_followups_cached") is True,
            value=benchmark_summary.get("all_followups_cached"),
            required=True,
            evidence=str(benchmark_path),
        ),
        check_result(
            "benchmark_followups_small_prefill",
            benchmark_summary.get("all_followups_small_prefill") is True,
            value=benchmark_summary.get("all_followups_small_prefill"),
            required=True,
            evidence=str(benchmark_path),
        ),
        check_result(
            "benchmark_outputs_preserve_name",
            benchmark_summary.get("all_outputs_preserve_name") is True,
            value=benchmark_summary.get("all_outputs_preserve_name"),
            required=True,
            evidence=str(benchmark_path),
        ),
        check_result(
            "comparison_status",
            comparison.get("status") == "pass",
            value=comparison.get("status"),
            required="pass",
            evidence=str(comparison_path),
        ),
        check_result(
            "readable_report_exists",
            readable_report_exists and readable_report_bytes > 0,
            value={"exists": readable_report_exists, "size_bytes": readable_report_bytes},
            required="existing non-empty Markdown report",
            evidence=str(readable_report_path),
        ),
        check_result(
            "lmstudio_preflight_ready",
            ready_for_live_validation is True,
            value=ready_for_live_validation,
            required=True,
            evidence=str(preflight_path),
        ),
        check_result(
            "live_lmstudio_validation_passed",
            live_passed,
            value=(
                None
                if live_validation is None
                else {
                    "status": live_validation.get("status"),
                    "validation_status": live_validation.get("validation_status"),
                    "passed": live_validation.get("passed"),
                    "success": live_validation.get("success"),
                }
            ),
            required="passing live LM Studio validation artifact",
            evidence=str(live_validation_path) if live_validation_path else "missing",
        ),
    ]
    status = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    return {
        "status": status,
        "promotion_status": "PROMOTION_READY" if status == "pass" else "NO_PROMOTION",
        "inputs": {
            "benchmark": str(benchmark_path),
            "comparison": str(comparison_path),
            "readable_report": str(readable_report_path),
            "preflight": str(preflight_path),
            "live_validation": str(live_validation_path) if live_validation_path else None,
            "min_samples": min_samples,
        },
        "checks": checks,
        "decision": {
            "runtime_changed": False,
            "live_lm_studio_validation_required": True,
            "reason": (
                "All promotion evidence is present and passing."
                if status == "pass"
                else "Promotion is blocked until every benchmark, report, preflight, and live LM Studio validation check passes."
            ),
        },
    }


def main() -> int:
    """Run the promotion gate and write a JSON report."""
    args = parse_args()
    live_validation = (
        load_json(args.live_validation) if args.live_validation is not None else None
    )
    report = build_gate_report(
        benchmark_path=args.benchmark,
        benchmark=load_json(args.benchmark),
        comparison_path=args.comparison,
        comparison=load_json(args.comparison),
        readable_report_path=args.readable_report,
        preflight_path=args.preflight,
        preflight=load_json(args.preflight),
        live_validation_path=args.live_validation,
        live_validation=live_validation,
        min_samples=args.min_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"Wrote LFM2.5 text-cache promotion gate to {args.output}")
    print(json.dumps({"status": report["status"], "promotion_status": report["promotion_status"]}, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
