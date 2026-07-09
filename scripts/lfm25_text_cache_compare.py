#!/usr/bin/env python3
"""Compare LFM2.5-VL text-cache benchmark reports."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Thresholds:
    """Comparison thresholds for text-cache promotion checks."""

    max_cache_ratio_regression: float
    max_prefill_ratio_regression: float
    max_ttft_regression_pct: float | None
    max_total_regression_pct: float | None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-cache-ratio-regression",
        type=float,
        default=0.01,
        help="Allowed absolute drop in follow-up cache-reuse ratio.",
    )
    parser.add_argument(
        "--max-prefill-ratio-regression",
        type=float,
        default=0.01,
        help="Allowed absolute increase in follow-up prefill ratio.",
    )
    parser.add_argument(
        "--max-ttft-regression-pct",
        type=float,
        default=None,
        help="Optional allowed percent regression in follow-up TTFT.",
    )
    parser.add_argument(
        "--max-total-regression-pct",
        type=float,
        default=None,
        help="Optional allowed percent regression in follow-up total latency.",
    )
    return parser.parse_args()


def load_report(path: Path) -> dict[str, Any]:
    """Load a JSON benchmark report."""
    return json.loads(path.read_text())


def summary_value(
    summary: dict[str, Any],
    key: str,
    field: str = "avg",
) -> float | None:
    """Return a numeric summary value if the report contains one."""
    bucket = summary.get(key)
    if not isinstance(bucket, dict):
        return None
    value = bucket.get(field)
    if isinstance(value, int | float):
        return float(value)
    return None


def ratio_value(
    summary: dict[str, Any],
    ratio_key: str,
    numerator_key: str,
    denominator_key: str,
) -> float | None:
    """Return a ratio summary value, computing it from token counts if needed."""
    value = summary_value(summary, ratio_key)
    if value is not None:
        return value

    numerator = summary_value(summary, numerator_key)
    denominator = summary_value(summary, denominator_key)
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def delta(candidate: float | None, baseline: float | None) -> float | None:
    """Return candidate-minus-baseline delta when both values are present."""
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


def pct_delta(candidate: float | None, baseline: float | None) -> float | None:
    """Return percent delta when both values are present and baseline is nonzero."""
    if candidate is None or baseline is None or baseline == 0:
        return None
    return ((candidate - baseline) / baseline) * 100.0


def metric_pair(
    baseline: float | None,
    candidate: float | None,
) -> dict[str, float | None]:
    """Build a baseline/candidate metric comparison object."""
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta(candidate, baseline),
        "delta_pct": pct_delta(candidate, baseline),
    }


def check_result(
    name: str,
    passed: bool,
    *,
    value: Any,
    threshold: Any = None,
) -> dict[str, Any]:
    """Build a pass/fail check result."""
    result = {"name": name, "status": "pass" if passed else "fail", "value": value}
    if threshold is not None:
        result["threshold"] = threshold
    return result


def compare_reports(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
    thresholds: Thresholds,
) -> dict[str, Any]:
    """Compare two LFM2.5 text-cache benchmark report payloads."""
    baseline_summary = baseline_report.get("summary", {})
    candidate_summary = candidate_report.get("summary", {})
    if not isinstance(baseline_summary, dict):
        baseline_summary = {}
    if not isinstance(candidate_summary, dict):
        candidate_summary = {}

    baseline_cache_ratio = ratio_value(
        baseline_summary,
        "followup_cache_reuse_ratio",
        "followup_cached_tokens",
        "followup_total_prompt_tokens",
    )
    candidate_cache_ratio = ratio_value(
        candidate_summary,
        "followup_cache_reuse_ratio",
        "followup_cached_tokens",
        "followup_total_prompt_tokens",
    )
    baseline_prefill_ratio = ratio_value(
        baseline_summary,
        "followup_prefill_ratio",
        "followup_prefill_tokens_processed",
        "followup_total_prompt_tokens",
    )
    candidate_prefill_ratio = ratio_value(
        candidate_summary,
        "followup_prefill_ratio",
        "followup_prefill_tokens_processed",
        "followup_total_prompt_tokens",
    )
    baseline_ttft = summary_value(baseline_summary, "followup_ttft_s")
    candidate_ttft = summary_value(candidate_summary, "followup_ttft_s")
    baseline_total = summary_value(baseline_summary, "followup_total_s")
    candidate_total = summary_value(candidate_summary, "followup_total_s")

    cache_regression = delta(baseline_cache_ratio, candidate_cache_ratio)
    prefill_regression = delta(candidate_prefill_ratio, baseline_prefill_ratio)
    ttft_regression_pct = pct_delta(candidate_ttft, baseline_ttft)
    total_regression_pct = pct_delta(candidate_total, baseline_total)

    checks = [
        check_result(
            "candidate_row_errors",
            candidate_summary.get("row_errors") == 0,
            value=candidate_summary.get("row_errors"),
            threshold=0,
        ),
        check_result(
            "candidate_followups_cached",
            candidate_summary.get("all_followups_cached") is True,
            value=candidate_summary.get("all_followups_cached"),
            threshold=True,
        ),
        check_result(
            "candidate_followups_small_prefill",
            candidate_summary.get("all_followups_small_prefill") is True,
            value=candidate_summary.get("all_followups_small_prefill"),
            threshold=True,
        ),
        check_result(
            "candidate_outputs_preserve_name",
            candidate_summary.get("all_outputs_preserve_name") is True,
            value=candidate_summary.get("all_outputs_preserve_name"),
            threshold=True,
        ),
        check_result(
            "cache_reuse_ratio_regression",
            cache_regression is not None
            and cache_regression <= thresholds.max_cache_ratio_regression,
            value=cache_regression,
            threshold=thresholds.max_cache_ratio_regression,
        ),
        check_result(
            "prefill_ratio_regression",
            prefill_regression is not None
            and prefill_regression <= thresholds.max_prefill_ratio_regression,
            value=prefill_regression,
            threshold=thresholds.max_prefill_ratio_regression,
        ),
    ]

    if thresholds.max_ttft_regression_pct is not None:
        checks.append(
            check_result(
                "followup_ttft_regression_pct",
                ttft_regression_pct is not None
                and ttft_regression_pct <= thresholds.max_ttft_regression_pct,
                value=ttft_regression_pct,
                threshold=thresholds.max_ttft_regression_pct,
            )
        )
    if thresholds.max_total_regression_pct is not None:
        checks.append(
            check_result(
                "followup_total_regression_pct",
                total_regression_pct is not None
                and total_regression_pct <= thresholds.max_total_regression_pct,
                value=total_regression_pct,
                threshold=thresholds.max_total_regression_pct,
            )
        )

    status = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    return {
        "status": status,
        "thresholds": {
            "max_cache_ratio_regression": thresholds.max_cache_ratio_regression,
            "max_prefill_ratio_regression": thresholds.max_prefill_ratio_regression,
            "max_ttft_regression_pct": thresholds.max_ttft_regression_pct,
            "max_total_regression_pct": thresholds.max_total_regression_pct,
        },
        "checks": checks,
        "metrics": {
            "sample_count": {
                "baseline": baseline_summary.get("sample_count"),
                "candidate": candidate_summary.get("sample_count"),
            },
            "followup_cache_reuse_ratio_avg": metric_pair(
                baseline_cache_ratio,
                candidate_cache_ratio,
            ),
            "followup_prefill_ratio_avg": metric_pair(
                baseline_prefill_ratio,
                candidate_prefill_ratio,
            ),
            "followup_ttft_s_avg": metric_pair(baseline_ttft, candidate_ttft),
            "followup_total_s_avg": metric_pair(baseline_total, candidate_total),
        },
    }


def build_comparison(
    baseline_path: Path,
    candidate_path: Path,
    thresholds: Thresholds,
) -> dict[str, Any]:
    """Load two reports and return a path-aware comparison payload."""
    comparison = compare_reports(
        load_report(baseline_path),
        load_report(candidate_path),
        thresholds,
    )
    comparison["baseline"] = str(baseline_path.resolve())
    comparison["candidate"] = str(candidate_path.resolve())
    return comparison


def main() -> int:
    """Run the comparison gate and write a JSON report."""
    args = parse_args()
    thresholds = Thresholds(
        max_cache_ratio_regression=args.max_cache_ratio_regression,
        max_prefill_ratio_regression=args.max_prefill_ratio_regression,
        max_ttft_regression_pct=args.max_ttft_regression_pct,
        max_total_regression_pct=args.max_total_regression_pct,
    )
    comparison = build_comparison(args.baseline, args.candidate, thresholds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2))
    print(f"Wrote LFM2.5 text-cache comparison to {args.output}")
    print(json.dumps({"status": comparison["status"], "checks": comparison["checks"]}, indent=2))
    return 0 if comparison["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
