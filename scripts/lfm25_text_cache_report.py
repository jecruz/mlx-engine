#!/usr/bin/env python3
"""Render readable LFM2.5 text-cache benchmark and comparison reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, action="append", default=[])
    parser.add_argument("--comparison", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="LFM2.5 Text-Cache Evidence")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON report from disk."""
    return json.loads(path.read_text())


def format_value(value: Any) -> str:
    """Format a report value for Markdown."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return ""
    return str(value)


def summary_avg(summary: dict[str, Any], key: str) -> Any:
    """Return the avg field for a summary bucket."""
    bucket = summary.get(key)
    if not isinstance(bucket, dict):
        return None
    return bucket.get("avg")


def render_benchmark(path: Path, report: dict[str, Any]) -> list[str]:
    """Render a benchmark JSON report as Markdown lines."""
    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    config = report.get("config", {})
    if not isinstance(config, dict):
        config = {}
    lines = [
        f"## Benchmark `{path}`",
        "",
        f"- Model path: `{report.get('model_path', '')}`",
        f"- Samples: `{format_value(summary.get('sample_count'))}`",
        f"- Row errors: `{format_value(summary.get('row_errors'))}`",
        f"- All followups cached: `{format_value(summary.get('all_followups_cached'))}`",
        f"- All followups small prefill: `{format_value(summary.get('all_followups_small_prefill'))}`",
        f"- All outputs preserve name: `{format_value(summary.get('all_outputs_preserve_name'))}`",
        f"- Prefill step size: `{format_value(config.get('prefill_step_size'))}`",
        f"- Story tokens: `{format_value(config.get('story_tokens'))}`",
        f"- Followup tokens: `{format_value(config.get('followup_tokens'))}`",
        "",
        "| Metric | Average |",
        "| --- | ---: |",
        f"| First-turn TTFT seconds | {format_value(summary_avg(summary, 'first_turn_ttft_s'))} |",
        f"| Follow-up TTFT seconds | {format_value(summary_avg(summary, 'followup_ttft_s'))} |",
        f"| Follow-up total seconds | {format_value(summary_avg(summary, 'followup_total_s'))} |",
        f"| Follow-up cached tokens | {format_value(summary_avg(summary, 'followup_cached_tokens'))} |",
        f"| Follow-up total prompt tokens | {format_value(summary_avg(summary, 'followup_total_prompt_tokens'))} |",
        f"| Follow-up prefill tokens | {format_value(summary_avg(summary, 'followup_prefill_tokens_processed'))} |",
        f"| Follow-up cache reuse ratio | {format_value(summary_avg(summary, 'followup_cache_reuse_ratio'))} |",
        f"| Follow-up prefill ratio | {format_value(summary_avg(summary, 'followup_prefill_ratio'))} |",
        "",
    ]

    samples = report.get("samples", [])
    if isinstance(samples, list) and samples:
        lines.extend(
            [
                "| Sample | Follow-up cached tokens | Follow-up prefill tokens | Follow-up TTFT seconds | Follow-up output | Error |",
                "| ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            followup = sample.get("followup", {})
            if not isinstance(followup, dict):
                followup = {}
            output = str(followup.get("output_text", "")).replace("|", "\\|")
            error = str(sample.get("error") or "").replace("|", "\\|")
            lines.append(
                "| "
                f"{format_value(sample.get('sample_index'))} | "
                f"{format_value(followup.get('cached_tokens'))} | "
                f"{format_value(followup.get('prefill_tokens_processed'))} | "
                f"{format_value(followup.get('ttft_s'))} | "
                f"{output} | "
                f"{error} |"
            )
        lines.append("")
    return lines


def render_metric_pair(name: str, metric: dict[str, Any]) -> str:
    """Render one comparison metric table row."""
    return (
        f"| {name} | "
        f"{format_value(metric.get('baseline'))} | "
        f"{format_value(metric.get('candidate'))} | "
        f"{format_value(metric.get('delta'))} | "
        f"{format_value(metric.get('delta_pct'))} |"
    )


def render_comparison(path: Path, report: dict[str, Any]) -> list[str]:
    """Render a comparison JSON report as Markdown lines."""
    lines = [
        f"## Comparison `{path}`",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Baseline: `{report.get('baseline', '')}`",
        f"- Candidate: `{report.get('candidate', '')}`",
        "",
        "| Check | Status | Value | Threshold |",
        "| --- | --- | ---: | ---: |",
    ]
    for check in report.get("checks", []):
        if not isinstance(check, dict):
            continue
        lines.append(
            "| "
            f"{check.get('name', '')} | "
            f"{check.get('status', '')} | "
            f"{format_value(check.get('value'))} | "
            f"{format_value(check.get('threshold'))} |"
        )

    lines.extend(["", "| Metric | Baseline | Candidate | Delta | Delta percent |", "| --- | ---: | ---: | ---: | ---: |"])
    metrics = report.get("metrics", {})
    if isinstance(metrics, dict):
        for name, metric in metrics.items():
            if isinstance(metric, dict) and {"baseline", "candidate"}.issubset(metric):
                lines.append(render_metric_pair(name, metric))
    lines.append("")
    return lines


def render_markdown(
    *,
    title: str,
    benchmarks: list[tuple[Path, dict[str, Any]]],
    comparisons: list[tuple[Path, dict[str, Any]]],
) -> str:
    """Render benchmark and comparison reports into one Markdown document."""
    lines = [
        f"# {title}",
        "",
        "This report is evidence only. Runtime promotion still requires repeated",
        "retained-workload wins, passing quality gates, candidate-vs-baseline",
        "deltas, and live LM Studio validation.",
        "",
    ]
    for path, report in benchmarks:
        lines.extend(render_benchmark(path, report))
    for path, report in comparisons:
        lines.extend(render_comparison(path, report))
    return "\n".join(lines)


def main() -> int:
    """Write a Markdown evidence report from JSON inputs."""
    args = parse_args()
    benchmarks = [(path, load_json(path)) for path in args.benchmark]
    comparisons = [(path, load_json(path)) for path in args.comparison]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_markdown(
            title=args.title,
            benchmarks=benchmarks,
            comparisons=comparisons,
        )
    )
    print(f"Wrote LFM2.5 text-cache Markdown report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
