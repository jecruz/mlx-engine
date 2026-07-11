#!/usr/bin/env python3
"""Render a readable diff between two upstream candidate scan JSON reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Upstream Candidate Scan Diff")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file from disk."""
    return json.loads(path.read_text())


def branch_by_name(scan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index candidate branches by name."""
    branches = scan.get("candidate_branches", [])
    if not isinstance(branches, list):
        return {}
    return {
        branch.get("branch"): branch
        for branch in branches
        if isinstance(branch, dict) and branch.get("branch")
    }


def summary_value(summary: dict[str, Any], key: str, field: str = "avg") -> Any:
    """Return a nested summary value if available."""
    bucket = summary.get(key)
    if not isinstance(bucket, dict):
        return None
    return bucket.get(field)


def nested_value(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    """Return a nested dictionary value if all keys are present."""
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def format_value(value: Any) -> str:
    """Format a value for Markdown tables."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return ""
    return str(value)


def change_surface(branch: dict[str, Any]) -> str:
    """Classify a branch by surface size using changed files and unmatched commits."""
    changed_files = branch.get("changed_files", [])
    unmatched = branch.get("unmatched_patch_ids", [])
    changed_count = len(changed_files) if isinstance(changed_files, list) else 0
    unmatched_count = len(unmatched) if isinstance(unmatched, list) else 0
    if changed_count >= 20 or unmatched_count >= 10:
        return "broad"
    if changed_count >= 6 or unmatched_count >= 4:
        return "moderate"
    return "small"


def render_scan_diff(
    baseline: dict[str, Any], candidate: dict[str, Any], *, title: str
) -> str:
    """Render a scan diff as Markdown."""
    baseline_branches = branch_by_name(baseline)
    candidate_branches = branch_by_name(candidate)
    all_branch_names = sorted(
        set(baseline_branches) | set(candidate_branches),
    )

    lines = [
        f"# {title}",
        "",
        "This report is evidence only. It highlights scan deltas; it does not",
        "authorize a cherry-pick or runtime promotion.",
        "",
        "## Summary",
        "",
        f"- Baseline head: `{format_value(baseline.get('head'))}`",
        f"- Candidate head: `{format_value(candidate.get('head'))}`",
        f"- Baseline upstream/main: `{format_value(baseline.get('upstream_main_head'))}`",
        f"- Candidate upstream/main: `{format_value(candidate.get('upstream_main_head'))}`",
        f"- Baseline origin branch: `{format_value(baseline.get('origin_branch_head'))}`",
        f"- Candidate origin branch: `{format_value(candidate.get('origin_branch_head'))}`",
        f"- Baseline candidate count: `{len(baseline.get('candidate_branches', []))}`",
        f"- Candidate candidate count: `{len(candidate.get('candidate_branches', []))}`",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    baseline_summary = baseline if isinstance(baseline, dict) else {}
    candidate_summary = candidate if isinstance(candidate, dict) else {}

    summary_keys = (
        ("head_vs_upstream_main.left", ("head_vs_upstream_main", "left")),
        ("head_vs_upstream_main.right", ("head_vs_upstream_main", "right")),
        ("head_vs_origin_branch.left", ("head_vs_origin_branch", "left")),
        ("head_vs_origin_branch.right", ("head_vs_origin_branch", "right")),
    )
    for label, path in summary_keys:
        baseline_value = nested_value(baseline_summary, path)
        candidate_value = nested_value(candidate_summary, path)
        delta = (
            candidate_value - baseline_value
            if isinstance(baseline_value, (int, float))
            and isinstance(candidate_value, (int, float))
            else None
        )
        lines.append(
            f"| {label} | {format_value(baseline_value)} | {format_value(candidate_value)} | {format_value(delta)} |"
        )

    lines.extend(
        [
            "",
            "## Branch Deltas",
            "",
            "| Branch | Baseline head | Candidate head | Status | Surface | Baseline files | Candidate files | Baseline unmatched | Candidate unmatched |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )

    for branch_name in all_branch_names:
        baseline_branch = baseline_branches.get(branch_name)
        candidate_branch = candidate_branches.get(branch_name)
        if baseline_branch is None:
            status = "new"
        elif candidate_branch is None:
            status = "removed"
        elif baseline_branch.get("head") == candidate_branch.get("head"):
            status = "unchanged"
        else:
            status = "changed"
        active_branch = candidate_branch or baseline_branch or {}
        lines.append(
            "| "
            f"`{branch_name}` | "
            f"`{format_value(baseline_branch.get('head') if baseline_branch else None)}` | "
            f"`{format_value(candidate_branch.get('head') if candidate_branch else None)}` | "
            f"{status} | "
            f"{change_surface(active_branch)} | "
            f"{len(baseline_branch.get('changed_files', [])) if isinstance(baseline_branch, dict) else 0} | "
            f"{len(candidate_branch.get('changed_files', [])) if isinstance(candidate_branch, dict) else 0} | "
            f"{len(baseline_branch.get('unmatched_patch_ids', [])) if isinstance(baseline_branch, dict) else 0} | "
            f"{len(candidate_branch.get('unmatched_patch_ids', [])) if isinstance(candidate_branch, dict) else 0} |"
        )

    summary_rows = [
        (
            "head_vs_upstream_main.left",
            nested_value(baseline_summary, ("head_vs_upstream_main", "left")),
            nested_value(candidate_summary, ("head_vs_upstream_main", "left")),
        ),
        (
            "head_vs_upstream_main.right",
            nested_value(baseline_summary, ("head_vs_upstream_main", "right")),
            nested_value(candidate_summary, ("head_vs_upstream_main", "right")),
        ),
        (
            "head_vs_origin_branch.left",
            nested_value(baseline_summary, ("head_vs_origin_branch", "left")),
            nested_value(candidate_summary, ("head_vs_origin_branch", "left")),
        ),
        (
            "head_vs_origin_branch.right",
            nested_value(baseline_summary, ("head_vs_origin_branch", "right")),
            nested_value(candidate_summary, ("head_vs_origin_branch", "right")),
        ),
    ]
    lines.extend(
        [
            "",
            "## Scan Summary Deltas",
            "",
            "| Metric | Baseline | Candidate | Delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for label, baseline_value, candidate_value in summary_rows:
        delta = (
            candidate_value - baseline_value
            if isinstance(baseline_value, (int, float))
            and isinstance(candidate_value, (int, float))
            else None
        )
        lines.append(
            f"| {label} | {format_value(baseline_value)} | {format_value(candidate_value)} | {format_value(delta)} |"
        )

    lines.extend(["", "## Candidate Notes", ""])
    candidate_summary = candidate.get("notes", [])
    if isinstance(candidate_summary, list) and candidate_summary:
        lines.extend(f"- {item}" for item in candidate_summary)
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    """Write a scan diff Markdown report."""
    args = parse_args()
    baseline = load_json(args.baseline)
    candidate = load_json(args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_scan_diff(baseline, candidate, title=args.title))
    print(f"Wrote upstream candidate scan diff to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
