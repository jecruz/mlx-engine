#!/usr/bin/env python3
"""Render a readable timeline from multiple upstream candidate scan JSON reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scans", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Upstream Candidate Scan History")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON report from disk."""
    return json.loads(path.read_text())


def branch_map(scan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index candidate branches by branch name."""
    branches = scan.get("candidate_branches", [])
    if not isinstance(branches, list):
        return {}
    return {
        branch.get("branch"): branch
        for branch in branches
        if isinstance(branch, dict) and branch.get("branch")
    }


def branch_surface(branch: dict[str, Any]) -> str:
    """Classify a branch surface using changed-file and unmatched-commit counts."""
    changed_files = branch.get("changed_files", [])
    unmatched = branch.get("unmatched_patch_ids", [])
    changed_count = len(changed_files) if isinstance(changed_files, list) else 0
    unmatched_count = len(unmatched) if isinstance(unmatched, list) else 0
    if changed_count >= 20 or unmatched_count >= 10:
        return "broad"
    if changed_count >= 6 or unmatched_count >= 4:
        return "moderate"
    return "small"


def format_value(value: Any) -> str:
    """Format a value for Markdown output."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return ""
    return str(value)


def scan_label(scan: dict[str, Any]) -> str:
    """Return a stable label for one scan."""
    timestamp = scan.get("timestamp_utc")
    head = scan.get("head")
    if timestamp and head:
        return f"{timestamp} ({head})"
    if timestamp:
        return str(timestamp)
    return str(head or "")


def branch_delta(previous: dict[str, Any], current: dict[str, Any]) -> str:
    """Return a simple branch-status label between two scans."""
    if previous.get("head") == current.get("head"):
        return "unchanged"
    return "changed"


def render_history(scans: list[dict[str, Any]], *, title: str) -> str:
    """Render a sequence of scan JSON payloads as Markdown."""
    if len(scans) < 2:
        raise ValueError("At least two scan reports are required")

    lines = [
        f"# {title}",
        "",
        "This report is evidence only. It records upstream scan history and",
        "does not classify promotion readiness.",
        "",
        "| Scan | Head | Upstream/main | Origin branch | Candidate branches |",
        "| --- | --- | --- | --- | ---: |",
    ]

    for scan in scans:
        lines.append(
            "| "
            f"{scan_label(scan)} | "
            f"`{format_value(scan.get('head'))}` | "
            f"`{format_value(scan.get('upstream_main_head'))}` | "
            f"`{format_value(scan.get('origin_branch_head'))}` | "
            f"{len(scan.get('candidate_branches', [])) if isinstance(scan.get('candidate_branches', []), list) else 0} |"
        )

    lines.extend(["", "## Scan-by-Scan Changes", ""])
    for index, scan in enumerate(scans):
        label = scan_label(scan)
        lines.extend(
            [
                f"### Scan `{label}`",
                "",
                f"- Repository: `{format_value(scan.get('repository'))}`",
                f"- Head: `{format_value(scan.get('head'))}`",
                f"- Upstream/main: `{format_value(scan.get('upstream_main_head'))}`",
                f"- Origin branch: `{format_value(scan.get('origin_branch_head'))}`",
                f"- Candidate branches: `{len(scan.get('candidate_branches', [])) if isinstance(scan.get('candidate_branches', []), list) else 0}`",
                "",
            ]
        )
        if index == 0:
            lines.append("- Baseline scan in this history sequence.")
        else:
            previous = scans[index - 1]
            prev_map = branch_map(previous)
            curr_map = branch_map(scan)
            names = sorted(set(prev_map) | set(curr_map))
            changed = 0
            new = 0
            removed = 0
            unchanged = 0
            for name in names:
                prev_branch = prev_map.get(name)
                curr_branch = curr_map.get(name)
                if prev_branch is None:
                    new += 1
                elif curr_branch is None:
                    removed += 1
                elif prev_branch.get("head") == curr_branch.get("head"):
                    unchanged += 1
                else:
                    changed += 1
            lines.extend(
                [
                    f"- Baseline scan: `{scan_label(previous)}`",
                    f"- Branch deltas: changed `{changed}`, new `{new}`, removed `{removed}`, unchanged `{unchanged}`",
                    "",
                    "| Branch | Previous head | Current head | Status | Surface | Previous files | Current files | Previous unmatched | Current unmatched |",
                    "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for name in names:
                prev_branch = prev_map.get(name)
                curr_branch = curr_map.get(name)
                if prev_branch is None:
                    status = "new"
                    active_branch = curr_branch or {}
                elif curr_branch is None:
                    status = "removed"
                    active_branch = prev_branch or {}
                elif prev_branch.get("head") == curr_branch.get("head"):
                    status = "unchanged"
                    active_branch = curr_branch
                else:
                    status = "changed"
                    active_branch = curr_branch
                lines.append(
                    "| "
                    f"`{name}` | "
                    f"`{format_value(prev_branch.get('head') if prev_branch else None)}` | "
                    f"`{format_value(curr_branch.get('head') if curr_branch else None)}` | "
                    f"{status} | "
                    f"{branch_surface(active_branch)} | "
                    f"{len(prev_branch.get('changed_files', [])) if isinstance(prev_branch, dict) else 0} | "
                    f"{len(curr_branch.get('changed_files', [])) if isinstance(curr_branch, dict) else 0} | "
                    f"{len(prev_branch.get('unmatched_patch_ids', [])) if isinstance(prev_branch, dict) else 0} | "
                    f"{len(curr_branch.get('unmatched_patch_ids', [])) if isinstance(curr_branch, dict) else 0} |"
                )
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- The history is factual only; it does not make a cherry-pick or promotion decision.",
            "- Any candidate still requires retained benchmarks, readable reports, and live LM Studio validation before promotion.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """Write a Markdown history report from multiple scan JSON files."""
    args = parse_args()
    scans = [load_json(path) for path in args.scans]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_history(scans, title=args.title))
    print(f"Wrote upstream candidate history to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
