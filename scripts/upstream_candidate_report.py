#!/usr/bin/env python3
"""Render a readable Markdown report from an upstream candidate scan JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--title",
        default="Upstream Candidate Scan",
        help="Markdown title for the generated report.",
    )
    return parser.parse_args()


def load_scan_report(path: Path) -> dict[str, Any]:
    """Load a JSON scan report from disk."""
    return json.loads(path.read_text())


def change_surface_label(branch: dict[str, Any]) -> str:
    """Return a boundedness label based on changed files and unmatched commits."""
    changed_file_count = len(branch.get("changed_files", []))
    unmatched_count = len(branch.get("unmatched_patch_ids", []))
    if changed_file_count >= 20 or unmatched_count >= 10:
        return "broad"
    if changed_file_count >= 6 or unmatched_count >= 4:
        return "moderate"
    return "small"


def branch_subject(branch: dict[str, Any]) -> str:
    """Return the current branch subject if the scan captured one."""
    ref_info = branch.get("ref_info") or {}
    return str(ref_info.get("subject") or "")


def render_markdown(scan: dict[str, Any], *, title: str) -> str:
    """Render the scan report as Markdown."""
    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        f"- Repository: `{scan.get('repository', '')}`",
        f"- Head: `{scan.get('head', '')}`",
        f"- Upstream main: `{scan.get('upstream_main', '')}` at `{scan.get('upstream_main_head', '')}`",
        f"- Origin branch: `{scan.get('origin_branch', '')}` at `{scan.get('origin_branch_head', '')}`",
        f"- HEAD vs upstream/main: `{scan.get('head_vs_upstream_main', {})}`",
        f"- HEAD vs origin branch: `{scan.get('head_vs_origin_branch', {})}`",
        "",
        "This report is readable evidence only. It does not make a promotion or",
        "cherry-pick decision. Runtime changes still require retained benchmarks,",
        "quality gates, candidate-vs-baseline deltas, and live LM Studio validation.",
        "",
        "## Candidate Branches",
        "",
        "| Branch | Head | Surface | Changed files | Unmatched commits | Subject |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]

    for branch in scan.get("candidate_branches", []):
        name = branch.get("branch", "")
        head = branch.get("head", "")
        changed_file_count = len(branch.get("changed_files", []))
        unmatched_count = len(branch.get("unmatched_patch_ids", []))
        surface = change_surface_label(branch)
        subject = branch_subject(branch).replace("|", "\\|")
        lines.append(
            f"| `{name}` | `{head}` | `{surface}` | {changed_file_count} | {unmatched_count} | {subject} |"
        )

    for branch in scan.get("candidate_branches", []):
        name = branch.get("branch", "")
        lines.extend(
            [
                "",
                f"## `{name}`",
                "",
                f"- Head: `{branch.get('head', '')}`",
                f"- Subject: {branch_subject(branch)}",
                f"- Ahead/behind vs upstream main: `{branch.get('ahead_behind_vs_upstream_main', {})}`",
                f"- Change surface: `{change_surface_label(branch)}`",
                f"- Changed files: `{len(branch.get('changed_files', []))}`",
                f"- Unmatched patch-id commits: `{len(branch.get('unmatched_patch_ids', []))}`",
                "",
                "### Changed Files",
                "",
            ]
        )
        changed_files = branch.get("changed_files", [])
        if changed_files:
            lines.extend(f"- `{entry}`" for entry in changed_files)
        else:
            lines.append("- None")

        lines.extend(["", "### Unmatched Patch IDs", ""])
        unmatched = branch.get("unmatched_patch_ids", [])
        if unmatched:
            lines.extend(f"- `{entry}`" for entry in unmatched)
        else:
            lines.append("- None")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    """Render a Markdown report from a scan JSON file."""
    args = parse_args()
    scan = load_scan_report(args.scan_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(scan, title=args.title))
    print(f"Wrote upstream candidate report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
