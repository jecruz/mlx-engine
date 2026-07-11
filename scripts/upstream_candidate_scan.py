#!/usr/bin/env python3
"""Generate a JSON upstream candidate scan report."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

DEFAULT_CANDIDATE_BRANCHES = [
    "upstream/neil/gemma4-tool-context",
    "upstream/yagil/dist",
    "upstream/yagil/mlx-dist-non-batched",
    "upstream/neil/vlm-parity-ci",
    "upstream/will/lfm-2.5-unified",
    "upstream/neil/img-caching",
]


@dataclass(frozen=True)
class GitResult:
    """Captured git command result."""

    command: list[str]
    returncode: int
    stdout: str
    stderr: str


GitRunner = Callable[[list[str]], GitResult]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream-main", default="upstream/main")
    parser.add_argument(
        "--origin-branch",
        default="origin/mlx-vlm-restore-eval-followup",
    )
    parser.add_argument(
        "--candidate-branch",
        action="append",
        dest="candidate_branches",
        help="Remote candidate branch to inspect. May be passed multiple times.",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Run `git fetch --all --prune` before collecting facts.",
    )
    parser.add_argument(
        "--recent-commit-limit",
        type=int,
        default=12,
        help="Recent commits retained per candidate branch.",
    )
    return parser.parse_args()


def run_git(args: list[str]) -> GitResult:
    """Run one git command and capture stdout/stderr."""
    completed = subprocess.run(
        ["git", *args],
        check=False,
        text=True,
        capture_output=True,
    )
    return GitResult(
        command=["git", *args],
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def split_nonempty_lines(text: str) -> list[str]:
    """Return stripped non-empty output lines."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def parse_rev_counts(result: GitResult) -> dict[str, int | None]:
    """Parse `git rev-list --left-right --count` output."""
    if result.returncode != 0:
        return {"left": None, "right": None}
    parts = result.stdout.strip().split()
    if len(parts) != 2:
        return {"left": None, "right": None}
    return {"left": int(parts[0]), "right": int(parts[1])}


def parse_for_each_ref_line(line: str) -> dict[str, str | None]:
    """Parse one pipe-delimited for-each-ref line."""
    parts = line.split("|", 3)
    if len(parts) != 4:
        return {
            "ref": None,
            "head": None,
            "committer_date": None,
            "subject": line,
        }
    ref, head, committer_date, subject = parts
    return {
        "ref": ref,
        "head": head,
        "committer_date": committer_date,
        "subject": subject,
    }


def command_payload(result: GitResult) -> dict[str, Any]:
    """Return a JSON-safe command result payload."""
    return {
        "command": result.command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def branch_summary(
    branch: str,
    *,
    upstream_main: str,
    recent_commit_limit: int,
    git_runner: GitRunner,
) -> dict[str, Any]:
    """Return a factual summary for one candidate branch."""
    head = git_runner(["rev-parse", "--short", branch])
    for_each_ref = git_runner(
        [
            "for-each-ref",
            "--format=%(refname:short)|%(objectname:short)|%(committerdate:iso8601)|%(subject)",
            f"refs/remotes/{branch}",
        ]
    )
    counts = git_runner(
        ["rev-list", "--left-right", "--count", f"{upstream_main}...{branch}"]
    )
    commits = git_runner(
        [
            "log",
            "--oneline",
            f"--max-count={recent_commit_limit}",
            f"{upstream_main}..{branch}",
        ]
    )
    diffstat = git_runner(["diff", "--stat", f"{upstream_main}..{branch}"])
    names = git_runner(["diff", "--name-status", f"{upstream_main}..{branch}"])
    cherry = git_runner(["cherry", "-v", "HEAD", branch])

    ref_info = (
        parse_for_each_ref_line(split_nonempty_lines(for_each_ref.stdout)[0])
        if for_each_ref.returncode == 0 and split_nonempty_lines(for_each_ref.stdout)
        else None
    )

    return {
        "branch": branch,
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "ref_info": ref_info,
        "ahead_behind_vs_upstream_main": parse_rev_counts(counts),
        "recent_commits": split_nonempty_lines(commits.stdout),
        "diffstat": diffstat.stdout.strip(),
        "changed_files": split_nonempty_lines(names.stdout),
        "unmatched_patch_ids": split_nonempty_lines(cherry.stdout),
        "commands": {
            "head": command_payload(head),
            "for_each_ref": command_payload(for_each_ref),
            "ahead_behind": command_payload(counts),
            "recent_commits": command_payload(commits),
            "diffstat": command_payload(diffstat),
            "changed_files": command_payload(names),
            "cherry": command_payload(cherry),
        },
    }


def build_report(
    *,
    upstream_main: str,
    origin_branch: str,
    candidate_branches: list[str],
    fetch: bool,
    recent_commit_limit: int,
    git_runner: GitRunner = run_git,
) -> dict[str, Any]:
    """Build a read-only upstream candidate scan report."""
    commands: dict[str, Any] = {}
    if fetch:
        commands["fetch"] = command_payload(git_runner(["fetch", "--all", "--prune"]))

    head = git_runner(["rev-parse", "--short", "HEAD"])
    upstream_head = git_runner(["rev-parse", "--short", upstream_main])
    origin_head = git_runner(["rev-parse", "--short", origin_branch])
    upstream_counts = git_runner(
        ["rev-list", "--left-right", "--count", f"HEAD...{upstream_main}"]
    )
    origin_counts = git_runner(
        ["rev-list", "--left-right", "--count", f"HEAD...{origin_branch}"]
    )
    refs = git_runner(
        [
            "for-each-ref",
            "--sort=-committerdate",
            "--format=%(refname:short)|%(objectname:short)|%(committerdate:iso8601)|%(subject)",
            "refs/remotes/upstream",
        ]
    )

    commands.update(
        {
            "head": command_payload(head),
            "upstream_head": command_payload(upstream_head),
            "origin_head": command_payload(origin_head),
            "upstream_counts": command_payload(upstream_counts),
            "origin_counts": command_payload(origin_counts),
            "upstream_refs": command_payload(refs),
        }
    )

    return {
        "repository": str(Path.cwd()),
        "upstream_main": upstream_main,
        "origin_branch": origin_branch,
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "upstream_main_head": upstream_head.stdout.strip()
        if upstream_head.returncode == 0
        else None,
        "origin_branch_head": origin_head.stdout.strip()
        if origin_head.returncode == 0
        else None,
        "head_vs_upstream_main": parse_rev_counts(upstream_counts),
        "head_vs_origin_branch": parse_rev_counts(origin_counts),
        "upstream_refs": [
            parse_for_each_ref_line(line) for line in split_nonempty_lines(refs.stdout)
        ],
        "candidate_branches": [
            branch_summary(
                branch,
                upstream_main=upstream_main,
                recent_commit_limit=recent_commit_limit,
                git_runner=git_runner,
            )
            for branch in candidate_branches
        ],
        "commands": commands,
        "notes": [
            "This report is factual only; it does not classify promotion readiness.",
            "A cherry-pick still requires human triage, retained benchmarks, quality gates, and live LM Studio validation before promotion.",
        ],
    }


def main() -> int:
    """Write an upstream candidate scan report."""
    args = parse_args()
    candidate_branches = args.candidate_branches or DEFAULT_CANDIDATE_BRANCHES
    report = build_report(
        upstream_main=args.upstream_main,
        origin_branch=args.origin_branch,
        candidate_branches=candidate_branches,
        fetch=args.fetch,
        recent_commit_limit=args.recent_commit_limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"Wrote upstream candidate scan to {args.output}")
    print(
        json.dumps(
            {
                "head": report["head"],
                "upstream_main_head": report["upstream_main_head"],
                "candidate_branch_count": len(report["candidate_branches"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
