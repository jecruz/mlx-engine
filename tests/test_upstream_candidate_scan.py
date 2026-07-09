"""Tests for the upstream candidate scan reporter."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "upstream_candidate_scan.py"
)


def _load_scan_module():
    spec = importlib.util.spec_from_file_location(
        "upstream_candidate_scan",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import sanity
        raise RuntimeError("Failed to load upstream_candidate_scan module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCAN = _load_scan_module()


def _fake_git(args: list[str]):
    command = ["git", *args]
    stdout = ""
    if args[:3] == ["rev-parse", "--short", "HEAD"]:
        stdout = "abc1234\n"
    elif args[:2] == ["rev-parse", "--short"] and args[2] == "upstream/main":
        stdout = "def5678\n"
    elif args[:2] == ["rev-parse", "--short"] and args[2] == "origin/current":
        stdout = "aaaaaaa\n"
    elif args[:2] == ["rev-parse", "--short"]:
        stdout = "bbbbbbb\n"
    elif args[:3] == ["rev-list", "--left-right", "--count"]:
        stdout = "3\t1\n"
    elif args and args[0] == "for-each-ref" and args[-1] == "refs/remotes/upstream":
        stdout = "upstream/main|def5678|2026-07-09 00:00:00 -0400|Main head\n"
    elif args and args[0] == "for-each-ref":
        stdout = "upstream/topic|bbbbbbb|2026-07-08 00:00:00 -0400|Topic head\n"
    elif args[:2] == ["log", "--oneline"]:
        stdout = "bbbbbbb Improve cache\nccccccc Fix stream\n"
    elif args[:2] == ["diff", "--stat"]:
        stdout = " file.py | 2 ++\n 1 file changed, 2 insertions(+)\n"
    elif args[:2] == ["diff", "--name-status"]:
        stdout = "M\tfile.py\n"
    elif args[:2] == ["cherry", "-v"]:
        stdout = "+ bbbbbbb Improve cache\n"
    return SCAN.GitResult(command=command, returncode=0, stdout=stdout, stderr="")


def test_parse_rev_counts_handles_expected_output():
    """Rev-list count parsing should preserve left and right counts."""
    result = SCAN.GitResult(
        command=["git", "rev-list"],
        returncode=0,
        stdout="12\t3\n",
        stderr="",
    )

    assert SCAN.parse_rev_counts(result) == {"left": 12, "right": 3}


def test_build_report_collects_candidate_branch_facts():
    """Report builder should include refs, counts, commits, files, and cherry output."""
    report = SCAN.build_report(
        upstream_main="upstream/main",
        origin_branch="origin/current",
        candidate_branches=["upstream/topic"],
        fetch=False,
        recent_commit_limit=2,
        git_runner=_fake_git,
    )

    assert report["head"] == "abc1234"
    assert report["upstream_main_head"] == "def5678"
    assert report["head_vs_upstream_main"] == {"left": 3, "right": 1}
    assert report["upstream_refs"][0]["ref"] == "upstream/main"
    branch = report["candidate_branches"][0]
    assert branch["branch"] == "upstream/topic"
    assert branch["head"] == "bbbbbbb"
    assert branch["recent_commits"] == ["bbbbbbb Improve cache", "ccccccc Fix stream"]
    assert branch["changed_files"] == ["M\tfile.py"]
    assert branch["unmatched_patch_ids"] == ["+ bbbbbbb Improve cache"]


def test_main_writes_json_report(tmp_path, monkeypatch):
    """CLI entrypoint should write a JSON report."""
    output = tmp_path / "scan.json"
    monkeypatch.setattr(SCAN, "run_git", _fake_git)
    monkeypatch.setattr(
        "sys.argv",
        [
            "upstream_candidate_scan.py",
            "--output",
            str(output),
            "--origin-branch",
            "origin/current",
            "--candidate-branch",
            "upstream/topic",
        ],
    )

    assert SCAN.main() == 0
    payload = json.loads(output.read_text())
    assert payload["candidate_branches"][0]["branch"] == "upstream/topic"
