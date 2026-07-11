"""Tests for the upstream candidate history renderer."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "upstream_candidate_history.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "upstream_candidate_history",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import sanity
        raise RuntimeError("Failed to load upstream_candidate_history module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HISTORY = _load_module()


def _scan(head: str, timestamp: str, branches: list[dict[str, object]]):
    return {
        "timestamp_utc": timestamp,
        "repository": "/repo",
        "head": head,
        "upstream_main_head": "up-1",
        "origin_branch_head": "or-1",
        "candidate_branches": branches,
    }


def test_render_history_includes_scan_table_and_branch_change_counts():
    """The history report should summarize scans and pairwise branch deltas."""
    scans = [
        _scan(
            "aaa1111",
            "2026-07-09T00:00:00Z",
            [
                {
                    "branch": "upstream/a",
                    "head": "a1",
                    "changed_files": ["M\ta.py"],
                    "unmatched_patch_ids": [],
                }
            ],
        ),
        _scan(
            "bbb2222",
            "2026-07-09T01:00:00Z",
            [
                {
                    "branch": "upstream/a",
                    "head": "a2",
                    "changed_files": ["M\ta.py", "M\tb.py"],
                    "unmatched_patch_ids": [],
                },
                {
                    "branch": "upstream/b",
                    "head": "b1",
                    "changed_files": ["M\tc.py"],
                    "unmatched_patch_ids": ["+ b1 fix"],
                },
            ],
        ),
    ]

    markdown = HISTORY.render_history(scans, title="History")

    assert "# History" in markdown
    assert (
        "| 2026-07-09T00:00:00Z (aaa1111) | `aaa1111` | `up-1` | `or-1` | 1 |"
        in markdown
    )
    assert (
        "- Branch deltas: changed `1`, new `1`, removed `0`, unchanged `0`" in markdown
    )
    assert (
        "| `upstream/a` | `a1` | `a2` | changed | small | 1 | 2 | 0 | 0 |" in markdown
    )
    assert "| `upstream/b` | `` | `b1` | new | small | 0 | 1 | 0 | 1 |" in markdown


def test_main_writes_markdown(tmp_path, monkeypatch):
    """CLI should write a Markdown history report."""
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output = tmp_path / "history.md"
    first.write_text(json.dumps(_scan("aaa1111", "2026-07-09T00:00:00Z", [])))
    second.write_text(json.dumps(_scan("bbb2222", "2026-07-09T01:00:00Z", [])))
    monkeypatch.setattr(
        "sys.argv",
        [
            "upstream_candidate_history.py",
            str(first),
            str(second),
            "--output",
            str(output),
            "--title",
            "CLI History",
        ],
    )

    assert HISTORY.main() == 0
    assert output.read_text().startswith("# CLI History")
