"""Tests for the upstream candidate scan diff renderer."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "upstream_candidate_scan_diff.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "upstream_candidate_scan_diff",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import sanity
        raise RuntimeError("Failed to load upstream_candidate_scan_diff module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DIFF = _load_module()


def _baseline():
    return {
        "head": "aaa1111",
        "upstream_main_head": "up11111",
        "origin_branch_head": "or11111",
        "head_vs_upstream_main": {"left": 1, "right": 0},
        "head_vs_origin_branch": {"left": 2, "right": 0},
        "candidate_branches": [
            {
                "branch": "upstream/old",
                "head": "old0001",
                "changed_files": ["M\ta.py"],
                "unmatched_patch_ids": [],
            }
        ],
    }


def _candidate():
    return {
        "head": "bbb2222",
        "upstream_main_head": "up22222",
        "origin_branch_head": "or22222",
        "head_vs_upstream_main": {"left": 2, "right": 1},
        "head_vs_origin_branch": {"left": 3, "right": 0},
        "candidate_branches": [
            {
                "branch": "upstream/old",
                "head": "old0002",
                "changed_files": ["M\ta.py", "M\tb.py"],
                "unmatched_patch_ids": ["+ old0002 fix"],
            },
            {
                "branch": "upstream/new",
                "head": "new0001",
                "changed_files": ["M\tc.py"],
                "unmatched_patch_ids": [],
            },
        ],
        "notes": ["scan refreshed"],
    }


def test_render_scan_diff_includes_branch_status_and_counts():
    """The diff should summarize changed, new, and removed branches."""
    markdown = DIFF.render_scan_diff(_baseline(), _candidate(), title="Scan Diff")

    assert "# Scan Diff" in markdown
    assert "- Baseline head: `aaa1111`" in markdown
    assert "| `upstream/old` | `old0001` | `old0002` | changed | small | 1 | 2 | 0 | 1 |" in markdown
    assert "| `upstream/new` | `` | `new0001` | new | small | 0 | 1 | 0 | 0 |" in markdown


def test_main_writes_markdown(tmp_path, monkeypatch):
    """CLI should render a Markdown diff file."""
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "diff.md"
    baseline.write_text(json.dumps(_baseline()))
    candidate.write_text(json.dumps(_candidate()))
    monkeypatch.setattr(
        "sys.argv",
        [
            "upstream_candidate_scan_diff.py",
            str(baseline),
            str(candidate),
            "--output",
            str(output),
            "--title",
            "CLI Diff",
        ],
    )

    assert DIFF.main() == 0
    assert output.read_text().startswith("# CLI Diff")
