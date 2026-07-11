"""Tests for the upstream candidate Markdown reporter."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "upstream_candidate_report.py"
)


def _load_report_module():
    spec = importlib.util.spec_from_file_location(
        "upstream_candidate_report",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import sanity
        raise RuntimeError("Failed to load upstream_candidate_report module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REPORT = _load_report_module()


def _scan_fixture():
    return {
        "repository": "/repo",
        "head": "abc1234",
        "upstream_main": "upstream/main",
        "upstream_main_head": "def5678",
        "origin_branch": "origin/current",
        "origin_branch_head": "aaaaaaa",
        "head_vs_upstream_main": {"left": 2, "right": 1},
        "head_vs_origin_branch": {"left": 3, "right": 0},
        "candidate_branches": [
            {
                "branch": "upstream/small",
                "head": "bbbbbbb",
                "ref_info": {"subject": "Improve cache"},
                "ahead_behind_vs_upstream_main": {"left": 0, "right": 1},
                "changed_files": ["M\tcache.py"],
                "unmatched_patch_ids": ["+ bbbbbbb Improve cache"],
            },
            {
                "branch": "upstream/broad",
                "head": "ccccccc",
                "ref_info": {"subject": "Large runtime | bridge"},
                "ahead_behind_vs_upstream_main": {"left": 4, "right": 20},
                "changed_files": [f"M\tfile_{idx}.py" for idx in range(20)],
                "unmatched_patch_ids": [],
            },
        ],
    }


def test_change_surface_label_uses_changed_files_and_unmatched_commits():
    """Surface labels should distinguish small, moderate, and broad branches."""
    assert REPORT.change_surface_label({"changed_files": ["M\ta.py"]}) == "small"
    assert (
        REPORT.change_surface_label(
            {"changed_files": ["M\ta.py"], "unmatched_patch_ids": ["+ a"] * 4}
        )
        == "moderate"
    )
    assert (
        REPORT.change_surface_label(
            {"changed_files": [f"M\tf{idx}.py" for idx in range(20)]}
        )
        == "broad"
    )


def test_render_markdown_includes_summary_table_and_detail_sections():
    """Markdown output should include scan summary and per-branch evidence."""
    markdown = REPORT.render_markdown(_scan_fixture(), title="M62 Scan")

    assert "# M62 Scan" in markdown
    assert (
        "| `upstream/small` | `bbbbbbb` | `small` | 1 | 1 | Improve cache |" in markdown
    )
    assert (
        "| `upstream/broad` | `ccccccc` | `broad` | 20 | 0 | Large runtime \\| bridge |"
        in markdown
    )
    assert "## `upstream/small`" in markdown
    assert "- `M\tcache.py`" in markdown
    assert "- `+ bbbbbbb Improve cache`" in markdown


def test_main_writes_markdown_report(tmp_path, monkeypatch):
    """CLI entrypoint should write a Markdown file."""
    scan = tmp_path / "scan.json"
    output = tmp_path / "scan.md"
    scan.write_text(__import__("json").dumps(_scan_fixture()))
    monkeypatch.setattr(
        "sys.argv",
        [
            "upstream_candidate_report.py",
            str(scan),
            "--output",
            str(output),
            "--title",
            "CLI Scan",
        ],
    )

    assert REPORT.main() == 0
    assert output.read_text().startswith("# CLI Scan")
