"""Tests for the LM Studio VLM live-validation preflight script."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "lmstudio_vlm_live_validation_preflight.py"
)


def _load_preflight_module():
    spec = importlib.util.spec_from_file_location(
        "lmstudio_vlm_live_validation_preflight",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import sanity
        raise RuntimeError(
            "Failed to load lmstudio_vlm_live_validation_preflight module spec"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PRECHECK = _load_preflight_module()


def test_model_visibility_uses_loadable_lmstudio_key():
    """The preflight should key off the LM Studio-visible model id."""
    visibility = PRECHECK._model_visible_in_lms(
        [{"modelKey": "lfm2.5-vl-1.6b-mlx"}],
        "lfm2.5-vl-1.6b-mlx",
    )

    assert visibility["visible"] is True
    assert visibility["reason"] is None


def test_main_passes_when_visible_key_and_repo_metadata_both_match(
    tmp_path, monkeypatch
):
    """Default preflight should pass once the visible key is present."""

    def fake_run_command(command, *, timeout):
        if command == ["lms", "ls", "--json"]:
            return {
                "command": command,
                "returncode": 0,
                "stdout": json.dumps(
                    [
                        {"modelKey": "lfm2.5-vl-1.6b-mlx"},
                        {"modelKey": "other-model"},
                    ]
                ),
                "stderr": "",
                "timed_out": False,
            }
        return {
            "command": command,
            "returncode": 0,
            "stdout": "ok",
            "stderr": "",
            "timed_out": False,
        }

    monkeypatch.setattr(PRECHECK, "_run_command", fake_run_command)
    monkeypatch.setattr(
        PRECHECK,
        "_model_dir_status",
        lambda _model_dir: {"complete": True, "path": "model-dir"},
    )
    monkeypatch.setattr(
        PRECHECK,
        "_lmstudio_store_status",
        lambda _store_dir, _model_repo: {"complete": True, "path": "store-dir"},
    )
    monkeypatch.setattr(
        PRECHECK,
        "_load_model_data",
        lambda _model_repo: {"contains_model_key": True, "entry": {"source": "hf"}},
    )
    output = tmp_path / "preflight.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lmstudio_vlm_live_validation_preflight.py",
            "--output",
            str(output),
        ],
    )

    assert PRECHECK.main() == 0
    payload = json.loads(output.read_text())

    assert payload["ready_for_live_validation"] is True
    assert payload["model_key"] == "lfm2.5-vl-1.6b-mlx"
    assert payload["model_repo"] == "lmstudio-community/LFM2.5-VL-1.6B-MLX-8bit"
    assert payload["model_visible_to_lms"]["visible"] is True
    assert payload["model_visible_to_lms"]["reason"] is None
