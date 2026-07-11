"""Tests for the LM Studio VLM download probe script."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "lmstudio_vlm_download_probe.py"
)


def _load_probe_module():
    spec = importlib.util.spec_from_file_location(
        "lmstudio_vlm_download_probe",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import sanity
        raise RuntimeError("Failed to load lmstudio_vlm_download_probe module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROBE = _load_probe_module()


def test_run_probe_records_zero_progress_timeout(monkeypatch):
    """Timeouts should preserve clean progress and command evidence."""

    def fake_run(*_args, **_kwargs):
        output = (
            "\x1b[?25l\r"
            "↓ To download: LFM2.5 VL 1.6B 8BIT [MLX] - 2.09 GB\x1b[0K\r"
            "⠙ [▏                     ] 0.00%          "
        )
        raise subprocess.TimeoutExpired(
            cmd=["lms", "get"],
            timeout=60,
            output=output,
            stderr="",
        )

    monkeypatch.setattr(PROBE.subprocess, "run", fake_run)
    payload = PROBE.run_probe(
        lms_bin="lms",
        hf_url="https://example.invalid/model",
        timeout=60,
        tail_lines=10,
    )

    assert payload["command"] == [
        "lms",
        "get",
        "https://example.invalid/model",
        "--mlx",
        "-y",
    ]
    assert payload["success"] is False
    assert payload["timed_out"] is True
    assert payload["returncode"] is None
    assert payload["resolved_artifact"] == "LFM2.5 VL 1.6B 8BIT [MLX] - 2.09 GB"
    assert payload["progress"]["max_percent"] == 0.0
    assert payload["stalled_at_zero"] is True
    assert all("\x1b" not in line for line in payload["output_tail"])


def test_run_probe_records_success(monkeypatch):
    """Successful downloads should return success with observed progress."""

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["lms", "get"],
            returncode=0,
            stdout=(
                "↓ To download: LFM2.5 VL 1.6B 8BIT [MLX] - 2.09 GB\n"
                "Downloading 2.09 GB...\n"
                "100.00%\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(PROBE.subprocess, "run", fake_run)
    payload = PROBE.run_probe(
        lms_bin="lms",
        hf_url="https://example.invalid/model",
        timeout=60,
        tail_lines=10,
    )

    assert payload["success"] is True
    assert payload["timed_out"] is False
    assert payload["returncode"] == 0
    assert payload["progress"]["max_percent"] == 100.0
    assert payload["stalled_at_zero"] is False


def test_resolved_artifact_survives_long_spinner_tail(monkeypatch):
    """Artifact extraction must not depend on the retained output tail length."""

    def fake_run(*_args, **_kwargs):
        output = "↓ To download: LFM2.5 VL 1.6B 8BIT [MLX] - 2.09 GB\n" + "\n".join(
            "⠙ [▏                     ] 0.00%" for _ in range(250)
        )
        raise subprocess.TimeoutExpired(
            cmd=["lms", "get"],
            timeout=60,
            output=output,
            stderr="",
        )

    monkeypatch.setattr(PROBE.subprocess, "run", fake_run)
    payload = PROBE.run_probe(
        lms_bin="lms",
        hf_url="https://example.invalid/model",
        timeout=60,
        tail_lines=20,
    )

    assert payload["resolved_artifact"] == "LFM2.5 VL 1.6B 8BIT [MLX] - 2.09 GB"
    assert len(payload["output_tail"]) == 20
