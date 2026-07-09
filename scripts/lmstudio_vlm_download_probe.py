#!/usr/bin/env python3
"""Probe the supported LM Studio VLM download path with bounded JSON evidence."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

DEFAULT_HF_URL = "https://huggingface.co/lmstudio-community/LFM2.5-VL-1.6B-MLX-8bit"
DEFAULT_OUTPUT = Path(".planning/lmstudio-vlm-download-probe.json")
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PROGRESS_RE = re.compile(r"(?P<percent>\d+(?:\.\d+)?)%")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-url",
        default=DEFAULT_HF_URL,
        help=f"Supported LM Studio Hugging Face URL ({DEFAULT_HF_URL})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON report path ({DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Timeout in seconds for `lms get`",
    )
    parser.add_argument(
        "--lms-bin",
        default="lms",
        help="LM Studio CLI executable",
    )
    parser.add_argument(
        "--tail-lines",
        type=int,
        default=80,
        help="Number of sanitized output lines to retain in the JSON report",
    )
    return parser.parse_args()


def _coerce_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _sanitize_output(text: str) -> str:
    return ANSI_RE.sub("", text).replace("\r", "\n")


def _tail_nonempty_lines(text: str, *, limit: int) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if limit <= 0:
        return []
    return lines[-limit:]


def _extract_progress(clean_output: str) -> dict[str, Any]:
    percents = [
        float(match.group("percent"))
        for match in PROGRESS_RE.finditer(clean_output)
    ]
    return {
        "max_percent": max(percents) if percents else None,
        "last_percent": percents[-1] if percents else None,
        "samples": percents[-10:],
    }


def _extract_resolved_artifact(clean_output: str) -> str | None:
    lines = [line.strip() for line in clean_output.splitlines() if line.strip()]
    for line in reversed(lines):
        if "To download:" in line:
            return line.split("To download:", 1)[1].strip()
    return None


def run_probe(
    *,
    lms_bin: str,
    hf_url: str,
    timeout: float,
    tail_lines: int,
) -> dict[str, Any]:
    """Run `lms get` once and return a machine-readable probe payload."""
    command = [lms_bin, "get", hf_url, "--mlx", "-y"]
    start = time.monotonic()
    timed_out = False
    returncode: int | None
    stdout = ""
    stderr = ""
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = None
        stdout = _coerce_text(exc.stdout)
        stderr = _coerce_text(exc.stderr)

    duration_s = time.monotonic() - start
    combined = _sanitize_output(stdout + "\n" + stderr)
    progress = _extract_progress(combined)
    success = returncode == 0 and not timed_out
    stalled_at_zero = (
        not success
        and progress["max_percent"] is not None
        and progress["max_percent"] <= 0.0
    )

    return {
        "command": command,
        "timeout_s": timeout,
        "duration_s": duration_s,
        "returncode": returncode,
        "timed_out": timed_out,
        "success": success,
        "resolved_artifact": _extract_resolved_artifact(combined),
        "progress": progress,
        "stalled_at_zero": stalled_at_zero,
        "stdout_bytes": len(stdout.encode("utf-8")),
        "stderr_bytes": len(stderr.encode("utf-8")),
        "output_tail": _tail_nonempty_lines(combined, limit=tail_lines),
        "notes": [
            "This script only runs the supported `lms get <hf-url> --mlx -y` path.",
            "It does not edit LM Studio index/cache files.",
            "Run scripts/lmstudio_vlm_live_validation_preflight.py after a successful probe.",
        ],
    }


def main() -> int:
    """Write a bounded LM Studio download probe report."""
    args = _parse_args()
    payload = run_probe(
        lms_bin=args.lms_bin,
        hf_url=args.hf_url,
        timeout=args.timeout,
        tail_lines=args.tail_lines,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    print(f"Wrote LM Studio VLM download probe to {args.output}")
    print(f"success={str(payload['success']).lower()}")
    print(f"timed_out={str(payload['timed_out']).lower()}")
    print(f"returncode={payload['returncode']}")
    print(f"max_progress_percent={payload['progress']['max_percent']}")
    if payload["resolved_artifact"]:
        print(f"resolved_artifact={payload['resolved_artifact']}")
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
