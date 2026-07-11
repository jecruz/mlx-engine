#!/usr/bin/env python3
"""Preflight live LM Studio VLM validation for retained mlx-engine lanes.

The M31 VLM prompt-cache fix requires live LM Studio validation before any
broader promotion claim. This script intentionally does not edit LM Studio
internal indexes, register backends, start servers, load models, or download
artifacts. It records whether the retained VLM is currently visible to
``lms load`` and prints the supported next command when it is not.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_MODEL_KEY = "lfm2.5-vl-1.6b-mlx"
DEFAULT_MODEL_REPO = "lmstudio-community/LFM2.5-VL-1.6B-MLX-8bit"
DEFAULT_MODEL_DIR = Path(
    "/Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/"
    "lmstudio-community/LFM2.5-VL-1.6B-MLX-8bit"
)
DEFAULT_LMSTUDIO_MODEL_STORE = Path.home() / ".lmstudio/models"
DEFAULT_HF_URL = f"https://huggingface.co/{DEFAULT_MODEL_REPO}"
DEFAULT_OUTPUT = Path(".planning/lmstudio-vlm-live-validation-preflight.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-key",
        default=DEFAULT_MODEL_KEY,
        help=f"LM Studio model key expected to appear in `lms ls` ({DEFAULT_MODEL_KEY})",
    )
    parser.add_argument(
        "--model-repo",
        default=DEFAULT_MODEL_REPO,
        help=(
            "Canonical LM Studio Hugging Face repo used for store/model-data "
            f"diagnostics ({DEFAULT_MODEL_REPO})"
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Existing local MLX model directory ({DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--lmstudio-model-store",
        type=Path,
        default=DEFAULT_LMSTUDIO_MODEL_STORE,
        help=(
            "LM Studio user model store to inspect for an already-copied model "
            f"({DEFAULT_LMSTUDIO_MODEL_STORE})"
        ),
    )
    parser.add_argument(
        "--hf-url",
        default=DEFAULT_HF_URL,
        help=f"Supported `lms get` URL to register/download the VLM ({DEFAULT_HF_URL})",
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
        default=20.0,
        help="Timeout in seconds for each `lms` query command",
    )
    return parser.parse_args()


def _run_command(command: list[str], *, timeout: float) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timed_out": True,
        }


def _json_or_none(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _load_model_data(model_repo: str) -> dict[str, Any]:
    model_data_path = Path.home() / ".lmstudio/.internal/model-data.json"
    if not model_data_path.exists():
        return {
            "path": str(model_data_path),
            "exists": False,
            "contains_model_key": False,
            "entry": None,
        }

    data = json.loads(model_data_path.read_text())
    entry = None
    for key, metadata in data.get("json", []):
        if key == model_repo:
            entry = metadata
            break
    return {
        "path": str(model_data_path),
        "exists": True,
        "contains_model_key": entry is not None,
        "entry": entry,
    }


def _model_dir_status(model_dir: Path) -> dict[str, Any]:
    required_files = [
        "config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    present = {name: (model_dir / name).exists() for name in required_files}
    return {
        "path": str(model_dir),
        "exists": model_dir.exists(),
        "is_dir": model_dir.is_dir(),
        "required_files": present,
        "complete": model_dir.is_dir() and all(present.values()),
    }


def _lmstudio_store_status(store_dir: Path, model_repo: str) -> dict[str, Any]:
    model_dir = store_dir / model_repo
    status = _model_dir_status(model_dir)
    status["store_dir"] = str(store_dir)
    status["note"] = (
        "Informational only. A complete directory in this store is not enough; "
        "`lms ls --json` must expose the model key before live validation."
    )
    return status


def _model_visible_in_lms(lms_ls_json: Any, model_key: str) -> dict[str, Any]:
    if not isinstance(lms_ls_json, list):
        return {
            "visible": False,
            "matching_entries": [],
            "reason": "lms ls --json did not return a JSON list",
        }
    matches = [
        entry
        for entry in lms_ls_json
        if isinstance(entry, dict) and entry.get("modelKey") == model_key
    ]
    return {
        "visible": bool(matches),
        "matching_entries": matches,
        "reason": None if matches else "model key is absent from lms ls --json",
    }


def main() -> int:
    """Write a live LM Studio VLM validation preflight report."""
    args = _parse_args()
    commands = {
        "runtime_ls": _run_command(["lms", "runtime", "ls"], timeout=args.timeout),
        "server_status": _run_command(
            ["lms", "server", "status"], timeout=args.timeout
        ),
        "loaded_models": _run_command(["lms", "ps"], timeout=args.timeout),
        "lms_ls_json": _run_command(["lms", "ls", "--json"], timeout=args.timeout),
    }
    lms_ls_json = _json_or_none(commands["lms_ls_json"]["stdout"])
    visibility = _model_visible_in_lms(lms_ls_json, args.model_key)
    model_dir = _model_dir_status(args.model_dir)
    lmstudio_store_model_dir = _lmstudio_store_status(
        args.lmstudio_model_store,
        args.model_key,
    )
    model_data = _load_model_data(args.model_repo)
    ready = (
        commands["lms_ls_json"]["returncode"] == 0
        and visibility["visible"]
        and model_dir["complete"]
    )

    payload: dict[str, Any] = {
        "model_key": args.model_key,
        "model_repo": args.model_repo,
        "hf_url": args.hf_url,
        "ready_for_live_validation": ready,
        "model_visible_to_lms": visibility,
        "model_dir": model_dir,
        "lmstudio_store_model_dir": lmstudio_store_model_dir,
        "model_data": model_data,
        "commands": commands,
        "next_supported_commands": [
            f"lms get {args.hf_url} --mlx -y",
            f"lms load {args.model_key} --identifier m31-lfm25-vl --ttl 300 -y",
            "lms server start",
        ],
        "notes": [
            "Do not hand-edit LM Studio model-index cache files.",
            "Do not treat a copied `~/.lmstudio/models/...` directory as loadable "
            "unless `lms ls --json` also exposes the model key.",
            "Do not force `lms import` for MLX model.safetensors files; "
            "the CLI treats that as a non-model file and prompts interactively.",
            "Proceed to backend registration and live /v1/chat/completions "
            "validation only after this report says ready_for_live_validation=true.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    print(f"Wrote LM Studio VLM preflight to {args.output}")
    print(f"ready_for_live_validation={str(ready).lower()}")
    print(f"model_visible_to_lms={str(visibility['visible']).lower()}")
    print(f"model_dir_complete={str(model_dir['complete']).lower()}")
    if not ready:
        print(f"next_supported_command=lms get {args.hf_url} --mlx -y")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
