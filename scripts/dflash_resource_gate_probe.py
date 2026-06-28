#!/usr/bin/env python3
"""Record live evidence for the M14 DFlash resource gate.

This script probes the local host for evidence that distinguishes a cloud-only
LLMDYNAMIX listener on ``127.0.0.1:12444`` from a local MLX/Metal-heavy load.
It writes a structured JSON report so the gate's decision can be audited
without relying on a verbal user claim.

Run from the mlx-engine repo root so ``mlx_engine`` resolves.

    .venv-py312/bin/python scripts/dflash_resource_gate_probe.py \\
        --target /Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/lmstudio-community/Qwen3.6-27B-MLX-8bit \\
        --drafter /Volumes/StudioStackSSD4TB/Development/LLM/huggingface/hub/models--z-lab--Qwen3.5-27B-DFlash/snapshots/25ee0025ff950496a634e100b75c2db4515e9824 \\
        --output .planning/dflash-resource-gate-evidence.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlx_engine.utils.dflash_boundary import (  # noqa: E402
    DFlashBoundaryOptions,
    build_port_blocker,
    probe_all_listener_evidence,
    probe_dflash_readiness,
    validate_dflash_preload_compatibility,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(
            "/Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/"
            "lmstudio-community/Qwen3.6-27B-MLX-8bit"
        ),
        help="DFlash target model path",
    )
    parser.add_argument(
        "--drafter",
        type=Path,
        default=Path(
            "/Volumes/StudioStackSSD4TB/Development/LLM/huggingface/hub/"
            "models--z-lab--Qwen3.5-27B-DFlash/snapshots/"
            "25ee0025ff950496a634e100b75c2db4515e9824"
        ),
        help="DFlash drafter snapshot path",
    )
    parser.add_argument(
        "--max-draft-tokens",
        type=int,
        default=4,
        help="Maximum draft tokens per round (default: 4)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".planning/dflash-resource-gate-evidence.json"),
        help="Where to write the JSON evidence report",
    )
    return parser.parse_args()


def _evidence_to_dict(evidence) -> dict[str, object]:
    return {
        "port": evidence.port,
        "classification": evidence.classification.value,
        "is_allowed": evidence.is_allowed(),
        "pid": evidence.pid,
        "comm": evidence.comm,
        "command": evidence.command,
        "cloud_backend_count": evidence.cloud_backend_count,
        "local_heavy_backend_count": evidence.local_heavy_backend_count,
        "config_path": str(evidence.config_path) if evidence.config_path else None,
        "blocker": build_port_blocker(evidence),
        "notes": list(evidence.notes),
    }


def _report_to_dict(report) -> dict[str, object]:
    return {
        "enabled": report.enabled,
        "dependency_available": report.dependency_available,
        "target_family": report.target_family,
        "drafter_family": report.drafter_family,
        "target_profile": (
            {
                "model_path": str(report.target_profile.model_path),
                "config_path": str(report.target_profile.config_path),
                "vocab_size": report.target_profile.vocab_size,
                "tokenizer_vocab_size": report.target_profile.tokenizer_vocab_size,
                "num_hidden_layers": report.target_profile.num_hidden_layers,
                "architectures": list(report.target_profile.architectures),
                "model_type": report.target_profile.model_type,
            }
            if report.target_profile is not None
            else None
        ),
        "cache_mode_blockers": list(report.cache_mode_blockers),
        "route_blockers": list(report.route_blockers),
        "resource_blockers": list(report.resource_blockers),
        "blockers": list(report.blockers),
        "listener_evidence": [
            _evidence_to_dict(ev) for ev in report.listener_evidence
        ],
    }


def main() -> int:
    args = _parse_args()
    listener_evidence = probe_all_listener_evidence()
    options = DFlashBoundaryOptions(
        enabled=True,
        target_model_path=args.target,
        drafter_model_path=args.drafter,
        max_draft_tokens=args.max_draft_tokens,
    )
    try:
        report = validate_dflash_preload_compatibility(
            options=options,
            loaded_model_path=args.target,
            is_vlm_route=False,
            vocab_only=False,
            distributed=False,
            max_seq_nums=1,
            kv_bits=None,
            kv_group_size=None,
            quantized_kv_start=None,
            vlm_prompt_cache_storage_root=None,
            vlm_prompt_cache_min_save_tokens=None,
        )
        readiness_error = None
    except Exception as exc:  # pragma: no cover - reporting probe path
        report = probe_dflash_readiness(options)
        readiness_error = f"{type(exc).__name__}: {exc}"

    payload = {
        "target": str(args.target),
        "drafter": str(args.drafter),
        "max_draft_tokens": args.max_draft_tokens,
        "ready_for_dflash_smoke": readiness_error is None,
        "readiness_error": readiness_error,
        "report": _report_to_dict(report),
        "listener_evidence_summary": [
            _evidence_to_dict(ev) for ev in listener_evidence
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote DFlash resource gate evidence to {args.output}")
    print(
        "ready_for_dflash_smoke=",
        payload["ready_for_dflash_smoke"],
        "cloud_only_listener=",
        any(
            ev.classification.value == "cloud-only-llmdynamix"
            for ev in listener_evidence
        ),
        "blocked_listener=",
        any(
            not ev.is_allowed()
            for ev in listener_evidence
            if ev.classification.value != "empty"
        ),
    )
    return 0 if payload["ready_for_dflash_smoke"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
