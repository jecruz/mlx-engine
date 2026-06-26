"""Scripted local-runtime streaming smoke for the M9 cheetara replacement.

This smoke validates that the ``vmlx_engine.cli serve`` local-runtime
compatibility layer (started through
``VMLX_ENGINE_LOCAL_COMPAT=1 python3.14 -m vmlx_engine.cli serve ...``)
preserves BOTH the OpenAI Responses surface (``POST /v1/responses``)
and the OpenAI Chat Completions surface (``POST /v1/chat/completions``)
on ``127.0.0.1:3181`` without ever touching ``vmlx.app.asar``.

The smoke is the M9 streaming-surface-parity evidence capture. Each
mode produces a structured JSON report with per-mode pass/fail,
HTTP status, SSE event counts, and content text. The script uses only
the Python standard library (``urllib.request`` + ``json``) so it can
run under either the cheetara ``.venv`` or the mlx-engine ``.venv-py312``
without an extra dependency.

Subcommands
-----------
- ``connect``   GET /v1/models  (cheetara local-session connect probe)
- ``chat``      POST /v1/chat/completions with ``stream=true`` (OpenAI
                Chat Completions surface; incremental ``data: <json>``
                chunks plus the terminal ``data: [DONE]`` marker).
- ``responses`` POST /v1/responses with ``stream=true`` (OpenAI
                Responses surface; typed ``event: <type>\\ndata: <json>``
                events plus the terminal ``data: [DONE]`` marker).
- ``health``    GET /health for diagnostic verification.

- ``all``       Runs ``connect``, ``chat``, ``responses``, and ``health``
                in sequence and produces a single JSON report that
                closes the M9 streaming-surface-parity assertion.

Output
------
A single JSON object is written to ``--output`` (default: stdout) with
the schema::

    {
      "base_url": str,
      "model": str,
      "results": {
        "connect":   {"status": "pass"|"fail", "details": {...}},
        "chat":      {"status": "pass"|"fail", "details": {...}},
        "responses": {"status": "pass"|"fail", "details": {...}},
        "health":    {"status": "pass"|"fail", "details": {...}}
      },
      "summary": {"total": int, "passed": int, "failed": int}
    }

The script exits 0 only when every requested subcommand reports
``status == "pass"``; otherwise it exits 1. This matches the M7
``cheetara_compat_smoke.py`` exit discipline so the local-runtime
contract is judged the same way as the external endpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional


# --- HTTP helpers -------------------------------------------------------------


def _http_get_json(
    base_url: str,
    path: str,
    *,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 30.0,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    url = f"{base_url.rstrip('/')}{path}"
    request = urllib.request.Request(url, method="GET", headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            response_headers = {key: value for key, value in response.headers.items()}
            status = response.getcode()
    except urllib.error.HTTPError as exc:
        raw = exc.read() if hasattr(exc, "read") else b""
        response_headers = {key: value for key, value in (exc.headers or {}).items()}
        status = exc.code
    body: dict[str, Any] = {}
    if raw:
        try:
            body = json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            body = {"_raw": raw.decode("utf-8", errors="replace")}
    return status, body, response_headers


def _stream_sse(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    *,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 240.0,
) -> dict[str, Any]:
    """Stream an SSE response and return a structured capture.

    Tracks BOTH typed events (``event: <type>\\ndata: <json>``) and
    untyped events (``data: <json>``) so the smoke can verify the
    OpenAI Responses surface (typed events) and the OpenAI Chat
    Completions surface (untyped data lines) through one helper.
    """
    url = f"{base_url.rstrip('/')}{path}"
    encoded = json.dumps(payload).encode("utf-8")
    request_headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    request_headers.update(headers or {})
    request = urllib.request.Request(
        url, data=encoded, method="POST", headers=request_headers
    )
    typed_events: list[dict[str, Any]] = []
    untyped_chunks: list[dict[str, Any]] = []
    raw_pieces: list[bytes] = []
    finish_reasons: list[Optional[str]] = []
    content_pieces: list[str] = []
    error_text: Optional[str] = None
    done_seen = False
    status = 0
    response_headers: dict[str, str] = {}

    try:
        response = urllib.request.urlopen(request, timeout=timeout)
        status = response.getcode()
        response_headers = {key: value for key, value in response.headers.items()}
        buffer = b""
        current_event: Optional[str] = None
        while True:
            piece = response.read(4096)
            if not piece:
                break
            raw_pieces.append(piece)
            buffer += piece
            while b"\n" in buffer:
                line_bytes, buffer = buffer.split(b"\n", 1)
                line = line_bytes.decode("utf-8", errors="replace").rstrip("\r")
                if not line:
                    current_event = None
                    continue
                if line.startswith("event:"):
                    current_event = line[len("event:") :].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                payload_str = line[len("data:") :].strip()
                if payload_str == "[DONE]":
                    done_seen = True
                    current_event = None
                    continue
                if not payload_str:
                    current_event = None
                    continue
                try:
                    event = json.loads(payload_str)
                except json.JSONDecodeError:
                    current_event = None
                    continue
                if current_event:
                    event["_event_type"] = current_event
                    typed_events.append(event)
                else:
                    untyped_chunks.append(event)
                if "error" in event and isinstance(event["error"], dict):
                    if error_text is None:
                        error_text = str(event["error"].get("message", ""))
                for choice in event.get("choices", []) or []:
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        content_pieces.append(content)
                    if choice.get("finish_reason") is not None:
                        finish_reasons.append(choice.get("finish_reason"))
                # Responses surface: collect the typed text deltas.
                if event.get("delta") and isinstance(event["delta"], str):
                    content_pieces.append(event["delta"])
                current_event = None
        response.close()
    except urllib.error.HTTPError as exc:
        status = exc.code
        response_headers = {key: value for key, value in (exc.headers or {}).items()}
        raw_pieces.append(exc.read() if hasattr(exc, "read") else b"")
    return {
        "status": status,
        "typed_events": typed_events,
        "untyped_chunks": untyped_chunks,
        "done_seen": done_seen,
        "finish_reasons": finish_reasons,
        "content_text": "".join(content_pieces),
        "raw": b"".join(raw_pieces).decode("utf-8", errors="replace"),
        "error_text": error_text,
        "response_headers": response_headers,
    }


# --- Subcommand implementations ---------------------------------------------


def _run_connect(base_url: str, model: str) -> dict[str, Any]:
    """Local connect probe: ``GET /v1/models`` returns the served model."""
    started = time.time()
    status, body, headers = _http_get_json(base_url, "/v1/models")
    elapsed_s = round(time.time() - started, 4)
    if status != 200:
        return {
            "status": "fail",
            "details": {
                "http_status": status,
                "elapsed_s": elapsed_s,
                "reason": f"GET /v1/models returned HTTP {status}",
                "body": body,
                "headers": dict(headers),
            },
        }
    data = body.get("data") or []
    if not isinstance(data, list) or len(data) == 0:
        return {
            "status": "fail",
            "details": {
                "http_status": status,
                "elapsed_s": elapsed_s,
                "reason": "/v1/models returned no models",
                "body": body,
            },
        }
    model_ids = [entry.get("id") for entry in data if isinstance(entry, dict)]
    if model not in model_ids:
        return {
            "status": "fail",
            "details": {
                "http_status": status,
                "elapsed_s": elapsed_s,
                "reason": (
                    f"requested model {model!r} not present in /v1/models"
                    f" (available: {model_ids!r})"
                ),
                "object": body.get("object"),
                "model_count": len(data),
                "model_ids": model_ids,
                "served_model": model,
                "selectable": False,
                "headers": dict(headers),
                "body": body,
            },
        }
    return {
        "status": "pass",
        "details": {
            "http_status": status,
            "elapsed_s": elapsed_s,
            "object": body.get("object"),
            "model_count": len(data),
            "model_ids": model_ids,
            "served_model": model,
            "selectable": True,
            "headers": dict(headers),
            "body": body,
        },
    }


def _run_health(base_url: str, model: str) -> dict[str, Any]:
    """Local ``/health`` probe: ready-state metadata matches ``/v1/models``."""
    started = time.time()
    status, body, headers = _http_get_json(base_url, "/health")
    elapsed_s = round(time.time() - started, 4)
    if status != 200:
        return {
            "status": "fail",
            "details": {
                "http_status": status,
                "elapsed_s": elapsed_s,
                "reason": f"GET /health returned HTTP {status}",
                "body": body,
            },
        }
    health_status = body.get("status")
    served_model = body.get("served_model")
    started_at = body.get("started_at")
    now = body.get("now")
    if health_status != "ok":
        return {
            "status": "fail",
            "details": {
                "http_status": status,
                "elapsed_s": elapsed_s,
                "reason": (
                    f"/health returned status={health_status!r}; expected 'ok'"
                ),
                "body": body,
            },
        }
    models_status, models_body, _ = _http_get_json(base_url, "/v1/models")
    consistency_check: dict[str, Any] = {
        "models_http_status": models_status,
        "models_requested_model_present": False,
        "models_consistent": False,
    }
    if models_status == 200:
        models_data = (models_body or {}).get("data") or []
        if isinstance(models_data, list):
            model_ids = [
                entry.get("id") for entry in models_data if isinstance(entry, dict)
            ]
            consistency_check["models_requested_model_present"] = (
                model in model_ids
            )
            consistency_check["models_consistent"] = (
                isinstance(served_model, str)
                and served_model in model_ids
                and model in model_ids
                and served_model == model
            )
    if not consistency_check["models_consistent"]:
        return {
            "status": "fail",
            "details": {
                "http_status": status,
                "elapsed_s": elapsed_s,
                "reason": "served_model mismatch between /health and /v1/models",
                "served_model": served_model,
                "model_path": body.get("model_path"),
                "model_type": body.get("model_type"),
                "supports_vision": body.get("supports_vision"),
                "started_at": started_at,
                "now": now,
                "uptime_s": (now - started_at) if isinstance(now, int) and isinstance(started_at, int) else None,
                "consistency_check": consistency_check,
                "headers": dict(headers),
                "body": body,
            },
        }
    return {
        "status": "pass",
        "details": {
            "http_status": status,
            "elapsed_s": elapsed_s,
            "health_status": health_status,
            "served_model": served_model,
            "model_path": body.get("model_path"),
            "model_type": body.get("model_type"),
            "supports_vision": body.get("supports_vision"),
            "started_at": started_at,
            "now": now,
            "uptime_s": (now - started_at) if isinstance(now, int) and isinstance(started_at, int) else None,
            "consistency_check": consistency_check,
            "headers": dict(headers),
            "body": body,
        },
    }


def _run_chat(
    base_url: str,
    model: str,
    *,
    expected_keyword: str = "ok",
    prompt: str = "Reply with the single word ok.",
) -> dict[str, Any]:
    """Local chat probe: incremental SSE chunks plus terminal ``[DONE]``."""
    started = time.time()
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": 32,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    capture = _stream_sse(base_url, "/v1/chat/completions", payload)
    elapsed_s = round(time.time() - started, 4)
    content_text = capture["content_text"]
    if capture["status"] != 200:
        return {
            "status": "fail",
            "details": {
                "http_status": capture["status"],
                "elapsed_s": elapsed_s,
                "reason": (
                    f"POST /v1/chat/completions returned HTTP {capture['status']}"
                ),
                "response_headers": capture.get("response_headers", {}),
                "error_text": capture.get("error_text"),
                "raw_excerpt": capture.get("raw", "")[:2000],
            },
        }
    if not capture["done_seen"]:
        return {
            "status": "fail",
            "details": {
                "http_status": capture["status"],
                "elapsed_s": elapsed_s,
                "reason": "stream terminated without [DONE] marker",
                "chunk_count": len(capture["untyped_chunks"]),
                "finish_reasons": capture["finish_reasons"],
                "content_text": content_text,
                "raw_excerpt": capture.get("raw", "")[:2000],
            },
        }
    if not capture["untyped_chunks"]:
        return {
            "status": "fail",
            "details": {
                "http_status": capture["status"],
                "elapsed_s": elapsed_s,
                "reason": "no SSE chunks received",
                "content_text": content_text,
            },
        }
    if capture.get("error_text"):
        return {
            "status": "fail",
            "details": {
                "http_status": capture["status"],
                "elapsed_s": elapsed_s,
                "reason": f"stream reported error: {capture['error_text']}",
                "content_text": content_text,
            },
        }
    if expected_keyword and expected_keyword.lower() not in content_text.lower():
        return {
            "status": "fail",
            "details": {
                "http_status": capture["status"],
                "elapsed_s": elapsed_s,
                "reason": (
                    f"final assistant text did not contain expected keyword"
                    f" {expected_keyword!r}"
                ),
                "content_text": content_text,
                "chunk_count": len(capture["untyped_chunks"]),
                "finish_reasons": capture["finish_reasons"],
            },
        }
    return {
        "status": "pass",
        "details": {
            "http_status": capture["status"],
            "elapsed_s": elapsed_s,
            "stream_done_seen": True,
            "chunk_count": len(capture["untyped_chunks"]),
            "finish_reasons": capture["finish_reasons"],
            "content_text": content_text,
            "raw_excerpt": capture.get("raw", "")[:2000],
        },
    }


def _run_responses(
    base_url: str,
    model: str,
    *,
    expected_keyword: str = "ok",
    prompt: str = "Reply with the single word ok.",
) -> dict[str, Any]:
    """Local Responses probe: typed event sequence plus terminal ``[DONE]``."""
    started = time.time()
    payload: dict[str, Any] = {
        "model": model,
        "input": prompt,
        "stream": True,
        "max_output_tokens": 32,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    capture = _stream_sse(base_url, "/v1/responses", payload)
    elapsed_s = round(time.time() - started, 4)
    content_text = capture["content_text"]
    if capture["status"] != 200:
        return {
            "status": "fail",
            "details": {
                "http_status": capture["status"],
                "elapsed_s": elapsed_s,
                "reason": f"POST /v1/responses returned HTTP {capture['status']}",
                "response_headers": capture.get("response_headers", {}),
                "error_text": capture.get("error_text"),
                "raw_excerpt": capture.get("raw", "")[:2000],
            },
        }
    if not capture["done_seen"]:
        return {
            "status": "fail",
            "details": {
                "http_status": capture["status"],
                "elapsed_s": elapsed_s,
                "reason": "stream terminated without [DONE] marker",
                "typed_event_count": len(capture["typed_events"]),
                "content_text": content_text,
                "raw_excerpt": capture.get("raw", "")[:2000],
            },
        }
    if not capture["typed_events"]:
        return {
            "status": "fail",
            "details": {
                "http_status": capture["status"],
                "elapsed_s": elapsed_s,
                "reason": "no typed Responses events received",
                "content_text": content_text,
            },
        }
    if capture.get("error_text"):
        return {
            "status": "fail",
            "details": {
                "http_status": capture["status"],
                "elapsed_s": elapsed_s,
                "reason": f"stream reported error: {capture['error_text']}",
                "content_text": content_text,
            },
        }
    event_types = [
        event.get("_event_type") for event in capture["typed_events"]
    ]
    expected_prefix = [
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
    ]
    if event_types[: len(expected_prefix)] != expected_prefix:
        return {
            "status": "fail",
            "details": {
                "http_status": capture["status"],
                "elapsed_s": elapsed_s,
                "reason": (
                    f"missing canonical Responses event prefix; got {event_types!r}"
                ),
                "event_types": event_types,
                "content_text": content_text,
            },
        }
    expected_suffix = [
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    if event_types[-len(expected_suffix):] != expected_suffix:
        return {
            "status": "fail",
            "details": {
                "http_status": capture["status"],
                "elapsed_s": elapsed_s,
                "reason": (
                    f"missing canonical Responses event suffix; got {event_types!r}"
                ),
                "event_types": event_types,
                "content_text": content_text,
            },
        }
    delta_events = [
        event for event in capture["typed_events"]
        if event.get("_event_type") == "response.output_text.delta"
    ]
    if not delta_events:
        return {
            "status": "fail",
            "details": {
                "http_status": capture["status"],
                "elapsed_s": elapsed_s,
                "reason": "no response.output_text.delta events received",
                "event_types": event_types,
                "content_text": content_text,
            },
        }
    if expected_keyword and expected_keyword.lower() not in content_text.lower():
        return {
            "status": "fail",
            "details": {
                "http_status": capture["status"],
                "elapsed_s": elapsed_s,
                "reason": (
                    f"final assistant text did not contain expected keyword"
                    f" {expected_keyword!r}"
                ),
                "content_text": content_text,
                "delta_count": len(delta_events),
            },
        }
    return {
        "status": "pass",
        "details": {
            "http_status": capture["status"],
            "elapsed_s": elapsed_s,
            "stream_done_seen": True,
            "typed_event_count": len(capture["typed_events"]),
            "untyped_chunk_count": len(capture["untyped_chunks"]),
            "event_types": event_types,
            "delta_count": len(delta_events),
            "content_text": content_text,
            "raw_excerpt": capture.get("raw", "")[:2000],
        },
    }


# --- CLI ----------------------------------------------------------------------


def _aggregate(results: dict[str, dict[str, Any]]) -> dict[str, int]:
    total = len(results)
    passed = sum(1 for r in results.values() if r.get("status") == "pass")
    failed = total - passed
    return {"total": total, "passed": passed, "failed": failed}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scripted local-runtime streaming smoke for the M9 "
            "vmlx_engine.cli serve compatibility layer. Does not touch "
            "vmlx.app.asar."
        )
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:3181",
        help="Local-compat base URL (default: http://127.0.0.1:3181)",
    )
    parser.add_argument(
        "--model",
        default="cheetara-m9",
        help="Served model id (default: cheetara-m9)",
    )
    parser.add_argument(
        "--chat-prompt",
        default="Reply with the single word ok.",
        help="Prompt for the streaming chat smoke",
    )
    parser.add_argument(
        "--chat-expected-keyword",
        default="ok",
        help="Lowercased substring the final streamed chat text must contain",
    )
    parser.add_argument(
        "--responses-prompt",
        default="Reply with the single word ok.",
        help="Prompt for the streaming Responses smoke",
    )
    parser.add_argument(
        "--responses-expected-keyword",
        default="ok",
        help="Lowercased substring the final streamed Responses text must contain",
    )
    parser.add_argument(
        "--modes",
        default="all",
        help=(
            "Comma-separated list of modes to run. "
            "Choices: connect, chat, responses, health, all. "
            "Default: all"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON report to (in addition to stdout)",
    )
    args = parser.parse_args(argv)

    base_url = args.base_url.rstrip("/")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if "all" in modes:
        modes = ["connect", "chat", "responses", "health"]

    results: dict[str, dict[str, Any]] = {}
    for mode in modes:
        if mode == "connect":
            results["connect"] = _run_connect(base_url, args.model)
        elif mode == "chat":
            results["chat"] = _run_chat(
                base_url,
                args.model,
                expected_keyword=args.chat_expected_keyword,
                prompt=args.chat_prompt,
            )
        elif mode == "responses":
            results["responses"] = _run_responses(
                base_url,
                args.model,
                expected_keyword=args.responses_expected_keyword,
                prompt=args.responses_prompt,
            )
        elif mode == "health":
            results["health"] = _run_health(base_url, args.model)
        else:
            results[mode] = {
                "status": "fail",
                "details": {"reason": f"unknown mode: {mode}"},
            }

    summary = _aggregate(results)
    report = {
        "base_url": base_url,
        "model": args.model,
        "modes_requested": modes,
        "results": results,
        "summary": summary,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
