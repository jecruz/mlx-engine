"""Scripted cheetara-compatible remote-session smoke for the M7 adapter.

This module is the M7 cutover evidence runner. It exercises the
host-local mlx-engine OpenAI-compatible adapter at
``127.0.0.1:3180`` with cheetara-compatible request shapes so the
external-endpoint cutover can be proven without packaged GUI
automation. It does NOT touch ``vmlx.app.asar``.

Subcommands
-----------
- ``connect``    GET /v1/models  (cheetara remote-session connect probe)
- ``text``       POST /v1/chat/completions with stream=true, plain text
                 message; cheetara extras (top_k, min_p,
                 repetition_penalty, chat_template_kwargs,
                 enable_thinking, reasoning_effort, stream_options)
                 are forwarded so the request shape matches what a
                 real cheetara client sends.
- ``image``      POST /v1/chat/completions with stream=true and an
                 OpenAI-style multimodal content array carrying a
                 ``data:image/jpeg;base64,...`` URL plus text. Verifies
                 that the adapter routes the image into the VLM path
                 and the streamed reply is image-grounded.
- ``health``     GET /health for diagnostic verification.
- ``auth``       Probes the auth gate by attempting requests with no
                 header, a wrong bearer token, and (if a key is
                 configured) the correct bearer token. The result
                 records which auth mode the adapter is currently
                 running under; this satisfies the "auth behavior is
                 explicitly validated for the configured run mode"
                 requirement.
- ``all``        Runs ``connect``, ``text``, ``image``, ``health``,
                 and ``auth`` in sequence and produces a single JSON
                 report that closes the M7 external-cutover
                 assertions.

Output
------
A single JSON object is written to ``--output`` (default: stdout) with
the schema::

    {
      "base_url": str,
      "model": str,
      "auth_mode": "no-auth" | "bearer-auth" | "unknown",
      "results": {
        "connect": {"status": "pass"|"fail", "details": {...}},
        "text":    {"status": "pass"|"fail", "details": {...}},
        "image":   {"status": "pass"|"fail", "details": {...}},
        "health":  {"status": "pass"|"fail", "details": {...}},
        "auth":    {"status": "pass"|"fail", "details": {...}}
      },
      "summary": {
        "total": int,
        "passed": int,
        "failed": int
      }
    }

The script exits 0 only when every requested subcommand reports
``status == "pass"``; otherwise it exits 1.

Design notes
------------
- Uses only the Python standard library (``urllib.request`` + ``json``)
  so it can run under the cheetara Python 3.14 venv *or* the
  mlx-engine py312 venv without an extra dependency.
- Streams SSE responses via ``http.client`` so the smoke can observe
  the incremental ``data: { ... }\n\n`` chunks AND the terminal
  ``data: [DONE]\n\n`` marker. Verifying the terminal marker is the
  M7 streaming contract; trusting just the exit code is not enough.
- Forwards cheetara extras the way a real cheetara client would; the
  adapter is contractually obligated to ignore them safely, and the
  smoke records that tolerance in the per-mode details.
"""

from __future__ import annotations

import argparse
import base64
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
    """GET a JSON resource and return ``(status, body, response_headers)``."""
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


def _http_post_json(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    *,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 240.0,
) -> tuple[int, dict[str, Any], dict[str, str], bytes]:
    """POST a JSON payload and return ``(status, body, headers, raw)``."""
    url = f"{base_url.rstrip('/')}{path}"
    encoded = json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json"}
    request_headers.update(headers or {})
    request = urllib.request.Request(
        url, data=encoded, method="POST", headers=request_headers
    )
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
    return status, body, response_headers, raw


# --- SSE streamer -------------------------------------------------------------


def _stream_sse(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    *,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 240.0,
) -> dict[str, Any]:
    """Stream an SSE response and return a structured capture.

    Returns a dict with:
      ``status``: HTTP status (e.g. 200, 401)
      ``chunks``: list of decoded JSON payloads in stream order
      ``done_seen``: True iff the terminal ``[DONE]`` marker was seen
      ``finish_reasons``: list of finish_reason values seen in the stream
      ``content_text``: the concatenated assistant delta content
      ``raw``: the raw response body bytes (utf-8 decoded best-effort)
      ``error_text``: the error message of the first error chunk, if any
    """
    url = f"{base_url.rstrip('/')}{path}"
    encoded = json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    request_headers.update(headers or {})
    request = urllib.request.Request(
        url, data=encoded, method="POST", headers=request_headers
    )
    chunks: list[dict[str, Any]] = []
    finish_reasons: list[Optional[str]] = []
    content_pieces: list[str] = []
    error_text: Optional[str] = None
    done_seen = False
    raw_pieces: list[bytes] = []
    status = 0
    response_headers: dict[str, str] = {}

    try:
        response = urllib.request.urlopen(request, timeout=timeout)
        status = response.getcode()
        response_headers = {key: value for key, value in response.headers.items()}
        buffer = b""
        while True:
            piece = response.read(4096)
            if not piece:
                break
            raw_pieces.append(piece)
            buffer += piece
            while b"\n\n" in buffer:
                event_bytes, buffer = buffer.split(b"\n\n", 1)
                for line in event_bytes.splitlines():
                    line = line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload_str = line[len("data:") :].strip()
                    if payload_str == "[DONE]":
                        done_seen = True
                        continue
                    if not payload_str:
                        continue
                    try:
                        event = json.loads(payload_str)
                    except json.JSONDecodeError:
                        continue
                    chunks.append(event)
                    if "error" in event and isinstance(event["error"], dict):
                        if error_text is None:
                            error_text = str(event["error"].get("message", ""))
                        continue
                    for choice in event.get("choices", []) or []:
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            content_pieces.append(content)
                        if choice.get("finish_reason") is not None:
                            finish_reasons.append(choice.get("finish_reason"))
        response.close()
    except urllib.error.HTTPError as exc:
        status = exc.code
        response_headers = {key: value for key, value in (exc.headers or {}).items()}
        raw_pieces.append(exc.read() if hasattr(exc, "read") else b"")
    return {
        "status": status,
        "chunks": chunks,
        "done_seen": done_seen,
        "finish_reasons": finish_reasons,
        "content_text": "".join(content_pieces),
        "raw": b"".join(raw_pieces).decode("utf-8", errors="replace"),
        "error_text": error_text,
        "response_headers": response_headers,
    }


# --- Subcommand implementations ---------------------------------------------


def _run_connect(base_url: str, model: str) -> dict[str, Any]:
    """VAL-M7-001: cheetara remote-session connect through GET /v1/models.

    Hardened: the smoke now FAILS when the requested model id is absent
    from the ``/v1/models`` response. Previously a non-matching
    ``selectable`` was recorded as ``pass``; the reusable surface must
    fail loudly so cheetara can distinguish a real connect from a
    misconfigured server during the M7 user-testing pass.
    """
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
    """VAL-M7-005: adapter diagnostics expose a useful /health route.

    Hardened: the smoke now requires ``status == "ok"`` on the
    ``/health`` response AND verifies ``served_model`` consistency
    between ``/health`` and ``/v1/models``. A ``/health`` that 200s
    but reports ``status != "ok"`` is a diagnostic surface failure,
    and a mismatch between the health and discovery surfaces means
    one of them is stale or misconfigured.
    """
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
    # Cross-check served_model consistency with /v1/models so an
    # adapter that mis-reports its served model name is caught here
    # instead of leaking into chat errors.
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
                entry.get("id")
                for entry in models_data
                if isinstance(entry, dict)
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
                "reason": (
                    "served_model mismatch between /health and /v1/models"
                ),
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


def _cheetara_extras() -> dict[str, Any]:
    """The cheetara-style optional knobs the smoke forwards on every chat call.

    These match the cheetara / vmlx_engine client field set seen in
    ``vmlx_engine.api`` and ``vmlx_engine.loaders.dsv4_chat_encoder``
    (``top_k``, ``min_p``, ``repetition_penalty``,
    ``chat_template_kwargs`` with ``enable_thinking``,
    ``reasoning_effort``). The adapter is contractually obligated to
    ignore them safely; the smoke records that tolerance as evidence.
    """
    return {
        "top_k": 40,
        "min_p": 0.05,
        "repetition_penalty": 1.05,
        "repetition_context_size": 20,
        "chat_template_kwargs": {"enable_thinking": False},
        "enable_thinking": False,
        "reasoning_effort": "low",
        "stream_options": {"include_usage": True},
        "user": "cheetara-compat-smoke",
    }


def _run_text(
    base_url: str,
    model: str,
    *,
    expected_keyword: str = "ok",
    prompt: str = "Reply with the single word ok.",
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    """VAL-M7-002: streaming text chat works end to end."""
    started = time.time()
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": 32,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    payload.update(_cheetara_extras())
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    capture = _stream_sse(base_url, "/v1/chat/completions", payload, headers=headers)
    elapsed_s = round(time.time() - started, 4)
    content_text = capture["content_text"]
    if capture["status"] != 200:
        return {
            "status": "fail",
            "details": {
                "http_status": capture["status"],
                "elapsed_s": elapsed_s,
                "reason": f"POST /v1/chat/completions returned HTTP {capture['status']}",
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
                "chunk_count": len(capture["chunks"]),
                "finish_reasons": capture["finish_reasons"],
                "content_text": content_text,
                "raw_excerpt": capture.get("raw", "")[:2000],
            },
        }
    if not capture["chunks"]:
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
                "reason": f"final assistant text did not contain expected keyword {expected_keyword!r}",
                "content_text": content_text,
                "chunk_count": len(capture["chunks"]),
                "finish_reasons": capture["finish_reasons"],
            },
        }
    return {
        "status": "pass",
        "details": {
            "http_status": capture["status"],
            "elapsed_s": elapsed_s,
            "stream_done_seen": True,
            "chunk_count": len(capture["chunks"]),
            "finish_reasons": capture["finish_reasons"],
            "content_text": content_text,
            "request_extras": _cheetara_extras(),
            "raw_excerpt": capture.get("raw", "")[:2000],
        },
    }


def _encode_image_for_data_url(image_path: Path) -> str:
    """Read an image file and return ``data:image/<ext>;base64,<payload>``."""
    suffix = image_path.suffix.lower().lstrip(".")
    mime = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "gif": "image/gif",
    }.get(suffix, "application/octet-stream")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _run_image(
    base_url: str,
    model: str,
    *,
    image_path: Path,
    expected_keyword: str = "toucan",
    prompt: str = "What bird is this? Reply with the bird name in one short sentence.",
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    """VAL-M7-003: image-attachment VLM chat works end to end."""
    started = time.time()
    data_url = _encode_image_for_data_url(image_path)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "stream": True,
        "max_tokens": 64,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    payload.update(_cheetara_extras())
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    capture = _stream_sse(base_url, "/v1/chat/completions", payload, headers=headers)
    elapsed_s = round(time.time() - started, 4)
    content_text = capture["content_text"]
    if capture["status"] != 200:
        return {
            "status": "fail",
            "details": {
                "http_status": capture["status"],
                "elapsed_s": elapsed_s,
                "reason": f"POST /v1/chat/completions returned HTTP {capture['status']}",
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
                "chunk_count": len(capture["chunks"]),
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
                "reason": f"image-grounded answer did not contain expected keyword {expected_keyword!r}",
                "content_text": content_text,
                "chunk_count": len(capture["chunks"]),
                "finish_reasons": capture["finish_reasons"],
            },
        }
    return {
        "status": "pass",
        "details": {
            "http_status": capture["status"],
            "elapsed_s": elapsed_s,
            "stream_done_seen": True,
            "chunk_count": len(capture["chunks"]),
            "finish_reasons": capture["finish_reasons"],
            "content_text": content_text,
            "image_path": str(image_path),
            "image_size_bytes": image_path.stat().st_size,
            "image_mime": data_url.split(";")[0].split(":", 1)[1],
            "request_extras": _cheetara_extras(),
            "raw_excerpt": capture.get("raw", "")[:2000],
        },
    }


def _detect_auth_mode(base_url: str) -> str:
    """Return ``"bearer-auth"`` if the adapter enforces a bearer token,
    ``"no-auth"`` if it does not, or ``"unknown"`` if it cannot tell.
    """
    # A minimal valid chat request without auth: if it 200s, the adapter
    # is in no-auth mode. If it 401s, the adapter is in bearer-auth mode.
    payload = {
        "model": "cheetara-m7",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }
    status, _body, _headers, _raw = _http_post_json(
        base_url, "/v1/chat/completions", payload, timeout=15.0
    )
    if status == 200:
        return "no-auth"
    if status == 401:
        return "bearer-auth"
    return "unknown"


def _run_auth(
    base_url: str,
    model: str,
    *,
    expected_api_key: Optional[str] = None,
) -> dict[str, Any]:
    """VAL-M7-004: auth behavior is explicitly validated for the run mode.

    The smoke records the adapter's current auth posture. If the
    adapter is in ``bearer-auth`` mode and a key was supplied, the
    smoke additionally proves the gate by issuing requests with no
    header, a wrong token, and the correct token, and asserting that
    only the correct token returns 200. If the adapter is in
    ``no-auth`` mode, the smoke records that auth was intentionally
    disabled for this run.
    """
    started = time.time()
    auth_mode = _detect_auth_mode(base_url)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }
    probes: list[dict[str, Any]] = []
    if auth_mode == "bearer-auth":
        if not expected_api_key:
            return {
                "status": "fail",
                "details": {
                    "auth_mode": auth_mode,
                    "reason": (
                        "adapter enforces bearer auth but smoke was not "
                        "invoked with --api-key; cannot prove credential gating"
                    ),
                },
            }
        # No header: must 401.
        status_no_auth, _, _, _ = _http_post_json(
            base_url, "/v1/chat/completions", payload, timeout=15.0
        )
        probes.append({"name": "no_header", "http_status": status_no_auth})
        # Wrong token: must 401.
        status_wrong, _, _, _ = _http_post_json(
            base_url,
            "/v1/chat/completions",
            payload,
            headers={"Authorization": "Bearer wrong-key"},
            timeout=15.0,
        )
        probes.append({"name": "wrong_token", "http_status": status_wrong})
        # Correct token: must 200.
        status_correct, body_correct, _, _ = _http_post_json(
            base_url,
            "/v1/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {expected_api_key}"},
            timeout=15.0,
        )
        probes.append({"name": "correct_token", "http_status": status_correct})
        if status_no_auth != 401 or status_wrong != 401 or status_correct != 200:
            return {
                "status": "fail",
                "details": {
                    "auth_mode": auth_mode,
                    "elapsed_s": round(time.time() - started, 4),
                    "probes": probes,
                    "reason": "auth gate did not match expected behavior",
                },
            }
        return {
            "status": "pass",
            "details": {
                "auth_mode": auth_mode,
                "elapsed_s": round(time.time() - started, 4),
                "probes": probes,
                "credential_gating": "verified",
                "evidence_note": (
                    "bearer-auth active: missing/wrong tokens returned "
                    "401, correct token returned 200"
                ),
            },
        }
    if auth_mode == "no-auth":
        # No gate: must be 200 for the no-header probe.
        status_no_auth, body_no_auth, _, _ = _http_post_json(
            base_url, "/v1/chat/completions", payload, timeout=15.0
        )
        probes.append({"name": "no_header", "http_status": status_no_auth})
        return {
            "status": "pass",
            "details": {
                "auth_mode": auth_mode,
                "elapsed_s": round(time.time() - started, 4),
                "probes": probes,
                "credential_gating": "disabled",
                "evidence_note": (
                    "auth was intentionally disabled for this run; "
                    "no-auth request succeeded with HTTP 200"
                ),
                "sample_body": body_no_auth,
            },
        }
    return {
        "status": "fail",
        "details": {
            "auth_mode": auth_mode,
            "elapsed_s": round(time.time() - started, 4),
            "reason": (
                f"could not determine adapter auth posture (auth_mode={auth_mode})"
            ),
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
            "Scripted cheetara-compatible remote-session smoke for the "
            "M7 mlx-engine adapter. Does not touch vmlx.app.asar."
        )
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:3180",
        help="Adapter base URL (default: http://127.0.0.1:3180)",
    )
    parser.add_argument(
        "--model",
        default="cheetara-m7",
        help="Served model id (default: cheetara-m7)",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=Path("/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/demo-data/toucan.jpeg"),
        help="Image to use for the image-attachment smoke (default: demo-data/toucan.jpeg)",
    )
    parser.add_argument(
        "--text-prompt",
        default="Reply with the single word ok.",
        help="Prompt for the streaming text smoke",
    )
    parser.add_argument(
        "--text-expected-keyword",
        default="ok",
        help="Lowercased substring the final streamed text must contain",
    )
    parser.add_argument(
        "--image-prompt",
        default="What bird is this? Reply with the bird name in one short sentence.",
        help="Prompt for the image-attachment VLM smoke",
    )
    parser.add_argument(
        "--image-expected-keyword",
        default="toucan",
        help="Lowercased substring the image-grounded answer must contain",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "Bearer token to use when the adapter is started with --api-key. "
            "If the adapter rejects this token, the auth mode is reported as "
            "bearer-auth and the smoke probes missing/wrong/correct tokens."
        ),
    )
    parser.add_argument(
        "--modes",
        default="all",
        help=(
            "Comma-separated list of modes to run. "
            "Choices: connect, text, image, health, auth, all. "
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
        modes = ["connect", "text", "image", "health", "auth"]

    results: dict[str, dict[str, Any]] = {}
    for mode in modes:
        if mode == "connect":
            results["connect"] = _run_connect(base_url, args.model)
        elif mode == "text":
            results["text"] = _run_text(
                base_url,
                args.model,
                expected_keyword=args.text_expected_keyword,
                prompt=args.text_prompt,
                api_key=args.api_key,
            )
        elif mode == "image":
            if not args.image_path.exists():
                results["image"] = {
                    "status": "fail",
                    "details": {
                        "reason": f"image not found at {args.image_path}",
                    },
                }
            else:
                results["image"] = _run_image(
                    base_url,
                    args.model,
                    image_path=args.image_path,
                    expected_keyword=args.image_expected_keyword,
                    prompt=args.image_prompt,
                    api_key=args.api_key,
                )
        elif mode == "health":
            results["health"] = _run_health(base_url, args.model)
        elif mode == "auth":
            results["auth"] = _run_auth(
                base_url, args.model, expected_api_key=args.api_key
            )
        else:
            results[mode] = {
                "status": "fail",
                "details": {"reason": f"unknown mode: {mode}"},
            }

    auth_mode = results.get("auth", {}).get("details", {}).get("auth_mode")
    if not auth_mode:
        # Try to detect from a probe we just ran.
        auth_mode = _detect_auth_mode(base_url)

    summary = _aggregate(results)
    report = {
        "base_url": base_url,
        "model": args.model,
        "auth_mode": auth_mode,
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
