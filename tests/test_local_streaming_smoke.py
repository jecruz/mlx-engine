"""Tests for the M9 local-runtime streaming smoke runner.

These tests validate the M9 streaming-surface-parity smoke script
without standing up the real mlx-engine adapter or cheetara
``vmlx_engine.cli serve`` process. They use a thread-pooled fake
adapter bound to an ephemeral localhost port that emits both OpenAI
Chat Completions untyped SSE chunks AND OpenAI Responses typed
events so the smoke's per-mode pass/fail paths are exercised
end-to-end.

The smoke runner is the M9 evidence capture for VAL-M9-002
("local streaming responses and chat surfaces remain compatible"),
which requires that BOTH ``POST /v1/responses`` and
``POST /v1/chat/completions`` stream valid incremental output with
no protocol regression on the local 3181 surface.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional


# Load the smoke script as a module without requiring it to be on PYTHONPATH.
SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "local_streaming_smoke.py"
)


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location("local_streaming_smoke", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - import sanity
        raise RuntimeError("Failed to load local_streaming_smoke module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SMOKE = _load_smoke_module()


# --- Fake adapter HTTP server helpers --------------------------------------


class _FakeLocalAdapterHandler(BaseHTTPRequestHandler):
    """Minimal fake adapter for the local-streaming smoke tests.

    The behaviour is controlled by the ``config`` dict set on the
    server instance. The keys we honour:

    - ``models_status`` / ``models_returned_ids``: ``/v1/models`` shape.
    - ``health_status`` / ``health_status_field`` /
      ``health_served_model``: ``/health`` shape.
    - ``chat_status``: HTTP status for ``POST /v1/chat/completions``.
    - ``chat_chunks``: list of strings to emit as untyped SSE
      ``data:`` payloads; each chunk becomes one
      ``chat.completion.chunk`` SSE event followed by a terminal
      ``finish_reason: stop`` chunk and ``data: [DONE]``.
    - ``responses_status``: HTTP status for ``POST /v1/responses``.
    - ``responses_typed_events``: list of ``(event_type, payload)``
      tuples to emit on the Responses surface. If empty and status
      is 200, the handler emits a canonical happy-path sequence with
      three ``response.output_text.delta`` events totalling ``"ok"``.
    """

    config: dict[str, Any] = {}

    def log_message(self, *_args, **_kwargs) -> None:  # silence test noise
        return

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_raw(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server convention
        if self.path == "/health":
            self._send_json(
                self.config.get("health_status", 200),
                {
                    "status": self.config.get("health_status_field", "ok"),
                    "served_model": self.config.get(
                        "health_served_model", "cheetara-m9"
                    ),
                    "model_path": "/tmp/fake-local-model",
                    "model_type": "lfm2_vl",
                    "supports_vision": True,
                    "started_at": 1700000000,
                    "now": 1700000010,
                },
            )
            return
        if self.path == "/v1/models":
            model_ids = self.config.get("models_returned_ids", ["cheetara-m9"])
            self._send_json(
                self.config.get("models_status", 200),
                {
                    "object": "list",
                    "data": [
                        {
                            "id": model_id,
                            "object": "model",
                            "created": 1700000000,
                            "owned_by": "mlx-engine",
                            "supports_vision": True,
                        }
                        for model_id in model_ids
                    ],
                },
            )
            return
        self._send_json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802 - http.server convention
        length = int(self.headers.get("content-length", "0"))
        _ = self.rfile.read(length) if length else b"{}"

        if self.path == "/v1/chat/completions":
            self._handle_chat()
            return
        if self.path == "/v1/responses":
            self._handle_responses()
            return
        self._send_json(404, {"error": {"message": "not found"}})

    def _handle_chat(self) -> None:
        status = self.config.get("chat_status", 200)
        if status != 200:
            self._send_json(
                status, {"error": {"message": f"fake chat status {status}"}}
            )
            return
        chunks = self.config.get("chat_chunks") or ["ok"]
        rendered: list[str] = []
        for chunk in chunks:
            rendered.append(
                json.dumps(
                    {
                        "id": "chatcmpl-local-fake",
                        "object": "chat.completion.chunk",
                        "created": 1700000000,
                        "model": "cheetara-m9",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": chunk},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            )
        rendered.append(
            json.dumps(
                {
                    "id": "chatcmpl-local-fake",
                    "object": "chat.completion.chunk",
                    "created": 1700000000,
                    "model": "cheetara-m9",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )
        )
        body = "".join(f"data: {line}\n\n" for line in rendered) + "data: [DONE]\n\n"
        self._send_raw(200, body.encode("utf-8"), "text/event-stream")

    def _handle_responses(self) -> None:
        status = self.config.get("responses_status", 200)
        if status != 200:
            self._send_json(
                status, {"error": {"message": f"fake responses status {status}"}}
            )
            return
        custom = self.config.get("responses_typed_events")
        if custom is not None:
            events = custom
        else:
            events = self._default_happy_responses_events()
        body_pieces: list[str] = []
        for event_type, payload in events:
            body_pieces.append(f"event: {event_type}\ndata: {json.dumps(payload)}\n\n")
        body_pieces.append("data: [DONE]\n\n")
        self._send_raw(200, "".join(body_pieces).encode("utf-8"), "text/event-stream")

    def _default_happy_responses_events(self) -> list[tuple[str, dict[str, Any]]]:
        """Canonical happy-path Responses event sequence with three text deltas."""
        sequence = 0

        def _next() -> int:
            nonlocal sequence
            current = sequence
            sequence += 1
            return current

        return [
            (
                "response.created",
                {
                    "type": "response.created",
                    "sequence_number": _next(),
                    "response": {
                        "id": "resp_local_fake",
                        "object": "response",
                        "created_at": 1700000000,
                        "status": "in_progress",
                        "model": "cheetara-m9",
                        "output": [],
                    },
                },
            ),
            (
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": _next(),
                    "output_index": 0,
                    "item": {
                        "id": "item_local_fake",
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    },
                },
            ),
            (
                "response.content_part.added",
                {
                    "type": "response.content_part.added",
                    "sequence_number": _next(),
                    "item_id": "item_local_fake",
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                },
            ),
            (
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "sequence_number": _next(),
                    "item_id": "item_local_fake",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "o",
                },
            ),
            (
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "sequence_number": _next(),
                    "item_id": "item_local_fake",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "k",
                },
            ),
            (
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "sequence_number": _next(),
                    "item_id": "item_local_fake",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": ".",
                },
            ),
            (
                "response.content_part.done",
                {
                    "type": "response.content_part.done",
                    "sequence_number": _next(),
                    "item_id": "item_local_fake",
                    "output_index": 0,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "text": "ok.",
                        "annotations": [],
                    },
                },
            ),
            (
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "sequence_number": _next(),
                    "output_index": 0,
                    "item": {
                        "id": "item_local_fake",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "ok.",
                                "annotations": [],
                            }
                        ],
                    },
                },
            ),
            (
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": _next(),
                    "response": {
                        "id": "resp_local_fake",
                        "object": "response",
                        "created_at": 1700000000,
                        "status": "completed",
                        "model": "cheetara-m9",
                        "output": [
                            {
                                "id": "item_local_fake",
                                "type": "message",
                                "status": "completed",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "ok.",
                                        "annotations": [],
                                    }
                                ],
                            }
                        ],
                    },
                },
            ),
        ]


class _FakeLocalAdapterServer:
    """Context manager that starts a thread-pooled fake local-compat adapter."""

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        self.config = dict(config or {})
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.port: int = 0

    def __enter__(self) -> "_FakeLocalAdapterServer":
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeLocalAdapterHandler)
        self._server.config = self.config
        _FakeLocalAdapterHandler.config = self.config
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="fake-local-adapter", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


# --- Tests ------------------------------------------------------------------


def test_connect_reports_served_model() -> None:
    with _FakeLocalAdapterServer() as server:
        result = SMOKE._run_connect(server.base_url, "cheetara-m9")
    assert result["status"] == "pass"
    assert result["details"]["model_count"] == 1
    assert result["details"]["model_ids"] == ["cheetara-m9"]
    assert result["details"]["selectable"] is True


def test_connect_fails_when_requested_model_absent() -> None:
    with _FakeLocalAdapterServer({"models_returned_ids": ["other-model"]}) as server:
        result = SMOKE._run_connect(server.base_url, "cheetara-m9")
    assert result["status"] == "fail"
    assert "cheetara-m9" in result["details"]["reason"]
    assert "other-model" in result["details"]["reason"]


def test_health_reports_diagnostics() -> None:
    with _FakeLocalAdapterServer() as server:
        result = SMOKE._run_health(server.base_url, "cheetara-m9")
    assert result["status"] == "pass"
    assert result["details"]["health_status"] == "ok"
    assert result["details"]["served_model"] == "cheetara-m9"
    consistency = result["details"]["consistency_check"]
    assert consistency["models_consistent"] is True


def test_health_fails_when_status_field_is_not_ok() -> None:
    with _FakeLocalAdapterServer({"health_status_field": "degraded"}) as server:
        result = SMOKE._run_health(server.base_url, "cheetara-m9")
    assert result["status"] == "fail"
    assert "'degraded'" in result["details"]["reason"]


def test_health_fails_when_models_consistency_mismatches() -> None:
    with _FakeLocalAdapterServer({"health_served_model": "drift"}) as server:
        result = SMOKE._run_health(server.base_url, "cheetara-m9")
    assert result["status"] == "fail"
    assert "mismatch" in result["details"]["reason"].lower()


def test_chat_streams_incremental_chunks_and_done_marker() -> None:
    with _FakeLocalAdapterServer({"chat_chunks": ["o", "k", "."]}) as server:
        result = SMOKE._run_chat(server.base_url, "cheetara-m9")
    assert result["status"] == "pass"
    assert result["details"]["stream_done_seen"] is True
    assert result["details"]["content_text"] == "ok."
    assert result["details"]["chunk_count"] >= 3


def test_chat_fails_when_done_marker_missing() -> None:
    with _FakeLocalAdapterServer() as server:
        original_post = _FakeLocalAdapterHandler.do_POST

        def _patched(self):  # type: ignore[no-redef]
            if self.path != "/v1/chat/completions":
                self._send_json(404, {"error": {"message": "not found"}})
                return
            length = int(self.headers.get("content-length", "0"))
            self.rfile.read(length) if length else b""
            body_text = (
                "data: "
                + json.dumps(
                    {
                        "id": "chatcmpl-fake",
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "ok"},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                + "\n\n"
            )
            self._send_raw(200, body_text.encode("utf-8"), "text/event-stream")

        _FakeLocalAdapterHandler.do_POST = _patched  # type: ignore[method-assign]
        try:
            result = SMOKE._run_chat(server.base_url, "cheetara-m9")
        finally:
            _FakeLocalAdapterHandler.do_POST = original_post  # type: ignore[method-assign]
    assert result["status"] == "fail"
    assert "DONE" in result["details"]["reason"]


def test_chat_fails_when_keyword_missing() -> None:
    with _FakeLocalAdapterServer({"chat_chunks": ["different", "answer"]}) as server:
        result = SMOKE._run_chat(server.base_url, "cheetara-m9")
    assert result["status"] == "fail"
    assert "expected keyword" in result["details"]["reason"]


def test_responses_streams_typed_events_with_done_marker() -> None:
    with _FakeLocalAdapterServer() as server:
        result = SMOKE._run_responses(server.base_url, "cheetara-m9")
    assert result["status"] == "pass"
    assert result["details"]["stream_done_seen"] is True
    event_types = result["details"]["event_types"]
    assert event_types[0] == "response.created"
    assert event_types[-1] == "response.completed"
    assert result["details"]["content_text"] == "ok."
    assert result["details"]["delta_count"] == 3


def test_responses_fails_when_done_marker_missing() -> None:
    with _FakeLocalAdapterServer() as server:
        original_post = _FakeLocalAdapterHandler.do_POST

        def _patched(self):  # type: ignore[no-redef]
            if self.path != "/v1/responses":
                self._send_json(404, {"error": {"message": "not found"}})
                return
            length = int(self.headers.get("content-length", "0"))
            self.rfile.read(length) if length else b""
            body_text = (
                "event: response.created\n"
                "data: "
                + json.dumps(
                    {
                        "type": "response.created",
                        "sequence_number": 0,
                        "response": {"id": "resp_x", "object": "response"},
                    }
                )
                + "\n\n"
            )
            self._send_raw(200, body_text.encode("utf-8"), "text/event-stream")

        _FakeLocalAdapterHandler.do_POST = _patched  # type: ignore[method-assign]
        try:
            result = SMOKE._run_responses(server.base_url, "cheetara-m9")
        finally:
            _FakeLocalAdapterHandler.do_POST = original_post  # type: ignore[method-assign]
    assert result["status"] == "fail"
    assert "DONE" in result["details"]["reason"]


def test_responses_fails_when_typed_event_prefix_missing() -> None:
    """A Responses stream without the canonical event prefix is rejected."""
    bad_events = [
        (
            "response.created",
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {"id": "resp_x", "object": "response"},
            },
        ),
        (
            "response.completed",
            {
                "type": "response.completed",
                "sequence_number": 1,
                "response": {"id": "resp_x", "object": "response"},
            },
        ),
    ]
    with _FakeLocalAdapterServer({"responses_typed_events": bad_events}) as server:
        result = SMOKE._run_responses(server.base_url, "cheetara-m9")
    assert result["status"] == "fail"
    assert "prefix" in result["details"]["reason"].lower()


def test_responses_fails_when_typed_event_suffix_missing() -> None:
    """A Responses stream without the canonical event suffix is rejected."""
    truncated_events = [
        (
            "response.created",
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {"id": "resp_x", "object": "response"},
            },
        ),
        (
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": {"id": "item_x", "type": "message", "role": "assistant"},
            },
        ),
        (
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "sequence_number": 2,
                "item_id": "item_x",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": ""},
            },
        ),
        (
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "sequence_number": 3,
                "item_id": "item_x",
                "output_index": 0,
                "content_index": 0,
                "delta": "o",
            },
        ),
        (
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "sequence_number": 4,
                "item_id": "item_x",
                "output_index": 0,
                "content_index": 0,
                "delta": "k",
            },
        ),
    ]
    with _FakeLocalAdapterServer(
        {"responses_typed_events": truncated_events}
    ) as server:
        result = SMOKE._run_responses(server.base_url, "cheetara-m9")
    assert result["status"] == "fail"
    assert "suffix" in result["details"]["reason"].lower()


def test_responses_fails_when_no_text_deltas_emitted() -> None:
    """A Responses stream that skips ``response.output_text.delta`` is rejected."""
    no_delta_events = [
        (
            "response.created",
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {"id": "resp_x", "object": "response"},
            },
        ),
        (
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": {"id": "item_x", "type": "message", "role": "assistant"},
            },
        ),
        (
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "sequence_number": 2,
                "item_id": "item_x",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": ""},
            },
        ),
        (
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "sequence_number": 3,
                "item_id": "item_x",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "ok."},
            },
        ),
        (
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "sequence_number": 4,
                "output_index": 0,
                "item": {"id": "item_x", "type": "message", "role": "assistant"},
            },
        ),
        (
            "response.completed",
            {
                "type": "response.completed",
                "sequence_number": 5,
                "response": {"id": "resp_x", "object": "response"},
            },
        ),
    ]
    with _FakeLocalAdapterServer({"responses_typed_events": no_delta_events}) as server:
        result = SMOKE._run_responses(server.base_url, "cheetara-m9")
    assert result["status"] == "fail"
    assert "delta" in result["details"]["reason"].lower()


def test_main_aggregates_pass_fail_exit_code() -> None:
    """End-to-end CLI run against a healthy fake adapter exits 0 with
    all four M9 streaming modes reporting pass."""
    with _FakeLocalAdapterServer() as server:
        captured = _capture_main(
            [
                "--base-url",
                server.base_url,
                "--model",
                "cheetara-m9",
                "--modes",
                "all",
            ]
        )
    assert captured["exit_code"] == 0
    report = json.loads(captured["stdout"])
    assert report["summary"]["failed"] == 0
    assert report["summary"]["passed"] == 4
    assert set(report["results"].keys()) == {
        "connect",
        "chat",
        "responses",
        "health",
    }
    # Responses surface evidence: typed events with the canonical
    # prefix/suffix and incremental ``response.output_text.delta`` deltas.
    responses_details = report["results"]["responses"]["details"]
    event_types = responses_details["event_types"]
    assert event_types[0] == "response.created"
    assert event_types[-1] == "response.completed"
    assert responses_details["delta_count"] >= 1
    # Chat surface evidence: incremental chunks + terminal DONE marker.
    chat_details = report["results"]["chat"]["details"]
    assert chat_details["stream_done_seen"] is True
    assert chat_details["chunk_count"] >= 1


def test_main_returns_nonzero_on_responses_failure() -> None:
    """A Responses surface that emits no typed events fails (HTTP 500)."""
    with _FakeLocalAdapterServer({"responses_status": 500}) as server:
        captured = _capture_main(
            [
                "--base-url",
                server.base_url,
                "--model",
                "cheetara-m9",
                "--modes",
                "responses",
            ]
        )
    assert captured["exit_code"] == 1
    report = json.loads(captured["stdout"])
    assert report["results"]["responses"]["status"] == "fail"


def test_main_returns_nonzero_on_chat_failure() -> None:
    """A chat surface that returns HTTP 500 fails the smoke."""
    with _FakeLocalAdapterServer({"chat_status": 500}) as server:
        captured = _capture_main(
            [
                "--base-url",
                server.base_url,
                "--model",
                "cheetara-m9",
                "--modes",
                "chat",
            ]
        )
    assert captured["exit_code"] == 1
    report = json.loads(captured["stdout"])
    assert report["results"]["chat"]["status"] == "fail"


def test_main_returns_nonzero_on_connect_failure() -> None:
    """A connect probe against an absent served model fails."""
    with _FakeLocalAdapterServer({"models_returned_ids": ["only-other"]}) as server:
        captured = _capture_main(
            [
                "--base-url",
                server.base_url,
                "--model",
                "cheetara-m9",
                "--modes",
                "connect",
            ]
        )
    assert captured["exit_code"] == 1
    report = json.loads(captured["stdout"])
    assert report["results"]["connect"]["status"] == "fail"


def test_main_returns_nonzero_on_health_failure() -> None:
    """A health surface with a status mismatch fails the smoke."""
    with _FakeLocalAdapterServer({"health_served_model": "drift"}) as server:
        captured = _capture_main(
            [
                "--base-url",
                server.base_url,
                "--model",
                "cheetara-m9",
                "--modes",
                "health",
            ]
        )
    assert captured["exit_code"] == 1
    report = json.loads(captured["stdout"])
    assert report["results"]["health"]["status"] == "fail"


# --- helpers -----------------------------------------------------------------


def _capture_main(argv: list[str]) -> dict[str, Any]:
    """Invoke ``SMOKE.main`` while capturing stdout and exit code."""
    backup_stdout = sys.stdout
    try:
        captured = io.StringIO()
        sys.stdout = captured
        exit_code = SMOKE.main(argv)
    finally:
        sys.stdout = backup_stdout
    return {"exit_code": exit_code, "stdout": captured.getvalue()}
