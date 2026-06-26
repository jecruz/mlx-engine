"""Tests for the cheetara-compatible remote-session smoke runner.

These tests validate the M7 cutover smoke script without standing up
the real mlx-engine adapter. They use a fake urllib transport to
exercise each subcommand's pass/fail paths, the SSE streamer, the
auth-mode detector, and the cheetara-extras forwarding.
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
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "cheetara_compat_smoke.py"
)


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location(
        "cheetara_compat_smoke", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import sanity
        raise RuntimeError("Failed to load cheetara_compat_smoke module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SMOKE = _load_smoke_module()


# --- Fake adapter HTTP server helpers --------------------------------------


class _FakeAdapterHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible fake adapter for smoke testing.

    The behaviour is controlled by the ``config`` dict set on the
    server instance. The keys we honour for testing:

    - ``models_status``: HTTP status for ``GET /v1/models`` (default 200)
    - ``health_status``: HTTP status for ``GET /health`` (default 200)
    - ``chat_status``: HTTP status for ``POST /v1/chat/completions``
    - ``chat_chunks``: list of pre-rendered SSE ``data:`` payload strings
      to send back, followed by ``"data: [DONE]\\n\\n"``. If empty and
      ``chat_status`` is 200, the handler emits a single content chunk
      with the literal text "ok".
    - ``enforce_bearer``: when True, the handler requires
      ``Authorization: Bearer secret-key`` for all non-``/health``
      routes, otherwise it returns 401.
    - ``auth_probe_status``: HTTP status returned to a minimal no-header
      chat request; defaults to ``chat_status`` if not set.
    - ``image_required_keyword``: substring that the streamed chat
      reply must contain when the request carries an image_url; the
      handler always returns "toucan" by default so the smoke's
      expected-keyword check passes.
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

    def _send_sse(self, lines: list[str]) -> None:
        body = "".join(line if line.endswith("\n\n") else line + "\n\n" for line in lines)
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _check_bearer(self) -> bool:
        if not self.config.get("enforce_bearer"):
            return True
        auth = self.headers.get("authorization", "")
        return auth == "Bearer secret-key"

    def do_GET(self) -> None:  # noqa: N802 - http.server convention
        if self.path == "/health":
            self._send_json(self.config.get("health_status", 200), {
                "status": "ok",
                "served_model": "cheetara-m7",
                "model_path": "/tmp/fake-model",
                "model_type": "lfm2-vl",
                "supports_vision": True,
                "started_at": 1700000000,
                "now": 1700000010,
            })
            return
        if self.path == "/v1/models":
            self._send_json(self.config.get("models_status", 200), {
                "object": "list",
                "data": [
                    {
                        "id": "cheetara-m7",
                        "object": "model",
                        "created": 1700000000,
                        "owned_by": "mlx-engine",
                        "supports_vision": True,
                    }
                ],
            })
            return
        self._send_json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802 - http.server convention
        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": "not found"}})
            return
        if not self._check_bearer():
            self._send_json(401, {
                "error": {
                    "message": "Missing or invalid Authorization header",
                    "type": "invalid_request_error",
                    "code": "unauthorized",
                }
            })
            return
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        body = json.loads(raw.decode("utf-8"))
        # Detect the image smoke by the presence of an image_url part.
        has_image = any(
            isinstance(message.get("content"), list)
            and any(
                isinstance(part, dict) and part.get("type") in {"image_url", "image"}
                for part in message["content"]
            )
            for message in body.get("messages", [])
        )
        if "auth_probe" in body or body.get("max_tokens", 0) <= 1 and len(body.get("messages", [])) == 1 and not body.get("stream", False):
            status = self.config.get("auth_probe_status", self.config.get("chat_status", 200))
        else:
            status = self.config.get("chat_status", 200)
        if status != 200:
            self._send_json(status, {"error": {"message": f"fake status {status}"}})
            return
        if has_image:
            keyword = self.config.get("image_required_keyword", "toucan")
            text = f"It is a {keyword}."
        else:
            text = self.config.get("text_reply", "ok")
        chunks = self.config.get("chat_chunks") or [text]
        rendered = [
            json.dumps(
                {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion.chunk",
                    "created": 1700000000,
                    "model": "cheetara-m7",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": chunk},
                            "finish_reason": None,
                        }
                    ],
                }
            )
            for chunk in chunks
        ] + [json.dumps(
            {
                "id": "chatcmpl-fake",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "cheetara-m7",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )]
        self._send_sse([f"data: {line}\n\n" for line in rendered] + ["data: [DONE]\n\n"])


class _FakeAdapterServer:
    """Context manager that starts a thread-pooled fake adapter."""

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        self.config = dict(config or {})
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.port: int = 0

    def __enter__(self) -> "_FakeAdapterServer":
        # Bind to an ephemeral port on localhost.
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeAdapterHandler)
        self._server.config = self.config
        # The handler reads config from the class, so we patch both.
        _FakeAdapterHandler.config = self.config
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="fake-adapter", daemon=True
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
    with _FakeAdapterServer() as server:
        result = SMOKE._run_connect(server.base_url, "cheetara-m7")
    assert result["status"] == "pass"
    assert result["details"]["model_count"] == 1
    assert result["details"]["model_ids"] == ["cheetara-m7"]
    assert result["details"]["selectable"] is True


def test_connect_fails_when_no_models() -> None:
    with _FakeAdapterServer({"models_status": 200}) as server:
        # Override the handler to return an empty list.
        original_models = _FakeAdapterHandler.do_GET
        def _patched(self):  # type: ignore[no-redef]
            if self.path == "/v1/models":
                self._send_json(200, {"object": "list", "data": []})
                return
            original_models(self)
        _FakeAdapterHandler.do_GET = _patched  # type: ignore[method-assign]
        try:
            result = SMOKE._run_connect(server.base_url, "cheetara-m7")
        finally:
            _FakeAdapterHandler.do_GET = original_models  # type: ignore[method-assign]
    assert result["status"] == "fail"
    assert "no models" in result["details"]["reason"]


def test_health_reports_diagnostics() -> None:
    with _FakeAdapterServer() as server:
        result = SMOKE._run_health(server.base_url, "cheetara-m7")
    assert result["status"] == "pass"
    assert result["details"]["served_model"] == "cheetara-m7"
    assert result["details"]["uptime_s"] == 10


def test_text_streams_incremental_chunks_and_done_marker() -> None:
    with _FakeAdapterServer({"chat_chunks": ["ok"]}) as server:
        result = SMOKE._run_text(server.base_url, "cheetara-m7")
    assert result["status"] == "pass"
    assert result["details"]["stream_done_seen"] is True
    assert "ok" in result["details"]["content_text"]


def test_text_fails_when_keyword_missing() -> None:
    with _FakeAdapterServer({"chat_chunks": ["something else"]}) as server:
        result = SMOKE._run_text(server.base_url, "cheetara-m7")
    assert result["status"] == "fail"
    assert "expected keyword" in result["details"]["reason"]


def test_text_fails_when_no_done_marker() -> None:
    # Patch the handler to send chunks without a [DONE] terminator.
    with _FakeAdapterServer() as server:
        original_post = _FakeAdapterHandler.do_POST

        def _patched(self):  # type: ignore[no-redef]
            if self.path != "/v1/chat/completions":
                self._send_json(404, {"error": {"message": "not found"}})
                return
            length = int(self.headers.get("content-length", "0"))
            self.rfile.read(length) if length else b""
            body_text = "data: " + json.dumps({
                "id": "chatcmpl-fake",
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": 0, "delta": {"content": "ok"}, "finish_reason": None}
                ],
            }) + "\n\n"
            encoded = body_text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        _FakeAdapterHandler.do_POST = _patched  # type: ignore[method-assign]
        try:
            result = SMOKE._run_text(server.base_url, "cheetara-m7")
        finally:
            _FakeAdapterHandler.do_POST = original_post  # type: ignore[method-assign]
    assert result["status"] == "fail"
    assert "DONE" in result["details"]["reason"]


def test_image_streams_image_grounded_answer() -> None:
    image_bytes = b"\x89PNG\r\n\x1a\n" + b"fake-image-bytes"
    with _FakeAdapterServer({"image_required_keyword": "toucan"}) as server:
        result = SMOKE._run_image(
            server.base_url,
            "cheetara-m7",
            image_path=_write_temp_image(image_bytes),
            expected_keyword="toucan",
        )
    assert result["status"] == "pass"
    assert "toucan" in result["details"]["content_text"].lower()


def test_image_fails_when_keyword_missing() -> None:
    image_bytes = b"\x89PNG\r\n\x1a\n" + b"fake-image-bytes"
    with _FakeAdapterServer({"image_required_keyword": "elephant"}) as server:
        result = SMOKE._run_image(
            server.base_url,
            "cheetara-m7",
            image_path=_write_temp_image(image_bytes),
            expected_keyword="toucan",
        )
    assert result["status"] == "fail"
    assert "toucan" in result["details"]["reason"]


def test_auth_no_auth_mode_records_intentional_disable() -> None:
    with _FakeAdapterServer() as server:
        result = SMOKE._run_auth(server.base_url, "cheetara-m7")
    assert result["status"] == "pass"
    assert result["details"]["auth_mode"] == "no-auth"
    assert "intentionally disabled" in result["details"]["evidence_note"]


def test_auth_bearer_mode_verifies_gating_with_key() -> None:
    with _FakeAdapterServer({"enforce_bearer": True}) as server:
        result = SMOKE._run_auth(
            server.base_url, "cheetara-m7", expected_api_key="secret-key"
        )
    assert result["status"] == "pass"
    assert result["details"]["auth_mode"] == "bearer-auth"
    probes = {probe["name"]: probe for probe in result["details"]["probes"]}
    assert probes["no_header"]["http_status"] == 401
    assert probes["wrong_token"]["http_status"] == 401
    assert probes["correct_token"]["http_status"] == 200


def test_auth_bearer_mode_fails_without_supplied_key() -> None:
    with _FakeAdapterServer({"enforce_bearer": True}) as server:
        result = SMOKE._run_auth(server.base_url, "cheetara-m7")
    assert result["status"] == "fail"
    assert "smoke was not invoked with --api-key" in result["details"]["reason"]


def test_detect_auth_mode_distinguishes_modes() -> None:
    with _FakeAdapterServer() as server:
        assert SMOKE._detect_auth_mode(server.base_url) == "no-auth"
    with _FakeAdapterServer({"enforce_bearer": True}) as server:
        assert SMOKE._detect_auth_mode(server.base_url) == "bearer-auth"


def test_cheetara_extras_are_forwarded() -> None:
    """The chat payload must carry cheetara extras so the adapter is exercised
    against a real cheetara-shaped request, not a minimal one."""
    extras = SMOKE._cheetara_extras()
    assert extras["top_k"] == 40
    assert extras["min_p"] == 0.05
    assert extras["repetition_penalty"] == 1.05
    assert extras["chat_template_kwargs"] == {"enable_thinking": False}
    assert extras["enable_thinking"] is False
    assert extras["reasoning_effort"] == "low"
    assert extras["stream_options"] == {"include_usage": True}


def test_main_aggregates_pass_fail_exit_code() -> None:
    """End-to-end CLI run against a healthy fake adapter exits 0 and
    produces a JSON report that names the served model and pass/fail
    per mode."""
    with _FakeAdapterServer() as server:
        captured = _capture_main(
            [
                "--base-url",
                server.base_url,
                "--modes",
                "connect,text,health,auth",
                "--image-path",
                str(_write_temp_image(b"unused")),
            ]
        )
    assert captured["exit_code"] == 0
    report = json.loads(captured["stdout"])
    assert report["base_url"] == server.base_url
    assert report["summary"]["failed"] == 0
    assert report["auth_mode"] == "no-auth"
    assert set(report["results"].keys()) == {"connect", "text", "health", "auth"}


def test_main_returns_nonzero_on_failure() -> None:
    """When a subcommand fails, the aggregate exit code is non-zero."""
    with _FakeAdapterServer({"chat_chunks": ["nope"]}) as server:
        captured = _capture_main(
            [
                "--base-url",
                server.base_url,
                "--modes",
                "text",
            ]
        )
    assert captured["exit_code"] == 1
    report = json.loads(captured["stdout"])
    assert report["summary"]["failed"] == 1
    assert report["results"]["text"]["status"] == "fail"


# --- helpers -----------------------------------------------------------------


def _write_temp_image(payload: bytes) -> Path:
    """Write a small temp file the smoke can load as an image."""
    import tempfile

    tmp = tempfile.NamedTemporaryFile(
        prefix="cheetara-compat-smoke-", suffix=".png", delete=False
    )
    tmp.write(payload)
    tmp.close()
    return Path(tmp.name)


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
