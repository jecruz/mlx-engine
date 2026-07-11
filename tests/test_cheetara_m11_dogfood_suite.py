"""Tests for the M11 mixed text + vision dogfood suite."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "cheetara_m11_dogfood_suite.py"
)


def _load_suite_module():
    spec = importlib.util.spec_from_file_location(
        "cheetara_m11_dogfood_suite", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import sanity
        raise RuntimeError("Failed to load cheetara_m11_dogfood_suite module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SUITE = _load_suite_module()


class _FakeM11Handler(BaseHTTPRequestHandler):
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

    def _send_sse_chunks(self, text: str) -> None:
        chunks = [text[: len(text) // 2], text[len(text) // 2 :]]
        rendered: list[str] = []
        for chunk in chunks:
            if chunk:
                rendered.append(
                    json.dumps(
                        {
                            "id": "chatcmpl-m11-fake",
                            "object": "chat.completion.chunk",
                            "created": 1700000000,
                            "model": "cheetara-m11",
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
                    "id": "chatcmpl-m11-fake",
                    "object": "chat.completion.chunk",
                    "created": 1700000000,
                    "model": "cheetara-m11",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )
        )
        body = "".join(f"data: {line}\n\n" for line in rendered) + "data: [DONE]\n\n"
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802 - http.server convention
        if self.path == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.config.get("served_model", "cheetara-m7"),
                            "object": "model",
                            "created": 1700000000,
                            "owned_by": "mlx-engine",
                            "supports_vision": True,
                        }
                    ],
                },
            )
            return
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "status": self.config.get("health_status_field", "ok"),
                    "served_model": self.config.get("served_model", "cheetara-m7"),
                    "model_path": "/tmp/fake-model",
                    "model_type": "lfm2_vl",
                    "supports_vision": self.config.get("supports_vision", True),
                    "started_at": 1700000000,
                    "now": 1700000010,
                },
            )
            return
        self._send_json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802 - http.server convention
        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": "not found"}})
            return
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        body = json.loads(raw.decode("utf-8"))
        self.server.request_bodies.append(body)  # type: ignore[attr-defined]
        prompt = _extract_prompt(body)
        self._send_sse_chunks(_response_for_prompt(prompt))


class _FakeM11Server:
    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        self.config = dict(config or {})
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.port: int = 0
        self.request_bodies: list[dict[str, Any]] = []

    def __enter__(self) -> "_FakeM11Server":
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeM11Handler)
        self._server.config = self.config  # type: ignore[attr-defined]
        self._server.request_bodies = self.request_bodies  # type: ignore[attr-defined]
        _FakeM11Handler.config = self.config
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="fake-m11-adapter", daemon=True
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


def _extract_prompt(body: dict[str, Any]) -> str:
    message = (body.get("messages") or [{}])[0]
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
        return "\n".join(text_parts)
    return ""


def _response_for_prompt(prompt: str) -> str:
    lowered = prompt.lower()
    if "status update" in lowered:
        return "The session is ready and the adapter is responding."
    if "describe the bird" in lowered:
        return "It is a toucan with a bright beak."
    if "what bird is shown" in lowered:
        return "The bird is a toucan with a large bill."
    if "using both" in lowered:
        return "The session is ready and the bird is a toucan."
    if "status update" in lowered:
        return "The session is ready."
    return "ok"


def _capture_main(argv: list[str]) -> dict[str, Any]:
    backup_stdout = sys.stdout
    try:
        captured = io.StringIO()
        sys.stdout = captured
        exit_code = SUITE.main(argv)
    finally:
        sys.stdout = backup_stdout
    return {"exit_code": exit_code, "stdout": captured.getvalue()}


def _write_temp_image(payload: bytes) -> Path:
    tmp = tempfile.NamedTemporaryFile(
        prefix="m11-dogfood-", suffix=".png", delete=False
    )
    tmp.write(payload)
    tmp.close()
    return Path(tmp.name)


def test_suite_runs_all_m11_tasks_and_records_dependencies() -> None:
    image_path = _write_temp_image(b"\x89PNG\r\n\x1a\nfake-image-bytes")
    output_path = Path(tempfile.mkdtemp()) / "m11-report.json"
    with _FakeM11Server() as server:
        captured = _capture_main(
            [
                "--base-url",
                server.base_url,
                "--model",
                "cheetara-m7",
                "--path-label",
                "m7-external",
                "--image-path",
                str(image_path),
                "--output",
                str(output_path),
            ]
        )
    assert captured["exit_code"] == 0
    report = json.loads(captured["stdout"])
    assert report["suite_id"] == "m11-mixed-text-vision-dogfood"
    assert report["path_label"] == "m7-external"
    assert report["summary"] == {"total": 6, "passed": 6, "failed": 0}
    assert report["preflight"]["connect"]["status"] == "pass"
    assert report["preflight"]["health"]["status"] == "pass"
    task_ids = [task["task_id"] for task in report["tasks"]]
    assert task_ids == [
        "text_status_update",
        "image_description",
        "image_qna",
        "mixed_followup",
    ]
    mixed = report["tasks"][-1]
    assert mixed["dependencies"]["text_status_update"] == (
        "The session is ready and the adapter is responding."
    )
    assert "toucan" in mixed["dependencies"]["image_qna"].lower()
    assert "The session is ready and the adapter is responding." in mixed["prompt"]
    assert "toucan" in mixed["prompt"].lower()
    assert report["evidence_paths"]["report_json"] == str(output_path)
    assert output_path.exists()
    assert len(server.request_bodies) == 4
    image_bodies = [
        body
        for body in server.request_bodies
        if isinstance(body["messages"][0]["content"], list)
    ]
    assert len(image_bodies) == 3
    assert any(
        any(part.get("type") == "image_url" for part in body["messages"][0]["content"])
        for body in image_bodies
    )


def test_suite_fails_when_health_reports_no_vision_support() -> None:
    image_path = _write_temp_image(b"\x89PNG\r\n\x1a\nfake-image-bytes")
    with _FakeM11Server({"supports_vision": False}) as server:
        captured = _capture_main(
            [
                "--base-url",
                server.base_url,
                "--model",
                "cheetara-m7",
                "--image-path",
                str(image_path),
            ]
        )
    assert captured["exit_code"] == 1
    report = json.loads(captured["stdout"])
    assert report["preflight"]["health"]["status"] == "fail"
    assert "supports_vision" in report["preflight"]["health"]["details"]["reason"]
    assert report["summary"]["failed"] >= 1


def test_suite_records_skipped_mixed_followup_when_prereqs_missing(
    monkeypatch: Any,
) -> None:
    image_path = _write_temp_image(b"\x89PNG\r\n\x1a\nfake-image-bytes")
    output_path = Path(tempfile.mkdtemp()) / "m11-report.json"
    original_response_for_prompt = _response_for_prompt

    def _broken_response_for_prompt(prompt: str) -> str:
        if "status update" in prompt.lower():
            return "The session is pending."
        return original_response_for_prompt(prompt)

    monkeypatch.setattr(
        sys.modules[__name__], "_response_for_prompt", _broken_response_for_prompt
    )
    with _FakeM11Server() as server:
        captured = _capture_main(
            [
                "--base-url",
                server.base_url,
                "--model",
                "cheetara-m7",
                "--path-label",
                "m7-external",
                "--image-path",
                str(image_path),
                "--output",
                str(output_path),
            ]
        )

    assert captured["exit_code"] == 1
    report = json.loads(captured["stdout"])
    assert report["summary"]["passed"] == 4
    assert report["summary"]["failed"] == 1
    assert report["summary"]["skipped"] == 1
    mixed = report["tasks"][-1]
    assert mixed["status"] == "skipped"
    assert mixed["reason"] == "missing prerequisite outputs: text_status_update"
    assert mixed["missing_prerequisite_outputs"] == ["text_status_update"]
    assert mixed["dependencies"]["text_status_update"] == ""
    assert "image_qna" in mixed["dependencies"]
    assert output_path.exists()
