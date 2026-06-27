#!/usr/bin/env python3
"""Repeatable M11 cheetara mixed text + vision dogfood suite.

This runner defines the M11 task set for the staged cheetara cutover:

* text-only status update
* image-grounded description
* image-grounded question answering
* mixed follow-up that combines prior text context with image evidence

The suite is intentionally non-GUI. It speaks only to the already
validated cheetara-compatible HTTP surfaces on the host-local mlx-engine
adapter or the source-level local-compatibility surface. It does not
touch ``vmlx.app.asar`` and it does not involve LM Studio.

The same runner can be pointed at either:

* M7 external adapter, ``http://127.0.0.1:3180``
* M9 local compatibility, ``http://127.0.0.1:3181``

The output is a single JSON report that records invocation shapes,
expected outcomes, evidence paths, and cleanup rules.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

TASK_SET_ID = "m11-mixed-text-vision-dogfood"
TASK_SET_VERSION = 1
DEFAULT_IMAGE_PATH = Path(
    "/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/demo-data/toucan.jpeg"
)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    kind: str  # "text" or "image"
    prompt_template: str
    expected_all_keywords: tuple[str, ...]
    expected_any_groups: tuple[tuple[str, ...], ...] = ()
    max_tokens: int = 64

    def render_prompt(self, prior_outputs: dict[str, str]) -> str:
        return self.prompt_template.format(**prior_outputs)


TASK_SET: tuple[TaskSpec, ...] = (
    TaskSpec(
        task_id="text_status_update",
        kind="text",
        prompt_template=(
            "Write one concise status update for this session. Mention that "
            "the cheetara path is ready and that the adapter is responding."
        ),
        expected_all_keywords=("ready", "responding"),
        max_tokens=48,
    ),
    TaskSpec(
        task_id="image_description",
        kind="image",
        prompt_template=(
            "Describe the bird in the attached photo in one short sentence."
        ),
        expected_all_keywords=("toucan",),
        max_tokens=64,
    ),
    TaskSpec(
        task_id="image_qna",
        kind="image",
        prompt_template=(
            "What bird is shown, and what visible feature makes it easy to "
            "recognize? Answer in one short sentence."
        ),
        expected_all_keywords=("toucan",),
        expected_any_groups=(("beak", "bill"),),
        max_tokens=64,
    ),
    TaskSpec(
        task_id="mixed_followup",
        kind="image",
        prompt_template=(
            "Status note from earlier: {text_status_update}\n"
            "Image evidence from earlier: {image_qna}\n"
            "Using both, write one sentence that says the session is ready "
            "and names the bird."
        ),
        expected_all_keywords=("ready", "toucan"),
        max_tokens=64,
    ),
)

CLEANUP_RULES = (
    "Run only one validator at a time, never M7 and M9 simultaneously.",
    "Stop the adapter or local compatibility service with the matching "
    "services.yaml stop command before starting the next path.",
    "Do not start any Qwen LLMDYNAMIX or other MLX-heavy validation service "
    "during M11 dogfood capture.",
    "Do not modify or repack vmlx.app.asar.",
    "If any temporary persistent-cache experiment was used, clean "
    "/private/tmp/mlx-engine-vlm-cache-* before the next run.",
)


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


def _encode_image_for_data_url(image_path: Path) -> str:
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


def _contains_keywords(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return all(keyword.lower() in lowered for keyword in keywords)


def _contains_any_groups(text: str, groups: tuple[tuple[str, ...], ...]) -> bool:
    lowered = text.lower()
    for group in groups:
        if not any(keyword.lower() in lowered for keyword in group):
            return False
    return True


def _run_connect(base_url: str, model: str) -> dict[str, Any]:
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
                    f"requested model {model!r} not present in /v1/models "
                    f"(available: {model_ids!r})"
                ),
                "model_ids": model_ids,
                "model_count": len(data),
                "served_model": model,
                "body": body,
                "headers": dict(headers),
            },
        }
    return {
        "status": "pass",
        "details": {
            "http_status": status,
            "elapsed_s": elapsed_s,
            "model_ids": model_ids,
            "model_count": len(data),
            "served_model": model,
            "body": body,
            "headers": dict(headers),
        },
    }


def _run_health(base_url: str, model: str) -> dict[str, Any]:
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
    supports_vision = body.get("supports_vision")
    if health_status != "ok":
        return {
            "status": "fail",
            "details": {
                "http_status": status,
                "elapsed_s": elapsed_s,
                "reason": f"/health returned status={health_status!r}; expected 'ok'",
                "body": body,
            },
        }
    if supports_vision is not True:
        return {
            "status": "fail",
            "details": {
                "http_status": status,
                "elapsed_s": elapsed_s,
                "reason": "/health reported supports_vision != True",
                "body": body,
            },
        }
    models_status, models_body, _ = _http_get_json(base_url, "/v1/models")
    model_ids = [
        entry.get("id")
        for entry in (models_body.get("data") or [])
        if isinstance(entry, dict)
    ]
    if models_status != 200 or model not in model_ids or served_model != model:
        return {
            "status": "fail",
            "details": {
                "http_status": status,
                "elapsed_s": elapsed_s,
                "reason": "served_model mismatch between /health and /v1/models",
                "served_model": served_model,
                "model_ids": model_ids,
                "supports_vision": supports_vision,
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
            "supports_vision": supports_vision,
            "headers": dict(headers),
            "body": body,
        },
    }


def _build_request_payload(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    image_data_url: Optional[str] = None,
) -> dict[str, Any]:
    if image_data_url is None:
        messages = [{"role": "user", "content": prompt}]
        request_shape = {
            "endpoint": "/v1/chat/completions",
            "stream": True,
            "message_shape": "text",
            "has_image_url": False,
            "max_tokens": max_tokens,
        }
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        request_shape = {
            "endpoint": "/v1/chat/completions",
            "stream": True,
            "message_shape": "multimodal",
            "has_image_url": True,
            "max_tokens": max_tokens,
        }
    return {
        "payload": {
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
        },
        "request_shape": request_shape,
    }


def _run_task(
    base_url: str,
    model: str,
    spec: TaskSpec,
    *,
    image_data_url: Optional[str],
    prompt: str,
    dependencies: dict[str, str],
) -> dict[str, Any]:
    started = time.time()
    built = _build_request_payload(
        model=model,
        prompt=prompt,
        max_tokens=spec.max_tokens,
        image_data_url=image_data_url,
    )
    capture = _stream_sse(base_url, "/v1/chat/completions", built["payload"])
    elapsed_s = round(time.time() - started, 4)
    content_text = capture["content_text"]
    result: dict[str, Any] = {
        "task_id": spec.task_id,
        "kind": spec.kind,
        "prompt": prompt,
        "request_shape": built["request_shape"],
        "expected": {
            "all_keywords": list(spec.expected_all_keywords),
            "any_groups": [list(group) for group in spec.expected_any_groups],
        },
        "dependencies": dependencies,
        "elapsed_s": elapsed_s,
        "http_status": capture["status"],
        "chunk_count": len(capture["chunks"]),
        "finish_reasons": capture["finish_reasons"],
        "stream_done_seen": capture["done_seen"],
        "content_text": content_text,
        "raw_excerpt": capture.get("raw", "")[:2000],
    }

    if capture["status"] != 200:
        result["status"] = "fail"
        result["reason"] = (
            f"POST /v1/chat/completions returned HTTP {capture['status']}"
        )
        result["response_headers"] = capture.get("response_headers", {})
        result["error_text"] = capture.get("error_text")
        return result
    if not capture["done_seen"]:
        result["status"] = "fail"
        result["reason"] = "stream terminated without [DONE] marker"
        result["response_headers"] = capture.get("response_headers", {})
        result["error_text"] = capture.get("error_text")
        return result
    if capture.get("error_text"):
        result["status"] = "fail"
        result["reason"] = f"stream reported error: {capture['error_text']}"
        result["response_headers"] = capture.get("response_headers", {})
        return result
    if not _contains_keywords(content_text, spec.expected_all_keywords):
        result["status"] = "fail"
        result["reason"] = (
            f"final assistant text did not contain all expected keywords "
            f"{spec.expected_all_keywords!r}"
        )
        result["response_headers"] = capture.get("response_headers", {})
        return result
    if spec.expected_any_groups and not _contains_any_groups(
        content_text, spec.expected_any_groups
    ):
        result["status"] = "fail"
        result["reason"] = (
            f"final assistant text did not contain one keyword from each "
            f"expected group {spec.expected_any_groups!r}"
        )
        result["response_headers"] = capture.get("response_headers", {})
        return result
    result["status"] = "pass"
    result["response_headers"] = capture.get("response_headers", {})
    return result


def _aggregate(result_states: list[str]) -> dict[str, int]:
    total = len(result_states)
    passed = sum(1 for state in result_states if state == "pass")
    failed = total - passed
    return {"total": total, "passed": passed, "failed": failed}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Repeatable M11 mixed text + vision cheetara dogfood suite. "
            "Uses host-local HTTP surfaces only."
        )
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:3180",
        help="Base URL for the cheetara-compatible surface.",
    )
    parser.add_argument(
        "--model",
        default="cheetara-m7",
        help="Served model id for the target surface.",
    )
    parser.add_argument(
        "--path-label",
        default="m7-external",
        help="Human-readable label for the path under test.",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help="Image used for the image-grounded tasks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report output path.",
    )
    args = parser.parse_args(argv)

    image_path = args.image_path
    if not image_path.exists():
        report = {
            "suite_id": TASK_SET_ID,
            "suite_version": TASK_SET_VERSION,
            "base_url": args.base_url.rstrip("/"),
            "model": args.model,
            "path_label": args.path_label,
            "image_path": str(image_path),
            "cleanup_rules": list(CLEANUP_RULES),
            "results": {
                "image_path": {
                    "status": "fail",
                    "reason": f"image not found at {image_path}",
                }
            },
            "summary": {"total": 1, "passed": 0, "failed": 1},
            "evidence_paths": {
                "report_json": str(args.output) if args.output else "stdout"
            },
        }
        encoded = json.dumps(report, indent=2, sort_keys=True)
        print(encoded)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")
        return 1

    base_url = args.base_url.rstrip("/")
    preflight: dict[str, dict[str, Any]] = {}
    result_states: list[str] = []

    connect = _run_connect(base_url, args.model)
    preflight["connect"] = connect
    result_states.append(connect["status"])

    health = _run_health(base_url, args.model)
    preflight["health"] = health
    result_states.append(health["status"])

    image_data_url = _encode_image_for_data_url(image_path)
    prior_outputs: dict[str, str] = {}
    task_results: list[dict[str, Any]] = []

    for spec in TASK_SET:
        prompt = spec.render_prompt(prior_outputs)
        dependencies: dict[str, str] = {}
        if spec.task_id == "mixed_followup":
            dependencies = {
                "text_status_update": prior_outputs.get("text_status_update", ""),
                "image_qna": prior_outputs.get("image_qna", ""),
            }
        task_result = _run_task(
            base_url,
            args.model,
            spec,
            image_data_url=image_data_url if spec.kind == "image" else None,
            prompt=prompt,
            dependencies=dependencies,
        )
        task_results.append(task_result)
        result_states.append(task_result["status"])
        if task_result["status"] == "pass":
            prior_outputs[spec.task_id] = task_result["content_text"]

    report = {
        "suite_id": TASK_SET_ID,
        "suite_version": TASK_SET_VERSION,
        "path_label": args.path_label,
        "base_url": base_url,
        "model": args.model,
        "image_path": str(image_path),
        "cleanup_rules": list(CLEANUP_RULES),
        "preflight": preflight,
        "tasks": task_results,
        "summary": _aggregate(result_states),
        "evidence_paths": {
            "report_json": str(args.output) if args.output else "stdout"
        },
    }

    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
