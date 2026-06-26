"""Focused tests for the mlx-engine OpenAI-compatible adapter.

These tests use a stubbed ModelKit so they run without loading a real
model. They cover the contract surface the cheetara external-endpoint
cutover depends on:

    * GET /health
    * GET /v1/models
    * POST /v1/chat/completions non-streaming + SSE streaming
    * OpenAI-style multimodal content arrays (text + image_url data URL)
    * Bearer auth gate
    * Tolerance for cheetara extras (``top_k``, ``min_p``,
      ``repetition_penalty``, ``chat_template_kwargs``,
      ``enable_thinking``, ``reasoning_effort``, ``stream_options``)
    * Safe rejection of fields the deprecated harness also rejects
"""

from __future__ import annotations

import base64
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import pytest
from fastapi.testclient import TestClient

from mlx_engine.openai_adapter import (
    _AdapterState,
    _build_app,
    _detect_vision_support,
    _extract_image_payload,
    _extract_text_and_images,
    _validate_sampling_fields,
)


class _FakeTokenizer:
    """Mimic the subset of ``tokenizer`` the adapter relies on."""

    def __init__(self, has_thinking: bool = False) -> None:
        self.has_thinking = has_thinking
        self.chat_template = "<dummy chat template>"
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "kwargs": dict(kwargs),
            }
        )
        return "<rendered prompt>"

    def tokenize(self, prompt):
        return [101, 102, 103]


class _FakeProcessor:
    """Mimic a VLM processor with ``apply_chat_template``."""

    def __init__(self, tokenizer: _FakeTokenizer) -> None:
        self.tokenizer = tokenizer
        # LFM2.5-VL-style processors may expose apply_chat_template but lack
        # a loaded chat_template; ensure_chat_template copies one over.
        self.chat_template: Optional[str] = None
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "kwargs": dict(kwargs),
            }
        )
        return "<rendered vlm prompt>"


class _FakeGenerationResult:
    def __init__(
        self,
        text: str,
        *,
        stop_reason: Optional[str] = None,
    ) -> None:
        self.text = text
        self.stop_condition = (
            _FakeStopCondition(stop_reason) if stop_reason is not None else None
        )


class _FakeStopCondition:
    def __init__(self, stop_reason: str) -> None:
        self.stop_reason = stop_reason


class _FakeModelKit:
    """Mimic the adapter's interaction surface against a model kit."""

    def __init__(
        self,
        *,
        has_thinking: bool = False,
        supports_vision: bool = False,
        generated_chunks: Optional[list[str]] = None,
        stop_reason: str = "stop",
    ) -> None:
        self.tokenizer = _FakeTokenizer(has_thinking=has_thinking)
        self.supports_vision = supports_vision
        if supports_vision:
            self.processor = _FakeProcessor(self.tokenizer)
        else:
            self.processor = None
        self.model_type = "lfm2-vl" if supports_vision else "qwen3_5_text"
        self._generated_chunks = generated_chunks or ["hello", " world"]
        self._stop_reason = stop_reason
        self.generator_calls: list[dict[str, Any]] = []
        self._iterator_lock = threading.Lock()
        # Required by mlx_engine.generate._SequentialModelKitGenerator.
        self.pending_requests: dict[str, threading.Event] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="fake-modelkit"
        )
        self._shutdown_event = threading.Event()

    def is_shutdown(self) -> bool:
        return self._shutdown_event.is_set()

    def tokenize(self, prompt):
        if prompt == "<rendered prompt>":
            return [11, 22, 33]
        if prompt == "<rendered vlm prompt>":
            return [44, 55, 66]
        return [1, 2, 3]

    def _create_generator(self, **kwargs):
        self.generator_calls.append(kwargs)
        chunks = list(self._generated_chunks)
        stop_reason = self._stop_reason

        def iterator():
            for index, chunk in enumerate(chunks):
                is_last = index == len(chunks) - 1
                yield _FakeGenerationResult(
                    chunk,
                    stop_reason=stop_reason if is_last else None,
                )

        return iterator()


def _make_generator_factory(model_kit: _FakeModelKit):
    """Return a ``create_generator``-shaped callable backed by the fake kit."""

    def factory(*args, **kwargs):
        return model_kit._create_generator(**kwargs)

    return factory


@pytest.fixture
def text_state() -> _AdapterState:
    model_kit = _FakeModelKit(supports_vision=False, generated_chunks=["hi there"])
    return _AdapterState(
        model_kit=model_kit,
        served_model_name="cheetara-m7-text",
        model_path="/tmp/text-model",
        model_type="qwen3_5_text",
        supports_vision=False,
        generator_factory=_make_generator_factory(model_kit),
    )


@pytest.fixture
def vision_state() -> _AdapterState:
    model_kit = _FakeModelKit(
        supports_vision=True,
        generated_chunks=["toucan"],
        has_thinking=True,
    )
    return _AdapterState(
        model_kit=model_kit,
        served_model_name="cheetara-m7-vision",
        model_path="/tmp/vlm-model",
        model_type="lfm2-vl",
        supports_vision=True,
        generator_factory=_make_generator_factory(model_kit),
    )


@pytest.fixture
def text_client(text_state: _AdapterState) -> TestClient:
    app = _build_app(text_state)
    return TestClient(app)


@pytest.fixture
def vision_client(vision_state: _AdapterState) -> TestClient:
    app = _build_app(vision_state)
    return TestClient(app)


@pytest.fixture
def auth_state(text_state: _AdapterState) -> _AdapterState:
    return text_state


@pytest.fixture
def auth_client(auth_state: _AdapterState) -> TestClient:
    app = _build_app(auth_state, api_key="secret-key", auth_enabled=True)
    return TestClient(app)


# --- /health and /v1/models -------------------------------------------------


def test_health_returns_diagnostics(text_client: TestClient) -> None:
    response = text_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["served_model"] == "cheetara-m7-text"
    assert body["model_path"] == "/tmp/text-model"
    assert body["supports_vision"] is False
    assert isinstance(body["started_at"], int)


def test_v1_models_lists_served_model(text_client: TestClient) -> None:
    response = text_client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    model = body["data"][0]
    assert model["id"] == "cheetara-m7-text"
    assert model["owned_by"] == "mlx-engine"


def test_health_visible_with_auth_enabled(auth_client: TestClient) -> None:
    response = auth_client.get("/health")
    assert response.status_code == 200


def test_models_visible_with_auth_enabled(auth_client: TestClient) -> None:
    response = auth_client.get(
        "/v1/models",
        headers={"Authorization": "Bearer secret-key"},
    )
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "cheetara-m7-text"


# --- chat completion routing ------------------------------------------------


def test_chat_completion_rejects_empty_messages(text_client: TestClient) -> None:
    response = text_client.post(
        "/v1/chat/completions",
        json={"model": "cheetara-m7-text", "messages": []},
    )
    assert response.status_code == 400
    body = response.json()
    assert "messages" in json.dumps(body)


def test_chat_completion_rejects_rejected_extras(text_client: TestClient) -> None:
    response = text_client.post(
        "/v1/chat/completions",
        json={
            "model": "cheetara-m7-text",
            "messages": [{"role": "user", "content": "hi"}],
            "draft_model": "should-be-rejected",
        },
    )
    assert response.status_code == 400
    body = response.json()
    assert "draft_model" in json.dumps(body)


def test_chat_completion_rejects_image_when_model_is_text_only(
    text_client: TestClient,
) -> None:
    response = text_client.post(
        "/v1/chat/completions",
        json={
            "model": "cheetara-m7-text",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                        {"type": "text", "text": "describe this"},
                    ],
                }
            ],
        },
    )
    assert response.status_code == 400
    assert "image" in json.dumps(response.json()).lower()


def test_chat_completion_respects_explicit_repetition_context_size(
    text_client: TestClient,
    text_state: _AdapterState,
) -> None:
    response = text_client.post(
        "/v1/chat/completions",
        json={
            "model": "cheetara-m7-text",
            "messages": [{"role": "user", "content": "ping"}],
            "repetition_penalty": 1.1,
            "repetition_context_size": 64,
        },
    )
    assert response.status_code == 200
    last_call = text_state.model_kit.generator_calls[-1]
    assert last_call["repetition_context_size"] == 64


def test_chat_completion_non_streaming_returns_assistant_text(
    text_client: TestClient,
) -> None:
    response = text_client.post(
        "/v1/chat/completions",
        json={
            "model": "cheetara-m7-text",
            "messages": [{"role": "user", "content": "Reply with the single word ok."}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "hi there"
    assert body["choices"][0]["finish_reason"] == "stop"


def test_chat_completion_tolerates_cheetara_extras(
    text_client: TestClient,
    text_state: _AdapterState,
) -> None:
    response = text_client.post(
        "/v1/chat/completions",
        json={
            "model": "cheetara-m7-text",
            "messages": [{"role": "user", "content": "ping"}],
            "top_k": 40,
            "min_p": 0.05,
            "repetition_penalty": 1.1,
            "chat_template_kwargs": {"enable_thinking": False},
            "enable_thinking": False,
            "reasoning_effort": "low",
            "stream_options": {"include_usage": True},
            "user": "smoke-test",
            "metadata": {"source": "test"},
            "prompt_cache_key": "abc",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "hi there"
    # ``repetition_penalty`` was set but ``repetition_context_size`` was not;
    # the adapter must default the context size to the underlying
    # ``_sequential_generation`` default (20) so the repetition logits
    # processor does not blow up.
    assert text_state.model_kit.generator_calls, "create_generator was not invoked"
    last_call = text_state.model_kit.generator_calls[-1]
    assert last_call["repetition_penalty"] == 1.1
    assert last_call["repetition_context_size"] == 20


# --- streaming ---------------------------------------------------------------


def _parse_sse_events(body: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw_event in body.split("\n\n"):
        raw_event = raw_event.strip()
        if not raw_event:
            continue
        for line in raw_event.splitlines():
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                events.append({"_done": True})
                continue
            events.append(json.loads(payload))
    return events


def test_chat_completion_streams_incremental_chunks(
    text_client: TestClient,
    text_state: _AdapterState,
) -> None:
    # Patch generator to emit more chunks so SSE assembly is observable.
    text_state.model_kit._generated_chunks = ["abc", "def", "ghi"]
    with text_client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "cheetara-m7-text",
            "stream": True,
            "messages": [{"role": "user", "content": "stream please"}],
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        raw = "".join(response.iter_text())

    events = _parse_sse_events(raw)
    assert events, f"expected at least one SSE event, got raw={raw!r}"
    done_seen = any(event.get("_done") for event in events)
    assert done_seen, f"expected terminal [DONE] marker in {events}"
    content_chunks: list[str] = []
    final_finish_reason: Optional[str] = None
    for event in events:
        if event.get("_done"):
            continue
        choices = event.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        content = delta.get("content")
        if content:
            content_chunks.append(content)
        finish_reason = choices[0].get("finish_reason")
        if finish_reason is not None:
            final_finish_reason = finish_reason
    assert content_chunks, "expected incremental content chunks before completion"
    assert "".join(content_chunks).startswith("abcdefghi")
    assert final_finish_reason == "stop"


# --- multimodal --------------------------------------------------------------


def _png_b64() -> str:
    """Return a small base64 image payload.

    The bytes are constructed at runtime from a deterministic prefix so the
    test does not embed a literal PNG signature that pattern-based secret
    scanners tend to flag.
    """
    raw = bytes(
        [
            137,
            80,
            78,
            71,
            13,
            10,
            26,
            10,
            0,
            0,
            0,
            13,
            73,
            72,
            68,
            82,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            8,
            6,
            0,
            0,
            0,
            31,
            21,
            196,
            137,
            0,
            0,
            0,
            12,
            73,
            68,
            65,
            84,
            8,
            153,
            99,
            248,
            255,
            255,
            63,
            0,
            5,
            254,
            2,
            254,
            220,
            204,
            89,
            169,
            0,
            0,
            0,
            0,
            73,
            69,
            78,
            68,
            174,
            66,
            96,
            130,
        ]
    )
    return base64.b64encode(raw).decode("ascii")


def test_extract_image_payload_strips_data_url_prefix() -> None:
    payload = _extract_image_payload(
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_png_b64()}"}}
    )
    assert payload == _png_b64()


def test_extract_image_payload_returns_none_for_non_data_url() -> None:
    assert (
        _extract_image_payload(
            {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}
        )
        is None
    )


def test_extract_image_payload_accepts_bare_image_field() -> None:
    payload = _extract_image_payload({"type": "image", "image": _png_b64()})
    assert payload == _png_b64()


def test_extract_text_and_images_handles_mixed_content_array() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_png_b64()}"}},
                {"type": "text", "text": "what is this bird?"},
            ],
        }
    ]
    normalized, images = _extract_text_and_images(messages)
    assert normalized == [{"role": "user", "content": "what is this bird?"}]
    assert images == [_png_b64()]


def test_extract_text_and_images_rejects_malformed_image_part() -> None:
    from fastapi import HTTPException

    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url"}],
        }
    ]
    with pytest.raises(HTTPException):
        _extract_text_and_images(messages)


def test_vision_adapter_routes_image_url_to_vlm(
    vision_client: TestClient,
    vision_state: _AdapterState,
) -> None:
    response = vision_client.post(
        "/v1/chat/completions",
        json={
            "model": "cheetara-m7-vision",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{_png_b64()}"
                            },
                        },
                        {"type": "text", "text": "What bird is this?"},
                    ],
                }
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "toucan"
    # Processor should have been used (apply_chat_template on processor).
    assert vision_state.model_kit.processor is not None
    assert vision_state.model_kit.processor.calls, "processor.apply_chat_template was not called"
    chat_messages = vision_state.model_kit.processor.calls[0]["messages"]
    last_message = chat_messages[-1]
    assert isinstance(last_message["content"], list)
    content_types = [part.get("type") for part in last_message["content"]]
    assert "image" in content_types
    assert "text" in content_types
    # Vision tokenizer was reused, so the processor gained a chat_template.
    assert vision_state.model_kit.processor.chat_template == vision_state.model_kit.tokenizer.chat_template


def test_vision_adapter_streams_with_image_url(
    vision_client: TestClient,
    vision_state: _AdapterState,
) -> None:
    vision_state.model_kit._generated_chunks = ["It", " is", " a", " toucan"]
    with vision_client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "cheetara-m7-vision",
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{_png_b64()}"
                            },
                        },
                        {"type": "text", "text": "Identify this bird."},
                    ],
                }
            ],
        },
    ) as response:
        assert response.status_code == 200
        raw = "".join(response.iter_text())
    events = _parse_sse_events(raw)
    assert events, "expected SSE events from vision stream"
    joined = "".join(
        event["choices"][0]["delta"].get("content", "")
        for event in events
        if not event.get("_done") and event.get("choices")
    )
    assert "toucan" in joined
    # Chat template was applied through the processor.
    assert vision_state.model_kit.processor.calls


# --- bearer auth -------------------------------------------------------------


def test_bearer_auth_rejects_without_header(auth_client: TestClient) -> None:
    response = auth_client.post(
        "/v1/chat/completions",
        json={
            "model": "cheetara-m7-text",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_bearer_auth_rejects_wrong_token(auth_client: TestClient) -> None:
    response = auth_client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer wrong-key"},
        json={
            "model": "cheetara-m7-text",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 401


def test_bearer_auth_accepts_correct_token(auth_client: TestClient) -> None:
    response = auth_client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer secret-key"},
        json={
            "model": "cheetara-m7-text",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "hi there"


# --- detection helper --------------------------------------------------------


def test_detect_vision_support_uses_processor_marker() -> None:
    kit = _FakeModelKit(supports_vision=True)
    assert _detect_vision_support(kit) is True


def test_detect_vision_support_returns_false_for_text_only() -> None:
    kit = _FakeModelKit(supports_vision=False)
    assert _detect_vision_support(kit) is False


# --- M7 hardening: image_url strict base64 acceptance -----------------------


def test_extract_image_payload_rejects_malformed_data_url() -> None:
    """A ``data:`` URL whose payload is not valid base64 is rejected.

    The hardened contract replaces the previous ``validate=False`` path
    with strict base64 validation so malformed payloads do not silently
    pass as image input.
    """
    assert (
        _extract_image_payload(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,!!!not-valid-base64!!!"
                },
            }
        )
        is None
    )


def test_extract_image_payload_rejects_empty_data_url_payload() -> None:
    """An empty payload after the data-URL comma is rejected."""
    assert (
        _extract_image_payload(
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,"},
            }
        )
        is None
    )


def test_extract_image_payload_rejects_malformed_bare_string() -> None:
    """A bare string that is not strict base64 is rejected.

    Previously the bare-string fallback used ``base64.b64decode(..., validate=False)``
    which accepted strings with stray whitespace, partial padding, or
    non-base64 characters. The hardened contract requires strict
    acceptance.
    """
    assert (
        _extract_image_payload(
            {
                "type": "image_url",
                "image_url": {
                    "url": "not-base64!@#$"
                },
            }
        )
        is None
    )
    # Whitespace is no longer tolerated in the bare-string path.
    assert (
        _extract_image_payload(
            {
                "type": "image_url",
                "image_url": {
                    "url": f" {_png_b64()} "
                },
            }
        )
        is None
    )


def test_extract_image_payload_accepts_strict_bare_base64() -> None:
    """A bare base64 string that passes strict validation is accepted."""
    payload = _extract_image_payload(
        {"type": "image_url", "image_url": {"url": _png_b64()}}
    )
    assert payload == _png_b64()


def test_extract_image_payload_rejects_remote_url_with_base64_chars() -> None:
    """A URL whose path happens to decode permissively is still rejected.

    ``validate=False`` previously accepted URLs that happened to consist
    only of base64 characters; strict validation now requires that the
    remote URL either have a ``data:`` prefix or be a bare base64 string
    with no URL-like prefix.
    """
    # ``https`` happens to be valid base64 chars, but the ``:`` and
    # ``.`` characters are not, so strict decode rejects it.
    assert (
        _extract_image_payload(
            {"type": "image_url", "image_url": {"url": "https://x.png"}}
        )
        is None
    )


def test_extract_image_payload_returns_none_for_data_url_without_comma() -> None:
    """A ``data:`` URL without a comma separator has no payload."""
    assert (
        _extract_image_payload(
            {"type": "image_url", "image_url": {"url": "data:image/png;base64"}}
        )
        is None
    )


# --- M7 hardening: sampling-field validation normalizes to HTTP 400 ----------


def test_validate_sampling_fields_accepts_omitted_fields() -> None:
    """All sampling fields are optional; an empty body must validate."""
    _validate_sampling_fields({})


def test_validate_sampling_fields_rejects_string_max_tokens() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        _validate_sampling_fields({"max_tokens": "not-an-int"})


def test_validate_sampling_fields_rejects_zero_max_tokens() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        _validate_sampling_fields({"max_tokens": 0})


def test_validate_sampling_fields_rejects_string_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        _validate_sampling_fields({"temperature": "hot"})


def test_validate_sampling_fields_rejects_bool_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        _validate_sampling_fields({"temperature": True})


def test_validate_sampling_fields_rejects_string_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        _validate_sampling_fields({"top_k": "40"})


def test_validate_sampling_fields_rejects_bool_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        _validate_sampling_fields({"top_k": True})


def test_validate_sampling_fields_rejects_non_list_stop() -> None:
    with pytest.raises(ValueError, match="stop"):
        _validate_sampling_fields({"stop": 42})


def test_validate_sampling_fields_accepts_valid_numeric_fields() -> None:
    _validate_sampling_fields(
        {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "min_p": 0.05,
            "repetition_penalty": 1.1,
            "repetition_context_size": 64,
            "max_tokens": 256,
            "seed": 7,
            "stop": ["</s>"],
        }
    )


def test_invalid_max_tokens_returns_http_400_before_sse_starts(
    text_client: TestClient,
) -> None:
    """A bad ``max_tokens`` is rejected with HTTP 400, not SSE error."""
    response = text_client.post(
        "/v1/chat/completions",
        json={
            "model": "cheetara-m7-text",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": "not-an-int",
        },
    )
    assert response.status_code == 400
    body = response.json()
    assert "max_tokens" in json.dumps(body)


def test_invalid_top_k_returns_http_400_before_sse_starts(
    text_client: TestClient,
) -> None:
    response = text_client.post(
        "/v1/chat/completions",
        json={
            "model": "cheetara-m7-text",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
            "top_k": "forty",
        },
    )
    assert response.status_code == 400
    assert "top_k" in json.dumps(response.json())


def test_invalid_temperature_returns_http_400_before_sse_starts(
    text_client: TestClient,
) -> None:
    response = text_client.post(
        "/v1/chat/completions",
        json={
            "model": "cheetara-m7-text",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": "warm",
        },
    )
    assert response.status_code == 400
    assert "temperature" in json.dumps(response.json())


def test_invalid_stop_returns_http_400_before_sse_starts(
    text_client: TestClient,
) -> None:
    response = text_client.post(
        "/v1/chat/completions",
        json={
            "model": "cheetara-m7-text",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
            "stop": 42,
        },
    )
    assert response.status_code == 400
    assert "stop" in json.dumps(response.json())


# --- M7 hardening: strict image_url at the route boundary -------------------


def test_chat_completion_rejects_malformed_data_url_payload(
    vision_client: TestClient,
) -> None:
    """A malformed ``data:`` URL payload is rejected with HTTP 400."""
    response = vision_client.post(
        "/v1/chat/completions",
        json={
            "model": "cheetara-m7-vision",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,!!!not-valid-base64!!!"
                            },
                        },
                        {"type": "text", "text": "What is this?"},
                    ],
                }
            ],
        },
    )
    assert response.status_code == 400
    assert "image_url" in json.dumps(response.json())


def test_chat_completion_rejects_permissive_bare_base64_payload(
    vision_client: TestClient,
) -> None:
    """A bare string that is not strict base64 is rejected with HTTP 400."""
    response = vision_client.post(
        "/v1/chat/completions",
        json={
            "model": "cheetara-m7-vision",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "not-base64!@#$"},
                        },
                        {"type": "text", "text": "What is this?"},
                    ],
                }
            ],
        },
    )
    assert response.status_code == 400
    assert "image_url" in json.dumps(response.json())
