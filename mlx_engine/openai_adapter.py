"""mlx-engine host-local OpenAI-compatible adapter.

This module is the dedicated M7 cutover surface: a maintained
``127.0.0.1:3180`` adapter that exposes the subset of the OpenAI /
cheetara HTTP contract the cheetara app needs in external-endpoint mode.

Unlike ``mlx_engine.distributed_server`` (a deprecated debug harness for
the distributed MLX path), this module is a normal host-local adapter
that can be started without ``mx.distributed.init`` and without editing
``vmlx.app.asar``.

Routes:
    GET  /health                  diagnostics probe
    GET  /v1/models               OpenAI-style model discovery
    POST /v1/chat/completions     OpenAI-style chat completions, supports
                                  ``stream=true`` SSE, plain text and
                                  OpenAI multimodal content arrays
                                  (``text`` + ``image_url`` data URLs)

Request normalization reuses the safe helpers in
``mlx_engine.distributed_server`` (``normalize_messages``,
``format_prompt``, ``optional_int``/``optional_float``,
``max_tokens_from_value``) so the cheetara contract keeps the same
shape the deprecated debug harness already produced.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import logging
import os
import threading
import time
import uuid
from typing import Any, AsyncIterator, Iterable, Optional

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from mlx_engine.distributed_server import (
    max_tokens_from_value,
    normalize_stop,
    optional_float,
    optional_int,
    reject_unsupported_request_fields,
)
from mlx_engine.generate import create_generator, load_model, unload
from mlx_engine.utils.chat_template_args import resolve_chat_template_args
from mlx_engine.utils.generation_result import GenerationStopCondition

logger = logging.getLogger(__name__)


# Fields that the deprecated debug harness rejected. We still reject them
# here so the cheetara contract is consistent across surfaces.
_REJECTED_REQUEST_FIELDS = {"adapters", "adapter", "draft_model", "num_draft_tokens"}

# Cheetara / OpenAI chat completion routes through the adapter can carry
# extra sampling knobs (top_k, min_p, repetition_penalty, repetition_context_size,
# reasoning_effort, chat_template_kwargs, enable_thinking, ...). We accept them
# silently so the request does not 400 on unsupported optional knobs.
_KNOWN_OPTIONAL_FIELDS = {
    "model",
    "messages",
    "stream",
    "stream_options",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "repetition_context_size",
    "stop",
    "max_tokens",
    "seed",
    "user",
    "n",
    "presence_penalty",
    "frequency_penalty",
    "response_format",
    "tools",
    "tool_choice",
    "logit_bias",
    "metadata",
    "store",
    "chat_template_kwargs",
    "enable_thinking",
    "reasoning_effort",
    "prompt_cache_key",
    "safety_identifier",
    "verbosity",
    "service_tier",
    "parallel_tool_calls",
    "audio",
    "modalities",
    "prediction",
    "web_search_options",
    "prompt_cache_retention",
    "stream_compression",
    "include_reasoning",
}

# Sentinel for ``finish_reason`` when the generator still produces output.
_FINISH_REASON_LENGTH = "length"
_FINISH_REASON_STOP = "stop"
_FINISH_REASON_CANCELLED = "cancelled"


class _AdapterState:
    """Container for the loaded model and runtime metadata."""

    def __init__(
        self,
        model_kit: Any,
        served_model_name: str,
        model_path: str,
        model_type: str,
        supports_vision: bool,
        *,
        generator_factory: Optional[Any] = None,
    ) -> None:
        self.model_kit = model_kit
        self.served_model_name = served_model_name
        self.model_path = model_path
        self.model_type = model_type
        self.supports_vision = supports_vision
        self.started_at = int(time.time())
        # Optional override hook for tests; defaults to ``mlx_engine.create_generator``.
        self.generator_factory = generator_factory or create_generator


def _detect_vision_support(model_kit: Any) -> bool:
    """Return True if the loaded model_kit looks like a vision model kit."""
    if model_kit is None:
        return False
    if hasattr(model_kit, "processor") and getattr(model_kit, "processor") is not None:
        return True
    model_type = getattr(model_kit, "model_type", None) or ""
    return "vision" in model_type.lower() or "vl" in model_type.lower()


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible request body.

    The adapter tolerates cheetara extras (``top_k``, ``min_p``,
    ``repetition_penalty``, ``chat_template_kwargs``, ``enable_thinking``,
    ``reasoning_effort``, ``stream_options``, ...) so requests do not 400
    on unsupported optional knobs. ``extra="allow"`` keeps unknown
    fields accessible via ``model_dump()`` instead of dropping them
    silently, while still accepting the documented OpenAI surface.
    """

    model_config = ConfigDict(extra="allow")

    messages: list[Any]
    stream: bool = False


def _build_app(
    state: _AdapterState,
    *,
    api_key: Optional[str] = None,
    auth_enabled: bool = False,
) -> FastAPI:
    """Build the FastAPI app around an already-loaded ``_AdapterState``."""

    app = FastAPI(
        title="mlx-engine OpenAI-compatible adapter",
        version="1.0.0",
        description=(
            "Host-local mlx-engine adapter that exposes the subset of the "
            "OpenAI / cheetara HTTP contract needed for cheetara "
            "external-endpoint mode. Backed by mlx-engine; never touches "
            "vmlx.app.asar."
        ),
    )

    @app.middleware("http")
    async def _enforce_bearer_auth(request: Request, call_next):
        if not auth_enabled:
            return await call_next(request)
        # Allow unauthenticated probes that do not carry any model data.
        if request.url.path in {"/health"}:
            return await call_next(request)
        expected = f"Bearer {api_key or ''}"
        provided = request.headers.get("authorization", "")
        if not api_key or provided != expected:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": {
                        "message": "Missing or invalid Authorization header",
                        "type": "invalid_request_error",
                        "code": "unauthorized",
                    }
                },
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "served_model": state.served_model_name,
            "model_path": state.model_path,
            "model_type": state.model_type,
            "supports_vision": state.supports_vision,
            "started_at": state.started_at,
            "now": int(time.time()),
        }

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": state.served_model_name,
                    "object": "model",
                    "created": state.started_at,
                    "owned_by": "mlx-engine",
                    "supports_vision": state.supports_vision,
                }
            ],
        }

    class _ChatCompletionRequest(ChatCompletionRequest):
        pass

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest = Body(...)) -> Any:
        raw_body = body.model_dump(exclude_none=False)
        await _validate_and_dispatch(raw_body, state)
        if raw_body.get("stream") is True:
            return StreamingResponse(
                _stream_chat_completion(raw_body, state),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        return await _non_stream_chat_completion(raw_body, state)

    return app


async def _validate_and_dispatch(raw_body: dict[str, Any], state: _AdapterState) -> None:
    """Validate the request, raise HTTPException on user errors."""
    if not isinstance(raw_body.get("messages"), list) or len(raw_body["messages"]) == 0:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": "messages must be a non-empty list"}},
        )
    # Reject explicit cheetara fields that the deprecated harness rejected
    # (we keep behavior consistent across surfaces).
    try:
        reject_unsupported_request_fields(raw_body)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": str(exc)}},
        ) from exc
    # Normalize sampling-field errors to HTTP 400 before SSE streaming
    # begins. ``max_tokens_from_value`` and ``normalize_stop`` raise
    # ``ValueError`` on bad input; without this catch the exception
    # would either become an SSE server_error chunk or a 500 once the
    # stream is already wired up. Validating here returns a clean
    # OpenAI-style invalid_request_error instead.
    try:
        _validate_sampling_fields(raw_body)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": str(exc)}},
        ) from exc
    # Detect vision content.
    if _request_has_image_payload(raw_body) and not state.supports_vision:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": (
                        "Loaded model does not support image inputs; "
                        "request contained image_url parts."
                    )
                }
            },
        )


def _validate_sampling_fields(body: dict[str, Any]) -> None:
    """Validate OpenAI sampling-field types before SSE streaming begins.

    Mirrors the cheetara contract: only ``None`` or the documented type
    is accepted for each knob. ``ValueError`` messages are normalized
    by ``_validate_and_dispatch`` into HTTP 400 invalid_request_error
    responses before any SSE chunk is yielded, so the caller never sees
    a server_error chunk for a malformed sampling field.
    """
    # ``max_tokens`` uses ``max_tokens_from_value`` so we surface the
    # same error the deprecated debug harness produced.
    max_tokens_from_value(body.get("max_tokens"))
    # ``stop`` uses ``normalize_stop`` for the same reason — a non-list
    # of strings raises ValueError with a message suitable for HTTP 400.
    normalize_stop(body.get("stop"))
    # ``temperature`` / ``top_p`` / ``min_p`` / ``repetition_penalty``
    # accept numbers or None. Reject booleans explicitly so ``True`` /
    # ``False`` never coerce to ``1`` / ``0`` silently.
    for float_field in ("temperature", "top_p", "min_p", "repetition_penalty"):
        if not _is_optional_number(body.get(float_field)):
            raise ValueError(
                f"{float_field} must be a number or null"
            )
    # ``top_k`` / ``repetition_context_size`` / ``seed`` accept ints or
    # None. ``bool`` is a subclass of ``int`` in Python so we exclude it
    # explicitly to keep the contract strict.
    for int_field in ("top_k", "repetition_context_size", "seed"):
        if not _is_optional_int(body.get(int_field)):
            raise ValueError(
                f"{int_field} must be an integer or null"
            )


def _is_optional_number(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


def _is_optional_int(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    return isinstance(value, int)


def _extract_text_and_images(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], list[str]]:
    """Convert OpenAI multimodal content arrays into normalized text + images.

    Returns ``(normalized_messages, images_b64)``. ``normalized_messages`` is
    shaped like the standard text-only messages consumed by
    ``format_prompt``; ``images_b64`` contains the raw base64 payloads
    (without the ``data:image/...;base64,`` prefix) for the VLM path.
    """

    normalized: list[dict[str, str]] = []
    images: list[str] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": f"messages[{message_index}] must be an object",
                    }
                },
            )
        role = message.get("role")
        if not isinstance(role, str) or len(role) == 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": f"messages[{message_index}].role must be a non-empty string",
                    }
                },
            )
        content = message.get("content")
        if isinstance(content, str):
            normalized.append({"role": role, "content": content})
            continue
        if isinstance(content, list):
            text_parts: list[str] = []
            for part_index, part in enumerate(content):
                if not isinstance(part, dict):
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": {
                                "message": (
                                    f"messages[{message_index}].content"
                                    f"[{part_index}] must be an object"
                                )
                            }
                        },
                    )
                part_type = part.get("type")
                if part_type == "text":
                    text = part.get("text", "")
                    if not isinstance(text, str):
                        raise HTTPException(
                            status_code=400,
                            detail={
                                "error": {
                                    "message": (
                                        f"messages[{message_index}].content"
                                        f"[{part_index}].text must be a string"
                                    )
                                }
                            },
                        )
                    text_parts.append(text)
                    continue
                if part_type in {"image_url", "image"}:
                    image_payload = _extract_image_payload(part)
                    if image_payload is None:
                        raise HTTPException(
                            status_code=400,
                            detail={
                                "error": {
                                    "message": (
                                        f"messages[{message_index}].content"
                                        f"[{part_index}] missing image_url.url"
                                    )
                                }
                            },
                        )
                    images.append(image_payload)
                    continue
                # Unknown content part type: tolerate silently, like the
                # deprecated debug harness does, so cheetara extras do not
                # 400 the request.
                continue
            normalized.append({"role": role, "content": "".join(text_parts)})
            continue
        if content is None:
            normalized.append({"role": role, "content": ""})
            continue
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": (
                        f"messages[{message_index}].content must be a string or an array"
                    )
                }
            },
        )
    return normalized, images


def _extract_image_payload(part: dict[str, Any]) -> Optional[str]:
    """Return base64 image payload from an OpenAI ``image_url`` part.

    Only two forms of ``image_url`` payload are accepted:

    - ``data:<mime>;base64,<payload>`` — the canonical OpenAI data URL.
      The ``<payload>`` is rejected unless it passes strict base64
      validation (``validate=True``); permissive decoding is not allowed
      here so malformed payloads do not silently pass as image input.
    - ``data:<mime>,<text>`` — the rare non-base64 data URL form, used
      for inline ``image/svg+xml`` payloads. The text payload is
      returned as-is after a minimal non-empty check.
    - A bare base64 string (no ``data:`` prefix). The string is rejected
      unless it passes strict base64 validation. Anything that is not
      valid base64 is treated as a remote URL the VLM path cannot fetch,
      and ``None`` is returned so the caller can reject the request.

    Remote URLs (``https://...``, ``http://...``, ``file://...``, etc.)
    are NOT accepted: returning ``None`` causes the caller to raise a
    400 invalid_request_error before SSE streaming begins.
    """
    image_field = part.get("image_url")
    if image_field is None and "image" in part:
        image_field = part.get("image")
    url: Optional[str] = None
    if isinstance(image_field, dict):
        candidate = image_field.get("url")
        if isinstance(candidate, str):
            url = candidate
    elif isinstance(image_field, str):
        url = image_field
    if not isinstance(url, str) or len(url) == 0:
        return None
    if url.startswith("data:"):
        # ``data:image/png;base64,XXXX`` -> strip prefix and decode.
        comma_index = url.find(",")
        if comma_index < 0:
            return None
        prefix = url[:comma_index]
        payload = url[comma_index + 1 :]
        if len(payload) == 0:
            return None
        if ";base64" in prefix:
            try:
                # Strict base64 acceptance: reject any non-base64 chars
                # or padding/segment-length issues instead of silently
                # passing malformed payloads as image input.
                base64.b64decode(payload, validate=True)
            except (ValueError, binascii.Error):
                return None
            return payload
        # Non-base64 data URL (e.g. ``data:image/svg+xml,...``).
        return payload
    # Bare string with no ``data:`` prefix. The hardened contract only
    # accepts STRICT base64 — anything else is treated as a remote URL
    # the VLM path does not fetch, so ``None`` is returned to signal a
    # caller-side rejection rather than letting a malformed string pass
    # as image input.
    try:
        base64.b64decode(url, validate=True)
    except (ValueError, binascii.Error):
        return None
    return url


def _request_has_image_payload(body: dict[str, Any]) -> bool:
    """Return True if any message contains an ``image_url`` part with a data URL."""
    for message in body.get("messages", []) or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"image_url", "image"}:
                return True
    return False


def _build_chat_request(
    state: _AdapterState,
    body: dict[str, Any],
) -> tuple[list[int], Optional[list[str]], dict[str, Any]]:
    """Resolve prompt tokens, image payload, and sampling args for a request."""

    normalized_messages, images_b64 = _extract_text_and_images(body.get("messages", []))
    template_args = resolve_chat_template_args(
        getattr(state.model_kit, "tokenizer", None),
        {},
        body.get("chat_template_kwargs"),
    )
    if state.supports_vision and images_b64:
        renderer = getattr(state.model_kit, "processor", None) or state.model_kit.tokenizer
        if (
            hasattr(renderer, "apply_chat_template")
            and getattr(renderer, "chat_template", None) is None
            and getattr(state.model_kit.tokenizer, "chat_template", None) is not None
        ):
            renderer.chat_template = state.model_kit.tokenizer.chat_template
        content_blocks: list[dict[str, Any]] = [
            *({"type": "image"} for _ in images_b64),
            {
                "type": "text",
                "text": _last_user_text(normalized_messages),
            },
        ]
        chat_messages: list[dict[str, Any]] = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in normalized_messages
        ]
        # The final user turn carries media; replace plain content with blocks.
        if chat_messages:
            chat_messages[-1] = {
                "role": chat_messages[-1]["role"],
                "content": content_blocks,
            }
        prompt_text = renderer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_args,
        )
    else:
        prompt_text = state.model_kit.tokenizer.apply_chat_template(
            normalized_messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_args,
        )
    prompt_tokens = state.model_kit.tokenize(prompt_text)
    sampling = {
        "temperature": optional_float(body.get("temperature")),
        "top_p": optional_float(body.get("top_p")),
        "top_k": optional_int(body.get("top_k")),
        "min_p": optional_float(body.get("min_p")),
        "repetition_penalty": optional_float(body.get("repetition_penalty")),
        # ``repetition_context_size`` defaults to 20 in the underlying
        # ``_sequential_generation`` call; surface that default here so
        # requests that set ``repetition_penalty`` without an explicit
        # context size do not break the logits-processor init.
        "repetition_context_size": optional_int(
            body.get("repetition_context_size")
        )
        or 20,
        "max_tokens": max_tokens_from_value(body.get("max_tokens")),
        "stop_strings": normalize_stop(body.get("stop")),
        "seed": optional_int(body.get("seed")),
    }
    return prompt_tokens, (images_b64 or None), sampling


def _last_user_text(normalized_messages: list[dict[str, str]]) -> str:
    """Return concatenated text from the final user turn for VLM templates."""
    for message in reversed(normalized_messages):
        if message.get("role") == "user":
            return message.get("content") or ""
    return ""


def _resolve_request_id(body: dict[str, Any]) -> str:
    candidate = body.get("user")
    if isinstance(candidate, str) and len(candidate) > 0:
        return f"chatcmpl-{candidate}-{uuid.uuid4().hex[:8]}"
    return f"chatcmpl-{uuid.uuid4().hex}"


def _serialize_sse(payload: dict[str, Any]) -> bytes:
    return f"data: {__import__('json').dumps(payload, separators=(',', ':'))}\n\n".encode("utf-8")


def _to_openai_chunk(
    *,
    request_id: str,
    model: str,
    delta: dict[str, Any],
    finish_reason: Optional[str],
) -> dict[str, Any]:
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


async def _stream_chat_completion(
    body: dict[str, Any],
    state: _AdapterState,
) -> AsyncIterator[bytes]:
    """Async SSE stream for one chat completion request."""
    request_id = _resolve_request_id(body)
    model_name = state.served_model_name
    try:
        prompt_tokens, images_b64, sampling = _build_chat_request(state, body)
    except HTTPException as http_exc:
        yield _serialize_sse(
            {
                "error": {
                    "message": str(
                        http_exc.detail.get("error", {}).get("message", "Invalid request")
                    ),
                    "type": "invalid_request_error",
                }
            }
        )
        return
    except ValueError as exc:
        # Defense-in-depth: ``_validate_and_dispatch`` already converts
        # sampling-field ``ValueError`` into HTTP 400 before SSE starts,
        # but a residual ``ValueError`` from prompt construction should
        # still surface as an invalid_request_error SSE chunk rather
        # than a generic server_error or a 500 once the stream is wired
        # up.
        logger.warning(
            "Adapter request_id=%s pre-stream validation failed: %s",
            request_id,
            exc,
        )
        yield _serialize_sse(
            {
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error",
                }
            }
        )
        return

    def _run_generator() -> Iterable[Any]:
        return state.generator_factory(
            state.model_kit,
            prompt_tokens,
            max_tokens=sampling["max_tokens"],
            stop_strings=sampling["stop_strings"],
            temp=sampling["temperature"],
            top_p=sampling["top_p"],
            top_k=sampling["top_k"],
            min_p=sampling["min_p"],
            repetition_penalty=sampling["repetition_penalty"],
            repetition_context_size=sampling["repetition_context_size"],
            seed=sampling["seed"],
            images_b64=images_b64,
            request_id=request_id,
        )

    yield _serialize_sse(
        _to_openai_chunk(
            request_id=request_id,
            model=model_name,
            delta={"role": "assistant", "content": ""},
            finish_reason=None,
        )
    )

    finish_reason = _FINISH_REASON_STOP
    loop = asyncio.get_running_loop()
    try:
        generator = await loop.run_in_executor(None, _run_generator)
        sentinel = object()
        while True:
            try:
                result = await loop.run_in_executor(
                    None, _next_safely, generator, sentinel
                )
            except HTTPException as http_exc:
                logger.warning(
                    "Adapter request_id=%s generation failed: %s",
                    request_id,
                    http_exc.detail,
                )
                yield _serialize_sse(
                    {
                        "error": {
                            "message": str(
                                http_exc.detail.get("error", {}).get("message", "")
                            ),
                            "type": "server_error",
                        }
                    }
                )
                return
            if result is sentinel:
                break
            text = getattr(result, "text", "") or ""
            stop_condition = getattr(result, "stop_condition", None)
            next_finish_reason = _finish_reason_from_stop_condition(stop_condition)
            if next_finish_reason is not None:
                finish_reason = next_finish_reason
            if text:
                yield _serialize_sse(
                    _to_openai_chunk(
                        request_id=request_id,
                        model=model_name,
                        delta={"content": text},
                        finish_reason=None,
                    )
                )
    except Exception as exc:
        logger.exception("Adapter streaming failed request_id=%s", request_id)
        yield _serialize_sse(
            {
                "error": {
                    "message": f"{type(exc).__name__}: {exc}",
                    "type": "server_error",
                }
            }
        )
        return
    yield _serialize_sse(
        _to_openai_chunk(
            request_id=request_id,
            model=model_name,
            delta={},
            finish_reason=finish_reason,
        )
    )
    yield b"data: [DONE]\n\n"


def _next_safely(generator: Iterable[Any], sentinel: object) -> Any:
    """Call ``next`` and convert ``StopIteration`` to a sentinel return.

    PEP 479 forbids raising ``StopIteration`` into a Future, which is what
    happens when ``next`` is dispatched via ``run_in_executor``. Returning
    a sentinel keeps the worker thread happy.
    """
    try:
        return next(generator)
    except StopIteration:
        return sentinel


def _finish_reason_from_stop_condition(
    stop_condition: Optional[GenerationStopCondition],
) -> Optional[str]:
    if stop_condition is None:
        return None
    if stop_condition.stop_reason == "token_limit":
        return _FINISH_REASON_LENGTH
    if stop_condition.stop_reason == "user_cancelled":
        return _FINISH_REASON_CANCELLED
    return _FINISH_REASON_STOP


async def _non_stream_chat_completion(
    body: dict[str, Any],
    state: _AdapterState,
) -> dict[str, Any]:
    """Return a single JSON response for a non-streaming chat request."""
    request_id = _resolve_request_id(body)
    model_name = state.served_model_name
    prompt_tokens, images_b64, sampling = _build_chat_request(state, body)
    text_parts: list[str] = []
    finish_reason = _FINISH_REASON_STOP
    generator = state.generator_factory(
        state.model_kit,
        prompt_tokens,
        max_tokens=sampling["max_tokens"],
        stop_strings=sampling["stop_strings"],
        temp=sampling["temperature"],
        top_p=sampling["top_p"],
        top_k=sampling["top_k"],
        min_p=sampling["min_p"],
        repetition_penalty=sampling["repetition_penalty"],
        repetition_context_size=sampling["repetition_context_size"],
        seed=sampling["seed"],
        images_b64=images_b64,
        request_id=request_id,
    )
    for result in generator:
        text_parts.append(getattr(result, "text", "") or "")
        stop_condition = getattr(result, "stop_condition", None)
        next_finish_reason = _finish_reason_from_stop_condition(stop_condition)
        if next_finish_reason is not None:
            finish_reason = next_finish_reason
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "".join(text_parts)},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": sum(len(part) for part in text_parts),
            "total_tokens": len(prompt_tokens),
        },
    }


def _load_model_kit(args: argparse.Namespace) -> tuple[Any, str, bool]:
    """Load the model_kit via ``mlx_engine.load_model`` with adapter kwargs."""

    load_kwargs: dict[str, Any] = {
        "max_seq_nums": args.max_seq_nums,
    }
    if args.prefill_step_size is not None:
        load_kwargs["prefill_step_size"] = args.prefill_step_size
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    if args.vlm_prompt_cache_root:
        load_kwargs["vlm_prompt_cache_storage_root"] = args.vlm_prompt_cache_root
    if args.vlm_prompt_cache_namespace:
        load_kwargs["vlm_prompt_cache_namespace"] = args.vlm_prompt_cache_namespace
    if args.vlm_prompt_cache_min_save_tokens is not None:
        load_kwargs["vlm_prompt_cache_min_save_tokens"] = (
            args.vlm_prompt_cache_min_save_tokens
        )
    model_kit = load_model(args.model, **load_kwargs)
    served_model_name = args.served_model_name or os.path.basename(args.model.rstrip("/"))
    supports_vision = _detect_vision_support(model_kit)
    return model_kit, served_model_name, supports_vision


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "mlx-engine host-local OpenAI-compatible adapter. "
            "Use ``--api-key`` to enable Bearer auth; omit it for local-only "
            "cheetara external-endpoint mode."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3180)
    parser.add_argument("--model", required=True, help="Path to mlx-engine model dir")
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--max-seq-nums", type=int, default=1)
    parser.add_argument("--prefill-step-size", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--vlm-prompt-cache-root", default=None)
    parser.add_argument("--vlm-prompt-cache-namespace", default=None)
    parser.add_argument(
        "--vlm-prompt-cache-min-save-tokens",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "Enable Bearer auth and require ``Authorization: Bearer <key>`` "
            "for non-/health routes. Without this flag the adapter runs "
            "without auth (the documented cheetara cutover default)."
        ),
    )
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> None:
    """Entry point: load the model and start the uvicorn server."""
    args = _parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )
    auth_enabled = bool(args.api_key)
    logger.info(
        "mlx-engine OpenAI adapter starting model=%s host=%s port=%s auth=%s",
        args.model,
        args.host,
        args.port,
        "enabled" if auth_enabled else "disabled",
    )

    model_kit, served_model_name, supports_vision = _load_model_kit(args)
    logger.info(
        "Loaded model_kit=%s served_model_name=%s supports_vision=%s",
        type(model_kit).__name__,
        served_model_name,
        supports_vision,
    )
    state = _AdapterState(
        model_kit=model_kit,
        served_model_name=served_model_name,
        model_path=args.model,
        model_type=str(getattr(model_kit, "model_type", "")),
        supports_vision=supports_vision,
    )
    app = _build_app(state, api_key=args.api_key, auth_enabled=auth_enabled)

    config = uvicorn.Config(
        app=app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        access_log=False,
        lifespan="off",
    )
    server = uvicorn.Server(config)

    def _shutdown_handler(*_: Any) -> None:
        server.should_exit = True

    import signal

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    shutdown_thread = threading.Thread(
        target=server.run, daemon=True, name="mlx-engine-adapter"
    )
    shutdown_thread.start()
    try:
        while shutdown_thread.is_alive():
            shutdown_thread.join(timeout=0.5)
    finally:
        try:
            unload(model_kit, force=True)
        except Exception:
            logger.exception("Failed to unload model_kit during adapter shutdown")


if __name__ == "__main__":
    main()
