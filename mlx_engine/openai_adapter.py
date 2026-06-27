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
import json
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


class ResponsesRequest(BaseModel):
    """OpenAI Responses-style request body (``POST /v1/responses``).

    The Responses API differs from Chat Completions in two ways:

    * The user turn is carried as ``input`` (a bare string or a list of
      structured message parts) instead of ``messages=[{"role":"user",
      "content": "..."}]``. The M9 compatibility layer normalizes this
      into the same chat-shaped messages the chat-completions route
      consumes so we reuse the same generation backend.
    * Streamed output is a sequence of typed events
      (``response.created`` → ``response.output_item.added`` →
      ``response.content_part.added`` → ``response.output_text.delta``
      → ``response.content_part.done`` → ``response.output_item.done``
      → ``response.completed``), not the chat-completion chunk stream.

    Like :class:`ChatCompletionRequest`, this body uses ``extra="allow"``
    so cheetara extras (``top_k``, ``min_p``, ``repetition_penalty``,
    ``chat_template_kwargs``, ``enable_thinking``, ``reasoning_effort``,
    ``stream_options``, ...) are accepted without 400ing.
    """

    model_config = ConfigDict(extra="allow")

    model: str
    input: Any
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

    @app.post("/v1/responses")
    async def responses(body: ResponsesRequest = Body(...)) -> Any:
        """OpenAI Responses-style generation surface.

        Preserves the ``vmlx_engine.cli serve`` / cheetara local-runtime
        Responses surface. The M9 source-level compatibility layer reaches
        this route through the same mlx-engine adapter that handles
        chat completions on port 3181 (and the external adapter on 3180).
        """
        raw_body = body.model_dump(exclude_none=False)
        await _validate_and_dispatch_responses(raw_body, state)
        if raw_body.get("stream") is True:
            return StreamingResponse(
                _stream_responses(raw_body, state),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        return await _non_stream_responses(raw_body, state)

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


def _normalize_responses_input(raw_input: Any) -> list[dict[str, Any]]:
    """Convert a Responses-API ``input`` to the standard chat messages list.

    The Responses API accepts three input shapes:

    * A bare string: treated as a single user turn.
    * A list of strings: each string becomes a user turn in order.
    * A list of structured message parts (``{"role": ..., "content": ...}``
      or the Responses-API ``{"type": "message", "role": ..., "content": ...}``
      form). ``content`` may be a string or a list of typed parts
      (``{"type": "input_text", "text": ...}`` /
      ``{"type": "input_image", ...}``); text parts are joined, and
      image parts are folded into the multimodal extraction path via
      ``image_url`` aliases so they reuse the existing VLM route.

    Anything that is not string / list, or any structured part without
    a recognised ``type`` / ``role``, raises ``HTTPException`` with a
    400 invalid_request_error so the caller sees the rejection before
    SSE starts.
    """

    if isinstance(raw_input, str):
        text = raw_input.strip()
        if not text:
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": "input must be non-empty"}},
            )
        return [{"role": "user", "content": text}]
    if not isinstance(raw_input, list) or len(raw_input) == 0:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": "input must be a non-empty string or list"}},
        )
    normalized: list[dict[str, Any]] = []
    for index, part in enumerate(raw_input):
        if isinstance(part, str):
            normalized.append({"role": "user", "content": part})
            continue
        if not isinstance(part, dict):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": f"input[{index}] must be a string or object",
                    }
                },
            )
        # ``type == "message"`` wraps the OpenAI Responses shape; the
        # ``role``/``content`` keys live one level deeper.
        if part.get("type") == "message":
            role = part.get("role", "user")
            content = part.get("content", "")
        else:
            role = part.get("role", "user")
            content = part.get("content", "")
        if role not in {"user", "assistant", "system", "developer"}:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            f"input[{index}].role {role!r} is not supported"
                        ),
                    }
                },
            )
        # ``content`` may itself be a list of typed parts
        # (``{"type": "input_text", "text": ...}`` / image variants).
        if isinstance(content, list):
            text_pieces: list[str] = []
            for sub_index, sub_part in enumerate(content):
                if not isinstance(sub_part, dict):
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": {
                                "message": (
                                    f"input[{index}].content[{sub_index}]"
                                    " must be an object"
                                ),
                            }
                        },
                    )
                sub_type = sub_part.get("type")
                if sub_type in {"input_text", "text"}:
                    text_pieces.append(str(sub_part.get("text", "")))
                    continue
                if sub_type in {"input_image", "image", "image_url"}:
                    # Fold the Responses image shape into the same
                    # multimodal payload the chat-completions path
                    # consumes: ``{"type": "image_url", "image_url": {...}}``.
                    image_field = sub_part.get("image_url")
                    if image_field is None and "image" in sub_part:
                        image_field = sub_part.get("image")
                    normalized.append(
                        {
                            "role": role,
                            "content": [
                                {"type": "image_url", "image_url": image_field}
                            ],
                        }
                    )
                    continue
                # Unknown typed part: tolerate silently (matches the
                # chat-completions cheetara-extra tolerance) so the
                # Responses surface does not 400 on unknown optional
                # knobs the way the chat surface tolerates.
                continue
            normalized.append({"role": role, "content": "".join(text_pieces)})
            continue
        if isinstance(content, str):
            normalized.append({"role": role, "content": content})
            continue
        if content is None:
            normalized.append({"role": role, "content": ""})
            continue
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": (
                        f"input[{index}].content must be a string or an array"
                    ),
                }
            },
        )
    return normalized


def _responses_input_has_image_payload(raw_input: Any) -> bool:
    """Return True if a Responses ``input`` carries any image-bearing part."""
    if not isinstance(raw_input, list):
        return False
    for part in raw_input:
        if not isinstance(part, dict):
            continue
        content = part.get("content")
        if isinstance(content, list):
            for sub_part in content:
                if not isinstance(sub_part, dict):
                    continue
                if sub_part.get("type") in {
                    "input_image",
                    "image",
                    "image_url",
                }:
                    return True
        elif isinstance(content, dict):
            if content.get("type") in {"input_image", "image", "image_url"}:
                return True
    return False


async def _validate_and_dispatch_responses(
    raw_body: dict[str, Any],
    state: _AdapterState,
) -> None:
    """Validate a Responses-style request before SSE streaming begins.

    Mirrors the chat-completion validation discipline: bad ``input``,
    bad sampling fields, or image payloads on a text-only loaded model
    are rejected with HTTP 400 BEFORE the stream is wired up so the
    caller never sees a partial SSE failure.
    """
    raw_input = raw_body.get("input")
    if raw_input is None or (
        isinstance(raw_input, (list, str)) and len(raw_input) == 0
    ):
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": "input must be a non-empty string or list"}},
        )
    # Try the normalize step so we surface malformed input shapes as 400s
    # rather than mid-stream server_error chunks.
    try:
        _normalize_responses_input(raw_input)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": str(exc)}},
        ) from exc
    # Translate ``max_output_tokens`` (Responses-style) into the
    # ``max_tokens`` field that the chat-completions validation helper
    # consumes. The Responses-native field always wins: when a caller
    # sets both ``max_output_tokens`` and ``max_tokens`` on a Responses
    # request, ``max_output_tokens`` is the authoritative Responses
    # contract and ``max_tokens`` is treated as a deprecated chat-style
    # alias that ``max_output_tokens`` overrides. This matches the
    # documented Responses semantics so future workers and validators
    # are not misled by stale precedence comments.
    if raw_body.get("max_output_tokens") is not None:
        raw_body["max_tokens"] = raw_body["max_output_tokens"]
    try:
        _validate_sampling_fields(raw_body)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": str(exc)}},
        ) from exc
    if _responses_input_has_image_payload(raw_input) and not state.supports_vision:
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


def _resolve_responses_request_id(body: dict[str, Any]) -> str:
    """Return the OpenAI Responses-API request id (``resp_<hex>``)."""
    user_field = body.get("user")
    if isinstance(user_field, str) and len(user_field) > 0:
        return f"resp_{user_field}-{uuid.uuid4().hex[:8]}"
    return f"resp_{uuid.uuid4().hex[:12]}"


def _build_responses_chat_body(
    raw_body: dict[str, Any],
) -> dict[str, Any]:
    """Translate a Responses-API body into the chat-completion body shape.

    The Responses surface reuses the chat-completion generation
    backend, so we normalize ``input`` to ``messages`` and rename
    ``max_output_tokens`` to ``max_tokens`` here. As documented for
    the Responses surface, the Responses-native ``max_output_tokens``
    always wins over ``max_tokens`` when both are present, so we
    unconditionally let ``max_output_tokens`` overwrite the chat-style
    alias. The returned dict is the body the chat-completion helpers
    already understand; no other helper needs to know whether the
    caller used /v1/responses or /v1/chat/completions.
    """
    chat_body = dict(raw_body)
    chat_body.pop("input", None)
    chat_body["messages"] = _normalize_responses_input(raw_body.get("input"))
    if raw_body.get("max_output_tokens") is not None:
        chat_body["max_tokens"] = raw_body["max_output_tokens"]
    return chat_body


def _serialize_responses_event(event_type: str, payload: dict[str, Any]) -> bytes:
    """Encode one typed Responses-style SSE event with the typed-event header.

    The Responses API uses ``event: <type>\\ndata: <json>\\n\\n`` so
    clients can dispatch by event type without parsing the payload. The
    chat-completion ``data: <json>\\n\\n`` events remain unchanged for
    /v1/chat/completions.
    """
    encoded = json.dumps(payload, separators=(",", ":"))
    return f"event: {event_type}\ndata: {encoded}\n\n".encode("utf-8")


def _serialize_responses_terminal(data_event: Optional[str] = None) -> bytes:
    """Emit the terminal ``[DONE]`` marker for the Responses surface.

    Responses clients (e.g. the OpenAI SDK) accept a trailing
    ``data: [DONE]`` line regardless of the typed-event format, so we
    emit the same marker the chat-completion surface uses. The function
    is a no-op when ``data_event`` is ``None`` so callers can choose
    between a typed terminal event and the standard terminal marker.
    """
    if data_event is None:
        return b"data: [DONE]\n\n"
    return f"data: {data_event}\n\n".encode("utf-8")


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


async def _stream_responses(
    raw_body: dict[str, Any],
    state: _AdapterState,
) -> AsyncIterator[bytes]:
    """Stream an OpenAI Responses-style SSE response.

    Emits the canonical Responses event sequence:

    1. ``response.created`` — request accepted, status=in_progress.
    2. ``response.output_item.added`` — the assistant message item
       has been created with status=in_progress.
    3. ``response.content_part.added`` — the first ``output_text`` part
       has been added to the item.
    4. ``response.output_text.delta`` — one event per generation step,
       carrying the incremental text produced by the chat-completions
       generator. The chat-completion backend emits non-empty chunks
       per step; the Responses surface wraps each chunk in a typed
       ``response.output_text.delta`` event so cheetara-style clients
       can render streaming output incrementally without parsing the
       chat-completion ``choices[0].delta.content`` envelope.
    5. ``response.content_part.done`` — content_part finalized.
    6. ``response.output_item.done`` — output item finalized.
    7. ``response.completed`` — final response object with
       status=completed and the full output text.
    8. ``data: [DONE]`` — terminal marker, matching the
       chat-completion surface.

    All events share the same ``response_id`` and ``sequence_number``
    so the client can order and de-duplicate them. Any exception during
    the generation step surfaces as a typed
    ``error`` chunk in the SSE stream followed by ``[DONE]`` so the
    client never sees a half-finished response without a terminal
    marker (the M9 streaming contract).
    """
    request_id = _resolve_responses_request_id(raw_body)
    model_name = state.served_model_name
    item_id = f"item_{uuid.uuid4().hex[:12]}"
    chat_body = _build_responses_chat_body(raw_body)
    try:
        prompt_tokens, images_b64, sampling = _build_chat_request(state, chat_body)
    except HTTPException as http_exc:
        logger.warning(
            "Adapter Responses request_id=%s validation failed: %s",
            request_id,
            http_exc.detail,
        )
        yield _serialize_responses_event(
            "error",
            {
                "type": "error",
                "error": {
                    "message": str(
                        http_exc.detail.get("error", {}).get(
                            "message", "Invalid request"
                        )
                    ),
                    "type": "invalid_request_error",
                },
                "response_id": request_id,
            },
        )
        yield _serialize_responses_terminal()
        return
    except ValueError as exc:
        logger.warning(
            "Adapter Responses request_id=%s pre-stream validation failed: %s",
            request_id,
            exc,
        )
        yield _serialize_responses_event(
            "error",
            {
                "type": "error",
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error",
                },
                "response_id": request_id,
            },
        )
        yield _serialize_responses_terminal()
        return

    sequence_number = 0
    created_at = int(time.time())
    output_text_pieces: list[str] = []
    status = "completed"

    def _next_seq() -> int:
        nonlocal sequence_number
        current = sequence_number
        sequence_number += 1
        return current

    yield _serialize_responses_event(
        "response.created",
        {
            "type": "response.created",
            "sequence_number": _next_seq(),
            "response": {
                "id": request_id,
                "object": "response",
                "created_at": created_at,
                "status": "in_progress",
                "model": model_name,
                "output": [],
            },
        },
    )
    yield _serialize_responses_event(
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "sequence_number": _next_seq(),
            "output_index": 0,
            "item": {
                "id": item_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        },
    )
    yield _serialize_responses_event(
        "response.content_part.added",
        {
            "type": "response.content_part.added",
            "sequence_number": _next_seq(),
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
    )

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
                    "Adapter Responses request_id=%s generation failed: %s",
                    request_id,
                    http_exc.detail,
                )
                yield _serialize_responses_event(
                    "error",
                    {
                        "type": "error",
                        "error": {
                            "message": str(
                                http_exc.detail.get("error", {}).get(
                                    "message", ""
                                )
                            ),
                            "type": "server_error",
                        },
                        "response_id": request_id,
                    },
                )
                yield _serialize_responses_terminal()
                return
            if result is sentinel:
                break
            text = getattr(result, "text", "") or ""
            if text:
                output_text_pieces.append(text)
                yield _serialize_responses_event(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "sequence_number": _next_seq(),
                        "item_id": item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": text,
                    },
                )
            stop_condition = getattr(result, "stop_condition", None)
            next_finish_reason = _finish_reason_from_stop_condition(stop_condition)
            if next_finish_reason == _FINISH_REASON_LENGTH:
                status = "incomplete"
    except Exception as exc:
        logger.exception(
            "Adapter Responses streaming failed request_id=%s", request_id
        )
        yield _serialize_responses_event(
            "error",
            {
                "type": "error",
                "error": {
                    "message": f"{type(exc).__name__}: {exc}",
                    "type": "server_error",
                },
                "response_id": request_id,
            },
        )
        yield _serialize_responses_terminal()
        return

    full_text = "".join(output_text_pieces)
    yield _serialize_responses_event(
        "response.content_part.done",
        {
            "type": "response.content_part.done",
            "sequence_number": _next_seq(),
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": full_text,
                "annotations": [],
            },
        },
    )
    yield _serialize_responses_event(
        "response.output_item.done",
        {
            "type": "response.output_item.done",
            "sequence_number": _next_seq(),
            "output_index": 0,
            "item": {
                "id": item_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": full_text,
                        "annotations": [],
                    }
                ],
            },
        },
    )
    yield _serialize_responses_event(
        "response.completed",
        {
            "type": "response.completed",
            "sequence_number": _next_seq(),
            "response": {
                "id": request_id,
                "object": "response",
                "created_at": created_at,
                "status": status,
                "model": model_name,
                "output": [
                    {
                        "id": item_id,
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": full_text,
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": len(prompt_tokens),
                    "output_tokens": sum(len(part) for part in output_text_pieces),
                    "total_tokens": len(prompt_tokens)
                    + sum(len(part) for part in output_text_pieces),
                },
            },
        },
    )
    yield _serialize_responses_terminal()


async def _non_stream_responses(
    raw_body: dict[str, Any],
    state: _AdapterState,
) -> dict[str, Any]:
    """Return a single JSON response for a non-streaming Responses request.

    Mirrors the chat-completion non-streaming shape but emits the
    Responses-API ``output[]`` array instead of ``choices[]`` and uses
    ``input_tokens``/``output_tokens`` usage keys per the OpenAI
    Responses contract.
    """
    request_id = _resolve_responses_request_id(raw_body)
    model_name = state.served_model_name
    chat_body = _build_responses_chat_body(raw_body)
    prompt_tokens, images_b64, sampling = _build_chat_request(state, chat_body)
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
    full_text = "".join(text_parts)
    completion_tokens = sum(len(part) for part in text_parts)
    status = "incomplete" if finish_reason == _FINISH_REASON_LENGTH else "completed"
    return {
        "id": request_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": model_name,
        "output": [
            {
                "id": f"item_{uuid.uuid4().hex[:12]}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": full_text,
                        "annotations": [],
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": len(prompt_tokens),
            "output_tokens": completion_tokens,
            "total_tokens": len(prompt_tokens) + completion_tokens,
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
