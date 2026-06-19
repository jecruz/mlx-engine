import importlib
import logging
import threading
from typing import Any

import mlx.core as mx


mlx_lm_generate = importlib.import_module("mlx_lm.generate")


logger = logging.getLogger(__name__)
_thread_state = threading.local()


def _format_distributed_group(distributed_group: Any | None) -> str:
    if distributed_group is None:
        return "none"
    try:
        return f"{distributed_group.rank()}/{distributed_group.size()}"
    except Exception as caught_error:
        return f"unknown({caught_error})"


def prepare_mlx_lm_generation_stream(
    *,
    reason: str,
    request_id: str | None = None,
    distributed_group: Any | None = None,
    use_default_stream: bool = False,
):
    current_thread = threading.current_thread()
    thread_ident = current_thread.ident
    default_device = mx.default_device()
    default_stream = mx.default_stream(default_device)
    stream_source = "default" if use_default_stream else "thread-local"
    cached_stream = getattr(_thread_state, "generation_stream", None)
    cached_source = getattr(_thread_state, "stream_source", None)
    cached_device = getattr(_thread_state, "device", None)

    if (
        cached_stream is not None
        and cached_source == stream_source
        and cached_device == default_device
    ):
        mlx_lm_generate.generation_stream = cached_stream
        logger.debug(
            "Reusing MLX-LM generation stream reason=%s request_id=%s rank=%s "
            "thread=%s thread_ident=%s device=%s stream_source=%s stream=%r",
            reason,
            request_id,
            _format_distributed_group(distributed_group),
            current_thread.name,
            thread_ident,
            default_device,
            stream_source,
            cached_stream,
        )
        return cached_stream

    if use_default_stream:
        generation_stream = default_stream
    else:
        generation_stream = mx.new_thread_local_stream(default_device)
    _thread_state.generation_stream = generation_stream
    _thread_state.stream_source = stream_source
    _thread_state.device = default_device
    mlx_lm_generate.generation_stream = generation_stream
    logger.info(
        "Prepared MLX-LM generation stream reason=%s request_id=%s rank=%s "
        "thread=%s thread_ident=%s device=%s stream_source=%s default_stream=%r "
        "stream=%r",
        reason,
        request_id,
        _format_distributed_group(distributed_group),
        current_thread.name,
        thread_ident,
        default_device,
        stream_source,
        default_stream,
        generation_stream,
    )
    return generation_stream


def log_mlx_stream_state(
    *,
    reason: str,
    request_id: str | None = None,
    distributed_group: Any | None = None,
    details: str | None = None,
) -> None:
    current_thread = threading.current_thread()
    default_device = mx.default_device()
    logger.info(
        "MLX stream state reason=%s request_id=%s rank=%s thread=%s "
        "thread_ident=%s device=%s default_stream=%r generation_stream=%r details=%s",
        reason,
        request_id,
        _format_distributed_group(distributed_group),
        current_thread.name,
        current_thread.ident,
        default_device,
        mx.default_stream(default_device),
        getattr(mlx_lm_generate, "generation_stream", None),
        details,
    )


def log_mlx_generation_exception(
    *,
    reason: str,
    request_id: str | None = None,
    distributed_group: Any | None = None,
) -> None:
    current_thread = threading.current_thread()
    default_device = mx.default_device()
    logger.exception(
        "MLX generation failed after stream preparation reason=%s request_id=%s "
        "rank=%s thread=%s thread_ident=%s device=%s default_stream=%r stream=%r",
        reason,
        request_id,
        _format_distributed_group(distributed_group),
        current_thread.name,
        current_thread.ident,
        default_device,
        mx.default_stream(default_device),
        getattr(mlx_lm_generate, "generation_stream", None),
    )
