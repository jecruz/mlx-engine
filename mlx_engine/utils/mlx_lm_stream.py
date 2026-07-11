import importlib
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any

import mlx.core as mx


mlx_lm_generate = importlib.import_module("mlx_lm.generate")


logger = logging.getLogger(__name__)
_thread_state = threading.local()
_global_stream_lock = threading.Lock()
_global_thread_unsafe_streams: dict[str, Any] = {}
_THREAD_UNSAFE_STREAM_TOGGLE_FILE = Path("/tmp/mlx-engine-thread-unsafe-stream")


def _format_distributed_group(distributed_group: Any | None) -> str:
    if distributed_group is None:
        return "none"
    try:
        return f"{distributed_group.rank()}/{distributed_group.size()}"
    except Exception as caught_error:
        return f"unknown({caught_error})"


def _thread_unsafe_stream_experiment_enabled() -> bool:
    """Return whether the shared thread-unsafe stream experiment is enabled."""
    value = os.getenv("MLX_ENGINE_EXPERIMENTAL_THREAD_UNSAFE_STREAM", "")
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    return _THREAD_UNSAFE_STREAM_TOGGLE_FILE.exists()


def _runtime_supports_thread_unsafe_stream() -> bool:
    """Return whether the active MLX runtime exposes new_thread_unsafe_stream."""
    return hasattr(mx, "new_thread_unsafe_stream")


def _resolve_stream_source(use_default_stream: bool) -> str:
    """Choose the stream source for this request."""
    if use_default_stream:
        return "default"
    if (
        _thread_unsafe_stream_experiment_enabled()
        and _runtime_supports_thread_unsafe_stream()
    ):
        return "thread-unsafe"
    return "thread-local"


def describe_stream_configuration(use_default_stream: bool) -> str:
    """Return a stable summary of the current MLX stream selection inputs."""
    return (
        f"source={_resolve_stream_source(use_default_stream)} "
        f"use_default_stream={use_default_stream} "
        f"toggle_env={_thread_unsafe_stream_experiment_enabled()} "
        f"toggle_file={_THREAD_UNSAFE_STREAM_TOGGLE_FILE.exists()} "
        f"runtime_supports_thread_unsafe={_runtime_supports_thread_unsafe_stream()} "
        f"toggle_path={_THREAD_UNSAFE_STREAM_TOGGLE_FILE}"
    )


def emit_stream_configuration_probe(*, reason: str, use_default_stream: bool) -> None:
    """Emit a stderr-visible probe line when the stream experiment is enabled."""
    if not _thread_unsafe_stream_experiment_enabled():
        return
    sys.stderr.write(
        "[mlx-engine-stream-probe] "
        f"reason={reason} {describe_stream_configuration(use_default_stream)}\n"
    )
    sys.stderr.flush()


def _get_or_create_thread_unsafe_stream(default_device: Any) -> Any:
    """Return a shared thread-unsafe stream for the given device."""
    device_key = repr(default_device)
    with _global_stream_lock:
        generation_stream = _global_thread_unsafe_streams.get(device_key)
        if generation_stream is None:
            generation_stream = mx.new_thread_unsafe_stream(default_device)
            _global_thread_unsafe_streams[device_key] = generation_stream
        return generation_stream


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
    stream_source = _resolve_stream_source(use_default_stream)
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
    elif stream_source == "thread-unsafe":
        generation_stream = _get_or_create_thread_unsafe_stream(default_device)
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
