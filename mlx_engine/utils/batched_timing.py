"""Opt-in timing diagnostics for batched inference backends."""

from __future__ import annotations

import json
import logging
import os
import time

BATCHED_TIMING_ENV = "MLX_ENGINE_BATCHED_TIMING"
BATCHED_TIMING_LOG_PREFIX = "MLX_ENGINE_BATCHED_TIMING"


def batched_timing_enabled() -> bool:
    """Return whether batched inference timing logs are enabled."""
    value = os.environ.get(BATCHED_TIMING_ENV, "")
    return value.lower() in {"1", "true", "yes", "on"}


def elapsed_ms(start: float | None, end: float | None = None) -> float:
    """Return elapsed milliseconds from a perf-counter start time."""
    if start is None:
        return 0.0
    if end is None:
        end = time.perf_counter()
    return round((end - start) * 1000.0, 3)


def log_batched_timing(logger: logging.Logger, event: str, **fields) -> None:
    """Log a structured timing event when batched timing is enabled."""
    if not batched_timing_enabled():
        return
    payload = {"event": event, **fields}
    logger.warning(
        "%s %s", BATCHED_TIMING_LOG_PREFIX, json.dumps(payload, sort_keys=True)
    )
