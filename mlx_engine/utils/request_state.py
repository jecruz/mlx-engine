from __future__ import annotations

from typing import Any


def request_id_is_empty(request_id: Any) -> bool:
    """Return True when a request id is missing or blank."""
    return request_id is None or request_id == ""


def model_kit_has_active_requests(model_kit: Any) -> bool:
    """Return True when a model kit still has requests that must not be unloaded."""
    active_pending_requests = getattr(model_kit, "pending_requests", None)
    if isinstance(active_pending_requests, dict) and len(active_pending_requests) > 0:
        return True

    generation_state = getattr(model_kit, "_generation_thread_state", None)
    if generation_state is not None and any(
        (
            getattr(generation_state, "active", None),
            getattr(generation_state, "pending", None),
            getattr(generation_state, "ready", None),
            getattr(generation_state, "restoring", None),
        )
    ):
        return True

    batch_results = getattr(model_kit, "_batch_results", None)
    if isinstance(batch_results, dict) and len(batch_results) > 0:
        return True

    return False
