from types import SimpleNamespace

from mlx_engine.utils.request_state import (
    model_kit_has_active_requests,
    request_id_is_empty,
)


class _IdleKit:
    def __init__(self):
        self.pending_requests = {}
        self._generation_thread_state = SimpleNamespace(
            active={},
            pending=[],
            ready=[],
            restoring={},
        )
        self._batch_results = {}


class _BusySequentialKit(_IdleKit):
    def __init__(self):
        super().__init__()
        self.pending_requests = {"request-1": object()}


class _BusyBatchedVisionKit(_IdleKit):
    def __init__(self):
        super().__init__()
        self._generation_thread_state = SimpleNamespace(
            active={1: object()},
            pending=[],
            ready=[],
            restoring={},
        )


class _BusyBatchedKit(_IdleKit):
    def __init__(self):
        super().__init__()
        self._batch_results = {1: object()}


def test_model_kit_has_active_requests_returns_false_when_idle():
    assert not model_kit_has_active_requests(_IdleKit())


def test_model_kit_has_active_requests_detects_sequential_pending_request():
    assert model_kit_has_active_requests(_BusySequentialKit())


def test_model_kit_has_active_requests_detects_batched_vision_state():
    assert model_kit_has_active_requests(_BusyBatchedVisionKit())


def test_model_kit_has_active_requests_detects_batched_results():
    assert model_kit_has_active_requests(_BusyBatchedKit())


def test_model_kit_has_active_requests_returns_false_for_shutdown_only_state():
    kit = _IdleKit()
    kit._generation_thread_state = None
    kit.pending_requests = {}
    kit._batch_results = {}

    assert not model_kit_has_active_requests(kit)


def test_request_id_is_empty_detects_blank_and_missing_values():
    assert request_id_is_empty(None)
    assert request_id_is_empty("")
    assert not request_id_is_empty("request-1")
