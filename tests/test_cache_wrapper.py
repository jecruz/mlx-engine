from contextlib import nullcontext
import unittest
from unittest.mock import patch

import mlx.core as mx
from mlx_engine.cache_wrapper import (
    CacheWrapper,
    DEFAULT_CHECKPOINT_TAIL_TOKENS,
    DEFERRED_CLEAR_DELAY_STEPS,
    StopPromptProcessing,
)
from mlx_engine.utils.specprefill import SpecPrefillOptions, SpecPrefillResult
from tests.shared import CancellingReporter, RecordingReporter


class FakeCache:
    def __init__(self, offset: int = 0, trimmable: bool = True):
        self.offset = offset
        self._trimmable = trimmable

    @property
    def state(self):
        return []

    def is_trimmable(self):
        return self._trimmable

    def trim(self, n):
        if not self._trimmable:
            return 0
        n = min(self.offset, n)
        self.offset -= n
        return n

    @property
    def nbytes(self):
        return max(self.offset, 1)

    def advance(self, n):
        self.offset += n


class FakeModel:
    def __init__(self, *, cache_trimmable: bool = True):
        self.layers = [object()]
        self.calls = []
        self.cache_trimmable = cache_trimmable

    def make_cache(self):
        return [FakeCache(trimmable=self.cache_trimmable)]

    def __call__(self, tokens, cache):
        n_tokens = tokens.shape[1]
        self.calls.append(n_tokens)
        for entry in cache:
            entry.offset += n_tokens


class CloneableCache:
    def __init__(self, values):
        self.values = list(values)
        self.offset = len(self.values)

    @property
    def state(self):
        return self.values

    @state.setter
    def state(self, value):
        self.values = list(value)
        self.offset = len(self.values)

    @property
    def meta_state(self):
        return (str(self.offset),)

    @meta_state.setter
    def meta_state(self, value):
        self.offset = int(value[0])

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        self.values = self.values[: self.offset]
        return n

    @property
    def nbytes(self):
        return len(self.values)

    @classmethod
    def from_state(cls, state, meta_state):
        clone = cls.__new__(cls)
        clone.state = state
        clone.meta_state = meta_state
        return clone


def _no_op_stream(_stream):
    return nullcontext()


class TestCacheWrapper(unittest.TestCase):
    def _make_session(self, *, cache_trimmable=True, chunk_size=8, **kwargs):
        model = FakeModel(cache_trimmable=cache_trimmable)
        session = CacheWrapper(
            model=model,
            max_kv_size=None,
            chunk_size=chunk_size,
            **kwargs,
        )
        return session, model

    def _run_update_cache(
        self,
        session,
        prompt_tokens,
        reporter=None,
        **kwargs,
    ):
        reporter = reporter or RecordingReporter()
        with (
            patch("mlx_engine.cache_wrapper.mx.stream", side_effect=_no_op_stream),
            patch("mlx_engine.cache_wrapper.mx.eval"),
            patch("mlx_engine.cache_wrapper.mx.clear_cache"),
        ):
            result_tokens = session.update_cache(
                prompt_tokens=prompt_tokens,
                reporter=reporter,
                **kwargs,
            )
        return result_tokens, reporter

    def test_prompt_processing_cancellation(self):
        """Test that progress is saved when processing is cancelled and cache is reused on retry"""

        chunk_size = 20  # Small chunk size to ensure multiple progress callbacks
        model = FakeModel()
        cache_wrapper = CacheWrapper(
            model,
            max_kv_size=4096,
            chunk_size=chunk_size,
        )

        prompt_tokens = mx.array(list(range(1, 101)), dtype=mx.int32)

        # First attempt: Reporter that cancels after 3 events
        cancelling_reporter = CancellingReporter(cancel_after=3)

        with self.assertRaises(StopPromptProcessing):
            cache_wrapper.update_cache(
                prompt_tokens=prompt_tokens,
                reporter=cancelling_reporter,
            )

        # Second attempt: Reporter that doesn't cancel
        recording_reporter = RecordingReporter()

        result_tokens = cache_wrapper.update_cache(
            prompt_tokens=prompt_tokens,
            reporter=recording_reporter,
        )
        cached_before_cancel = cancelling_reporter.events[-1][
            "prefill_tokens_processed"
        ]
        retry_begin_event = recording_reporter.events[0]
        self.assertEqual(retry_begin_event["type"], "begin")
        self.assertEqual(retry_begin_event["cached_tokens"], cached_before_cancel)
        self.assertEqual(recording_reporter.events[-1]["type"], "finish")

        # Verify that the second attempt completed successfully
        self.assertIsNotNone(result_tokens)
        self.assertEqual(result_tokens.tolist(), prompt_tokens[-1:].tolist())

    def test_full_snapshot_reuse_requires_a_longer_prompt_without_checkpoint(self):
        session, _ = self._make_session(
            cache_trimmable=False,
            checkpoint_tail_tokens=100,
        )
        prompt = mx.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=mx.int32)

        self._run_update_cache(session, prompt)
        session.cache[0].advance(1)

        result_tokens, reporter = self._run_update_cache(
            session,
            mx.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=mx.int32),
        )
        self.assertEqual(reporter.events[0]["cached_tokens"], 8)
        self.assertEqual(result_tokens.tolist(), [9])

        _, reporter = self._run_update_cache(
            session,
            mx.array([1, 2, 3, 4], dtype=mx.int32),
        )
        self.assertEqual(reporter.events[0]["cached_tokens"], 0)

    def test_same_prompt_reuses_checkpoint_when_full_snapshot_cannot_trim(self):
        session, _ = self._make_session(cache_trimmable=False)
        prompt = mx.array(list(range(1, 16)), dtype=mx.int32)

        self._run_update_cache(session, prompt)
        session.cache[0].advance(1)

        result_tokens, reporter = self._run_update_cache(session, prompt)

        self.assertEqual(
            reporter.events[0]["cached_tokens"],
            len(prompt) - DEFAULT_CHECKPOINT_TAIL_TOKENS,
        )
        self.assertEqual(result_tokens.tolist(), prompt[-1:].tolist())

    def test_empty_prompt_update_returns_an_empty_tail(self):
        session, _ = self._make_session()

        result_tokens, reporter = self._run_update_cache(
            session,
            mx.array([], dtype=mx.int32),
        )

        self.assertEqual(reporter.events[0]["cached_tokens"], 0)
        self.assertEqual(reporter.events[0]["total_prompt_tokens"], 0)
        self.assertEqual(reporter.events[-1]["type"], "finish")
        self.assertEqual(result_tokens.tolist(), [])

    def test_same_prompt_trims_an_exact_hit_to_leave_one_seed_token(self):
        session, _ = self._make_session(
            cache_trimmable=True,
            checkpoint_tail_tokens=100,
        )
        prompt = mx.array([1, 2, 3, 4, 5, 6], dtype=mx.int32)

        self._run_update_cache(session, prompt)
        session.cache[0].advance(1)

        result_tokens, reporter = self._run_update_cache(session, prompt)

        self.assertEqual(reporter.events[0]["cached_tokens"], 5)
        self.assertEqual(result_tokens.tolist(), [6])

    def test_update_cache_splits_prefill_at_the_checkpoint_boundary(self):
        chunk_size = 8
        session, model = self._make_session(
            cache_trimmable=False,
            chunk_size=chunk_size,
        )
        prompt = mx.array(list(range(1, 16)), dtype=mx.int32)

        result_tokens, _ = self._run_update_cache(session, prompt)

        checkpoint_prefix_len = len(prompt) - DEFAULT_CHECKPOINT_TAIL_TOKENS
        prefillable_tokens = len(prompt) - 1
        expected_calls = [
            checkpoint_prefix_len,
            min(chunk_size, prefillable_tokens - checkpoint_prefix_len),
            prefillable_tokens - checkpoint_prefix_len - min(
                chunk_size, prefillable_tokens - checkpoint_prefix_len
            ),
        ]
        expected_calls = [count for count in expected_calls if count > 0]

        self.assertEqual(model.calls, expected_calls)
        self.assertEqual(result_tokens.tolist(), prompt[-1:].tolist())

    def test_live_tokens_are_stored_as_plain_python_tokens(self):
        session, _ = self._make_session(
            cache_trimmable=False,
            checkpoint_tail_tokens=100,
        )
        prompt = mx.array([1, 2, 3], dtype=mx.int32)

        self._run_update_cache(session, prompt)
        session.record_generated_token(4)
        session.cache[0].advance(1)

        result_tokens, reporter = self._run_update_cache(
            session,
            mx.array([1, 2, 3, 4, 5], dtype=mx.int32),
        )

        self.assertEqual(session._live_tokens, [1, 2, 3, 4, 5])
        self.assertEqual(reporter.events[0]["cached_tokens"], 4)
        self.assertEqual(result_tokens.tolist(), [5])

    def test_deferred_cache_clear_waits_for_generation_steps(self):
        session, _ = self._make_session(
            cache_trimmable=False,
            checkpoint_tail_tokens=100,
        )
        prompt = mx.array([1, 2, 3], dtype=mx.int32)

        with (
            patch("mlx_engine.cache_wrapper.mx.stream", side_effect=_no_op_stream),
            patch("mlx_engine.cache_wrapper.mx.eval"),
            patch.object(session, "_clear_cache_now") as clear_cache_now,
        ):
            session.update_cache(prompt_tokens=prompt, reporter=RecordingReporter())
            self.assertEqual(clear_cache_now.call_count, 0)

            for _ in range(DEFERRED_CLEAR_DELAY_STEPS - 1):
                session.record_generated_token(99)
            self.assertEqual(clear_cache_now.call_count, 0)

            session.record_generated_token(100)
            self.assertEqual(clear_cache_now.call_count, 1)

    def test_store_snapshot_clones_cache_entries(self):
        model = FakeModel()
        session = CacheWrapper(model, max_kv_size=None, chunk_size=1)
        live_cache = [CloneableCache([1, 2, 3])]

        session._store_snapshot([1, 2, 3], live_cache, cache_type="user")
        live_cache[0].values[0] = 99
        live_cache[0].offset = 1

        stored_cache, rest = session._history.fetch_nearest_cache(
            session._history_key,
            [1, 2, 3],
        )

        self.assertEqual(rest, [])
        self.assertIsNotNone(stored_cache)
        self.assertEqual(stored_cache[0].values, [1, 2, 3])
        self.assertEqual(stored_cache[0].offset, 3)

    def test_user_checkpoint_survives_assistant_snapshot_eviction_pressure(self):
        session, _ = self._make_session(
            cache_trimmable=False,
            history_capacity=3,
        )
        prompt = mx.array(list(range(1, 16)), dtype=mx.int32)

        # First request stores a reusable user checkpoint four tokens deep.
        self._run_update_cache(session, prompt)
        # These short follow-ups only flush assistant snapshots, creating eviction pressure.
        self._run_update_cache(session, mx.array([101, 102, 103], dtype=mx.int32))
        self._run_update_cache(session, mx.array([201, 202, 203], dtype=mx.int32))

        _, reporter = self._run_update_cache(session, prompt)

        self.assertEqual(
            reporter.events[0]["cached_tokens"],
            len(prompt) - DEFAULT_CHECKPOINT_TAIL_TOKENS,
        )

    def test_setting_a_draft_model_resets_cached_history(self):
        session, _ = self._make_session(
            cache_trimmable=True,
            checkpoint_tail_tokens=100,
        )
        prompt = mx.array([1, 2, 3], dtype=mx.int32)

        self._run_update_cache(session, prompt)
        session.cache[0].advance(1)
        session.set_draft_model(FakeModel())

        _, reporter = self._run_update_cache(session, prompt)

        self.assertEqual(reporter.events[0]["cached_tokens"], 0)

    def test_unsetting_draft_model_preserves_live_main_cache(self):
        session, _ = self._make_session(
            cache_trimmable=True,
            checkpoint_tail_tokens=100,
        )
        prompt = mx.array([1, 2, 3, 4], dtype=mx.int32)

        session.set_draft_model(FakeModel())
        self._run_update_cache(session, prompt)

        session.unset_draft_model()

        result_tokens, reporter = self._run_update_cache(session, prompt)

        self.assertEqual(reporter.events[0]["cached_tokens"], 3)
        self.assertEqual(len(session.cache), 1)
        self.assertEqual(result_tokens.tolist(), [4])

    def test_quantized_mode_reuses_full_snapshots_and_skips_same_prompt_checkpoint_reuse(
        self,
    ):
        model = FakeModel(cache_trimmable=False)
        session = CacheWrapper(
            model=model,
            max_kv_size=None,
            kv_bits=8,
            kv_group_size=64,
            quantized_kv_start=0,
            chunk_size=8,
        )
        prompt = mx.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=mx.int32)
        replacement = FakeCache(offset=0, trimmable=False)

        def quantize_cache(prompt_cache, **_):
            replacement.offset = prompt_cache[0].offset
            prompt_cache[0] = replacement

        with patch(
            "mlx_engine.cache_wrapper.maybe_quantize_kv_cache",
            side_effect=quantize_cache,
        ):
            result_tokens, _ = self._run_update_cache(session, prompt)
        self.assertEqual(model.calls, [7])
        self.assertIs(session.cache[0], replacement)
        self.assertEqual(result_tokens.tolist(), [8])

        session.cache[0].advance(1)
        with patch(
            "mlx_engine.cache_wrapper.maybe_quantize_kv_cache",
            side_effect=quantize_cache,
        ):
            _, reporter = self._run_update_cache(
                session,
                mx.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=mx.int32),
            )
        self.assertEqual(reporter.events[0]["cached_tokens"], 8)

        session, _ = self._make_session(
            cache_trimmable=False,
            kv_bits=8,
            kv_group_size=64,
            quantized_kv_start=0,
        )
        replacement = FakeCache(offset=0, trimmable=False)

        def quantize_same_prompt_cache(prompt_cache, **_):
            replacement.offset = prompt_cache[0].offset
            prompt_cache[0] = replacement

        with patch(
            "mlx_engine.cache_wrapper.maybe_quantize_kv_cache",
            side_effect=quantize_same_prompt_cache,
        ):
            self._run_update_cache(session, prompt)
        session.cache[0].advance(1)

        with patch(
            "mlx_engine.cache_wrapper.maybe_quantize_kv_cache",
            side_effect=quantize_same_prompt_cache,
        ):
            _, reporter = self._run_update_cache(session, prompt)
        self.assertEqual(reporter.events[0]["cached_tokens"], 0)

    def test_specprefill_skips_when_below_threshold(self):
        session, model = self._make_session(chunk_size=8)
        prompt = mx.array([1, 2, 3, 4, 5], dtype=mx.int32)
        options = SpecPrefillOptions(enabled=True, threshold=len(prompt))

        with patch("mlx_engine.cache_wrapper.try_sparse_prefill") as sparse_prefill:
            result_tokens, _ = self._run_update_cache(
                session,
                prompt,
                draft_model=FakeModel(),
                specprefill_options=options,
            )

        sparse_prefill.assert_not_called()
        self.assertEqual(model.calls, [4])
        self.assertEqual(result_tokens.tolist(), [5])

    def test_specprefill_failure_falls_back_to_full_prefill(self):
        session, model = self._make_session(chunk_size=8)
        prompt = mx.array(list(range(1, 9)), dtype=mx.int32)
        options = SpecPrefillOptions(enabled=True, threshold=1)

        with patch(
            "mlx_engine.cache_wrapper.try_sparse_prefill",
            side_effect=RuntimeError("sparse failed"),
        ) as sparse_prefill:
            result_tokens, _ = self._run_update_cache(
                session,
                prompt,
                draft_model=FakeModel(),
                specprefill_options=options,
            )

        sparse_prefill.assert_called_once()
        self.assertEqual(model.calls, [7])
        self.assertEqual(result_tokens.tolist(), [8])

    def test_specprefill_success_returns_seed_token_and_marks_sparse_cache(self):
        session, model = self._make_session(chunk_size=8)
        prompt = mx.array(list(range(1, 9)), dtype=mx.int32)
        sparse_cache = [FakeCache(offset=len(prompt), trimmable=True)]
        result = SpecPrefillResult(
            cache=sparse_cache,
            seed_tokens=mx.array([8], dtype=mx.int32),
            live_tokens=prompt.tolist(),
        )
        options = SpecPrefillOptions(enabled=True, threshold=1)

        with patch(
            "mlx_engine.cache_wrapper.try_sparse_prefill",
            return_value=result,
        ) as sparse_prefill:
            result_tokens, reporter = self._run_update_cache(
                session,
                prompt,
                draft_model=FakeModel(),
                specprefill_options=options,
            )

        sparse_prefill.assert_called_once()
        self.assertEqual(model.calls, [])
        self.assertEqual(session.cache, sparse_cache)
        self.assertTrue(session._sparse_cache_active)
        self.assertEqual(reporter.events[-1]["type"], "finish")
        self.assertEqual(result_tokens.tolist(), [8])

    def test_specprefill_sparse_cache_is_not_stored_in_prefix_history(self):
        session, _ = self._make_session(chunk_size=8)
        prompt = mx.array(list(range(1, 9)), dtype=mx.int32)
        sparse_cache = [FakeCache(offset=len(prompt), trimmable=True)]
        options = SpecPrefillOptions(enabled=True, threshold=1)

        with patch(
            "mlx_engine.cache_wrapper.try_sparse_prefill",
            return_value=SpecPrefillResult(
                cache=sparse_cache,
                seed_tokens=mx.array([8], dtype=mx.int32),
                live_tokens=prompt.tolist(),
            ),
        ):
            self._run_update_cache(
                session,
                prompt,
                draft_model=FakeModel(),
                specprefill_options=options,
            )

        session.record_generated_token(99)
        _, reporter = self._run_update_cache(session, prompt)

        self.assertFalse(session._sparse_cache_active)
        self.assertEqual(reporter.events[0]["cached_tokens"], 0)

    def test_specprefill_skips_when_system_tokens_cover_uncached_prompt(self):
        session, model = self._make_session(chunk_size=8)
        prompt = mx.array(list(range(1, 9)), dtype=mx.int32)
        options = SpecPrefillOptions(
            enabled=True,
            threshold=1,
            system_tokens=len(prompt),
        )

        with patch("mlx_engine.cache_wrapper.try_sparse_prefill") as sparse_prefill:
            result_tokens, _ = self._run_update_cache(
                session,
                prompt,
                draft_model=FakeModel(),
                specprefill_options=options,
            )

        sparse_prefill.assert_not_called()
        self.assertEqual(model.calls, [7])
        self.assertEqual(result_tokens.tolist(), [8])


if __name__ == "__main__":
    unittest.main(verbosity=2)
