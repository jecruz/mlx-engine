from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

import mlx.core as mx

from mlx_engine.generate import _sequential_generation, create_generator
from tests.shared import RecordingReporter


class FakeTokenizer:
    """Tokenizer test double for sequential generation cleanup tests."""

    def __init__(self):
        self.detokenizer = SimpleNamespace(finalize=lambda: None, last_segment="")
        self._tokenizer = object()
        self.eos_token_ids = []

    def decode(self, token):
        """Return a stable decoded token string."""
        return f"tok-{token}"


class FakeSequentialKit:
    """Sequential model-kit test double with SpecPrefill cleanup tracking."""

    def __init__(self):
        self.model = object()
        self.tokenizer = FakeTokenizer()
        self.draft_model = object()
        self.prefill_step_size = 8
        self.pending_requests = {}
        self.generation_lock = threading.Lock()
        self.cleanup_calls = 0
        self.process_prompt_calls = []

    def is_shutdown(self):
        """Report that the fake kit is running."""
        return False

    def process_prompt(self, prompt_tokens, *_args, **_kwargs):
        """Return the prompt as the stream seed."""
        self.process_prompt_calls.append(_kwargs)
        return mx.array(prompt_tokens, dtype=mx.int32), None

    def is_cross_prompt_cache_active(self):
        """Disable generated-token cache recording for this test."""
        return False

    def cleanup_specprefill(self):
        """Record cleanup calls."""
        self.cleanup_calls += 1


class FakeBatchedKit:
    """Batched model-kit marker used to test create_generator routing."""

    pass


class TestGenerateSpecPrefillCleanup(unittest.TestCase):
    """Tests that sequential generation cleans SpecPrefill state."""

    def test_cleanup_runs_after_specprefill_generation_finishes(self):
        kit = FakeSequentialKit()
        stream_result = SimpleNamespace(
            token=1,
            text="x",
            logprobs=mx.zeros((8,)),
            from_draft=False,
            finish_reason="length",
        )

        def fake_stream_generate(**_kwargs):
            yield stream_result

        with patch("mlx_engine.generate.stream_generate", side_effect=fake_stream_generate):
            results = list(
                _sequential_generation(
                    kit,
                    [1, 2, 3],
                    prompt_progress_reporter=RecordingReporter(),
                    max_tokens=1,
                    specprefill_toggle=True,
                    specprefill_threshold=1,
                    request_id="specprefill-cleanup",
                )
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(kit.cleanup_calls, 1)

    def test_below_threshold_specprefill_bypasses_specprefill_processing(self):
        kit = FakeSequentialKit()
        stream_result = SimpleNamespace(
            token=1,
            text="x",
            logprobs=mx.zeros((8,)),
            from_draft=False,
            finish_reason="length",
        )
        stream_calls = []

        def fake_stream_generate(**kwargs):
            stream_calls.append(kwargs)
            yield stream_result

        with patch("mlx_engine.generate.stream_generate", side_effect=fake_stream_generate):
            list(
                _sequential_generation(
                    kit,
                    [1, 2, 3],
                    prompt_progress_reporter=RecordingReporter(),
                    max_tokens=1,
                    specprefill_toggle=True,
                    specprefill_threshold=1024,
                    request_id="specprefill-below-threshold",
                )
            )

        self.assertIsNone(kit.process_prompt_calls[0]["specprefill_options"])
        self.assertIsNone(kit.process_prompt_calls[0]["draft_model_override"])
        self.assertIsNone(stream_calls[0]["draft_model"])

    def test_batched_generation_strips_disabled_specprefill_kwargs(self):
        batched_calls = []

        def fake_batched_generation(model_kit, prompt_tokens, **kwargs):
            batched_calls.append((model_kit, prompt_tokens, kwargs))
            return iter(())

        with (
            patch("mlx_engine.generate.BatchedModelKit", FakeBatchedKit),
            patch(
                "mlx_engine.generate._load_batched_vision_model_kit",
                return_value=type("UnusedVisionKit", (), {}),
            ),
            patch(
                "mlx_engine.generate._batched_generation",
                side_effect=fake_batched_generation,
            ),
        ):
            list(
                create_generator(
                    FakeBatchedKit(),
                    [1, 2, 3],
                    request_id="batched-no-specprefill",
                    specprefill_toggle=None,
                    specprefill_keep_pct=None,
                    specprefill_threshold=None,
                )
            )

        self.assertEqual(len(batched_calls), 1)
        batched_kwargs = batched_calls[0][2]
        self.assertNotIn("specprefill_toggle", batched_kwargs)
        self.assertNotIn("specprefill_keep_pct", batched_kwargs)
        self.assertNotIn("specprefill_threshold", batched_kwargs)

    def test_batched_generation_rejects_enabled_specprefill(self):
        with (
            patch("mlx_engine.generate.BatchedModelKit", FakeBatchedKit),
            patch(
                "mlx_engine.generate._load_batched_vision_model_kit",
                return_value=type("UnusedVisionKit", (), {}),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "SpecPrefill is only supported"):
                create_generator(
                    FakeBatchedKit(),
                    [1, 2, 3],
                    request_id="batched-specprefill",
                    specprefill_toggle=True,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
