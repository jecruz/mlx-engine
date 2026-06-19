import unittest

import mlx.core as mx

from mlx_engine.utils.prompt_processing import process_prompt_text_only
from mlx_engine.utils.specprefill import SpecPrefillOptions
from tests.shared import RecordingReporter


class FakeCacheWrapper:
    """Test double that records draft-cache and update-cache calls."""

    def __init__(self):
        self.cache = []
        self.set_calls = []
        self.unset_calls = 0
        self.update_calls = []

    def set_draft_model(self, draft_model):
        """Record that the draft cache path was enabled."""
        self.set_calls.append(draft_model)

    def unset_draft_model(self):
        """Record that the draft cache path was disabled."""
        self.unset_calls += 1

    def update_cache(self, prompt_tokens, reporter, **kwargs):
        """Record update kwargs and return the input prompt tokens."""
        self.update_calls.append(
            {
                "prompt_tokens": prompt_tokens,
                "reporter": reporter,
                **kwargs,
            }
        )
        return prompt_tokens


class TestPromptProcessingSpecPrefill(unittest.TestCase):
    """Tests for SpecPrefill prompt-processing draft-model routing."""

    def test_specprefill_uses_draft_for_scoring_without_enabling_draft_cache(self):
        cache_wrapper = FakeCacheWrapper()
        draft_model = object()
        options = SpecPrefillOptions(enabled=True, threshold=1)
        generate_args = {}
        prompt_tokens = mx.array([1, 2, 3], dtype=mx.int32)

        result = process_prompt_text_only(
            prompt_tokens,
            cache_wrapper,
            generate_args,
            draft_model,
            speculative_decoding_toggle=None,
            prompt_progress_reporter=RecordingReporter(),
            specprefill_options=options,
        )

        self.assertEqual(result.tolist(), [1, 2, 3])
        self.assertEqual(cache_wrapper.set_calls, [])
        self.assertEqual(cache_wrapper.unset_calls, 1)
        self.assertIs(cache_wrapper.update_calls[0]["draft_model"], draft_model)
        self.assertIs(cache_wrapper.update_calls[0]["specprefill_options"], options)
        self.assertEqual(generate_args["prompt_cache"], cache_wrapper.cache)

    def test_non_specprefill_keeps_existing_automatic_draft_cache_behavior(self):
        cache_wrapper = FakeCacheWrapper()
        draft_model = object()
        generate_args = {}

        process_prompt_text_only(
            mx.array([1, 2, 3], dtype=mx.int32),
            cache_wrapper,
            generate_args,
            draft_model,
            speculative_decoding_toggle=None,
            prompt_progress_reporter=RecordingReporter(),
        )

        self.assertEqual(cache_wrapper.set_calls, [draft_model])
        self.assertEqual(cache_wrapper.unset_calls, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
