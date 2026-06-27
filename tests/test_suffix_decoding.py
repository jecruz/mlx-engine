from types import SimpleNamespace
import os
import threading
import unittest
from unittest.mock import patch

import mlx.core as mx

from mlx_engine.generate import _sequential_generation, create_generator
from mlx_engine.utils.suffix_decoding import (
    SuffixDecodingProposal,
    propose_suffix_decoding_tokens,
)
from mlx_engine.utils.suffix_decoding_runtime import (
    resolve_suffix_decoding_options,
    suffix_stream_generate,
)


class FakeDetokenizer:
    def __init__(self):
        self.text = ""
        self.offset = 0

    def add_token(self, token):
        self.text += str(token)

    def finalize(self):
        return None

    @property
    def last_segment(self):
        segment = self.text[self.offset :]
        self.offset = len(self.text)
        return segment


class FakeTokenizer:
    def __init__(self):
        self.bos_token = None
        self.chat_template = None
        self.clean_up_tokenization_spaces = False
        self.eos_token_id = 999
        self.eos_token_ids = [self.eos_token_id]
        self.detokenizer = FakeDetokenizer()

    def get_vocab(self):
        return {}

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, tokens):
        if isinstance(tokens, int):
            tokens = [tokens]
        return "".join(str(token) for token in tokens)


class FakeCache:
    def __init__(self):
        self.trim_calls: list[int] = []

    @property
    def state(self):
        return []

    def is_trimmable(self):
        return True

    def trim(self, n):
        self.trim_calls.append(n)
        return n

    @property
    def nbytes(self):
        return 1


class FakeSuffixModel:
    def __init__(self, outputs_by_call, vocab_size=32):
        self.outputs_by_call = [list(output) for output in outputs_by_call]
        self.calls: list[list[int]] = []
        self.vocab_size = vocab_size
        self.layers = []

    def __call__(self, input_tokens, cache=None):
        tokens = input_tokens.tolist()[0]
        self.calls.append(tokens)
        outputs = self.outputs_by_call.pop(0)
        if len(outputs) != len(tokens):
            raise AssertionError(
                f"Expected {len(tokens)} outputs for call {len(self.calls)}, got {len(outputs)}"
            )
        logits = mx.full((1, len(outputs), self.vocab_size), -1e9)
        for index, token in enumerate(outputs):
            logits[0, index, token] = 0
        return logits


class FakeSequentialKit:
    def __init__(self):
        self.model = object()
        self.tokenizer = FakeTokenizer()
        self.draft_model = None
        self.prefill_step_size = 8
        self.pending_requests = {}
        self.generation_lock = threading.Lock()
        self.max_kv_size = None
        self.kv_bits = None
        self.kv_group_size = None
        self.quantized_kv_start = None

    def is_shutdown(self):
        return False

    def process_prompt(self, prompt_tokens, *_args, **_kwargs):
        return mx.array(prompt_tokens, dtype=mx.int32), None

    def is_cross_prompt_cache_active(self):
        return False

    def record_token_to_cache(self, _token):
        return None

    def cleanup_specprefill(self):
        return None


class FakeBatchedKit:
    pass


class FakeDistributedBatchKit:
    def uses_distributed_batching(self):
        return True


class TestSuffixDecodingOptions(unittest.TestCase):
    def test_defaults_off_without_env(self):
        with patch.dict(os.environ, {}, clear=True):
            options = resolve_suffix_decoding_options(None, None)

        self.assertFalse(options.enabled)

    def test_env_opt_in_enables_suffix_decoding(self):
        with patch.dict(
            os.environ,
            {
                "MLX_ENGINE_SUFFIX_DECODING": "1",
                "MLX_ENGINE_SUFFIX_DECODING_MAX_DRAFT_TOKENS": "3",
            },
            clear=True,
        ):
            options = resolve_suffix_decoding_options(None, None)

        self.assertTrue(options.enabled)
        self.assertEqual(options.max_draft_tokens, 3)


class TestSuffixDecodingHelper(unittest.TestCase):
    def test_returns_none_when_no_reusable_suffix_exists(self):
        proposal = propose_suffix_decoding_tokens(
            [1, 2, 3, 4],
            max_draft_tokens=4,
        )

        self.assertIsNone(proposal)

    def test_prefers_the_longest_matching_suffix_ngram(self):
        proposal = propose_suffix_decoding_tokens(
            [1, 2, 3, 4, 5, 6, 1, 2, 3, 7, 8, 1, 2, 3],
            max_draft_tokens=3,
        )

        self.assertEqual(
            proposal,
            SuffixDecodingProposal(
                source_start_index=6,
                matched_suffix_length=3,
                draft_tokens=(7, 8, 1),
            ),
        )

    def test_caps_proposed_draft_tokens(self):
        proposal = propose_suffix_decoding_tokens(
            [1, 2, 3, 4, 5, 6, 1, 2, 3, 7, 8, 1, 2, 3],
            max_draft_tokens=2,
        )

        self.assertEqual(
            proposal,
            SuffixDecodingProposal(
                source_start_index=6,
                matched_suffix_length=3,
                draft_tokens=(7, 8),
            ),
        )

    def test_uses_non_overlapping_source_windows(self):
        proposal = propose_suffix_decoding_tokens(
            [4, 5, 6, 7, 8, 4, 5, 6, 9, 10, 4, 5, 6],
            max_draft_tokens=2,
        )

        self.assertEqual(
            proposal,
            SuffixDecodingProposal(
                source_start_index=5,
                matched_suffix_length=3,
                draft_tokens=(9, 10),
            ),
        )
        self.assertLessEqual(
            proposal.source_start_index + proposal.matched_suffix_length,
            len([4, 5, 6, 7, 8, 4, 5, 6, 9, 10, 4, 5, 6])
            - proposal.matched_suffix_length,
        )

    def test_truncates_before_stop_and_eos_boundaries(self):
        scenarios = [
            (
                "stop",
                {"stop_token_ids": [99]},
                [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 99, 1, 2, 3],
            ),
            (
                "eos",
                {"eos_token_ids": [98]},
                [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 98, 1, 2, 3],
            ),
        ]

        for label, boundary_kwargs, history in scenarios:
            with self.subTest(boundary=label):
                proposal = propose_suffix_decoding_tokens(
                    history,
                    max_draft_tokens=4,
                    **boundary_kwargs,
                )

                self.assertEqual(
                    proposal,
                    SuffixDecodingProposal(
                        source_start_index=6,
                        matched_suffix_length=3,
                        draft_tokens=(4, 5),
                    ),
                )


class TestSuffixDecodingRouting(unittest.TestCase):
    def test_default_off_uses_existing_stream_generate_path(self):
        kit = FakeSequentialKit()
        response = SimpleNamespace(
            text="ok",
            token=7,
            logprobs=mx.zeros((8,)),
            from_draft=False,
            finish_reason="length",
        )

        def fake_stream_generate(**_kwargs):
            yield response

        with (
            patch("mlx_engine.generate.stream_generate", side_effect=fake_stream_generate) as stream_generate,
            patch(
                "mlx_engine.generate.suffix_stream_generate",
                side_effect=AssertionError("suffix path should stay disabled"),
            ),
        ):
            results = list(
                _sequential_generation(
                    kit,
                    [1],
                    max_tokens=1,
                    request_id="suffix-default-off",
                )
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].tokens[0].id, 7)
        stream_generate.assert_called_once()

    def test_rejects_loaded_draft_model_and_specprefill(self):
        kit = FakeSequentialKit()
        kit.draft_model = object()

        with self.assertRaisesRegex(ValueError, "SuffixDecoding cannot be combined"):
            list(
                _sequential_generation(
                    kit,
                    [1],
                    max_tokens=1,
                    request_id="suffix-draft-model",
                    suffix_decoding_toggle=True,
                )
            )

        kit = FakeSequentialKit()
        with self.assertRaisesRegex(ValueError, "SuffixDecoding cannot be combined"):
            list(
                _sequential_generation(
                    kit,
                    [1],
                    max_tokens=1,
                    request_id="suffix-specprefill",
                    suffix_decoding_toggle=True,
                    specprefill_toggle=True,
                )
            )

        kit = FakeSequentialKit()
        with self.assertRaisesRegex(ValueError, "SuffixDecoding cannot be combined"):
            list(
                _sequential_generation(
                    kit,
                    [1],
                    max_tokens=1,
                    request_id="suffix-num-draft-tokens",
                    suffix_decoding_toggle=True,
                    num_draft_tokens=2,
                )
            )

    def test_rejects_batched_distributed_and_vlm_surfaces(self):
        with patch("mlx_engine.generate.BatchedModelKit", FakeBatchedKit):
            with self.assertRaisesRegex(ValueError, "sequential text generation"):
                create_generator(
                    FakeBatchedKit(),
                    [1],
                    suffix_decoding_toggle=True,
                )

        with patch("mlx_engine.generate.DistributedModelKit", FakeDistributedBatchKit):
            with self.assertRaisesRegex(ValueError, "sequential text generation"):
                create_generator(
                    FakeDistributedBatchKit(),
                    [1],
                    suffix_decoding_toggle=True,
                )

        fake_vision_kit = type("FakeVisionKit", (), {})
        with patch("mlx_engine.generate._load_batched_vision_model_kit", return_value=fake_vision_kit):
            with self.assertRaisesRegex(ValueError, "sequential text generation"):
                create_generator(
                    fake_vision_kit(),
                    [1],
                    suffix_decoding_toggle=True,
                )

    def test_suffix_path_does_not_forward_input_embeddings(self):
        kit = FakeSequentialKit()
        response = SimpleNamespace(
            text="ok",
            token=7,
            logprobs=mx.zeros((8,)),
            from_draft=False,
            finish_reason="length",
        )
        captured = {}

        def fake_suffix_stream_generate(**kwargs):
            captured.update(kwargs)
            yield response

        with (
            patch(
                "mlx_engine.generate.suffix_stream_generate",
                side_effect=fake_suffix_stream_generate,
            ),
            patch(
                "mlx_engine.generate.stream_generate",
                side_effect=AssertionError("default stream path should stay disabled"),
            ),
        ):
            results = list(
                _sequential_generation(
                    kit,
                    [1],
                    max_tokens=1,
                    request_id="suffix-compat",
                    suffix_decoding_toggle=True,
                )
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].tokens[0].id, 7)
        self.assertNotIn("input_embeddings", captured)


class TestSuffixDecodingVerification(unittest.TestCase):
    def test_suffix_proposal_tokens_are_target_verified_before_emission(self):
        model = FakeSuffixModel(outputs_by_call=[[7], [8, 9]])
        cache = [FakeCache()]
        tokenizer = FakeTokenizer()

        responses = list(
            suffix_stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=[1],
                prompt_cache=cache,
                max_tokens=2,
                proposal_fn=lambda _history, **_kwargs: SuffixDecodingProposal(
                    source_start_index=0,
                    matched_suffix_length=1,
                    draft_tokens=(8,),
                ),
            )
        )

        self.assertEqual([response.token for response in responses], [7, 8])
        self.assertEqual([response.from_draft for response in responses], [False, True])
        self.assertEqual(model.calls, [[1], [7, 8]])
        self.assertEqual(cache[0].trim_calls, [])

    def test_suffix_proposal_receives_max_draft_tokens(self):
        model = FakeSuffixModel(outputs_by_call=[[7], [8, 9]])
        cache = [FakeCache()]
        tokenizer = FakeTokenizer()
        captured = {}

        def fake_proposal(history, *, max_draft_tokens):
            captured["history"] = list(history)
            captured["max_draft_tokens"] = max_draft_tokens
            return SuffixDecodingProposal(
                source_start_index=0,
                matched_suffix_length=1,
                draft_tokens=(8,),
            )

        responses = list(
            suffix_stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=[1],
                prompt_cache=cache,
                max_tokens=2,
                max_draft_tokens=2,
                proposal_fn=fake_proposal,
            )
        )

        self.assertEqual([response.token for response in responses], [7, 8])
        self.assertEqual(captured["max_draft_tokens"], 2)
        self.assertEqual(captured["history"], [1, 7])

    def test_suffix_proposal_falls_back_to_verified_target_token_on_mismatch(self):
        model = FakeSuffixModel(outputs_by_call=[[7], [9, 10]])
        cache = [FakeCache()]
        tokenizer = FakeTokenizer()

        responses = list(
            suffix_stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=[1],
                prompt_cache=cache,
                max_tokens=2,
                proposal_fn=lambda _history, **_kwargs: SuffixDecodingProposal(
                    source_start_index=0,
                    matched_suffix_length=1,
                    draft_tokens=(8,),
                ),
            )
        )

        self.assertEqual([response.token for response in responses], [7, 9])
        self.assertEqual([response.from_draft for response in responses], [False, False])
        self.assertEqual(model.calls, [[1], [7, 8]])
        self.assertEqual(cache[0].trim_calls, [1])

    def test_suffix_proposal_none_or_empty_does_not_duplicate_initial_target_token(self):
        scenarios = [
            ("none", None),
            (
                "empty",
                SuffixDecodingProposal(
                    source_start_index=0,
                    matched_suffix_length=1,
                    draft_tokens=(),
                ),
            ),
        ]

        for label, proposal in scenarios:
            with self.subTest(proposal=label):
                model = FakeSuffixModel(outputs_by_call=[[7], [8]])
                cache = [FakeCache()]
                tokenizer = FakeTokenizer()
                proposal_calls = 0

                def fake_proposal(history, *, max_draft_tokens):
                    nonlocal proposal_calls
                    proposal_calls += 1
                    self.assertEqual(history, [1, 7])
                    self.assertEqual(max_draft_tokens, 2)
                    return proposal

                responses = list(
                    suffix_stream_generate(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=[1],
                        prompt_cache=cache,
                        max_tokens=2,
                        max_draft_tokens=2,
                        proposal_fn=fake_proposal,
                    )
                )

                self.assertEqual(proposal_calls, 1)
                self.assertEqual([response.token for response in responses], [7, 8])
                self.assertEqual([response.from_draft for response in responses], [False, False])
                self.assertEqual(model.calls, [[1], [7]])
                self.assertEqual(cache[0].trim_calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)