from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx

from mlx_engine.utils.dflash_boundary import DFlashBoundaryOptions
from mlx_engine.utils.dflash_runtime import dflash_stream_generate


class FakeTokenizer:
    def __init__(self):
        self.eos_token_ids = [999]

    def decode(self, token):
        if isinstance(token, int):
            return str(token)
        return "".join(str(value) for value in token)


class FakePromptCacheLayer:
    def __init__(self, layer_id: int):
        self.layer_id = layer_id
        self.history: list[int] = []
        self.lengths = mx.array([0], dtype=mx.int32)


class FakeDraftModel:
    def __init__(self, draft_tokens: list[int], layer_count: int = 2):
        self._draft_tokens = tuple(draft_tokens)
        self.config = SimpleNamespace(
            target_layer_ids=list(range(layer_count)),
            block_size=len(draft_tokens) + 1,
        )
        self.reset_calls = []
        self.accept_lens = []
        self.draft_lens = []

    def reset(self, model):
        self.reset_calls.append(model)
        return [SimpleNamespace(lengths=mx.array([0])) for _ in self.config.target_layer_ids]

    def draft_block(self, last_bonus, hidden, cache, block_size, sampler, token_dtype):
        self.draft_lens.append((last_bonus, block_size))
        self.asserted_block_size = block_size
        return mx.array([list(self._draft_tokens)], dtype=token_dtype)


class FakeTargetModel:
    def __init__(
        self,
        prompt_tokens: list[int],
        spec_input_tokens: list[int],
        spec_output_tokens: list[int],
        layer_count: int = 2,
    ):
        self._prompt_tokens = tuple(prompt_tokens)
        self._spec_input_tokens = tuple(spec_input_tokens)
        self._spec_output_tokens = tuple(spec_output_tokens)
        self._layer_count = layer_count
        self.calls = []
        self.rollback_calls = []
        self.call_index = 0
        self.language_model = self

    def __call__(self, tokens, **kwargs):
        seq_tokens = tuple(tokens.reshape(-1).tolist())
        seq_len = len(seq_tokens)
        prompt_cache = kwargs["cache"]
        base_lengths = tuple(len(layer.history) for layer in prompt_cache)
        expected_tokens = self._prompt_tokens if self.call_index == 0 else self._spec_input_tokens
        output_tokens = (
            (self._spec_input_tokens[0],)
            if self.call_index == 0
            else self._spec_output_tokens
        )
        self.call_index += 1
        self.calls.append((seq_tokens, kwargs))
        assert seq_tokens == expected_tokens, (
            f"unexpected verify input {seq_tokens!r}, expected {expected_tokens!r}"
        )
        for layer in prompt_cache:
            layer.history.extend(seq_tokens)
            layer.lengths = mx.array([len(layer.history)], dtype=mx.int32)
        vocab_size = max(max(seq_tokens), max(self._spec_output_tokens), 255) + 8
        logits = mx.full((1, seq_len, vocab_size), -100.0)
        for index, token in enumerate(output_tokens):
            logits[:, index, token] = 100.0
        hidden = [
            mx.full((1, seq_len, 2), float(layer.layer_id + 1))
            for layer in prompt_cache
        ]
        gdn_states = [
            SimpleNamespace(
                layer_id=layer.layer_id,
                base_history_len=base_length,
                verify_tokens=seq_tokens,
            )
            for layer, base_length in zip(prompt_cache, base_lengths)
        ]
        return SimpleNamespace(logits=logits, hidden_states=hidden, gdn_states=gdn_states)

    def rollback_speculative_cache(self, prompt_cache, gdn_states, accepted, block_size):
        self.rollback_calls.append((accepted, block_size, len(gdn_states)))
        for layer, gdn_state in zip(prompt_cache, gdn_states):
            keep = gdn_state.base_history_len + accepted + 1
            layer.history = layer.history[:keep]
            layer.lengths = mx.array([len(layer.history)], dtype=mx.int32)


class FakeKit:
    def __init__(self, draft_tokens: list[int], spec_input_tokens: list[int], spec_output_tokens: list[int]):
        self.model = FakeTargetModel([1], spec_input_tokens, spec_output_tokens)
        self.tokenizer = FakeTokenizer()
        self.cache_wrapper = SimpleNamespace(
            cache=[FakePromptCacheLayer(layer_id) for layer_id in range(2)]
        )
        self.pending_requests = {}
        self.max_kv_size = None
        self.kv_bits = None
        self.kv_group_size = None
        self.quantized_kv_start = None
        self.received_cache_tokens = []
        self._prompt_tokens = [1]
        self._draft_tokens = draft_tokens

    def process_prompt(
        self,
        prompt_tokens,
        images_b64,
        prompt_progress_reporter,
        generate_args,
        max_image_size,
        **_kwargs,
    ):
        generate_args["prompt_cache"] = self.cache_wrapper.cache
        return mx.array(self._prompt_tokens, dtype=mx.int32), None

    def is_cross_prompt_cache_active(self):
        return False

    def record_token_to_cache(self, token):
        self.received_cache_tokens.append(token)


class TestDFlashRuntime(unittest.TestCase):
    def test_rolls_back_partial_rejections_and_preserves_live_history(self):
        cases = (
            {
                "name": "all-accepted",
                "draft_tokens": [12, 13, 14],
                "spec_input_tokens": [11, 12, 13, 14],
                "spec_output_tokens": [12, 13, 14, 99],
                "accepted": 3,
                "rollback": None,
            },
            {
                "name": "first-token-rejection",
                "draft_tokens": [12],
                "spec_input_tokens": [11, 12],
                "spec_output_tokens": [21, 99],
                "accepted": 0,
                "rollback": (0, 2),
            },
            {
                "name": "middle-rejection",
                "draft_tokens": [12, 13, 14],
                "spec_input_tokens": [11, 12, 13, 14],
                "spec_output_tokens": [12, 21, 99, 98],
                "accepted": 1,
                "rollback": (1, 3),
            },
            {
                "name": "tail-rejection",
                "draft_tokens": [12, 13, 14],
                "spec_input_tokens": [11, 12, 13, 14],
                "spec_output_tokens": [12, 13, 21, 99],
                "accepted": 2,
                "rollback": (2, 4),
            },
        )

        for case in cases:
            with self.subTest(case=case["name"]):
                kit = FakeKit(
                    case["draft_tokens"],
                    case["spec_input_tokens"],
                    case["spec_output_tokens"],
                )
                draft_model = FakeDraftModel(case["draft_tokens"])
                observed_proposals = []

                results = list(
                    dflash_stream_generate(
                        kit,
                        [1],
                        request_id=f"dflash-runtime-{case['name']}",
                        max_tokens=case["accepted"] + 2,
                        dflash_options=DFlashBoundaryOptions(
                            enabled=True,
                            target_model_path=Path("/tmp/target"),
                            drafter_model_path=Path("/tmp/drafter"),
                            max_draft_tokens=len(case["draft_tokens"]) + 1,
                        ),
                        dflash_draft_model=draft_model,
                        proposal_observer=lambda history, proposal: observed_proposals.append(
                            (tuple(history), tuple(proposal))
                        ),
                    )
                )

                emitted_ids = [token.id for result in results for token in result.tokens]
                emitted_from_draft = [
                    token.from_draft for result in results for token in result.tokens
                ]
                accepted = case["accepted"]
                draft_tokens = case["draft_tokens"]
                spec_output_tokens = case["spec_output_tokens"]
                expected_block_size = min(len(draft_tokens) + 1, accepted + 2)
                expected_emitted_ids = (
                    [11] + draft_tokens[:accepted] + [spec_output_tokens[accepted]]
                )
                expected_from_draft = (
                    [False] + [True] * accepted + [False]
                )
                expected_history = [1, 11] + draft_tokens[:accepted]

                self.assertEqual(emitted_ids, expected_emitted_ids)
                self.assertEqual(emitted_from_draft, expected_from_draft)
                self.assertEqual(observed_proposals, [((11,), tuple(draft_tokens))])
                self.assertEqual(len(kit.model.calls), 2)
                self.assertTrue(all(call[1]["target_verify"] for call in kit.model.calls))
                self.assertTrue(
                    all(call[1]["capture_layer_ids"] == [0, 1] for call in kit.model.calls)
                )
                self.assertEqual(draft_model.draft_lens[0], (11, expected_block_size))
                self.assertEqual(draft_model.accept_lens, [accepted])
                self.assertEqual(
                    kit.model.rollback_calls,
                    [] if case["rollback"] is None else [case["rollback"] + (2,)],
                )
                self.assertEqual(
                    [list(layer.history) for layer in kit.cache_wrapper.cache],
                    [expected_history, expected_history],
                )
                self.assertEqual(
                    [int(layer.lengths.tolist()[0]) for layer in kit.cache_wrapper.cache],
                    [len(expected_history), len(expected_history)],
                )
                rejected_tokens = draft_tokens[accepted:]
                for layer in kit.cache_wrapper.cache:
                    for rejected_token in rejected_tokens:
                        self.assertNotIn(rejected_token, layer.history)
                self.assertEqual(kit.received_cache_tokens, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
