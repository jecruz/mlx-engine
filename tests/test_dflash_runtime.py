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


class FakeDraftModel:
    def __init__(self):
        self.config = SimpleNamespace(target_layer_ids=[0])
        self.reset_calls = []
        self.accept_lens = []
        self.draft_lens = []

    def reset(self, model):
        self.reset_calls.append(model)
        return [SimpleNamespace(lengths=mx.array([1]))]

    def draft_block(self, last_bonus, hidden, cache, block_size, sampler, token_dtype):
        self.draft_lens.append((last_bonus, block_size))
        return mx.array([[12, 13]], dtype=token_dtype)


class FakeTargetModel:
    def __init__(self):
        self.calls = []
        self.rollback_calls = []
        self.language_model = self

    def __call__(self, tokens, **kwargs):
        seq_len = int(tokens.shape[1])
        self.calls.append((tokens, kwargs))
        logits = mx.full((1, seq_len, 128), -100.0)
        if seq_len == 1:
            logits[:, 0, 11] = 100.0
        else:
            logits[:, 0, 12] = 100.0
            logits[:, 1, 99] = 100.0
            logits[:, 2, 13] = 100.0
        hidden = [mx.ones((1, seq_len, 2))]
        return SimpleNamespace(logits=logits, hidden_states=hidden, gdn_states=hidden)

    def rollback_speculative_cache(self, prompt_cache, gdn_states, accepted, block_size):
        self.rollback_calls.append((prompt_cache, gdn_states, accepted, block_size))


class FakeKit:
    def __init__(self):
        self.model = FakeTargetModel()
        self.tokenizer = FakeTokenizer()
        self.cache_wrapper = SimpleNamespace(cache=[SimpleNamespace(lengths=mx.array([1]))])
        self.pending_requests = {}
        self.max_kv_size = None
        self.kv_bits = None
        self.kv_group_size = None
        self.quantized_kv_start = None
        self.received_cache_tokens = []

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
        return mx.array(prompt_tokens, dtype=mx.int32), None

    def is_cross_prompt_cache_active(self):
        return False

    def record_token_to_cache(self, token):
        self.received_cache_tokens.append(token)


class TestDFlashRuntime(unittest.TestCase):
    def test_emits_only_verified_tokens_and_keeps_proposals_out_of_history(self):
        kit = FakeKit()
        draft_model = FakeDraftModel()
        observed_proposals = []

        results = list(
            dflash_stream_generate(
                kit,
                [1],
                request_id="dflash-runtime",
                max_tokens=3,
                dflash_options=DFlashBoundaryOptions(
                    enabled=True,
                    target_model_path=Path("/tmp/target"),
                    drafter_model_path=Path("/tmp/drafter"),
                    max_draft_tokens=3,
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

        self.assertEqual(emitted_ids, [11, 12, 99])
        self.assertEqual(emitted_from_draft, [False, True, False])
        self.assertEqual(observed_proposals, [((11,), (12, 13))])
        self.assertEqual(len(kit.model.calls), 2)
        self.assertTrue(all(call[1]["target_verify"] for call in kit.model.calls))
        self.assertEqual(len(kit.model.rollback_calls), 1)
        self.assertEqual(kit.received_cache_tokens, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
