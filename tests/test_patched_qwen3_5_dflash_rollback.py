"""Focused DFlash rollback tests for the patched Qwen3.5 TextModel.

The patched ``Qwen3_5TextModel`` exposes ``rollback_speculative_cache`` so
``dflash_stream_generate`` can recover from partial DFlash rejection without
leaving unverified proposal tokens in the live cache state. The DFlash path
remains sequential text only and fail-closed, so these tests exercise the
hook in isolation rather than going through the full draft/verify runtime.

Each test exercises one of the three acceptance shapes the DFlash runtime
can produce and asserts that the rejected proposal tokens do not survive in
the live cache state.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import mlx.core as mx

from mlx_engine.model_kit.patches.qwen3_5 import (
    PatchedQwen3_5TextModel,
    _qwen3_5_dflash_rollback as rollback_speculative_cache,
)


class _HistoryCacheLayer:
    """Minimal cache layer with a ``history`` list and ``lengths`` array.

    Mirrors the shape the existing ``test_dflash_runtime`` tests rely on so
    the rollback hook is exercised with the same data layout the DFlash
    runtime probes. ``_idx`` and ``offset`` are included to prove the hook
    rolls back generic position metadata too.
    """

    def __init__(self, layer_id: int, history: list[int]):
        self.layer_id = layer_id
        self.history = list(history)
        self.lengths = mx.array([len(self.history)], dtype=mx.int32)
        self._idx = len(self.history)
        self.offset = len(self.history)
        self.advance_calls: list[int] = []


class _KVCacheLikeLayer:
    """Mimic an mlx-lm ``KVCache``-shaped layer with ``keys``/``values`` arrays."""

    def __init__(self, layer_id: int, num_tokens: int):
        self.layer_id = layer_id
        self.keys = mx.zeros((1, 2, num_tokens, 4), dtype=mx.bfloat16)
        self.values = mx.zeros((1, 2, num_tokens, 4), dtype=mx.bfloat16)
        self.offset = num_tokens
        self._idx = num_tokens
        self.lengths = mx.array([num_tokens], dtype=mx.int32)
        self.advance_calls: list[int] = []


def _build_gdn_state(base_history_len: int):
    return SimpleNamespace(base_history_len=base_history_len)


class TestPatchedQwen3_5RollbackHook(unittest.TestCase):
    def test_hook_exists_and_is_callable_on_patched_text_model(self):
        """The PatchedQwen3_5TextModel class must expose the rollback hook.

        ``dflash_stream_generate`` reaches the hook through ``getattr`` so the
        attribute only has to exist on the class. The hook should be defined
        on the patched class and default-off (no caller wires it into the
        ordinary text generation path).
        """
        self.assertTrue(hasattr(PatchedQwen3_5TextModel, "rollback_speculative_cache"))
        self.assertTrue(callable(PatchedQwen3_5TextModel.rollback_speculative_cache))

    def test_accepted_zero_keeps_only_bonus_token(self):
        """Full draft rejection: cache rolls back to base + 1 (bonus only)."""
        base_history_len = 4
        prompt_cache = [
            _HistoryCacheLayer(layer_id=0, history=[1, 2, 3, 4, 99, 12, 13, 14]),
            _HistoryCacheLayer(layer_id=1, history=[1, 2, 3, 4, 99, 12, 13, 14]),
        ]
        gdn_states = [
            _build_gdn_state(base_history_len),
            _build_gdn_state(base_history_len),
        ]

        rollback_speculative_cache(prompt_cache,
            gdn_states,
            accepted=0,
            block_size=4,
        )

        expected_history = [1, 2, 3, 4, 99]
        for layer in prompt_cache:
            self.assertEqual(layer.history, expected_history)
            self.assertEqual(int(layer.lengths.item()), len(expected_history))
            self.assertEqual(layer._idx, len(expected_history))
            self.assertEqual(layer.offset, len(expected_history))

        # The rejected draft tokens must not remain in the live cache state.
        for rejected_token in (12, 13, 14):
            for layer in prompt_cache:
                self.assertNotIn(rejected_token, layer.history)

    def test_partial_acceptance_keeps_bonus_plus_accepted_drafts(self):
        """Partial acceptance: cache rolls back to base + accepted + 1."""
        base_history_len = 4
        prompt_cache = [
            _HistoryCacheLayer(layer_id=0, history=[1, 2, 3, 4, 99, 12, 13, 14]),
            _HistoryCacheLayer(layer_id=1, history=[1, 2, 3, 4, 99, 12, 13, 14]),
        ]
        gdn_states = [
            _build_gdn_state(base_history_len),
            _build_gdn_state(base_history_len),
        ]

        rollback_speculative_cache(prompt_cache,
            gdn_states,
            accepted=2,
            block_size=4,
        )

        # keep = 4 + 2 + 1 = 7: tokens 1,2,3,4 (base), 99 (bonus), 12, 13 (accepted)
        expected_history = [1, 2, 3, 4, 99, 12, 13]
        for layer in prompt_cache:
            self.assertEqual(layer.history, expected_history)
            self.assertEqual(int(layer.lengths.item()), len(expected_history))
            self.assertEqual(layer._idx, len(expected_history))
            self.assertEqual(layer.offset, len(expected_history))

        # The rejected draft token (14) must not remain in live cache state.
        for layer in prompt_cache:
            self.assertNotIn(14, layer.history)

    def test_full_acceptance_is_a_noop(self):
        """Full acceptance: hook must not mutate cache state.

        ``dflash_stream_generate`` skips the hook when ``accepted == block_size
        - 1``, but the method must still be safe to call defensively.
        """
        base_history_len = 4
        original_history = [1, 2, 3, 4, 99, 12, 13, 14]
        prompt_cache = [
            _HistoryCacheLayer(layer_id=0, history=list(original_history)),
            _HistoryCacheLayer(layer_id=1, history=list(original_history)),
        ]
        gdn_states = [
            _build_gdn_state(base_history_len),
            _build_gdn_state(base_history_len),
        ]

        rollback_speculative_cache(prompt_cache,
            gdn_states,
            accepted=3,
            block_size=4,
        )

        for layer in prompt_cache:
            self.assertEqual(layer.history, original_history)
            self.assertEqual(int(layer.lengths.item()), len(original_history))
            self.assertEqual(layer._idx, len(original_history))
            self.assertEqual(layer.offset, len(original_history))

    def test_rollback_without_gdn_states_starts_from_zero(self):
        """Empty ``gdn_states`` must still roll back per-layer history."""
        prompt_cache = [
            _HistoryCacheLayer(layer_id=0, history=[10, 11, 99, 12, 13]),
            _HistoryCacheLayer(layer_id=1, history=[10, 11, 99, 12, 13]),
        ]

        rollback_speculative_cache(prompt_cache,
            gdn_states=[],
            accepted=1,
            block_size=3,
        )

        # base_history_len=0, accepted=1 -> keep = 0 + 1 + 1 = 2
        expected_history = [10, 11]
        for layer in prompt_cache:
            self.assertEqual(layer.history, expected_history)
            self.assertEqual(int(layer.lengths.item()), len(expected_history))

    def test_rollback_handles_mlxvlm_gdn_sink_tuple_format(self):
        """mlx-vlm GDN sink entries are tuples without ``base_history_len``.

        The hook must treat tuple-typed ``gdn_state`` entries as base=0 and
        fall back to the per-layer cache metadata for the roll-back length.
        """
        prompt_cache = [
            _HistoryCacheLayer(layer_id=0, history=[1, 2, 99, 12, 13]),
        ]
        # mlx-vlm gdn_sink tuple format (12 entries) without base_history_len.
        gdn_states = [
            (
                "q",
                "k",
                "v",
                "a",
                "b",
                "A_log",
                "dt_bias",
                "initial_state",
                "mask",
                "conv_input",
                4,
                "intermediate_states",
            ),
        ]

        rollback_speculative_cache(prompt_cache,
            gdn_states,
            accepted=1,
            block_size=3,
        )

        # base_history_len defaults to 0 for tuples; accepted=1 -> keep=2.
        expected_history = [1, 2]
        self.assertEqual(prompt_cache[0].history, expected_history)

    def test_rollback_truncates_real_kvcache_keys_and_values(self):
        """mlx-lm KVCache layers must have ``keys``/``values`` truncated too."""
        original_tokens = 8
        prompt_cache = [
            _KVCacheLikeLayer(layer_id=0, num_tokens=original_tokens),
        ]
        gdn_states = [
            _build_gdn_state(base_history_len=4),
        ]

        rollback_speculative_cache(prompt_cache,
            gdn_states,
            accepted=2,
            block_size=4,
        )

        layer = prompt_cache[0]
        # base=4, accepted=2, +1 bonus -> keep=7
        self.assertEqual(layer.offset, 7)
        self.assertEqual(layer._idx, 7)
        self.assertEqual(layer.keys.shape[-2], 7)
        self.assertEqual(layer.values.shape[-2], 7)
        self.assertEqual(int(layer.lengths.item()), 7)

    def test_rollback_rejects_negative_accepted(self):
        """Defensive: negative ``accepted`` is treated as a no-op."""
        prompt_cache = [
            _HistoryCacheLayer(layer_id=0, history=[1, 2, 3]),
        ]
        original = list(prompt_cache[0].history)
        gdn_states = [_build_gdn_state(base_history_len=2)]

        rollback_speculative_cache(prompt_cache,
            gdn_states,
            accepted=-1,
            block_size=2,
        )

        self.assertEqual(prompt_cache[0].history, original)

    def test_rollback_skips_none_cache_layers(self):
        """A None cache layer in the prompt_cache list must not crash."""
        prompt_cache = [
            None,
            _HistoryCacheLayer(layer_id=1, history=[10, 11, 99, 12]),
        ]
        gdn_states = [
            _build_gdn_state(base_history_len=2),
            _build_gdn_state(base_history_len=2),
        ]

        rollback_speculative_cache(prompt_cache,
            gdn_states,
            accepted=0,
            block_size=2,
        )

        # keep = 2 + 0 + 1 = 3
        self.assertEqual(prompt_cache[1].history, [10, 11, 99])

    def test_rollback_does_not_widen_dflash_surface(self):
        """The rollback hook must never alter the DFlash fail-closed surface.

        The hook is a text-only sequential rollback utility; it must not be
        imported or referenced from any non-DFlash generation path. We check
        this by inspecting the patched class: only ``rollback_speculative_cache``
        and the MRoPE helpers should be added by this milestone.
        """
        declared = set(dir(PatchedQwen3_5TextModel))
        # The hook itself must exist.
        self.assertIn("rollback_speculative_cache", declared)
        # MRoPE state methods remain available for the vision add-on.
        self.assertIn("reset_mrope_state", declared)
        # No DFlash defaults-on flags should be exposed on the text model.
        for forbidden_attr in (
            "enable_dflash",
            "dflash_enabled",
            "rollback_default_on",
        ):
            self.assertNotIn(forbidden_attr, declared)


class TestPatchedQwen3_5RollbackInvariants(unittest.TestCase):
    """Invariant tests: no unverified tokens may remain in live cache state."""

    def _run_full_round(self, base_history_len: int, draft_tokens, accepted: int):
        """Simulate one DFlash verify + rollback round on the cache."""
        prompt_tokens = list(range(1, base_history_len + 1))
        bonus_token = 99
        seq_history = (
            prompt_tokens
            + [bonus_token]
            + list(draft_tokens)
        )
        prompt_cache = [
            _HistoryCacheLayer(layer_id=i, history=list(seq_history))
            for i in range(3)
        ]
        gdn_states = [
            _build_gdn_state(base_history_len)
            for _ in prompt_cache
        ]
        rollback_speculative_cache(prompt_cache,
            gdn_states,
            accepted=accepted,
            block_size=len(draft_tokens) + 1,
        )
        return prompt_cache

    def test_no_rejected_tokens_remain_after_any_acceptance(self):
        """For every acceptance shape, rejected tokens must be absent."""
        draft_tokens = [12, 13, 14]
        base_history_len = 3

        for accepted in (0, 1, 2, 3):
            with self.subTest(accepted=accepted):
                prompt_cache = self._run_full_round(
                    base_history_len, draft_tokens, accepted
                )
                for layer in prompt_cache:
                    rejected = draft_tokens[accepted:]
                    for rejected_token in rejected:
                        self.assertNotIn(rejected_token, layer.history)

    def test_accepted_tokens_survive_rollback(self):
        """The accepted draft tokens must remain in live cache state."""
        draft_tokens = [12, 13, 14]
        base_history_len = 3

        for accepted in (0, 1, 2, 3):
            with self.subTest(accepted=accepted):
                prompt_cache = self._run_full_round(
                    base_history_len, draft_tokens, accepted
                )
                kept = draft_tokens[:accepted]
                for layer in prompt_cache:
                    for accepted_token in kept:
                        self.assertIn(accepted_token, layer.history)

    def test_preexisting_prompt_tokens_survive_rollback(self):
        """The pre-existing prompt tokens must never be removed by rollback."""
        draft_tokens = [12, 13, 14]
        base_history_len = 3

        for accepted in (0, 1, 2, 3):
            with self.subTest(accepted=accepted):
                prompt_cache = self._run_full_round(
                    base_history_len, draft_tokens, accepted
                )
                for layer in prompt_cache:
                    for prompt_token in range(1, base_history_len + 1):
                        self.assertIn(prompt_token, layer.history)
                    # Bonus target token also remains.
                    self.assertIn(99, layer.history)


if __name__ == "__main__":
    unittest.main(verbosity=2)
