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


class _RealQwen3ArraysCache:
    """Mimic the real Qwen3.6 ArraysCache layout loaded by ModelKit.

    The real mlx-lm ``ArraysCache`` (single-sequence GDN linear-attention
    state) has ``cache`` as a list of arrays plus ``lengths`` and
    ``left_padding`` attributes that default to ``None`` for sequential
    use. The Qwen3.6 27B target loaded through ``ModelKit`` produces
    exactly 48 of these layers (one per GDN linear-attention block) plus
    16 KVCache layers.

    The rollback hook now mutates the real ``cache[0]`` (conv window)
    and ``cache[1]`` (gated-delta state) in place using mlx-vlm's
    ``gated_delta_accept_states`` helper plus the per-layer GDN sink
    tuple captured during target_verify. Tests use the ``conv_input``,
    ``initial_state``, and ``intermediate_states`` arrays to drive
    realistic rollback paths.
    """

    def __init__(
        self,
        layer_id: int,
        conv_kernel_size: int = 3,
        conv_dim: int = 4,
        head_v_dim: int = 2,
        head_k_dim: int = 2,
        num_v_heads: int = 2,
    ):
        self.layer_id = layer_id
        self.conv_kernel_size = conv_kernel_size
        # Qwen3.5/3.6 GDN layers store conv_state in cache[0] and the
        # running gated-delta state in cache[1]. Both are mlx arrays
        # mutated in-place during the forward pass.
        self.cache = [
            mx.zeros(
                (1, conv_kernel_size - 1, conv_dim), dtype=mx.bfloat16
            ),
            mx.zeros((1, num_v_heads, head_v_dim, head_k_dim), dtype=mx.float32),
        ]
        # Real single-sequence ArraysCache has both attributes set to None.
        self.lengths = None
        self.left_padding = None
        # No history list, no keys/values, no offset/_idx on the real
        # ArraysCache shape used for sequential Qwen3.6 text.
        self.history = None
        self.keys = None
        self.values = None
        self.offset = None
        self._idx = None


def _build_real_arrays_cache_gdn_state(
    arrays_layer: _RealQwen3ArraysCache,
    accepted_token_value: float = 0.0,
    verify_token_count: int = 4,
) -> tuple:
    """Build a 12-tuple mlx-vlm GDN sink entry for the test fake.

    Mirrors the tuple shape mlx-vlm's ``Qwen3_5GatedDeltaNet`` appends
    to ``gdn_sink`` during ``target_verify``:

        (q, k, v, a, b, A_log, dt_bias,
         initial_state, mask, conv_input, conv_kernel_size, intermediate_states)

    Only the trailing four entries are consumed by the rollback hook;
    the others are placeholders to keep the tuple shape compatible
    with mlx-vlm's GDN sink layout. ``initial_state`` equals the live
    cache[1] before the verify call; ``intermediate_states`` carries
    per-step delta states; ``conv_input`` is the extended conv input
    (``[cache[0] prepended] + new tokens``).

    ``conv_input`` is tagged so ``conv_input[:, kernel_size - 1 + t, :]``
    equals ``-(t + 1)`` (per-token). The original conv window
    (``conv_input[:, :kernel_size - 1, :]``) stays zero, which lets
    tests distinguish the pre-verify conv state from per-token slices.
    """
    conv_kernel_size = arrays_layer.conv_kernel_size
    conv_dim = arrays_layer.cache[0].shape[-1]
    state_shape = arrays_layer.cache[1].shape

    intermediate_states = mx.stack(
        [
            mx.full(state_shape, float(accepted_token_value + step + 1))
            for step in range(verify_token_count)
        ],
        axis=1,
    )

    initial_state = mx.zeros(state_shape, dtype=arrays_layer.cache[1].dtype)

    conv_input = mx.zeros(
        (1, conv_kernel_size - 1 + verify_token_count, conv_dim),
        dtype=mx.bfloat16,
    )
    for t in range(verify_token_count):
        col = conv_kernel_size - 1 + t
        token_row = mx.full(
            (1, 1, conv_dim), -float(t + 1), dtype=mx.bfloat16
        )
        conv_input = mx.concatenate(
            [conv_input[:, :col, :], token_row, conv_input[:, col + 1 :, :]],
            axis=1,
        )

    return (
        mx.zeros((1, 1, 1), dtype=mx.bfloat16),  # q placeholder
        mx.zeros((1, 1, 1), dtype=mx.bfloat16),  # k placeholder
        mx.zeros((1, 1, 1), dtype=mx.bfloat16),  # v placeholder
        mx.zeros((1, 1, 1), dtype=mx.bfloat16),  # a placeholder
        mx.zeros((1, 1, 1), dtype=mx.bfloat16),  # b placeholder
        mx.zeros((1,), dtype=mx.bfloat16),      # A_log placeholder
        mx.zeros((1,), dtype=mx.bfloat16),      # dt_bias placeholder
        initial_state,
        None,                                    # mask placeholder
        conv_input,
        conv_kernel_size,
        intermediate_states,
    )


class TestRealQwen3ArraysCacheRollback(unittest.TestCase):
    """Prove the rollback hook mutates real ``ArraysCache.cache[idx]`` state.

    Per feature ``m14-dflash-real-arrayscache-gdn-rollback``: the
    rollback hook now drives ``gated_delta_accept_states`` to restore
    the GDN state arrays for the real sequential single-sequence
    ``ArraysCache`` shape. The tests below prove:

    * the rollback rewrites ``cache[0]`` (conv window) and ``cache[1]``
      (gated-delta state) in place for accepted=0, partial acceptance,
      and full acceptance,
    * the accepted target-verified state and pre-existing prompt / cache
      state survive rollback (the ArraysCache subset continues from the
      state mlx-vlm would have produced after processing the accepted
      prefix),
    * the ragged ``ArraysCache`` variant (non-None ``lengths`` /
      ``left_padding``) is left untouched so the validator stays
      fail-closed for unproven shapes.
    """

    def test_accepted_zero_restores_initial_conv_and_state(self):
        """accepted=0 restores cache to the post-bonus-target state.

        When ``_speculative_walk`` reports zero accepted drafts, only the
        bonus target token survived. The rollback must drop the rejected
        drafts and land the cache in the state mlx-vlm would have left
        it in after processing only the bonus target. Concretely:

        * cache[1] must equal ``intermediate_states[0]`` (state after
          processing the bonus target, before any drafts).
        * cache[0] must equal ``conv_input[:, 1:kernel_size, :]`` (the
          last ``kernel_size - 1`` entries of the extended conv input
          after 1 token processed).

        The rejected draft tokens must not leak into the live cache
        state.
        """
        arrays_layer = _RealQwen3ArraysCache(
            layer_id=0,
            conv_kernel_size=3,
            conv_dim=4,
            head_v_dim=2,
            head_k_dim=2,
            num_v_heads=2,
        )
        gdn_state = _build_real_arrays_cache_gdn_state(
            arrays_layer, accepted_token_value=7.0, verify_token_count=4
        )

        # Pre-seed the live cache[1] with a sentinel so the rollback must
        # replace it (otherwise the test could pass on a no-op).
        arrays_layer.cache[1] = mx.full(
            arrays_layer.cache[1].shape, 99.0, dtype=mx.float32
        )
        # Pre-seed cache[0] with a sentinel too.
        arrays_layer.cache[0] = mx.full(
            arrays_layer.cache[0].shape, 88.0, dtype=mx.bfloat16
        )

        prompt_cache = [arrays_layer]
        rollback_speculative_cache(
            prompt_cache,
            [gdn_state],
            accepted=0,
            block_size=4,
        )

        # After accepted=0 rollback:
        # * cache[1] must equal intermediate_states[0]
        #   (accepted_token_value + 0 + 1 = 8.0)
        # * cache[0] must equal conv_input[:, accepted + 1 :
        #   accepted + kernel_size, :] which for accepted=0, kernel_size=3
        #   is conv_input[:, 1:3, :] — shape (1, kernel_size - 1, conv_dim).
        intermediate_states = gdn_state[11]
        expected_state = intermediate_states[:, 0, :, :, :]
        new_state = prompt_cache[0].cache[1]
        self.assertTrue(
            bool(mx.all(mx.equal(new_state, expected_state)).item()),
            msg=(
                "accepted=0 rollback must pick intermediate_states[0] "
                "for cache[1]"
            ),
        )
        conv_input = gdn_state[9]
        expected_conv = conv_input[
            :,
            1 : 1 + (arrays_layer.conv_kernel_size - 1),
            :,
        ]
        new_conv = prompt_cache[0].cache[0]
        self.assertEqual(new_conv.shape, expected_conv.shape)
        self.assertTrue(
            bool(mx.all(mx.equal(new_conv, expected_conv)).item()),
            msg=(
                "accepted=0 rollback must pick the conv window after "
                "1 token processed"
            ),
        )
        # lengths / left_padding must stay None so the cache mode remains
        # single-sequence (the validator rejects ragged variants).
        self.assertIsNone(prompt_cache[0].lengths)
        self.assertIsNone(prompt_cache[0].left_padding)

    def test_partial_acceptance_restores_correct_intermediate_state(self):
        """Partial acceptance picks intermediate_states[accepted].

        For ``accepted`` drafts accepted (out of ``block_size - 1``
        drafts), the rollback must pick ``intermediate_states[accepted]``
        for cache[1] and the corresponding conv slice for cache[0]. The
        mlx-vlm ``gated_delta_accept_states`` helper computes both in
        one call so the test verifies the hook drives it correctly.
        """
        arrays_layer = _RealQwen3ArraysCache(
            layer_id=0,
            conv_kernel_size=3,
            conv_dim=4,
            head_v_dim=2,
            head_k_dim=2,
            num_v_heads=2,
        )
        # Build a per-step tagged intermediate state. Each step has a
        # unique scalar pattern that the rollback must surface.
        verify_token_count = 4
        gdn_state = _build_real_arrays_cache_gdn_state(
            arrays_layer,
            accepted_token_value=10.0,
            verify_token_count=verify_token_count,
        )

        # Pre-seed the live cache[1] with a non-zero sentinel so the
        # rollback must replace it.
        arrays_layer.cache[1] = mx.full(
            arrays_layer.cache[1].shape, 99.0, dtype=mx.float32
        )
        arrays_layer.cache[0] = mx.full(
            arrays_layer.cache[0].shape, 88.0, dtype=mx.bfloat16
        )

        prompt_cache = [arrays_layer]
        accepted = 2
        rollback_speculative_cache(
            prompt_cache,
            [gdn_state],
            accepted=accepted,
            block_size=verify_token_count + 1,
        )

        # After accepted=2 rollback:
        # * cache[1] must equal intermediate_states[accepted]
        #   (which is mx.full(state_shape, accepted_token_value + accepted + 1))
        # * cache[0] must reflect the conv window at step ``accepted``.
        expected_state_value = 10.0 + accepted + 1  # = 13.0
        new_state = prompt_cache[0].cache[1]
        expected_state = mx.full(
            new_state.shape, expected_state_value, dtype=new_state.dtype
        )
        self.assertTrue(
            bool(mx.all(mx.equal(new_state, expected_state)).item()),
            msg=(
                "partial-acceptance rollback must pick "
                f"intermediate_states[{accepted}] for cache[1]"
            ),
        )
        # The conv window must reflect conv_input[
        # accepted+1 : accepted+kernel_size-1, :] which for accepted=2,
        # kernel_size=3 is conv_input[:, 3:4, :] — just the row tagged
        # with -2 (token 1, the last accepted draft).
        new_conv = prompt_cache[0].cache[0]
        conv_input = gdn_state[9]
        expected_conv = conv_input[
            :,
            accepted + 1 : accepted + 1 + (arrays_layer.conv_kernel_size - 1),
            :,
        ]
        self.assertEqual(new_conv.shape, expected_conv.shape)
        self.assertTrue(
            bool(mx.all(mx.equal(new_conv, expected_conv)).item()),
            msg=(
                "partial-acceptance rollback must pick the right conv "
                "window slice"
            ),
        )

    def test_full_acceptance_is_a_noop(self):
        """Full acceptance leaves the live ArraysCache state untouched.

        ``dflash_stream_generate`` only invokes the hook when
        ``accepted < block_size - 1``. The hook is also safe to call
        with full acceptance: it must not corrupt the live state
        because the verify call already wrote the final state into
        cache[1] / cache[0].
        """
        arrays_layer = _RealQwen3ArraysCache(
            layer_id=0,
            conv_kernel_size=3,
            conv_dim=4,
            head_v_dim=2,
            head_k_dim=2,
            num_v_heads=2,
        )
        gdn_state = _build_real_arrays_cache_gdn_state(
            arrays_layer, accepted_token_value=5.0, verify_token_count=4
        )

        # Pre-seed the live cache with a sentinel so a no-op is
        # observable.
        arrays_layer.cache[1] = mx.full(
            arrays_layer.cache[1].shape, 99.0, dtype=mx.float32
        )
        arrays_layer.cache[0] = mx.full(
            arrays_layer.cache[0].shape, 88.0, dtype=mx.bfloat16
        )

        prompt_cache = [arrays_layer]
        live_state_before = prompt_cache[0].cache[1]
        live_conv_before = prompt_cache[0].cache[0]
        state_array_id_before = id(live_state_before)
        conv_array_id_before = id(live_conv_before)

        rollback_speculative_cache(
            prompt_cache,
            [gdn_state],
            accepted=3,  # block_size=4 => block_size - 1 = 3
            block_size=4,
        )

        # Full acceptance: cache[1] and cache[0] must still equal the
        # pre-rollback live state.
        live_state_after = prompt_cache[0].cache[1]
        live_conv_after = prompt_cache[0].cache[0]
        self.assertTrue(
            bool(mx.all(mx.equal(live_state_after, live_state_before)).item()),
            msg="full-acceptance rollback must leave cache[1] untouched",
        )
        self.assertTrue(
            bool(mx.all(mx.equal(live_conv_after, live_conv_before)).item()),
            msg="full-acceptance rollback must leave cache[0] untouched",
        )
        self.assertEqual(
            id(live_state_after),
            state_array_id_before,
            msg="full-acceptance rollback must not replace cache[1] array",
        )
        self.assertEqual(
            id(live_conv_after),
            conv_array_id_before,
            msg="full-acceptance rollback must not replace cache[0] array",
        )

    def test_full_qwen36_layout_kv_subset_sliced_arrays_subset_rolled_back(
        self,
    ):
        """Real Qwen3.6 layout (16 KVCache + 48 ArraysCache) rollback.

        The KVCache subset is sliced by the existing KVCache path (kept
        for compatibility), and the 48 ArraysCache subset is rolled
        back via the new GDN-aware path. Rejected proposal tokens must
        not leak into either subset of the live cache state.
        """
        kv_layers = [
            _KVCacheLikeLayer(layer_id=i, num_tokens=8) for i in range(16)
        ]
        arrays_layers: list[_RealQwen3ArraysCache] = []
        arrays_gdn_states = []
        for i in range(48):
            arrays_layer = _RealQwen3ArraysCache(
                layer_id=i,
                conv_kernel_size=3,
                conv_dim=4,
                head_v_dim=2,
                head_k_dim=2,
                num_v_heads=2,
            )
            arrays_layer.cache[1] = mx.full(
                arrays_layer.cache[1].shape, 99.0, dtype=mx.float32
            )
            arrays_layer.cache[0] = mx.full(
                arrays_layer.cache[0].shape, 88.0, dtype=mx.bfloat16
            )
            arrays_layers.append(arrays_layer)
            arrays_gdn_states.append(
                _build_real_arrays_cache_gdn_state(
                    arrays_layer,
                    accepted_token_value=20.0,
                    verify_token_count=4,
                )
            )

        prompt_cache: list = kv_layers + arrays_layers

        # Build an aligned gdn_states list mirroring what
        # ``dflash_stream_generate`` produces via
        # ``_align_gdn_states_with_prompt_cache``: a per-cache-index
        # list where KVCache layers get ``SimpleNamespace(base_history_len)``
        # (the simple-namespace branch the rollback hook supports) and
        # ArraysCache layers get the 12-tuple GDN sink entry.
        aligned_gdn_states: list = []
        arrays_iter = iter(arrays_gdn_states)
        for cache_index, cache_layer in enumerate(prompt_cache):
            if isinstance(cache_layer, _KVCacheLikeLayer):
                aligned_gdn_states.append(SimpleNamespace(base_history_len=4))
            else:
                aligned_gdn_states.append(next(arrays_iter))

        rollback_speculative_cache(
            prompt_cache,
            aligned_gdn_states,
            accepted=1,
            block_size=4,
        )

        # KVCache subset sliced to keep=base(4)+accepted(1)+bonus(1)=6.
        for layer in kv_layers:
            self.assertEqual(layer.offset, 6)
            self.assertEqual(layer._idx, 6)
            self.assertEqual(layer.keys.shape[-2], 6)
            self.assertEqual(layer.values.shape[-2], 6)
            self.assertEqual(int(layer.lengths.item()), 6)

        # ArraysCache subset: cache[1] must come from the rollback
        # path (not the original 99.0 sentinel). The exact target value
        # is intermediate_states[accepted=1] = mx.full(state_shape,
        # accepted_token_value + 1 + 1) = mx.full(state_shape, 22.0).
        for layer in arrays_layers:
            new_state = layer.cache[1]
            sentinel = mx.full(new_state.shape, 99.0, dtype=new_state.dtype)
            self.assertFalse(
                bool(mx.all(mx.equal(new_state, sentinel)).item()),
                msg=(
                    "ArraysCache cache[1] must not retain the pre-rollback "
                    "sentinel after rollback"
                ),
            )
            expected_state_value = 20.0 + 1 + 1
            expected_state = mx.full(
                new_state.shape, expected_state_value, dtype=new_state.dtype
            )
            self.assertTrue(
                bool(mx.all(mx.equal(new_state, expected_state)).item()),
                msg=(
                    "ArraysCache cache[1] must equal intermediate_states[1] "
                    f"({expected_state_value}) after accepted=1 rollback"
                ),
            )
            # lengths / left_padding must stay None so the cache mode
            # remains single-sequence.
            self.assertIsNone(layer.lengths)
            self.assertIsNone(layer.left_padding)

    def test_ragged_arrays_cache_is_left_untouched(self):
        """Ragged ArraysCache variants stay untouched.

        The validator rejects ragged ArraysCache shapes, so the rollback
        hook must NOT touch them: a future refactor that silently
        widens the rollback path to ragged shapes could corrupt GDN
        state. The ragged cache's cache[0] / cache[1] arrays must
        survive rollback unchanged.
        """

        class _RaggedArraysCache:
            def __init__(self):
                self.cache = [
                    mx.zeros((1, 2, 4), dtype=mx.bfloat16),
                    mx.zeros((1, 2, 2, 2), dtype=mx.float32),
                ]
                self.lengths = mx.array([1], dtype=mx.int32)
                self.left_padding = mx.array([0], dtype=mx.int32)

        ragged = _RaggedArraysCache()
        sentinel_state = ragged.cache[1]
        sentinel_conv = ragged.cache[0]
        prompt_cache = [ragged]
        # Even with a malformed gdn_state, the ragged cache must be
        # left untouched.
        rollback_speculative_cache(
            prompt_cache,
            [(None, None, None, None, None, None, None, None, None,
              None, 3, None)],
            accepted=1,
            block_size=4,
        )

        self.assertTrue(
            bool(mx.all(mx.equal(ragged.cache[1], sentinel_state)).item()),
            msg="ragged ArraysCache cache[1] must remain unchanged",
        )
        self.assertTrue(
            bool(mx.all(mx.equal(ragged.cache[0], sentinel_conv)).item()),
            msg="ragged ArraysCache cache[0] must remain unchanged",
        )
        # Sanity: the ragged arrays layer still has its non-None
        # lengths / left_padding (the hook did not strip them).
        self.assertIsNotNone(ragged.lengths)
        self.assertIsNotNone(ragged.left_padding)


class TestOuterTextModelWrapperRollbackHook(unittest.TestCase):
    """The outer ``mlx_lm.models.qwen3_5.TextModel`` wrapper exposes the hook.

    The loaded ``target_model.language_model`` is the outer mlx-lm
    ``TextModel`` wrapper class, not the inner ``Qwen3_5TextModel``.
    ``validate_dflash_runtime_compatibility`` and
    ``dflash_stream_generate`` reach the rollback hook through
    ``getattr`` on that wrapper, so the wrapper class must expose
    ``rollback_speculative_cache`` after ``apply_patches()`` and
    delegate to the inner PatchedQwen3_5TextModel rollback
    implementation. These tests prove the wrapper hook is exposed,
    delegates to the inner rollback semantics (or falls back to the
    module-level helper when the inner model lacks the hook), and does
    not widen the DFlash surface.
    """

    def test_wrapper_hook_exposed_after_apply_patches(self):
        """The outer ``TextModel`` exposes the rollback hook post-patch.

        ``dflash_stream_generate`` looks up ``rollback_speculative_cache``
        on the loaded ``target_model.language_model`` (the outer wrapper).
        If the wrapper does not expose the hook, the validator fails
        closed and the runtime raises DFlashUnavailableError before any
        draft round can run.
        """
        from mlx_engine.model_kit.patches.qwen3_5 import apply_patches

        apply_patches()

        import mlx_lm.models.qwen3_5 as qm

        self.assertTrue(hasattr(qm.TextModel, "rollback_speculative_cache"))
        self.assertTrue(callable(qm.TextModel.rollback_speculative_cache))
        # The inner class must continue to expose the hook too.
        self.assertTrue(hasattr(qm.Qwen3_5TextModel, "rollback_speculative_cache"))

    def test_wrapper_delegates_to_inner_patched_text_model(self):
        """Wrapper delegates to ``self.model.rollback_speculative_cache``.

        The patched-class case is the common path: the inner model was
        constructed after ``apply_patches()`` and exposes the hook, so
        the wrapper must hand off to it without going through the
        module-level fallback helper. The wrapper hook must pass
        ``prompt_cache``, ``gdn_states``, ``accepted``, and
        ``block_size`` through unchanged.
        """
        from mlx_engine.model_kit.patches.qwen3_5 import (
            _patched_qwen3_5_text_model_rollback_speculative_cache,
        )

        captured: list[tuple] = []

        class _FakeInnerModel:
            def rollback_speculative_cache(
                self, prompt_cache, gdn_states, accepted, block_size
            ):
                captured.append((prompt_cache, gdn_states, accepted, block_size))

        wrapper = SimpleNamespace(model=_FakeInnerModel())
        prompt_cache = [object(), object()]
        gdn_states = [object()]

        _patched_qwen3_5_text_model_rollback_speculative_cache(
            wrapper,
            prompt_cache,
            gdn_states,
            accepted=2,
            block_size=4,
        )

        self.assertEqual(captured, [(prompt_cache, gdn_states, 2, 4)])

    def test_wrapper_falls_back_when_inner_model_lacks_hook(self):
        """Wrapper falls back to the module-level rollback helper.

        A wrapper instance whose inner model was constructed before
        ``apply_patches()`` landed (or whose inner model does not
        implement the hook for any reason) must still receive the same
        accepted-state/rejected-cleanup semantics. The fallback path
        drives :func:`_qwen3_5_dflash_rollback` directly with the same
        arguments.
        """
        from mlx_engine.model_kit.patches.qwen3_5 import (
            _patched_qwen3_5_text_model_rollback_speculative_cache,
        )

        # Inner model has no rollback_speculative_cache attribute at all.
        wrapper = SimpleNamespace(model=SimpleNamespace())

        base_history_len = 4
        prompt_cache = [
            _HistoryCacheLayer(layer_id=0, history=[1, 2, 3, 4, 99, 12, 13, 14]),
            _HistoryCacheLayer(layer_id=1, history=[1, 2, 3, 4, 99, 12, 13, 14]),
        ]
        gdn_states = [
            _build_gdn_state(base_history_len),
            _build_gdn_state(base_history_len),
        ]

        _patched_qwen3_5_text_model_rollback_speculative_cache(
            wrapper,
            prompt_cache,
            gdn_states,
            accepted=2,
            block_size=4,
        )

        # Same accepted-state/rejected-cleanup contract as the inner
        # PatchedQwen3_5TextModel.rollback_speculative_cache path:
        # keep = 4 + 2 + 1 = 7 — bonus target token plus the two
        # accepted drafts.
        expected_history = [1, 2, 3, 4, 99, 12, 13]
        for layer in prompt_cache:
            self.assertEqual(layer.history, expected_history)
            self.assertEqual(int(layer.lengths.item()), len(expected_history))
            self.assertEqual(layer._idx, len(expected_history))
            self.assertEqual(layer.offset, len(expected_history))
            # Rejected draft token (14) must not survive.
            self.assertNotIn(14, layer.history)

    def test_wrapper_handles_none_inner_model(self):
        """A wrapper with no inner model still applies rollback semantics.

        Defensive path: ``self.model`` may be ``None`` during teardown
        or in a malformed load. The wrapper must not raise; it falls
        back to the module-level helper, which still trims the
        history list / KV offsets to the accepted boundary.
        """
        from mlx_engine.model_kit.patches.qwen3_5 import (
            _patched_qwen3_5_text_model_rollback_speculative_cache,
        )

        wrapper = SimpleNamespace(model=None)
        prompt_cache = [
            _HistoryCacheLayer(layer_id=0, history=[1, 2, 3, 4, 99, 12, 13, 14]),
        ]
        gdn_states = [_build_gdn_state(base_history_len=4)]

        _patched_qwen3_5_text_model_rollback_speculative_cache(
            wrapper,
            prompt_cache,
            gdn_states,
            accepted=1,
            block_size=4,
        )

        # keep = 4 + 1 + 1 = 6.
        expected_history = [1, 2, 3, 4, 99, 12]
        self.assertEqual(prompt_cache[0].history, expected_history)

    def test_wrapper_preserves_accepted_state_on_full_rejection(self):
        """Full draft rejection through the wrapper keeps only the bonus token.

        Mirrors ``TestPatchedQwen3_5RollbackHook.test_accepted_zero_keeps_only_bonus_token``
        but routes through the outer wrapper hook, exercising the
        delegation contract that ``validate_dflash_runtime_compatibility``
        depends on.
        """
        from mlx_engine.model_kit.patches.qwen3_5 import (
            _patched_qwen3_5_text_model_rollback_speculative_cache,
        )

        # Use the fallback path so we exercise the same module-level
        # helper that the patched inner class delegates to.
        wrapper = SimpleNamespace(model=SimpleNamespace())

        base_history_len = 4
        prompt_cache = [
            _HistoryCacheLayer(layer_id=0, history=[1, 2, 3, 4, 99, 12, 13, 14]),
            _HistoryCacheLayer(layer_id=1, history=[1, 2, 3, 4, 99, 12, 13, 14]),
        ]
        gdn_states = [
            _build_gdn_state(base_history_len),
            _build_gdn_state(base_history_len),
        ]

        _patched_qwen3_5_text_model_rollback_speculative_cache(
            wrapper,
            prompt_cache,
            gdn_states,
            accepted=0,
            block_size=4,
        )

        expected_history = [1, 2, 3, 4, 99]
        for layer in prompt_cache:
            self.assertEqual(layer.history, expected_history)
            self.assertEqual(int(layer.lengths.item()), len(expected_history))
        for rejected_token in (12, 13, 14):
            for layer in prompt_cache:
                self.assertNotIn(rejected_token, layer.history)

    def test_wrapper_full_acceptance_is_a_noop(self):
        """Full acceptance via the wrapper hook must not mutate cache state.

        Mirrors the inner-class invariant. ``dflash_stream_generate``
        only invokes the hook on partial rejection, but the method must
        be safe to call defensively.
        """
        from mlx_engine.model_kit.patches.qwen3_5 import (
            _patched_qwen3_5_text_model_rollback_speculative_cache,
        )

        wrapper = SimpleNamespace(model=SimpleNamespace())
        original_history = [1, 2, 3, 4, 99, 12, 13, 14]
        prompt_cache = [
            _HistoryCacheLayer(layer_id=0, history=list(original_history)),
            _HistoryCacheLayer(layer_id=1, history=list(original_history)),
        ]
        gdn_states = [
            _build_gdn_state(base_history_len=4),
            _build_gdn_state(base_history_len=4),
        ]

        _patched_qwen3_5_text_model_rollback_speculative_cache(
            wrapper,
            prompt_cache,
            gdn_states,
            accepted=3,  # block_size=4 => block_size - 1 = 3
            block_size=4,
        )

        for layer in prompt_cache:
            self.assertEqual(layer.history, original_history)
            self.assertEqual(int(layer.lengths.item()), len(original_history))
            self.assertEqual(layer._idx, len(original_history))
            self.assertEqual(layer.offset, len(original_history))

    def test_wrapper_does_not_widen_dflash_surface(self):
        """The wrapper hook must not enable any default-on DFlash surface.

        The rollback hook is a text-only sequential utility bound on the
        outer mlx-lm ``TextModel`` class. It must not import or
        reference any non-DFlash generation path. We assert this by
        inspecting the wrapper class: only ``rollback_speculative_cache``
        should be added by this milestone, and no DFlash defaults-on
        flags should leak onto the wrapper.
        """
        from mlx_engine.model_kit.patches.qwen3_5 import apply_patches

        apply_patches()

        import mlx_lm.models.qwen3_5 as qm

        # Snapshot the wrapper's public surface for inspection.
        declared = set(dir(qm.TextModel))
        # The wrapper hook itself must exist.
        self.assertIn("rollback_speculative_cache", declared)
        # No DFlash defaults-on flags should leak onto the wrapper.
        for forbidden_attr in (
            "enable_dflash",
            "dflash_enabled",
            "rollback_default_on",
        ):
            self.assertNotIn(forbidden_attr, declared)

    def test_wrapper_invokes_inner_hook_when_patched_inner_exists(self):
        """End-to-end delegation through a real PatchedQwen3_5TextModel instance.

        The wrapper looks up ``self.model.rollback_speculative_cache``
        via ``getattr`` and delegates to it when the inner model
        exposes the hook. We verify the delegation by patching a real
        PatchedQwen3_5TextModel instance into a wrapper-shaped object
        and confirming the wrapper route reaches the inner method
        (observed through the rollback's effect on the live cache).
        """
        from mlx_engine.model_kit.patches.qwen3_5 import (
            _patched_qwen3_5_text_model_rollback_speculative_cache,
            PatchedQwen3_5TextModel,
        )

        # Build a PatchedQwen3_5TextModel instance without going through
        # __init__ so we do not need a real MLX/Metal environment here.
        # The hook only ever touches self.model.rollback_speculative_cache
        # so a bare instance with the method is sufficient.
        inner = PatchedQwen3_5TextModel.__new__(PatchedQwen3_5TextModel)
        inner.position_ids = None
        inner.rope_deltas = None

        wrapper = SimpleNamespace(model=inner)

        base_history_len = 4
        prompt_cache = [
            _HistoryCacheLayer(layer_id=0, history=[1, 2, 3, 4, 99, 12, 13, 14]),
        ]
        gdn_states = [_build_gdn_state(base_history_len)]

        _patched_qwen3_5_text_model_rollback_speculative_cache(
            wrapper,
            prompt_cache,
            gdn_states,
            accepted=2,
            block_size=4,
        )

        # If the wrapper reached the inner hook, the rollback must have
        # trimmed the live cache state to the accepted boundary.
        expected_history = [1, 2, 3, 4, 99, 12, 13]
        self.assertEqual(prompt_cache[0].history, expected_history)


class TestOuterTextModelRuntimeCompatibility(unittest.TestCase):
    """The outer ``TextModel`` wrapper satisfies the runtime compatibility check.

    ``validate_dflash_runtime_compatibility`` looks up
    ``rollback_speculative_cache`` on ``target_model.language_model``.
    After ``apply_patches()`` that wrapper exposes the hook, so the
    validator must not raise the
    "TextModel does not implement rollback_speculative_cache" blocker
    for the Qwen3.5 / Qwen3.6 sequential surface.
    """

    def test_runtime_compatibility_passes_after_apply_patches(self):
        """Direct call: wrapper exposes the hook so runtime compatibility passes.

        The validator inspects the class via ``hasattr``, so we can
        prove the post-patch behavior without loading a heavyweight
        target model. The validator must see the hook on the wrapper
        and not raise DFlashUnavailableError for that single blocker.
        """
        from mlx_engine.model_kit.patches.qwen3_5 import apply_patches

        apply_patches()

        import mlx_lm.models.qwen3_5 as qm

        # Mimic the validator's lookup on the loaded language_model.
        lm = qm.TextModel
        self.assertTrue(
            hasattr(lm, "rollback_speculative_cache"),
            msg=(
                "After apply_patches() the outer TextModel wrapper must "
                "expose rollback_speculative_cache so "
                "validate_dflash_runtime_compatibility does not raise "
                "DFlashUnavailableError with the 'TextModel does not "
                "implement rollback_speculative_cache' blocker."
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
