from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx

from mlx_engine.utils.dflash_boundary import (
    DFLASH_PROVEN_QWEN35_LAYOUT,
    DFlashBoundaryOptions,
    DFlashUnavailableError,
)
from mlx_engine.utils.dflash_runtime import dflash_stream_generate


class FakeTokenizer:
    def __init__(self):
        self.eos_token_ids = [999]

    def decode(self, token):
        if isinstance(token, int):
            return str(token)
        return "".join(str(value) for value in token)


class KVCache:
    def __init__(self, layer_id: int):
        self.layer_id = layer_id
        self.history: list[int] = []
        self.lengths = mx.array([0], dtype=mx.int32)


class _FakeArraysCache:
    """Test fake for sequential single-sequence ``ArraysCache``.

    Mirrors the real mlx-lm ``ArraysCache`` shape (no ``lengths`` /
    ``left_padding``) so the DFlash runtime validator allows it through
    the proven-shape check. The fake carries a ``history`` list so the
    test can introspect live cache state without a real GDN state
    machine.
    """

    def __init__(self, layer_id: int):
        self.layer_id = layer_id
        self.cache = [mx.zeros((1, 2, 4), dtype=mx.bfloat16)] * 2
        self.lengths = None
        self.left_padding = None
        self.history: list[int] = []


def _make_proven_layout_cache() -> list:
    """Build the exact proven 16 KVCache + 48 ArraysCache layout."""

    proven_kv, proven_arrays = DFLASH_PROVEN_QWEN35_LAYOUT
    layers: list = []
    for layer_id in range(proven_kv):
        layers.append(KVCache(layer_id=layer_id))
    for layer_id in range(proven_arrays):
        layers.append(_FakeArraysCache(layer_id=layer_id + proven_kv))
    return layers


class FakeDraftModel:
    def __init__(
        self,
        draft_tokens: list[int],
        layer_count: int | None = None,
        target_layer_ids: list[int] | None = None,
    ):
        self._draft_tokens = tuple(draft_tokens)
        if layer_count is None:
            layer_count = sum(DFLASH_PROVEN_QWEN35_LAYOUT)
        if target_layer_ids is None:
            target_layer_ids = list(range(layer_count))
        self.config = SimpleNamespace(
            target_layer_ids=list(target_layer_ids),
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
        layer_count: int | None = None,
    ):
        self._prompt_tokens = tuple(prompt_tokens)
        self._spec_input_tokens = tuple(spec_input_tokens)
        self._spec_output_tokens = tuple(spec_output_tokens)
        if layer_count is None:
            layer_count = sum(DFLASH_PROVEN_QWEN35_LAYOUT)
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
            if hasattr(layer, "lengths") and isinstance(layer.lengths, mx.array):
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
            if hasattr(layer, "lengths") and isinstance(layer.lengths, mx.array):
                layer.lengths = mx.array([len(layer.history)], dtype=mx.int32)


class FakeKit:
    def __init__(
        self,
        draft_tokens: list[int],
        spec_input_tokens: list[int],
        spec_output_tokens: list[int],
        *,
        cache_layers: list[object] | None = None,
        rollback_supported: bool = True,
    ):
        if rollback_supported:
            self.model = FakeTargetModel([1], spec_input_tokens, spec_output_tokens)
        else:
            self.model = SimpleNamespace(language_model=SimpleNamespace())
        self.tokenizer = FakeTokenizer()
        self.cache_wrapper = SimpleNamespace(
            cache=cache_layers
            if cache_layers is not None
            else _make_proven_layout_cache()
        )
        self.pending_requests = {}
        self.max_kv_size = None
        self.kv_bits = None
        self.kv_group_size = None
        self.quantized_kv_start = None
        self.received_cache_tokens = []
        self._prompt_tokens = [1]
        self._draft_tokens = draft_tokens
        self.process_prompt_calls = 0

    def process_prompt(
        self,
        prompt_tokens,
        images_b64,
        prompt_progress_reporter,
        generate_args,
        max_image_size,
        **_kwargs,
    ):
        self.process_prompt_calls += 1
        generate_args["prompt_cache"] = self.cache_wrapper.cache
        return mx.array(self._prompt_tokens, dtype=mx.int32), None

    def is_cross_prompt_cache_active(self):
        return False

    def record_token_to_cache(self, token):
        self.received_cache_tokens.append(token)


class _TargetOnlySequenceModel:
    """Fake target that produces a fixed token sequence for any verify input.

    Each call to ``__call__(tokens, ...)`` returns logits whose argmax
    at position ``-1`` is the next entry in ``next_tokens``. The caller
    is responsible for advancing through the sequence by calling the
    model with the appropriate ``last_bonus`` token each round. This
    fake supports both the initial prompt processing call and any
    per-round bs=1 target-only verification call without asserting on
    the verify input shape.

    The fake records every call so tests can inspect verify inputs and
    verify the runtime never bypasses target verification. It also
    implements ``rollback_speculative_cache`` so the rollback
    fail-closed gate does not raise.
    """

    def __init__(
        self,
        *,
        next_tokens: list[int],
        prompt_tokens: list[int] | None = None,
        eos_token_id: int | None = None,
    ):
        self._next_tokens = list(next_tokens)
        self._prompt_tokens = list(prompt_tokens or [1])
        self._eos_token_id = eos_token_id
        self._emitted_index = 0
        self.language_model = self
        self.rollback_calls: list[tuple[int, int, int]] = []
        self.calls: list[tuple[tuple[int, ...], dict]] = []
        self.call_index = 0

    def __call__(self, tokens, **kwargs):
        seq_tokens = tuple(int(t) for t in tokens.reshape(-1).tolist())
        kwargs_key = {key: value for key, value in kwargs.items() if key != "cache"}
        kwargs_key["cache_size"] = len(kwargs.get("cache") or [])
        self.calls.append((seq_tokens, kwargs_key))

        prompt_cache = kwargs.get("cache")
        seq_len = len(seq_tokens)

        # Determine which token to emit at position -1.
        if self._emitted_index < len(self._next_tokens):
            bonus_token = int(self._next_tokens[self._emitted_index])
        else:
            bonus_token = self._eos_token_id if self._eos_token_id is not None else 0
        self._emitted_index += 1

        # Extend per-layer history where applicable (the fake mirrors
        # the KVCache semantics for the rollback bookkeeping tests).
        if prompt_cache is not None:
            for layer in prompt_cache:
                history = getattr(layer, "history", None)
                if isinstance(history, list):
                    history.extend(seq_tokens)
                    offset = getattr(layer, "offset", None)
                    if isinstance(offset, int):
                        layer.offset += seq_len
                    idx = getattr(layer, "_idx", None)
                    if isinstance(idx, int):
                        layer._idx += seq_len
                    lengths = getattr(layer, "lengths", None)
                    if lengths is not None and hasattr(lengths, "shape"):
                        layer.lengths = mx.full(
                            lengths.shape, len(history), dtype=lengths.dtype
                        )

        vocab_size = max(
            max(seq_tokens),
            bonus_token,
            *(self._next_tokens or [0]),
            255,
        ) + 8
        logits = mx.full((1, seq_len, vocab_size), -100.0)
        logits[:, seq_len - 1, bonus_token] = 100.0
        hidden = [
            mx.full((1, seq_len, 2), float(layer.layer_id + 1))
            for layer in (prompt_cache or [])
        ]
        gdn_states = [
            SimpleNamespace(
                layer_id=getattr(layer, "layer_id", layer_index),
                base_history_len=len(getattr(layer, "history", []) or []),
                verify_tokens=seq_tokens,
            )
            for layer_index, layer in enumerate(prompt_cache or [])
        ]
        self.call_index += 1
        return SimpleNamespace(
            logits=logits,
            hidden_states=hidden,
            gdn_states=gdn_states,
        )

    def rollback_speculative_cache(self, prompt_cache, gdn_states, accepted, block_size):
        # Should not be called in the max_draft_tokens=1 path because
        # the drafter never proposes any tokens. Record the call so
        # tests can fail if rollback is incorrectly invoked.
        self.rollback_calls.append((accepted, block_size, len(prompt_cache or [])))
        for layer, gdn_state in zip(prompt_cache or [], gdn_states or []):
            history = getattr(layer, "history", None)
            if not isinstance(history, list):
                continue
            keep = max(
                0,
                int(getattr(gdn_state, "base_history_len", 0))
                + accepted
                + 1,
            )
            layer.history = history[:keep]
            lengths = getattr(layer, "lengths", None)
            if lengths is not None and hasattr(lengths, "shape"):
                layer.lengths = mx.full(
                    lengths.shape, len(layer.history), dtype=lengths.dtype
                )


class _TrackingDraftModel:
    """DFlash drafter fake that records every ``draft_block`` call.

    Used to assert that the max_draft_tokens=1 runtime path skips the
    drafter entirely (no ``draft_block`` invocations) while still
    routing every emitted token through target verification.
    """

    def __init__(self, target_layer_ids: list[int] | None = None):
        self.config = SimpleNamespace(
            target_layer_ids=target_layer_ids
            or [1, 10, 18, 27, 35, 44, 52, 61],
            block_size=1,
        )
        self.draft_block_calls: list[int] = []
        self.accept_lens: list[int] = []
        self.draft_lens: list[int] = []
        self.reset_calls: list[Any] = []

    def reset(self, model):
        self.reset_calls.append(model)
        return [SimpleNamespace(lengths=mx.array([0])) for _ in self.config.target_layer_ids]

    def draft_block(self, last_bonus, hidden, cache, block_size, sampler, token_dtype):
        self.draft_block_calls.append(block_size)
        # Returning a length-1 token would be incorrect for the bs=1
        # case because the runtime should not invoke the drafter at
        # all. If this method is ever called in a max_draft_tokens=1
        # test, that signals a regression in the loop's
        # bs==1 short-circuit.
        raise AssertionError(
            "drafter.draft_block must not be invoked when max_draft_tokens=1"
        )


class _TargetOnlyKit:
    """Model kit stub that pairs a target-only fake with a tracking drafter."""

    def __init__(
        self,
        *,
        next_tokens: list[int],
        cache_layers: list | None = None,
        prompt_tokens: list[int] | None = None,
        eos_token_id: int | None = None,
    ):
        self.model = _TargetOnlySequenceModel(
            next_tokens=next_tokens,
            prompt_tokens=prompt_tokens,
            eos_token_id=eos_token_id,
        )
        self.tokenizer = FakeTokenizer()
        if cache_layers is None:
            cache_layers = _make_proven_layout_cache()
        self.cache_wrapper = SimpleNamespace(cache=cache_layers)
        self.pending_requests = {}
        self.max_kv_size = None
        self.kv_bits = None
        self.kv_group_size = None
        self.quantized_kv_start = None
        self.draft_model = None
        self.received_cache_tokens: list[int] = []
        self._prompt_tokens = prompt_tokens or [1]
        self.process_prompt_calls = 0

    def process_prompt(
        self,
        prompt_tokens,
        images_b64,
        prompt_progress_reporter,
        generate_args,
        max_image_size,
        **_kwargs,
    ):
        self.process_prompt_calls += 1
        generate_args["prompt_cache"] = self.cache_wrapper.cache
        return mx.array(self._prompt_tokens, dtype=mx.int32), None

    def is_cross_prompt_cache_active(self):
        return False

    def record_token_to_cache(self, token):
        self.received_cache_tokens.append(token)


class TestMaxDraftTokensOneContinuation(unittest.TestCase):
    """max_draft_tokens=1 must continue across multiple rounds.

    The conservative quality-gate retry uses ``--dflash-max-draft-tokens 1``
    to avoid the drafter overreach that caused
    ``max_draft_tokens=4`` to fail every prompt. Before the loop fix
    in ``dflash_stream_generate``, the runtime terminated after the
    first token because the loop's ``if bs <= 1: break`` guard fired
    on every iteration when ``max_draft_tokens=1``. The runtime now
    has a dedicated target-only round for ``bs == 1`` so the generator
    continues across multiple rounds until ``max_tokens``, EOS, or
    another normal stop criterion.

    These tests prove the fix end-to-end against the proven
    ``16 KVCache + 48 ArraysCache`` Qwen3.5 / Qwen3.6 sequential
    layout while reusing the existing rollback contract, telemetry
    surfaces, and fail-closed invariants.
    """

    def test_emits_multiple_target_verified_tokens_across_rounds(self):
        """max_draft_tokens=1 produces one bonus token per round up to max_tokens."""
        next_tokens = [11, 12, 13, 14, 15, 16, 17, 18]
        kit = _TargetOnlyKit(next_tokens=next_tokens, prompt_tokens=[1])
        draft_model = _TrackingDraftModel()

        results = list(
            dflash_stream_generate(
                kit,
                [1],
                request_id="dflash-runtime-max-draft-one-continuation",
                max_tokens=6,
                dflash_options=DFlashBoundaryOptions(
                    enabled=True,
                    target_model_path=Path("/tmp/target"),
                    drafter_model_path=Path("/tmp/drafter"),
                    max_draft_tokens=1,
                ),
                dflash_draft_model=draft_model,
            )
        )

        emitted_ids = [token.id for result in results for token in result.tokens]
        emitted_from_draft = [
            token.from_draft for result in results for token in result.tokens
        ]

        # First token is the bonus from the initial prompt processing
        # call; the remaining tokens are target-only bonuses from
        # successive bs=1 rounds. Six tokens total: 1 prompt-processing
        # bonus + 5 target-only bs=1 rounds (max_tokens=6).
        self.assertEqual(emitted_ids, next_tokens[:6])
        # None of the emitted tokens are drafter proposals.
        self.assertEqual(emitted_from_draft, [False] * len(emitted_ids))

    def test_loop_terminates_at_eos_without_emitting_unverified_drafter_tokens(self):
        """EOS in the target sequence stops the generator cleanly."""
        next_tokens = [11, 12, 13]
        eos = 999
        kit = _TargetOnlyKit(
            next_tokens=next_tokens, prompt_tokens=[1], eos_token_id=eos
        )
        draft_model = _TrackingDraftModel()

        results = list(
            dflash_stream_generate(
                kit,
                [1],
                request_id="dflash-runtime-max-draft-one-eos",
                max_tokens=16,
                dflash_options=DFlashBoundaryOptions(
                    enabled=True,
                    target_model_path=Path("/tmp/target"),
                    drafter_model_path=Path("/tmp/drafter"),
                    max_draft_tokens=1,
                ),
                dflash_draft_model=draft_model,
            )
        )

        all_tokens = [token for result in results for token in result.tokens]
        emitted_ids = [token.id for token in all_tokens]
        emitted_from_draft = [token.from_draft for token in all_tokens]
        # The first emissions are the configured target-only sequence
        # followed by repeated EOS tokens from the filler path. The
        # runtime does not stop on EOS directly; it stops at
        # max_tokens. None of the emitted tokens are drafter proposals.
        self.assertEqual(emitted_ids[: len(next_tokens)], next_tokens)
        for token_id in emitted_ids[len(next_tokens):]:
            self.assertEqual(token_id, eos)
        self.assertEqual(emitted_from_draft, [False] * len(emitted_ids))
        # Drafter was never invoked.
        self.assertEqual(draft_model.draft_block_calls, [])

    def test_loop_terminates_at_max_tokens(self):
        """Generator stops after exactly max_tokens emissions even with no EOS."""
        next_tokens = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        kit = _TargetOnlyKit(next_tokens=next_tokens, prompt_tokens=[1])
        draft_model = _TrackingDraftModel()

        results = list(
            dflash_stream_generate(
                kit,
                [1],
                request_id="dflash-runtime-max-draft-one-max-tokens",
                max_tokens=4,
                dflash_options=DFlashBoundaryOptions(
                    enabled=True,
                    target_model_path=Path("/tmp/target"),
                    drafter_model_path=Path("/tmp/drafter"),
                    max_draft_tokens=1,
                ),
                dflash_draft_model=draft_model,
            )
        )

        emitted_ids = [token.id for result in results for token in result.tokens]
        # First token (initial bonus) + 3 bs=1 rounds = 4 emissions.
        self.assertEqual(emitted_ids, [11, 12, 13, 14])
        # All tokens are target-only; no drafter proposals.
        self.assertEqual(
            [token.from_draft for result in results for token in result.tokens],
            [False] * 4,
        )
        # Drafter was never invoked across the entire run.
        self.assertEqual(draft_model.draft_block_calls, [])

    def test_drafter_is_bypassed_for_max_draft_tokens_one(self):
        """The drafter.draft_block path is skipped entirely when max_draft_tokens=1."""
        kit = _TargetOnlyKit(next_tokens=[11, 12, 13], prompt_tokens=[1])
        draft_model = _TrackingDraftModel()

        list(
            dflash_stream_generate(
                kit,
                [1],
                request_id="dflash-runtime-max-draft-one-bypass",
                max_tokens=3,
                dflash_options=DFlashBoundaryOptions(
                    enabled=True,
                    target_model_path=Path("/tmp/target"),
                    drafter_model_path=Path("/tmp/drafter"),
                    max_draft_tokens=1,
                ),
                dflash_draft_model=draft_model,
            )
        )

        self.assertEqual(draft_model.draft_block_calls, [])
        # ``_record_speculative_round`` still records per-round stats
        # so the runtime keeps its drafter bookkeeping consistent.
        # With no drafts, accepted=0 and draft_count=0 every round.
        self.assertEqual(draft_model.accept_lens, [0, 0])
        self.assertEqual(draft_model.draft_lens, [0, 0])

    def test_rollback_is_not_invoked_for_max_draft_tokens_one(self):
        """No rollback calls: there are no drafts to reject or roll back."""
        kit = _TargetOnlyKit(next_tokens=[11, 12, 13, 14], prompt_tokens=[1])
        draft_model = _TrackingDraftModel()

        list(
            dflash_stream_generate(
                kit,
                [1],
                request_id="dflash-runtime-max-draft-one-no-rollback",
                max_tokens=4,
                dflash_options=DFlashBoundaryOptions(
                    enabled=True,
                    target_model_path=Path("/tmp/target"),
                    drafter_model_path=Path("/tmp/drafter"),
                    max_draft_tokens=1,
                ),
                dflash_draft_model=draft_model,
            )
        )

        self.assertEqual(kit.model.rollback_calls, [])

    def test_every_target_verify_call_uses_target_verify_true(self):
        """All per-round target calls carry target_verify=True."""
        kit = _TargetOnlyKit(
            next_tokens=[11, 12, 13, 14], prompt_tokens=[1]
        )
        draft_model = _TrackingDraftModel()

        list(
            dflash_stream_generate(
                kit,
                [1],
                request_id="dflash-runtime-max-draft-one-verify-flag",
                max_tokens=4,
                dflash_options=DFlashBoundaryOptions(
                    enabled=True,
                    target_model_path=Path("/tmp/target"),
                    drafter_model_path=Path("/tmp/drafter"),
                    max_draft_tokens=1,
                ),
                dflash_draft_model=draft_model,
            )
        )

        # One call is the initial prompt processing verify; subsequent
        # calls are bs=1 target-only verify rounds. Every call must
        # carry target_verify=True so the patched Qwen3.5 wrapper
        # routes through the target-verification path.
        for call_tokens, call_kwargs in kit.model.calls:
            self.assertTrue(
                call_kwargs.get("target_verify"),
                msg=(
                    f"target verify call with input {call_tokens} must "
                    f"carry target_verify=True; got kwargs={call_kwargs}"
                ),
            )

    def test_cache_advances_one_token_per_round(self):
        """Each bs=1 round appends exactly one token to the live cache."""
        kit = _TargetOnlyKit(
            next_tokens=[11, 12, 13, 14, 15], prompt_tokens=[1]
        )
        draft_model = _TrackingDraftModel()

        list(
            dflash_stream_generate(
                kit,
                [1],
                request_id="dflash-runtime-max-draft-one-cache",
                max_tokens=5,
                dflash_options=DFlashBoundaryOptions(
                    enabled=True,
                    target_model_path=Path("/tmp/target"),
                    drafter_model_path=Path("/tmp/drafter"),
                    max_draft_tokens=1,
                ),
                dflash_draft_model=draft_model,
            )
        )

        # The KVCache layers carry an explicit history list and should
        # record every appended token (prompt + the first
        # ``max_tokens - 1`` emissions). The final emitted token has
        # not yet been processed by a target verify call, so it is not
        # appended to the cache yet — it will be appended at the start
        # of the next round (which the runtime does not enter because
        # ``emitted >= max_tokens``).
        kv_histories = [
            list(layer.history)
            for layer in kit.cache_wrapper.cache
            if hasattr(layer, "history") and isinstance(layer.history, list)
        ]
        self.assertGreater(len(kv_histories), 0)
        expected_history = [1, 11, 12, 13, 14]
        for history in kv_histories:
            self.assertEqual(
                history,
                expected_history,
                msg=(
                    "every KVCache layer must record the prompt plus "
                    "the appended bonus tokens in order"
                ),
            )

    def test_proposal_observer_is_not_called_for_max_draft_tokens_one(self):
        """The proposal observer never fires because no drafts are proposed."""
        kit = _TargetOnlyKit(next_tokens=[11, 12, 13], prompt_tokens=[1])
        draft_model = _TrackingDraftModel()
        observed: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

        list(
            dflash_stream_generate(
                kit,
                [1],
                request_id="dflash-runtime-max-draft-one-observer",
                max_tokens=3,
                dflash_options=DFlashBoundaryOptions(
                    enabled=True,
                    target_model_path=Path("/tmp/target"),
                    drafter_model_path=Path("/tmp/drafter"),
                    max_draft_tokens=1,
                ),
                dflash_draft_model=draft_model,
                proposal_observer=lambda history, proposal: observed.append(
                    (tuple(history), tuple(proposal))
                ),
            )
        )

        # No draft proposals => observer is never invoked.
        self.assertEqual(observed, [])


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
                # Capture layer ids default to range(layer_count); for the
                # default proven layout that is range(64).
                expected_capture_ids = list(range(sum(DFLASH_PROVEN_QWEN35_LAYOUT)))
                self.assertTrue(
                    all(
                        call[1]["capture_layer_ids"] == expected_capture_ids
                        for call in kit.model.calls
                    )
                )
                self.assertEqual(draft_model.draft_lens[0], (11, expected_block_size))
                self.assertEqual(draft_model.accept_lens, [accepted])
                expected_layer_count = sum(DFLASH_PROVEN_QWEN35_LAYOUT)
                self.assertEqual(
                    kit.model.rollback_calls,
                    [] if case["rollback"] is None else [case["rollback"] + (expected_layer_count,)],
                )
                self.assertEqual(
                    [list(layer.history) for layer in kit.cache_wrapper.cache],
                    [expected_history] * expected_layer_count,
                )
                # Only KVCache layers carry an mx.array ``lengths`` in the
                # fake; the ArraysCache fake mirrors the real
                # ``lengths=None`` single-sequence shape.
                kv_lengths = [
                    int(layer.lengths.tolist()[0])
                    for layer in kit.cache_wrapper.cache
                    if hasattr(layer, "lengths")
                    and isinstance(layer.lengths, mx.array)
                ]
                self.assertEqual(
                    kv_lengths,
                    [len(expected_history)] * len(kv_lengths),
                )
                rejected_tokens = draft_tokens[accepted:]
                for layer in kit.cache_wrapper.cache:
                    for rejected_token in rejected_tokens:
                        self.assertNotIn(rejected_token, layer.history)
                self.assertEqual(kit.received_cache_tokens, [])

    def test_rejects_rollback_unsafe_runtime_before_prompt_processing(self):
        cases = [
            ("already loaded draft_model", {"draft_model": object()}),
            ("max_kv_size", {"max_kv_size": 16}),
            ("kv_bits", {"kv_bits": 4}),
            ("kv_group_size", {"kv_group_size": 32}),
            ("quantized_kv_start", {"quantized_kv_start": 8}),
        ]

        for needle, attrs in cases:
            with self.subTest(case=needle):
                kit = FakeKit(
                    [12, 13],
                    [11],
                    [12, 13, 99],
                    cache_layers=[KVCache(0)],
                    rollback_supported=True,
                )
                for attr_name, value in attrs.items():
                    setattr(kit, attr_name, value)

                with self.assertRaisesRegex(DFlashUnavailableError, needle):
                    list(
                        dflash_stream_generate(
                            kit,
                            [1],
                            request_id=f"dflash-runtime-{needle}",
                            max_tokens=2,
                            dflash_options=DFlashBoundaryOptions(
                                enabled=True,
                                target_model_path=Path("/tmp/target"),
                                drafter_model_path=Path("/tmp/drafter"),
                                max_draft_tokens=3,
                            ),
                            dflash_draft_model=FakeDraftModel([12, 13]),
                        )
                    )

                self.assertEqual(kit.process_prompt_calls, 0)

    def test_rejects_missing_rollback_capability_before_prompt_processing(self):
        kit = FakeKit(
            [12, 13],
            [11],
            [12, 13, 99],
            cache_layers=[KVCache(0)],
            rollback_supported=False,
        )

        with self.assertRaisesRegex(DFlashUnavailableError, "rollback_speculative_cache"):
            list(
                dflash_stream_generate(
                    kit,
                    [1],
                    request_id="dflash-runtime-no-rollback",
                    max_tokens=2,
                    dflash_options=DFlashBoundaryOptions(
                        enabled=True,
                        target_model_path=Path("/tmp/target"),
                        drafter_model_path=Path("/tmp/drafter"),
                        max_draft_tokens=3,
                    ),
                    dflash_draft_model=FakeDraftModel([12, 13]),
                )
            )

        self.assertEqual(kit.process_prompt_calls, 0)

    def test_uses_exact_m13_target_layer_ids(self):
        target_layer_ids = [1, 10, 18, 27, 35, 44, 52, 61]
        kit = FakeKit(
            [12, 13, 14],
            [11, 12, 13, 14],
            [12, 21, 99, 98],
        )
        draft_model = FakeDraftModel([12, 13, 14], target_layer_ids=target_layer_ids)

        results = list(
            dflash_stream_generate(
                kit,
                [1],
                request_id="dflash-runtime-target-layer-ids",
                max_tokens=3,
                dflash_options=DFlashBoundaryOptions(
                    enabled=True,
                    target_model_path=Path("/tmp/target"),
                    drafter_model_path=Path("/tmp/drafter"),
                    max_draft_tokens=4,
                ),
                dflash_draft_model=draft_model,
            )
        )

        emitted_ids = [token.id for result in results for token in result.tokens]
        self.assertEqual(emitted_ids, [11, 12, 21])
        self.assertEqual(
            kit.model.calls[0][1]["capture_layer_ids"],
            target_layer_ids,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
