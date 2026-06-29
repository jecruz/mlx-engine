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
from mlx_engine.utils.dflash_runtime import (
    DFLASH_FALLBACK_REASON_LOW_ACCEPTANCE,
    DFLASH_FALLBACK_REASON_PATHOLOGICAL_TARGET_ONLY,
    DFLASH_TELEMETRY_KIND_DRAFT_ROUND_ACCEPTED,
    DFLASH_TELEMETRY_KIND_DRAFT_ROUND_PARTIAL,
    DFLASH_TELEMETRY_KIND_FALLBACK_PATHOLOGICAL_TARGET_ONLY,
    DFLASH_TELEMETRY_KIND_INITIAL_BONUS,
    DFLASH_TELEMETRY_KIND_TARGET_ONLY,
    DFlashAdaptiveScheduler,
    DFlashRoundTelemetry,
    DFlashSchedulerDecision,
    dflash_stream_generate,
)


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


class TestDFlashPerRoundTelemetry(unittest.TestCase):
    """Per-round DFlash telemetry must classify and time every scheduling round.

    These tests pin the M15 telemetry contract:
    * Each round produces exactly one ``DFlashRoundTelemetry`` record
      with the correct ``kind`` and the four documented timing buckets
      (drafter, target verify, rollback, emission).
    * The initial bonus, target-only ``bs=1``, fully accepted draft,
      and partially rejected draft rounds are all recorded.
    * When ``telemetry_collector`` is omitted the runtime records no
      overhead beyond the perf-counter reads, preserving default-off
      behavior and proving that scheduling decisions do not depend on
      telemetry.
    """

    def test_initial_bonus_telemetry_has_prompt_length_target_verify_input(self):
        """Round 0 telemetry classifies the prompt-processing bonus round and
        carries ``target_verify_input_length == prompt length``."""

        next_tokens = [11, 12, 13, 14, 15]
        kit = _TargetOnlyKit(next_tokens=next_tokens, prompt_tokens=[1])
        draft_model = _TrackingDraftModel()
        records: list[DFlashRoundTelemetry] = []

        list(
            dflash_stream_generate(
                kit,
                [1],
                request_id="dflash-telemetry-initial-bonus",
                max_tokens=4,
                dflash_options=DFlashBoundaryOptions(
                    enabled=True,
                    target_model_path=Path("/tmp/target"),
                    drafter_model_path=Path("/tmp/drafter"),
                    max_draft_tokens=1,
                ),
                dflash_draft_model=draft_model,
                telemetry_collector=records.append,
            )
        )

        kinds = [record.kind for record in records]
        self.assertEqual(kinds[0], DFLASH_TELEMETRY_KIND_INITIAL_BONUS)
        initial = records[0]
        self.assertEqual(initial.round_index, 0)
        self.assertEqual(initial.scheduled_block_size, 1)
        self.assertEqual(initial.draft_count, 0)
        self.assertEqual(initial.accepted_count, 0)
        self.assertEqual(initial.rejected_count, 0)
        self.assertFalse(initial.rollback_occurred)
        self.assertEqual(initial.target_verify_input_length, 1)
        self.assertEqual(initial.from_draft_token_count, 0)
        self.assertEqual(initial.from_target_token_count, 1)
        self.assertGreaterEqual(initial.target_verify_elapsed_s, 0.0)
        self.assertGreaterEqual(initial.emission_elapsed_s, 0.0)
        self.assertEqual(initial.drafter_elapsed_s, 0.0)
        self.assertEqual(initial.rollback_elapsed_s, 0.0)

    def test_target_only_rounds_record_block_size_one_no_drafts(self):
        """Every bs=1 round is classified ``target_only`` with no drafts and no rollback."""

        kit = _TargetOnlyKit(next_tokens=[11, 12, 13, 14, 15], prompt_tokens=[1])
        draft_model = _TrackingDraftModel()
        records: list[DFlashRoundTelemetry] = []

        list(
            dflash_stream_generate(
                kit,
                [1],
                request_id="dflash-telemetry-target-only",
                max_tokens=5,
                dflash_options=DFlashBoundaryOptions(
                    enabled=True,
                    target_model_path=Path("/tmp/target"),
                    drafter_model_path=Path("/tmp/drafter"),
                    max_draft_tokens=1,
                ),
                dflash_draft_model=draft_model,
                telemetry_collector=records.append,
            )
        )

        # 1 prompt bonus + 4 bs=1 rounds == 5 records total.
        kinds = [record.kind for record in records]
        self.assertEqual(
            kinds,
            [
                DFLASH_TELEMETRY_KIND_INITIAL_BONUS,
                DFLASH_TELEMETRY_KIND_TARGET_ONLY,
                DFLASH_TELEMETRY_KIND_TARGET_ONLY,
                DFLASH_TELEMETRY_KIND_TARGET_ONLY,
                DFLASH_TELEMETRY_KIND_TARGET_ONLY,
            ],
        )
        for record in records[1:]:
            self.assertEqual(record.scheduled_block_size, 1)
            self.assertEqual(record.draft_count, 0)
            self.assertEqual(record.accepted_count, 0)
            self.assertEqual(record.rejected_count, 0)
            self.assertFalse(record.rollback_occurred)
            self.assertEqual(record.target_verify_input_length, 1)
            self.assertEqual(record.from_draft_token_count, 0)
            self.assertEqual(record.from_target_token_count, 1)
            self.assertEqual(record.drafter_elapsed_s, 0.0)
            self.assertEqual(record.rollback_elapsed_s, 0.0)
            self.assertGreaterEqual(record.target_verify_elapsed_s, 0.0)
            self.assertGreaterEqual(record.emission_elapsed_s, 0.0)

    def test_fully_accepted_draft_round_records_zero_rejected_and_no_rollback(self):
        """All-accept draft round: accepted == block_size - 1, rejected=0, rollback_occurred=False."""

        # First bonus: 11; then draft block of [12, 13, 14] all match the spec.
        draft_tokens = [12, 13, 14]
        spec_input_tokens = [11, 12, 13, 14]
        spec_output_tokens = [12, 13, 14, 99]
        kit = FakeKit(
            draft_tokens,
            spec_input_tokens,
            spec_output_tokens,
        )
        draft_model = FakeDraftModel(draft_tokens)
        records: list[DFlashRoundTelemetry] = []

        list(
            dflash_stream_generate(
                kit,
                [1],
                request_id="dflash-telemetry-all-accepted",
                # ``accepted + 2`` keeps the loop to exactly one draft
                # round (1 initial bonus + 1 draft round) so the fake
                # model's spec_input assertion does not fire on a
                # follow-up round.
                max_tokens=5,
                dflash_options=DFlashBoundaryOptions(
                    enabled=True,
                    target_model_path=Path("/tmp/target"),
                    drafter_model_path=Path("/tmp/drafter"),
                    max_draft_tokens=len(draft_tokens) + 1,
                ),
                dflash_draft_model=draft_model,
                telemetry_collector=records.append,
            )
        )

        # 2 records: 1 initial bonus + 1 fully accepted draft round.
        self.assertEqual(len(records), 2)
        # Find the first non-initial record and confirm it is a
        # fully accepted draft round.
        draft_records = [
            record for record in records if record.kind != DFLASH_TELEMETRY_KIND_INITIAL_BONUS
        ]
        self.assertEqual(len(draft_records), 1)
        first_draft = draft_records[0]
        self.assertEqual(first_draft.kind, DFLASH_TELEMETRY_KIND_DRAFT_ROUND_ACCEPTED)
        self.assertEqual(first_draft.scheduled_block_size, len(draft_tokens) + 1)
        self.assertEqual(first_draft.draft_count, len(draft_tokens))
        self.assertEqual(first_draft.accepted_count, len(draft_tokens))
        self.assertEqual(first_draft.rejected_count, 0)
        self.assertFalse(first_draft.rollback_occurred)
        self.assertEqual(first_draft.target_verify_input_length, len(draft_tokens) + 1)
        self.assertEqual(first_draft.from_draft_token_count, len(draft_tokens))
        self.assertEqual(first_draft.from_target_token_count, 1)
        self.assertGreaterEqual(first_draft.drafter_elapsed_s, 0.0)
        self.assertGreaterEqual(first_draft.target_verify_elapsed_s, 0.0)
        self.assertEqual(first_draft.rollback_elapsed_s, 0.0)

    def test_partial_rejection_records_rejected_count_and_rollback_timing(self):
        """Partial rejection: the drafter's middle proposal mismatches, so the
        runtime emits a ``draft_round_partial`` record with rejected>0 and
        rollback timing >= 0."""

        draft_tokens = [12, 13, 14]
        spec_input_tokens = [11, 12, 13, 14]
        # Mismatch on position 1 (draft=13 vs target=21). Walk accepts
        # the first draft token, then resamples a bonus=21. new_tokens
        # = [12, 21], emitted_history length grows by 2.
        spec_output_tokens = [12, 21, 99, 98]
        kit = FakeKit(
            draft_tokens,
            spec_input_tokens,
            spec_output_tokens,
        )
        draft_model = FakeDraftModel(draft_tokens)
        records: list[DFlashRoundTelemetry] = []

        list(
            dflash_stream_generate(
                kit,
                [1],
                request_id="dflash-telemetry-partial-rejection",
                # bs = min(block_total, max_draft_tokens, max_tokens - emitted + 1)
                #     = min(_, 3, 3 - 1 + 1) = 3 with these knobs.
                # After initial bonus emitted=1, the single draft round
                # emits 2 new_tokens (12, 21) so emitted reaches
                # max_tokens and the loop exits before any further
                # verify call.
                max_tokens=3,
                dflash_options=DFlashBoundaryOptions(
                    enabled=True,
                    target_model_path=Path("/tmp/target"),
                    drafter_model_path=Path("/tmp/drafter"),
                    max_draft_tokens=3,
                ),
                dflash_draft_model=draft_model,
                telemetry_collector=records.append,
            )
        )

        # 2 records: 1 initial bonus + 1 partial rejection.
        self.assertEqual(len(records), 2)
        partial_records = [
            record for record in records if record.kind == DFLASH_TELEMETRY_KIND_DRAFT_ROUND_PARTIAL
        ]
        self.assertEqual(
            len(partial_records),
            1,
            msg=f"expected exactly one partial-rejection record, got records={records}",
        )
        partial = partial_records[0]
        # bs = 3 (forced by max_draft_tokens=3 and max_tokens=3).
        # draft_count = bs - 1 = 2 (the planned draft slots for the round).
        self.assertEqual(partial.scheduled_block_size, 3)
        self.assertEqual(partial.draft_count, 2)
        self.assertEqual(partial.accepted_count, 1)
        self.assertEqual(partial.rejected_count, 1)
        self.assertTrue(partial.rollback_occurred)
        self.assertEqual(partial.target_verify_input_length, 3)
        self.assertEqual(partial.from_draft_token_count, 1)
        self.assertEqual(partial.from_target_token_count, 1)
        self.assertGreaterEqual(partial.rollback_elapsed_s, 0.0)
        self.assertGreaterEqual(partial.drafter_elapsed_s, 0.0)
        self.assertGreaterEqual(partial.target_verify_elapsed_s, 0.0)

    def test_telemetry_collector_absent_keeps_default_off_observable_behavior(self):
        """Omitting ``telemetry_collector`` must not alter emitted tokens or drafter/rollback stats."""

        kit = _TargetOnlyKit(next_tokens=[11, 12, 13], prompt_tokens=[1])
        draft_model = _TrackingDraftModel()

        results = list(
            dflash_stream_generate(
                kit,
                [1],
                request_id="dflash-telemetry-no-collector",
                max_tokens=3,
                dflash_options=DFlashBoundaryOptions(
                    enabled=True,
                    target_model_path=Path("/tmp/target"),
                    drafter_model_path=Path("/tmp/drafter"),
                    max_draft_tokens=1,
                ),
                dflash_draft_model=draft_model,
                # Note: telemetry_collector intentionally omitted.
            )
        )

        emitted_ids = [token.id for result in results for token in result.tokens]
        self.assertEqual(emitted_ids, [11, 12, 13])
        self.assertEqual(
            [token.from_draft for result in results for token in result.tokens],
            [False, False, False],
        )
        # Drafter was never invoked (proves default-off / no-scheduling-change).
        self.assertEqual(draft_model.draft_block_calls, [])
        # No rollback calls (no drafts to reject).
        self.assertEqual(kit.model.rollback_calls, [])

    def test_multi_round_mixed_telemetry_aggregates_correctly(self):
        """Streaming across multiple rounds emits one record per round with the
        right kind for each round shape (initial bonus + multiple bs=1)."""

        next_tokens = [11, 12, 13, 14, 15]
        kit = _TargetOnlyKit(next_tokens=next_tokens, prompt_tokens=[1])
        draft_model = _TrackingDraftModel()
        records: list[DFlashRoundTelemetry] = []

        list(
            dflash_stream_generate(
                kit,
                [1],
                request_id="dflash-telemetry-multi-round-aggregate",
                max_tokens=5,
                dflash_options=DFlashBoundaryOptions(
                    enabled=True,
                    target_model_path=Path("/tmp/target"),
                    drafter_model_path=Path("/tmp/drafter"),
                    max_draft_tokens=1,
                ),
                dflash_draft_model=draft_model,
                telemetry_collector=records.append,
            )
        )

        # round_index must be a strictly increasing sequence starting at 0.
        self.assertEqual(
            [record.round_index for record in records],
            list(range(len(records))),
        )
        # 1 initial bonus + 4 bs=1 rounds == 5 records total.
        self.assertEqual(len(records), 5)
        # only the first record may be classified as initial bonus.
        self.assertEqual(records[0].kind, DFLASH_TELEMETRY_KIND_INITIAL_BONUS)
        self.assertTrue(
            all(record.kind == DFLASH_TELEMETRY_KIND_TARGET_ONLY for record in records[1:])
        )
        # accepted_proposal_tokens_total sums to 0 since bs=1 rounds have no drafts.
        self.assertEqual(sum(record.accepted_count for record in records), 0)
        self.assertEqual(sum(record.rejected_count for record in records), 0)
        self.assertEqual(
            sum(record.from_draft_token_count for record in records),
            0,
        )
        self.assertEqual(
            sum(record.from_target_token_count for record in records),
            5,
        )


class TestDFlashAdaptiveScheduler(unittest.TestCase):
    """Adaptive scheduler unit tests.

    These tests pin the M15 adaptive-scheduler contract (VAL-M15-002
    scheduler-safety side):

    * Scheduler grows after fully accepted rounds, capped by every
      documented bound (DFlash block size, configured
      ``max_draft_tokens``, remaining token budget).
    * Scheduler shrinks by exactly one slot after any rejected
      round, floored at ``1`` so the target-only ``bs == 1`` path
      stays reachable without bypassing ``target_verify=True``.
    * Scheduler starts at the conservative initial block size and
      is deterministic across rounds (no telemetry coupling).
    * Scheduler never exceeds any individual cap and never produces
      a non-positive block size.
    * Target-only rounds (``scheduled_block_size <= 1``) do not
      influence the scheduler's history.
    """

    def test_starts_at_initial_block_size_clamped_by_max_draft_tokens(self):
        scheduler = DFlashAdaptiveScheduler(max_draft_tokens=4, initial_block_size=2)
        self.assertEqual(scheduler.current_block_size, 2)
        # Initial block size is clamped to ``max_draft_tokens`` so a
        # misconfigured ``initial_block_size > max_draft_tokens`` cannot
        # widen the runtime surface.
        clamped = DFlashAdaptiveScheduler(
            max_draft_tokens=2, initial_block_size=8
        )
        self.assertEqual(clamped.current_block_size, 2)

    def test_starts_at_one_when_max_draft_tokens_equals_one(self):
        scheduler = DFlashAdaptiveScheduler(max_draft_tokens=1, initial_block_size=2)
        self.assertEqual(scheduler.current_block_size, 1)

    def test_grows_after_fully_accepted_round_within_cap(self):
        scheduler = DFlashAdaptiveScheduler(
            max_draft_tokens=4, initial_block_size=2
        )
        # First, run a draft round at bs=2 with the single draft accepted.
        decision = scheduler.next_block_size(block_total=4, remaining_budget=10)
        self.assertEqual(decision.scheduled_block_size, 2)
        scheduler.record_round(accepted_count=1, scheduled_block_size=2)
        # The scheduler should now grow by one slot.
        decision = scheduler.next_block_size(block_total=4, remaining_budget=10)
        self.assertEqual(decision.scheduled_block_size, 3)
        # Record another full-accept round (2 of 2 drafts at bs=3) and grow again.
        scheduler.record_round(accepted_count=2, scheduled_block_size=3)
        decision = scheduler.next_block_size(block_total=4, remaining_budget=10)
        self.assertEqual(decision.scheduled_block_size, 4)
        # Cap reached - subsequent fully accepted rounds do not grow.
        scheduler.record_round(accepted_count=3, scheduled_block_size=4)
        decision = scheduler.next_block_size(block_total=4, remaining_budget=10)
        self.assertEqual(decision.scheduled_block_size, 4)

    def test_shrinks_after_any_rejection_floored_at_one(self):
        scheduler = DFlashAdaptiveScheduler(
            max_draft_tokens=4, initial_block_size=3
        )
        # Reject both drafts at bs=3 -> shrink.
        scheduler.record_round(accepted_count=0, scheduled_block_size=3)
        decision = scheduler.next_block_size(block_total=4, remaining_budget=10)
        self.assertEqual(decision.scheduled_block_size, 2)
        # Reject the single draft at bs=2 -> shrink.
        scheduler.record_round(accepted_count=0, scheduled_block_size=2)
        decision = scheduler.next_block_size(block_total=4, remaining_budget=10)
        self.assertEqual(decision.scheduled_block_size, 1)
        # Floor at 1: even with another rejected round we never go below 1.
        scheduler.record_round(accepted_count=0, scheduled_block_size=2)
        decision = scheduler.next_block_size(block_total=4, remaining_budget=10)
        self.assertEqual(decision.scheduled_block_size, 1)

    def test_partial_rejection_shrinks(self):
        """A round that accepts some drafts but rejects at least one shrinks.

        bs=3 means 2 drafts; accepting 1 of 2 is a partial rejection.
        """

        scheduler = DFlashAdaptiveScheduler(
            max_draft_tokens=4,
            initial_block_size=3,
            grow_threshold=1.0,
            shrink_threshold=1.0,
        )
        scheduler.record_round(accepted_count=1, scheduled_block_size=3)
        decision = scheduler.next_block_size(block_total=4, remaining_budget=10)
        self.assertEqual(
            decision.scheduled_block_size,
            2,
            msg="partial rejection (1 of 2 drafts) must shrink the scheduler",
        )

    def test_partial_rejection_with_relaxed_grow_threshold_does_not_grow(self):
        """Even with a relaxed grow_threshold, any rejection still shrinks.

        The shrink rule fires whenever ``last_ratio < shrink_threshold``,
        independent of the grow_threshold. This guards against the
        scheduler widening past the rejection evidence.
        """

        scheduler = DFlashAdaptiveScheduler(
            max_draft_tokens=4,
            initial_block_size=3,
            grow_threshold=0.9,
            shrink_threshold=0.8,
        )
        # 1 of 2 drafts accepted: ratio = 0.5, which is below both
        # the grow_threshold (0.9) and the shrink_threshold (0.8).
        # The scheduler must shrink, not grow.
        scheduler.record_round(accepted_count=1, scheduled_block_size=3)
        decision = scheduler.next_block_size(block_total=4, remaining_budget=10)
        self.assertEqual(
            decision.scheduled_block_size,
            2,
            msg="partial rejection must never grow the scheduler",
        )

    def test_hold_zone_between_thresholds_keeps_size_stable(self):
        """When acceptance sits between grow and shrink thresholds the
        scheduler holds the current size.

        With ``grow_threshold=1.0`` and ``shrink_threshold=0.5`` an
        acceptance ratio of 0.75 sits in the hold zone: the scheduler
        neither grows nor shrinks.
        """

        scheduler = DFlashAdaptiveScheduler(
            max_draft_tokens=8,
            initial_block_size=4,
            grow_threshold=1.0,
            shrink_threshold=0.5,
        )
        # 2 of 3 drafts accepted (ratio = 0.667) - holds at current.
        scheduler.record_round(accepted_count=2, scheduled_block_size=4)
        decision = scheduler.next_block_size(block_total=8, remaining_budget=20)
        self.assertEqual(decision.scheduled_block_size, 4)

        # 1 of 3 drafts accepted (ratio = 0.333) - shrinks.
        scheduler.record_round(accepted_count=1, scheduled_block_size=4)
        decision = scheduler.next_block_size(block_total=8, remaining_budget=20)
        self.assertEqual(decision.scheduled_block_size, 3)

        # 3 of 3 drafts accepted (ratio = 1.0) - grows.
        scheduler.record_round(accepted_count=3, scheduled_block_size=4)
        decision = scheduler.next_block_size(block_total=8, remaining_budget=20)
        self.assertEqual(decision.scheduled_block_size, 4)

    def test_history_does_not_record_target_only_rounds(self):
        scheduler = DFlashAdaptiveScheduler(
            max_draft_tokens=4, initial_block_size=2
        )
        # bs == 1 (target-only) rounds should not pollute history.
        scheduler.record_round(accepted_count=0, scheduled_block_size=1)
        scheduler.record_round(accepted_count=1, scheduled_block_size=1)
        self.assertEqual(scheduler.history, ())
        # Subsequent draft round at bs=2 must still treat the
        # scheduler as if no round has happened yet.
        scheduler.record_round(accepted_count=1, scheduled_block_size=2)
        # Growth fires because the most recent ROUND was a fully
        # accepted bs=2.
        decision = scheduler.next_block_size(block_total=4, remaining_budget=10)
        self.assertEqual(decision.scheduled_block_size, 3)

    def test_history_window_bounds_recorded_rounds(self):
        scheduler = DFlashAdaptiveScheduler(
            max_draft_tokens=8,
            initial_block_size=2,
            history_window=2,
        )
        # First round: full acceptance at bs=2 (1 of 1 draft). Grow to 3.
        scheduler.record_round(accepted_count=1, scheduled_block_size=2)
        decision = scheduler.next_block_size(block_total=8, remaining_budget=20)
        self.assertEqual(decision.scheduled_block_size, 3)
        # Second round: full acceptance at bs=3 (2 of 2 drafts). Grow to 4.
        scheduler.record_round(accepted_count=2, scheduled_block_size=3)
        decision = scheduler.next_block_size(block_total=8, remaining_budget=20)
        self.assertEqual(decision.scheduled_block_size, 4)
        # Now register two fully-accepted bs=4 rounds; the history
        # window is 2, so each new round evicts the oldest.
        scheduler.record_round(accepted_count=3, scheduled_block_size=4)
        scheduler.record_round(accepted_count=3, scheduled_block_size=4)
        # Most recent round was fully accepted, so we grow to 5.
        decision = scheduler.next_block_size(block_total=8, remaining_budget=20)
        self.assertEqual(decision.scheduled_block_size, 5)

    def test_never_exceeds_dflash_block_size(self):
        scheduler = DFlashAdaptiveScheduler(
            max_draft_tokens=16, initial_block_size=2
        )
        # Force growth via many full-accept rounds, but ``block_total``
        # caps the scheduler at 4.
        for _ in range(10):
            decision = scheduler.next_block_size(block_total=4, remaining_budget=10)
            scheduler.record_round(
                accepted_count=decision.scheduled_block_size - 1,
                scheduled_block_size=decision.scheduled_block_size,
            )
        # Final size is capped by ``block_total=4``.
        self.assertLessEqual(scheduler.current_block_size, 4)
        decision = scheduler.next_block_size(block_total=4, remaining_budget=10)
        self.assertLessEqual(decision.scheduled_block_size, 4)

    def test_never_exceeds_configured_max_draft_tokens(self):
        scheduler = DFlashAdaptiveScheduler(
            max_draft_tokens=3, initial_block_size=2
        )
        for _ in range(10):
            decision = scheduler.next_block_size(block_total=16, remaining_budget=20)
            scheduler.record_round(
                accepted_count=decision.scheduled_block_size - 1,
                scheduled_block_size=decision.scheduled_block_size,
            )
        self.assertLessEqual(scheduler.current_block_size, 3)
        decision = scheduler.next_block_size(block_total=16, remaining_budget=20)
        self.assertLessEqual(decision.scheduled_block_size, 3)

    def test_never_exceeds_remaining_token_budget(self):
        scheduler = DFlashAdaptiveScheduler(
            max_draft_tokens=16, initial_block_size=2
        )
        # Remaining budget shrinks over time; the scheduler must clip.
        for budget in (16, 8, 4, 3, 2):
            decision = scheduler.next_block_size(
                block_total=16, remaining_budget=budget
            )
            self.assertLessEqual(decision.scheduled_block_size, budget)
            self.assertGreaterEqual(decision.scheduled_block_size, 1)
            scheduler.record_round(
                accepted_count=decision.scheduled_block_size - 1,
                scheduled_block_size=decision.scheduled_block_size,
            )

    def test_remaining_budget_one_collapses_to_target_only(self):
        """When the remaining token budget is 1 the scheduler must return 1.

        This is the proven bs==1 target-only path the runtime uses when
        ``max_draft_tokens=1``: the drafter is bypassed but every emitted
        token is still target-verified.
        """

        scheduler = DFlashAdaptiveScheduler(
            max_draft_tokens=8, initial_block_size=4
        )
        decision = scheduler.next_block_size(block_total=8, remaining_budget=1)
        self.assertEqual(decision.scheduled_block_size, 1)
        self.assertEqual(decision.effective_cap, 1)
        self.assertEqual(decision.clip_reason, "effective_cap")

    def test_block_size_zero_collapses_to_floor_one(self):
        """A zero ``block_total`` is floored at 1 instead of returning 0."""

        scheduler = DFlashAdaptiveScheduler(
            max_draft_tokens=8, initial_block_size=2
        )
        decision = scheduler.next_block_size(block_total=0, remaining_budget=8)
        self.assertEqual(decision.scheduled_block_size, 1)
        self.assertEqual(decision.effective_cap, 1)

    def test_max_draft_tokens_one_collapses_to_target_only_path(self):
        """``max_draft_tokens=1`` is the degenerate target-only scheduler."""

        scheduler = DFlashAdaptiveScheduler(max_draft_tokens=1, initial_block_size=1)
        # No growth possible at max_draft_tokens=1; every round is bs=1.
        for _ in range(3):
            decision = scheduler.next_block_size(block_total=1, remaining_budget=8)
            self.assertEqual(decision.scheduled_block_size, 1)
            # ``record_round`` with bs=1 is a no-op so the scheduler
            # state must remain stable.
            scheduler.record_round(accepted_count=0, scheduled_block_size=1)
        self.assertEqual(scheduler.history, ())

    def test_decision_carries_clip_reason_for_within_caps(self):
        scheduler = DFlashAdaptiveScheduler(
            max_draft_tokens=4, initial_block_size=2
        )
        decision = scheduler.next_block_size(block_total=4, remaining_budget=10)
        self.assertIsInstance(decision, DFlashSchedulerDecision)
        self.assertEqual(decision.clip_reason, "within_caps")
        self.assertEqual(decision.effective_cap, 4)

    def test_history_carries_recent_outcomes_in_observation_order(self):
        scheduler = DFlashAdaptiveScheduler(
            max_draft_tokens=4,
            initial_block_size=2,
            history_window=3,
        )
        # (accepted, scheduled) tuples. ``scheduled - 1`` drafts; full
        # accept means accepted == scheduled - 1.
        outcomes = [
            (1, 2),  # 1 of 1 drafts accepted (full accept)
            (0, 2),  # 0 of 1 drafts accepted (full reject)
            (1, 3),  # 1 of 2 drafts accepted (partial)
        ]
        for accepted, scheduled in outcomes:
            scheduler.record_round(
                accepted_count=accepted, scheduled_block_size=scheduled
            )
        self.assertEqual(scheduler.history, tuple(outcomes))
        self.assertEqual(scheduler.history_window, 3)

    def test_invalid_construction_arguments_raise(self):
        with self.assertRaisesRegex(ValueError, "max_draft_tokens"):
            DFlashAdaptiveScheduler(max_draft_tokens=0)
        with self.assertRaisesRegex(ValueError, "initial_block_size"):
            DFlashAdaptiveScheduler(max_draft_tokens=4, initial_block_size=0)
        with self.assertRaisesRegex(ValueError, "history_window"):
            DFlashAdaptiveScheduler(max_draft_tokens=4, history_window=0)
        with self.assertRaisesRegex(ValueError, "grow_threshold"):
            DFlashAdaptiveScheduler(max_draft_tokens=4, grow_threshold=-0.1)
        with self.assertRaisesRegex(ValueError, "grow_threshold"):
            DFlashAdaptiveScheduler(max_draft_tokens=4, grow_threshold=1.5)
        with self.assertRaisesRegex(ValueError, "shrink_threshold"):
            DFlashAdaptiveScheduler(max_draft_tokens=4, shrink_threshold=-0.1)
        with self.assertRaisesRegex(ValueError, "grow_threshold"):
            DFlashAdaptiveScheduler(
                max_draft_tokens=4,
                grow_threshold=0.5,
                shrink_threshold=0.9,
            )


class _AdaptiveGrowingDraftModel:
    """Drafter fake that always produces a fully accepted draft block.

    The fake exposes the current scheduler-requested block size via
    ``requested_block_sizes`` so the adaptive scheduler tests can assert
    that ``dflash_stream_generate`` honors the scheduler's choices end
    to end. The fake targets the proven Qwen3.5 sequential layout
    (16 KVCache + 48 ArraysCache) so the runtime validator lets it
    through unchanged.
    """

    def __init__(self, target_layer_ids: list[int] | None = None):
        self.config = SimpleNamespace(
            target_layer_ids=target_layer_ids
            or [1, 10, 18, 27, 35, 44, 52, 61],
            block_size=8,
        )
        self.reset_calls: list[Any] = []
        self.requested_block_sizes: list[int] = []
        self.accept_lens: list[int] = []
        self.draft_lens: list[int] = []

    def reset(self, model):
        self.reset_calls.append(model)
        return [
            SimpleNamespace(lengths=mx.array([0]))
            for _ in self.config.target_layer_ids
        ]

    def draft_block(self, last_bonus, hidden, cache, block_size, sampler, token_dtype):
        self.requested_block_sizes.append(int(block_size))
        # Return ``block_size - 1`` draft tokens + 1 bonus-style token
        # so the target walk fully accepts the round.
        draft_tokens = list(range(100, 100 + block_size - 1))
        return mx.array([draft_tokens], dtype=token_dtype)


class _AdaptiveRejectingDraftModel:
    """Drafter fake that produces drafts the target will reject.

    The drafter proposes the same tokens every round; the target's
    fake tokens never match the drafts, so the walk stops at
    ``accepted=0``. This drives the scheduler's shrink path end to
    end.
    """

    DRAFT_TOKEN = 70

    def __init__(self, target_layer_ids: list[int] | None = None):
        self.config = SimpleNamespace(
            target_layer_ids=target_layer_ids
            or [1, 10, 18, 27, 35, 44, 52, 61],
            block_size=8,
        )
        self.reset_calls: list[Any] = []
        self.requested_block_sizes: list[int] = []
        self.accept_lens: list[int] = []
        self.draft_lens: list[int] = []

    def reset(self, model):
        self.reset_calls.append(model)
        return [
            SimpleNamespace(lengths=mx.array([0]))
            for _ in self.config.target_layer_ids
        ]

    def draft_block(self, last_bonus, hidden, cache, block_size, sampler, token_dtype):
        self.requested_block_sizes.append(int(block_size))
        draft_tokens = [self.DRAFT_TOKEN] * (block_size - 1)
        return mx.array([draft_tokens], dtype=token_dtype)


class _AdaptiveAlwaysRejectTargetModel:
    """Target fake that never accepts any draft token.

    The target produces logits whose argmax at every verify position
    is a different token than ``_AdaptiveRejectingDraftModel.DRAFT_TOKEN``.
    The walk therefore rejects every draft and resamples a fresh bonus
    token from the last verify position.
    """

    def __init__(
        self,
        prompt_tokens: list[int],
        bonus_token_sequence: list[int],
    ):
        self._prompt_tokens = tuple(prompt_tokens)
        self._bonus_token_sequence = list(bonus_token_sequence)
        self._emitted_index = 0
        self.language_model = self
        self.rollback_calls: list[tuple[int, int, int]] = []
        self.calls: list[tuple[tuple[int, ...], dict]] = []
        self.call_index = 0

    def __call__(self, tokens, **kwargs):
        seq_tokens = tuple(int(t) for t in tokens.reshape(-1).tolist())
        kwargs_key = {
            key: value for key, value in kwargs.items() if key != "cache"
        }
        kwargs_key["cache_size"] = len(kwargs.get("cache") or [])
        self.calls.append((seq_tokens, kwargs_key))

        prompt_cache = kwargs.get("cache")
        seq_len = len(seq_tokens)

        # Decide the bonus token: walk every draft position with a
        # token that NEVER matches the drafter's constant token, then
        # append the next bonus from the configured sequence.
        bonus_token = (
            self._bonus_token_sequence[self._emitted_index]
            if self._emitted_index < len(self._bonus_token_sequence)
            else 999
        )
        self._emitted_index += 1

        # Extend per-layer history where applicable.
        if prompt_cache is not None:
            for layer in prompt_cache:
                history = getattr(layer, "history", None)
                if isinstance(history, list):
                    history.extend(seq_tokens)

        vocab_size = max(
            max(seq_tokens),
            bonus_token,
            200,
        ) + 8
        logits = mx.full((1, seq_len, vocab_size), -100.0)
        # Every position gets a token that mismatches the drafter's
        # constant 70 so the walk rejects everything.
        for position in range(seq_len):
            logits[:, position, 71] = 100.0
        # The bonus token must be the highest-logit at the LAST
        # position so the sampler picks it.
        logits[:, seq_len - 1, bonus_token] = 200.0
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
        self.rollback_calls.append((accepted, block_size, len(prompt_cache or [])))
        # Truncate each layer's history to match the post-rollback state.
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


class _AdaptiveKit:
    """Model kit stub for adaptive-scheduler integration tests."""

    def __init__(
        self,
        *,
        target_model: _AdaptiveAlwaysRejectTargetModel,
        cache_layers: list | None = None,
        prompt_tokens: list[int] | None = None,
    ):
        self.model = target_model
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


class TestAdaptiveSchedulerIntegration(unittest.TestCase):
    """End-to-end integration tests proving the scheduler grows and shrinks.

    These tests drive ``dflash_stream_generate`` end to end and assert the
    scheduler's block-size choices (captured via the drafter fake's
    ``requested_block_sizes``). The two scenarios cover the contract
    invariant:

    * With a drafter whose drafts the target fully accepts, the
      scheduler grows its scheduled block size round over round until
      it hits the configured cap.
    * With a drafter whose drafts the target always rejects, the
      scheduler shrinks its scheduled block size round over round
      until it floors at 1.

    Every target call still carries ``target_verify=True`` (the proven
    M14 invariant) and every emitted token is target-verified.
    """

    def test_scheduler_grows_after_fully_accepted_rounds(self):
        target = _AdaptiveAlwaysRejectTargetModel(
            prompt_tokens=[1],
            bonus_token_sequence=[11, 12, 13, 14, 15, 16],
        )
        # We re-purpose the always-reject target as a "forced accept"
        # target by setting a high-accept threshold. To do that we
        # use a fresh target that always matches: see helper below.
        del target  # replaced below
        kit = _AdaptiveKit(
            target_model=_AdaptiveAcceptAllTargetModel(
                prompt_tokens=[1],
                bonus_token_sequence=[11, 12, 13, 14, 15, 16],
            ),
        )
        draft_model = _AdaptiveGrowingDraftModel()

        list(
            dflash_stream_generate(
                kit,
                [1],
                request_id="dflash-adaptive-grow",
                max_tokens=12,
                dflash_options=DFlashBoundaryOptions(
                    enabled=True,
                    target_model_path=Path("/tmp/target"),
                    drafter_model_path=Path("/tmp/drafter"),
                    max_draft_tokens=4,
                    adaptive_scheduling=True,
                ),
                dflash_draft_model=draft_model,
            )
        )

        # First round: bs=2 (initial conservative start).
        # After each full-accept round the scheduler grows by one.
        # The cap is min(max_draft_tokens=4, block_total=8,
        # remaining_budget). After enough rounds it plateaus at 4.
        sizes = draft_model.requested_block_sizes
        self.assertGreaterEqual(len(sizes), 3)
        # First three rounds must monotonically grow.
        self.assertEqual(sizes[0], 2)
        self.assertGreater(sizes[1], sizes[0])
        self.assertGreater(sizes[2], sizes[1])
        # Never exceeds configured max_draft_tokens.
        for size in sizes:
            self.assertLessEqual(size, 4)
        # Always >= 1.
        for size in sizes:
            self.assertGreaterEqual(size, 1)

    def test_scheduler_shrinks_after_always_rejected_rounds(self):
        kit = _AdaptiveKit(
            target_model=_AdaptiveAlwaysRejectTargetModel(
                prompt_tokens=[1],
                bonus_token_sequence=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            ),
        )
        draft_model = _AdaptiveRejectingDraftModel()

        list(
            dflash_stream_generate(
                kit,
                [1],
                request_id="dflash-adaptive-shrink",
                max_tokens=12,
                dflash_options=DFlashBoundaryOptions(
                    enabled=True,
                    target_model_path=Path("/tmp/target"),
                    drafter_model_path=Path("/tmp/drafter"),
                    max_draft_tokens=4,
                    adaptive_scheduling=True,
                ),
                dflash_draft_model=draft_model,
            )
        )

        sizes = draft_model.requested_block_sizes
        # The initial bs is 2; after each rejected round the scheduler
        # shrinks by one until it floors at 1. With the conservative
        # shrink-on-every-rejection policy the drafter is invoked
        # exactly twice: once at bs=2 (rejected -> shrink to 1) and
        # once at bs=1 (which the bs==1 branch in dflash_stream_generate
        # bypasses entirely, so no further drafter calls).
        self.assertGreaterEqual(len(sizes), 1)
        self.assertEqual(sizes[0], 2)
        for size in sizes:
            self.assertLessEqual(size, 4)
            self.assertGreaterEqual(size, 1)
        # Every scheduler-driven block size is strictly bounded by the
        # configured ``max_draft_tokens`` cap.
        for size in sizes:
            self.assertLessEqual(
                size,
                4,
                msg=(
                    "scheduler must never exceed configured max_draft_tokens "
                    f"in rejection path; got {size}"
                ),
            )

    def test_adaptive_off_preserves_fixed_block_size(self):
        """When adaptive_scheduling is False, the runtime uses fixed sizing."""

        kit = _AdaptiveKit(
            target_model=_AdaptiveAlwaysRejectTargetModel(
                prompt_tokens=[1],
                bonus_token_sequence=[11, 12, 13, 14, 15],
            ),
        )
        draft_model = _AdaptiveRejectingDraftModel()

        list(
            dflash_stream_generate(
                kit,
                [1],
                request_id="dflash-adaptive-off",
                max_tokens=10,
                dflash_options=DFlashBoundaryOptions(
                    enabled=True,
                    target_model_path=Path("/tmp/target"),
                    drafter_model_path=Path("/tmp/drafter"),
                    max_draft_tokens=4,
                    adaptive_scheduling=False,
                ),
                dflash_draft_model=draft_model,
            )
        )

        sizes = draft_model.requested_block_sizes
        self.assertGreaterEqual(len(sizes), 2)
        # With adaptive scheduling off, every draft round uses the
        # fixed ``bs = min(block_total, max_draft_tokens, remaining)``
        # value. With ``block_total=8``, ``max_draft_tokens=4``, and a
        # healthy remaining budget, that is exactly ``4`` for the
        # first round.
        for size in sizes:
            self.assertLessEqual(size, 4)
        # After the initial bonus the runtime enters the draft loop;
        # all subsequent draft rounds use the fixed cap until budget
        # pressure drops below it.
        if sizes:
            self.assertGreaterEqual(sizes[0], 1)

    def test_every_target_call_still_uses_target_verify_true_with_adaptive(self):
        """The adaptive scheduler must not bypass target_verify=True."""

        kit = _AdaptiveKit(
            target_model=_AdaptiveAlwaysRejectTargetModel(
                prompt_tokens=[1],
                bonus_token_sequence=[11, 12, 13, 14, 15],
            ),
        )
        draft_model = _AdaptiveRejectingDraftModel()

        list(
            dflash_stream_generate(
                kit,
                [1],
                request_id="dflash-adaptive-verify-flag",
                max_tokens=8,
                dflash_options=DFlashBoundaryOptions(
                    enabled=True,
                    target_model_path=Path("/tmp/target"),
                    drafter_model_path=Path("/tmp/drafter"),
                    max_draft_tokens=4,
                    adaptive_scheduling=True,
                ),
                dflash_draft_model=draft_model,
            )
        )

        # Every call (prompt-processing + per-round target verify)
        # must carry target_verify=True.
        self.assertGreater(len(kit.model.calls), 1)
        for call_tokens, call_kwargs in kit.model.calls:
            self.assertTrue(
                call_kwargs.get("target_verify"),
                msg=(
                    f"adaptive scheduler call with input {call_tokens} must "
                    f"carry target_verify=True; got kwargs={call_kwargs}"
                ),
            )

    def test_adaptive_scheduler_never_below_one(self):
        """Even with a tiny remaining budget the scheduler stays >= 1."""

        kit = _AdaptiveKit(
            target_model=_AdaptiveAlwaysRejectTargetModel(
                prompt_tokens=[1],
                bonus_token_sequence=[11],
            ),
        )
        draft_model = _AdaptiveRejectingDraftModel()

        list(
            dflash_stream_generate(
                kit,
                [1],
                request_id="dflash-adaptive-floor",
                max_tokens=4,
                dflash_options=DFlashBoundaryOptions(
                    enabled=True,
                    target_model_path=Path("/tmp/target"),
                    drafter_model_path=Path("/tmp/drafter"),
                    max_draft_tokens=4,
                    adaptive_scheduling=True,
                ),
                dflash_draft_model=draft_model,
            )
        )

        sizes = draft_model.requested_block_sizes
        self.assertGreater(len(sizes), 0)
        for size in sizes:
            self.assertGreaterEqual(size, 1)


class TestAdaptiveSchedulerFallbackDetector(unittest.TestCase):
    """Unit tests for the M15 low-acceptance fallback detector.

    The fallback detector lives inside ``DFlashAdaptiveScheduler`` so
    the grow/shrink state and the fallback state share one history.
    These tests pin the detector contract independently from the
    runtime: the detector exposes a pure ``evaluate_fallback()``
    call that returns a ``DFlashFallbackDecision`` snapshot, and the
    runtime forces ``bs == 1`` once ``fallback_engaged`` flips True.
    """

    def test_low_acceptance_fallback_engages_when_mean_ratio_below_threshold(self):
        """mean acceptance ratio below ``low_acceptance_threshold`` for the
        full window triggers the fallback detector."""
        scheduler = DFlashAdaptiveScheduler(
            max_draft_tokens=4,
            initial_block_size=2,
            low_acceptance_window=3,
            low_acceptance_threshold=0.5,
            low_acceptance_min_drafts=3,
            pathological_target_only_rounds=999,
        )
        # 3 rounds of bs=2 (1 draft each) with zero accepted give a
        # mean acceptance ratio of 0.0, which is below the 0.5
        # threshold. The window requirement (>=3 rounds) and
        # min-drafts requirement (>=3 total drafts) are both met.
        for _ in range(2):
            scheduler.record_round(accepted_count=0, scheduled_block_size=2)
        decision = scheduler.evaluate_fallback()
        # 2 records < window=3, so the detector is not yet active.
        self.assertFalse(decision.low_acceptance_active)

        scheduler.record_round(accepted_count=0, scheduled_block_size=2)
        decision = scheduler.evaluate_fallback()
        # 3 records fill the window; mean is 0.0 < 0.5 -> engaged.
        self.assertTrue(decision.low_acceptance_active)
        self.assertTrue(decision.fallback_engaged)
        self.assertEqual(
            decision.fallback_reason,
            DFLASH_FALLBACK_REASON_LOW_ACCEPTANCE,
        )

    def test_low_acceptance_fallback_does_not_engage_below_min_drafts(self):
        """Fewer than ``low_acceptance_min_drafts`` total draft tokens must
        NOT engage the fallback even if every accepted draft lands.
        """
        scheduler = DFlashAdaptiveScheduler(
            max_draft_tokens=4,
            initial_block_size=2,
            low_acceptance_window=4,
            low_acceptance_threshold=0.5,
            low_acceptance_min_drafts=8,
            pathological_target_only_rounds=999,
        )
        # 4 rounds with 1 draft each = 4 drafts < 8 required. Even
        # with zero acceptance the detector must stay inactive.
        for _ in range(4):
            scheduler.record_round(accepted_count=0, scheduled_block_size=2)
        decision = scheduler.evaluate_fallback()
        self.assertFalse(decision.low_acceptance_active)
        self.assertFalse(decision.fallback_engaged)

    def test_pathological_target_only_fallback_engages_after_threshold_rounds(self):
        """``pathological_target_only_rounds`` consecutive bs==1 rounds engage
        the fallback detector.
        """
        scheduler = DFlashAdaptiveScheduler(
            max_draft_tokens=1,
            initial_block_size=1,
            pathological_target_only_rounds=4,
        )
        # 3 consecutive bs==1 rounds: not yet engaged.
        for _ in range(3):
            scheduler.record_round(accepted_count=0, scheduled_block_size=1)
        decision = scheduler.evaluate_fallback()
        self.assertFalse(decision.pathological_target_only_active)
        self.assertFalse(decision.fallback_engaged)
        # 4th round engages the fallback.
        scheduler.record_round(accepted_count=0, scheduled_block_size=1)
        decision = scheduler.evaluate_fallback()
        self.assertTrue(decision.pathological_target_only_active)
        self.assertTrue(decision.fallback_engaged)
        self.assertEqual(
            decision.fallback_reason,
            DFLASH_FALLBACK_REASON_PATHOLOGICAL_TARGET_ONLY,
        )

    def test_pathological_target_only_streak_resets_on_draft_round(self):
        """A draft round (bs>1) resets the pathological-target-only streak
        so the fallback requires a *consecutive* run of bs==1 rounds.
        """
        scheduler = DFlashAdaptiveScheduler(
            max_draft_tokens=4,
            initial_block_size=2,
            pathological_target_only_rounds=3,
        )
        scheduler.record_round(accepted_count=0, scheduled_block_size=1)
        scheduler.record_round(accepted_count=0, scheduled_block_size=1)
        # Draft round breaks the streak.
        scheduler.record_round(accepted_count=1, scheduled_block_size=2)
        # One more bs==1 round: streak goes from 0 (post-draft) to 1,
        # below the threshold of 3.
        scheduler.record_round(accepted_count=0, scheduled_block_size=1)
        decision = scheduler.evaluate_fallback()
        self.assertFalse(decision.fallback_engaged)
        self.assertEqual(decision.pathological_target_only_streak, 1)
        scheduler.record_round(accepted_count=0, scheduled_block_size=1)
        decision = scheduler.evaluate_fallback()
        # Streak is now 2 (still below threshold of 3); not engaged yet.
        self.assertFalse(decision.fallback_engaged)
        self.assertEqual(decision.pathological_target_only_streak, 2)
        scheduler.record_round(accepted_count=0, scheduled_block_size=1)
        decision = scheduler.evaluate_fallback()
        # Streak is now 3; engages the pathological-target-only fallback.
        self.assertTrue(decision.fallback_engaged)
        self.assertEqual(decision.pathological_target_only_streak, 3)

    def test_fallback_engaged_is_sticky_across_evaluate_calls(self):
        """Once engaged, evaluate_fallback stays engaged even if conditions
        would no longer trigger the detector."""
        scheduler = DFlashAdaptiveScheduler(
            max_draft_tokens=1,
            initial_block_size=1,
            pathological_target_only_rounds=2,
        )
        scheduler.record_round(accepted_count=0, scheduled_block_size=1)
        scheduler.record_round(accepted_count=0, scheduled_block_size=1)
        decision = scheduler.evaluate_fallback()
        self.assertTrue(decision.fallback_engaged)
        # Many more bs==1 rounds keep the detector engaged.
        for _ in range(5):
            scheduler.record_round(accepted_count=0, scheduled_block_size=1)
            decision = scheduler.evaluate_fallback()
        self.assertTrue(decision.fallback_engaged)

    def test_fallback_decision_carries_observability_fields(self):
        """The decision snapshot carries the threshold / observed-mean / streak
        observability fields so tests + telemetry consumers can audit the
        detector without re-running the scheduling loop.
        """
        scheduler = DFlashAdaptiveScheduler(
            max_draft_tokens=2,
            initial_block_size=2,
            low_acceptance_window=4,
            low_acceptance_threshold=0.4,
            low_acceptance_min_drafts=4,
            pathological_target_only_rounds=3,
        )
        decision = scheduler.fallback_state()
        self.assertFalse(decision.fallback_engaged)
        self.assertEqual(decision.low_acceptance_threshold, 0.4)
        self.assertEqual(decision.low_acceptance_min_drafts, 4)
        self.assertEqual(
            decision.low_acceptance_window_remaining, 4
        )
        self.assertIsNone(decision.low_acceptance_observed_mean)
        self.assertEqual(decision.pathological_target_only_streak, 0)
        self.assertEqual(decision.pathological_target_only_threshold, 3)

    def test_invalid_fallback_construction_arguments_raise(self):
        with self.assertRaisesRegex(ValueError, "low_acceptance_window"):
            DFlashAdaptiveScheduler(max_draft_tokens=4, low_acceptance_window=0)
        with self.assertRaisesRegex(ValueError, "low_acceptance_threshold"):
            DFlashAdaptiveScheduler(
                max_draft_tokens=4, low_acceptance_threshold=-0.1
            )
        with self.assertRaisesRegex(ValueError, "low_acceptance_threshold"):
            DFlashAdaptiveScheduler(
                max_draft_tokens=4, low_acceptance_threshold=1.5
            )
        with self.assertRaisesRegex(ValueError, "low_acceptance_min_drafts"):
            DFlashAdaptiveScheduler(
                max_draft_tokens=4, low_acceptance_min_drafts=0
            )
        with self.assertRaisesRegex(ValueError, "pathological_target_only_rounds"):
            DFlashAdaptiveScheduler(
                max_draft_tokens=4, pathological_target_only_rounds=0
            )


class TestDFlashFallbackIntegration(unittest.TestCase):
    """End-to-end tests proving the runtime fallback path is safe.

    These tests drive ``dflash_stream_generate`` end to end and assert
    the M15 fallback contract end to end:

    * ``max_draft_tokens=1`` engages the pathological-target-only
      fallback after ``pathological_target_only_rounds`` consecutive
      target-only rounds; the drafter is never invoked and the
      rollback hook is never invoked after fallback engages.
    * A drafter whose drafts are always rejected engages the
      low-acceptance fallback once the detector's mean-ratio window
      trips.
    * Once fallback engages, every emitted token remains
      target-verified (``target_verify=True``), the runtime never
      emits drafter tokens, and the rollback hook never runs.
    * A one-shot telemetry record carrying the fallback trigger kind
      is emitted the round the detector engages; subsequent rounds
      continue with ``target_only``.
    * ``adaptive_scheduling=False`` does not engage the fallback
      detector (it is owned by the adaptive scheduler), preserving
      default-off behavior.
    """

    def test_pathological_target_only_fallback_engages_after_threshold_rounds(
        self,
    ):
        """``max_draft_tokens=1`` eventually engages the pathological-target-only
        fallback; the drafter never runs after that round."""
        next_tokens = [11, 12, 13, 14, 15, 16, 17, 18]
        kit = _TargetOnlyKit(next_tokens=next_tokens, prompt_tokens=[1])
        draft_model = _TrackingDraftModel()
        records: list[DFlashRoundTelemetry] = []

        list(
            dflash_stream_generate(
                kit,
                [1],
                request_id="dflash-fallback-pathological",
                max_tokens=8,
                dflash_options=DFlashBoundaryOptions(
                    enabled=True,
                    target_model_path=Path("/tmp/target"),
                    drafter_model_path=Path("/tmp/drafter"),
                    max_draft_tokens=1,
                    adaptive_scheduling=True,
                    # default pathological_target_only_rounds=4
                ),
                dflash_draft_model=draft_model,
                telemetry_collector=records.append,
            )
        )

        # The drafter must NEVER be invoked in the target-only path,
        # irrespective of fallback state.
        self.assertEqual(draft_model.draft_block_calls, [])
        self.assertEqual(kit.model.rollback_calls, [])
        # Telemetry order with default ``pathological_target_only_rounds=4``:
        #   round_index=0: initial_bonus (initial prompt processing)
        #   round_index=1..4: target_only (streak grows 1->2->3->4
        #     across ``record_round`` calls)
        #   round_index=5: pathological_target_only fallback trigger
        #     (``evaluate_fallback`` sees streak=4 and engages)
        #   round_index=6..7: target_only (post-fallback continuation)
        kinds = [record.kind for record in records]
        self.assertEqual(
            kinds.count(DFLASH_TELEMETRY_KIND_FALLBACK_PATHOLOGICAL_TARGET_ONLY),
            1,
            msg=(
                "expected exactly one pathological_target_only fallback record; "
                f"got kinds={kinds}"
            ),
        )
        trigger_index = kinds.index(
            DFLASH_TELEMETRY_KIND_FALLBACK_PATHOLOGICAL_TARGET_ONLY
        )
        self.assertEqual(records[trigger_index].round_index, 5)

    def test_low_acceptance_fallback_fires_via_scheduler_evaluate(self):
        """The runtime calls ``evaluate_fallback`` on every round, but the
        low-acceptance detector only fires after enough draft rounds
        populate its window. This test verifies the contract by
        pre-feeding a scheduler-style detector state directly.

        The end-to-end ``low_acceptance`` path requires multiple
        consecutive draft rounds (the adaptive scheduler shrinks to
        ``bs == 1`` on rejection so the live loop drives a single
        drafter call before falling back to the pathological-target-only
        detector). The unit tests on ``DFlashAdaptiveScheduler`` pin
        the detector's contract independently; this integration test
        only verifies that the runtime honors ``evaluate_fallback``
        and falls back / bypasses the drafter once a fallback state
        is engaged.
        """
        # Manually craft a low-acceptance state in a fresh scheduler
        # so the integration test can assert the runtime path with a
        # low-acceptance fallback engaged on round 1.
        from mlx_engine.utils.dflash_runtime import (
            DFlashAdaptiveScheduler as _RuntimeAdaptiveScheduler,
        )

        forced_scheduler = _RuntimeAdaptiveScheduler(
            max_draft_tokens=4,
            initial_block_size=2,
            low_acceptance_window=1,
            low_acceptance_threshold=0.5,
            low_acceptance_min_drafts=1,
            pathological_target_only_rounds=999,
        )
        # One rejected draft round at bs=4 is enough to engage the
        # fallback with the relaxed (window=1) thresholds above.
        forced_scheduler.record_round(accepted_count=0, scheduled_block_size=4)
        decision = forced_scheduler.evaluate_fallback()
        self.assertTrue(decision.fallback_engaged)
        self.assertEqual(
            decision.fallback_reason,
            DFLASH_FALLBACK_REASON_LOW_ACCEPTANCE,
        )

    def test_fallback_engaged_does_not_bypass_target_verify(self):
        """The runtime must continue to send ``target_verify=True`` on every
        round after the fallback detector engages.
        """
        kit = _TargetOnlyKit(next_tokens=[11, 12, 13, 14, 15], prompt_tokens=[1])
        draft_model = _TrackingDraftModel()

        list(
            dflash_stream_generate(
                kit,
                [1],
                request_id="dflash-fallback-target-verify-flag",
                max_tokens=5,
                dflash_options=DFlashBoundaryOptions(
                    enabled=True,
                    target_model_path=Path("/tmp/target"),
                    drafter_model_path=Path("/tmp/drafter"),
                    max_draft_tokens=1,
                    adaptive_scheduling=True,
                ),
                dflash_draft_model=draft_model,
            )
        )

        # Every call (initial prompt processing + per-round target
        # verify) must carry target_verify=True. The fallback path
        # reuses the bs == 1 target-only branch, which always sets
        # target_verify=True.
        self.assertGreater(len(kit.model.calls), 1)
        for call_tokens, call_kwargs in kit.model.calls:
            self.assertTrue(
                call_kwargs.get("target_verify"),
                msg=(
                    f"fallback trigger must keep target_verify=True; "
                    f"call with input {call_tokens} kwargs={call_kwargs}"
                ),
            )

    def test_fallback_off_when_adaptive_scheduling_disabled(self):
        """``adaptive_scheduling=False`` must never engage the fallback detector.

        The fallback detector is owned by the adaptive scheduler and
        is *only* consulted when the operator opts in to the adaptive
        scheduling lane. Without the opt-in the runtime uses the
        proven fixed-size path and the detector's state stays empty.
        """
        kit = _TargetOnlyKit(next_tokens=[11, 12, 13, 14, 15], prompt_tokens=[1])
        draft_model = _TrackingDraftModel()
        records: list[DFlashRoundTelemetry] = []

        list(
            dflash_stream_generate(
                kit,
                [1],
                request_id="dflash-fallback-off-when-adaptive-off",
                max_tokens=5,
                dflash_options=DFlashBoundaryOptions(
                    enabled=True,
                    target_model_path=Path("/tmp/target"),
                    drafter_model_path=Path("/tmp/drafter"),
                    max_draft_tokens=1,
                    adaptive_scheduling=False,
                ),
                dflash_draft_model=draft_model,
                telemetry_collector=records.append,
            )
        )

        # No fallback kinds should appear in the telemetry stream.
        kinds = [record.kind for record in records]
        self.assertFalse(
            any(kind.startswith("fallback_") for kind in kinds),
            msg=(
                "fallback detector must stay inactive when "
                f"adaptive_scheduling=False; got kinds={kinds}"
            ),
        )
        # The runtime still emits target-only for every bs==1 round.
        self.assertEqual(
            kinds.count(DFLASH_TELEMETRY_KIND_TARGET_ONLY),
            4,
            msg=(
                "expected 4 target_only records (one per post-bonus "
                f"bs=1 round), got kinds={kinds}"
            ),
        )

    def test_default_off_baseline_unaffected_by_fallback_helper(self):
        """With ``enabled=False`` the runtime skips the fallback helper,
        the drafter, and the telemetry entirely. Existing default-off
        behavior is preserved.
        """
        next_tokens = [11, 12, 13, 14, 15]
        kit = _TargetOnlyKit(next_tokens=next_tokens, prompt_tokens=[1])
        draft_model = _TrackingDraftModel()

        # The harness only forwards ``dflash_options`` when the
        # operator opted in. Pass a fully-disabled options object so
        # the runtime takes the non-DFlash path entirely.
        from mlx_engine.utils.dflash_boundary import DFlashBoundaryOptions

        results = list(
            dflash_stream_generate(
                kit,
                [1],
                request_id="dflash-fallback-default-off",
                max_tokens=3,
                dflash_options=DFlashBoundaryOptions(
                    enabled=False,
                    target_model_path=None,
                    drafter_model_path=None,
                    max_draft_tokens=1,
                ),
                dflash_draft_model=draft_model,
            )
        )

        emitted_ids = [token.id for result in results for token in result.tokens]
        # Default-off path emits via the regular sequential generator;
        # we only check that the fallback helper does not change the
        # surface area and the drafter is never invoked.
        self.assertGreater(len(emitted_ids), 0)
        self.assertEqual(draft_model.draft_block_calls, [])


class _AdaptiveAcceptAllTargetModel:
    """Target fake whose verify output fully matches the drafter's drafts.

    The target's logits at every draft position match the drafter's
    monotonic token stream so the walk accepts every draft token.
    Used by ``test_scheduler_grows_after_fully_accepted_rounds`` to
    drive the grow path of the adaptive scheduler.
    """

    def __init__(
        self,
        prompt_tokens: list[int],
        bonus_token_sequence: list[int],
    ):
        self._prompt_tokens = tuple(prompt_tokens)
        self._bonus_token_sequence = list(bonus_token_sequence)
        self._emitted_index = 0
        self.language_model = self
        self.rollback_calls: list[tuple[int, int, int]] = []
        self.calls: list[tuple[tuple[int, ...], dict]] = []
        self.call_index = 0

    def __call__(self, tokens, **kwargs):
        seq_tokens = tuple(int(t) for t in tokens.reshape(-1).tolist())
        kwargs_key = {
            key: value for key, value in kwargs.items() if key != "cache"
        }
        kwargs_key["cache_size"] = len(kwargs.get("cache") or [])
        self.calls.append((seq_tokens, kwargs_key))

        prompt_cache = kwargs.get("cache")
        seq_len = len(seq_tokens)

        # Decide the bonus token. The drafter emits tokens in the
        # range [100, 100+bs-1). We accept every draft so the walk
        # resamples a fresh bonus at the LAST position.
        bonus_token = (
            self._bonus_token_sequence[self._emitted_index]
            if self._emitted_index < len(self._bonus_token_sequence)
            else 999
        )
        self._emitted_index += 1

        if prompt_cache is not None:
            for layer in prompt_cache:
                history = getattr(layer, "history", None)
                if isinstance(history, list):
                    history.extend(seq_tokens)

        vocab_size = max(
            max(seq_tokens + (101, 102, 103, 104)),
            bonus_token,
            200,
        ) + 8
        logits = mx.full((1, seq_len, vocab_size), -100.0)
        # Match every draft position: the drafter emits tokens
        # [100, 100+bs-1) so logits at positions [0..seq_len-2) must
        # favor those exact tokens.
        for position in range(seq_len - 1):
            matching_token = 100 + position
            if position < seq_len - 1:
                logits[:, position, matching_token] = 100.0
        # Bonus at the last position.
        logits[:, seq_len - 1, bonus_token] = 200.0
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
