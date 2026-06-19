from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mlx.core as mx

from mlx_engine.utils.specprefill import (
    _OffsetAdjustedRoPE,
    SpecPrefillOptions,
    cleanup_rope,
    manual_rope,
    select_chunks,
    try_sparse_prefill,
)
from tests.shared import RecordingReporter


class FakeRope:
    """RoPE test double with the attributes SpecPrefill inspects."""

    def __init__(self):
        self.dims = 4
        self.base = 10000.0
        self.scale = 1.0

    def __call__(self, x, offset=0):
        """Return the input unchanged for tests."""
        return x


class FakeAttention:
    """Attention test double with a RoPE attribute."""

    def __init__(self):
        self.rope = FakeRope()


class FakeModel:
    """Model test double with one self-attention layer."""

    def __init__(self):
        self.layers = [SimpleNamespace(self_attn=FakeAttention())]
        self.calls = []

    def __call__(self, tokens, cache):
        """Record token counts and advance fake cache offsets."""
        self.calls.append(int(tokens.shape[1]))
        for entry in cache:
            entry.offset += int(tokens.shape[1])
        return mx.zeros((1, int(tokens.shape[1]), 8))


class FakeCache:
    """Cache test double with offset and empty state."""

    def __init__(self):
        self.offset = 0

    @property
    def state(self):
        return []


class TestSpecPrefillHelpers(unittest.TestCase):
    """Tests for SpecPrefill helper functions ported from oMLX."""

    def test_select_chunks_keeps_high_importance_chunks_in_prompt_order(self):
        importance = mx.zeros(128)
        importance = importance.at[32:64].add(2.0)
        importance = importance.at[96:128].add(1.0)

        selected = select_chunks(importance, keep_pct=0.5, chunk_size=32)

        self.assertEqual(selected.tolist(), list(range(32, 64)) + list(range(96, 128)))

    def test_manual_rope_preserves_unrotated_dimensions(self):
        values = mx.random.normal((1, 2, 4, 8))

        rotated = manual_rope(values, mx.arange(4), dims=4)

        self.assertEqual(rotated.shape, values.shape)
        self.assertTrue(mx.allclose(rotated[..., 4:], values[..., 4:]).item())

    def test_cleanup_rope_restores_offset_adjusted_rope(self):
        model = FakeModel()
        original = model.layers[0].self_attn.rope
        model.layers[0].self_attn.rope = _OffsetAdjustedRoPE(original, 3)

        cleanup_rope(model)

        self.assertIs(model.layers[0].self_attn.rope, original)

    def test_try_sparse_prefill_uses_selected_tokens_and_returns_final_seed(self):
        model = FakeModel()
        cache = [FakeCache()]
        prompt = mx.array(list(range(10)), dtype=mx.int32)
        options = SpecPrefillOptions(enabled=True, keep_pct=0.5, threshold=2)

        with (
            patch(
                "mlx_engine.utils.specprefill.score_tokens",
                return_value=mx.array([0.1, 5.0, 4.0, 0.2, 0.1, 0.1]),
            ),
            patch(
                "mlx_engine.utils.specprefill.select_chunks",
                return_value=mx.array([1, 2, 5], dtype=mx.int32),
            ),
        ):
            result = try_sparse_prefill(
                model=model,
                draft_model=object(),
                prompt_tokens=prompt,
                uncached_tokens=prompt[4:],
                cache=cache,
                cached_tokens=4,
                options=options,
                chunk_size=8,
                reporter=RecordingReporter(),
            )

        self.assertIsNotNone(result)
        self.assertEqual(result.seed_tokens.tolist(), [9])
        self.assertEqual(result.live_tokens, prompt.tolist())
        self.assertEqual(model.calls, [1, 1])
        self.assertIsInstance(model.layers[0].self_attn.rope, _OffsetAdjustedRoPE)

    def test_try_sparse_prefill_cleans_rope_when_sparse_prefill_fails(self):
        model = FakeModel()
        original = model.layers[0].self_attn.rope
        model.layers[0].self_attn.rope = _OffsetAdjustedRoPE(original, 3)

        with (
            patch(
                "mlx_engine.utils.specprefill.score_tokens",
                return_value=mx.array([0.1, 5.0, 4.0, 0.2]),
            ),
            patch(
                "mlx_engine.utils.specprefill.select_chunks",
                return_value=mx.array([1, 2], dtype=mx.int32),
            ),
            patch(
                "mlx_engine.utils.specprefill.sparse_prefill",
                side_effect=RuntimeError("failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                try_sparse_prefill(
                    model=model,
                    draft_model=object(),
                    prompt_tokens=mx.array(list(range(8)), dtype=mx.int32),
                    uncached_tokens=mx.array(list(range(4, 8)), dtype=mx.int32),
                    cache=[FakeCache()],
                    cached_tokens=4,
                    options=SpecPrefillOptions(enabled=True, threshold=2),
                    chunk_size=8,
                    reporter=RecordingReporter(),
                )

        self.assertIs(model.layers[0].self_attn.rope, original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
