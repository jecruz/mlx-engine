from mlx_engine.model_kit.batched_vision.prompt_cache.chunks import (
    build_prefix_cache_chunks,
    extend_prefix_cache_chunks,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.types import (
    DEFAULT_PREFIX_CHUNK_SIZE,
    PromptImageSpan,
)


C = DEFAULT_PREFIX_CHUNK_SIZE


def _chunk_bounds(prompt_len: int, image_spans=None) -> list[tuple[int, int]]:
    chunks = build_prefix_cache_chunks(list(range(prompt_len)), image_spans or [])
    return [(chunk.start, chunk.end) for chunk in chunks]


def test_chunks_drop_short_tail():
    """The final seed token stays out of the reusable cache chain."""
    assert _chunk_bounds(C - 1) == [(0, C - 2)]
    assert _chunk_bounds(C) == [(0, C - 1)]
    assert _chunk_bounds((2 * C) - 1) == [(0, C), (C, (2 * C) - 2)]
    assert _chunk_bounds(2 * C) == [(0, C), (C, (2 * C) - 1)]


def test_chunks_include_image_identity_without_growing_bounds():
    """Image spans crossing a boundary mark both reusable-prefix chunks."""
    image_spans = [PromptImageSpan(start=C - 6, end=C + 44, image_hash="image-a")]

    assert _chunk_bounds((2 * C) + 88, image_spans) == [
        (0, C),
        (C, 2 * C),
        (2 * C, (2 * C) + 87),
    ]

    changed_image_spans = [
        PromptImageSpan(start=C - 6, end=C + 44, image_hash="image-b")
    ]
    chunks_a = build_prefix_cache_chunks(list(range((2 * C) + 88)), image_spans)
    chunks_b = build_prefix_cache_chunks(
        list(range((2 * C) + 88)),
        changed_image_spans,
    )

    assert chunks_a[0].key != chunks_b[0].key
    assert chunks_a[1].key != chunks_b[1].key


def test_chunks_later_images_do_not_invalidate_earlier_chunks():
    """Later image changes preserve earlier chunk keys and change later keys."""
    prompt_input_ids = list(range((2 * C) + 88))
    image_a = [PromptImageSpan(start=C + 44, end=C + 64, image_hash="image-a")]
    image_b = [PromptImageSpan(start=C + 44, end=C + 64, image_hash="image-b")]

    chunks_a = build_prefix_cache_chunks(prompt_input_ids, image_a)
    chunks_b = build_prefix_cache_chunks(prompt_input_ids, image_b)

    assert chunks_a[0].key == chunks_b[0].key
    assert chunks_a[1].key != chunks_b[1].key
    assert chunks_a[2].key != chunks_b[2].key


def test_chunks_later_tokens_do_not_invalidate_earlier_chunks():
    """Later token changes preserve earlier chunk keys and change later keys."""
    prompt_a = list(range((2 * C) + 88))
    prompt_b = list(prompt_a)
    prompt_b[C + 44] = -1

    chunks_a = build_prefix_cache_chunks(prompt_a, [])
    chunks_b = build_prefix_cache_chunks(prompt_b, [])

    assert chunks_a[0].key == chunks_b[0].key
    assert chunks_a[1].key != chunks_b[1].key
    assert chunks_a[2].key != chunks_b[2].key


def test_chunks_extend_incrementally_matches_full_build():
    """Decode can append only newly completed chunks without rebuilding all."""
    prompt_input_ids = list(range((2 * C) + 188))
    image_spans = [PromptImageSpan(start=C + 44, end=C + 64, image_hash="image-a")]
    chunks = build_prefix_cache_chunks(prompt_input_ids[: C + 1], image_spans)

    extend_prefix_cache_chunks(prompt_input_ids, image_spans, chunks)

    assert [(chunk.start, chunk.end) for chunk in chunks] == [
        (0, C),
        (C, 2 * C),
        (2 * C, (2 * C) + 187),
    ]
    assert chunks == build_prefix_cache_chunks(prompt_input_ids, image_spans)
