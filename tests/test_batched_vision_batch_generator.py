import contextlib
import io
import logging
from types import SimpleNamespace

import mlx.core as mx

from mlx_engine.model_kit.batched_vision import batch_generator as batcher
from mlx_engine.model_kit.batched_vision.batch_generator import (
    BatchGenerator,
    GenerationBatch,
    _PrefixCacheSaveState,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.chunks import (
    build_prefix_cache_chunks,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.types import (
    DEFAULT_PREFIX_CHUNK_SIZE,
    PromptImageSpan,
)


C = DEFAULT_PREFIX_CHUNK_SIZE


def _argmax_sampler(logprobs):
    return mx.argmax(logprobs, axis=-1).astype(mx.int32)


def _logits(batch_size: int, seq_len: int, vocab_size: int = 8):
    return mx.zeros((batch_size, seq_len, vocab_size), dtype=mx.float32)


def _bump(logits, token: int):
    bump = [0.0] * logits.shape[-1]
    bump[token] = 100.0
    return logits + mx.array([bump], dtype=mx.float32)


def _prefix_cache_save_states(count: int):
    return [
        _PrefixCacheSaveState([], 0, DEFAULT_PREFIX_CHUNK_SIZE, [], None)
        for _ in range(count)
    ]


class _HistoryProcessor:
    def __init__(self, token: int):
        self.token = token
        self.calls = []

    def __call__(self, tokens, logits):
        self.calls.append(tokens.tolist())
        return _bump(logits, self.token)


class _IntLastTokenProcessor:
    def __init__(self, token: int):
        self.token = token
        self.calls = []
        self.last_token_calls = []

    def process_last_token(self, last_token, logits):
        if not isinstance(last_token, int):
            raise TypeError("last_token must be an int")
        self.last_token_calls.append(last_token)
        return _bump(logits, self.token)

    def __call__(self, tokens, logits):
        self.calls.append(tokens.tolist())
        return _bump(logits, self.token)


class _FakeBatchCache:
    keys = True

    def __init__(self, name: str = "cache"):
        self.name = name
        self.state = mx.array([0], dtype=mx.int32)
        self.extended = []
        self.filtered = []
        self.extracted = []

    def extract(self, idx: int):
        self.extracted.append(idx)
        return _FakeScalarCache(f"{self.name}:{idx}")

    def extend(self, other):
        self.extended.append(other)

    def filter(self, keep):
        self.filtered.append(keep.tolist())


class ArraysCache(_FakeBatchCache):
    pass


class _FakeScalarCache:
    def __init__(self, name: str = "scalar"):
        self.name = name
        self.state = mx.array([0], dtype=mx.int32)
        self.merge_calls = []

    def merge(self, caches):
        self.merge_calls.append(caches)
        return _FakeBatchCache(f"merged:{self.name}")


class _FakeModel:
    def __init__(self):
        self.calls = []
        self.model_type = None
        self.config = SimpleNamespace(use_bidirectional_attention=None)

    def __call__(self, input_ids, cache=None, inputs_embeds=None, **kwargs):
        self.calls.append(
            {
                "input_ids": input_ids.tolist(),
                "inputs_embeds_shape": (
                    None if inputs_embeds is None else inputs_embeds.shape
                ),
                "n_to_process": kwargs.get("n_to_process"),
                "position_ids": (
                    None
                    if kwargs.get("position_ids") is None
                    else kwargs["position_ids"].tolist()
                ),
                "rope_deltas": (
                    None
                    if kwargs.get("rope_deltas") is None
                    else kwargs["rope_deltas"].tolist()
                ),
                "mm_token_type_ids": (
                    None
                    if kwargs.get("mm_token_type_ids") is None
                    else kwargs["mm_token_type_ids"].tolist()
                ),
            }
        )
        batch_size, seq_len = input_ids.shape
        return SimpleNamespace(logits=_logits(batch_size, seq_len))


def _gemma4_unified_model():
    model = _FakeModel()
    model.model_type = "gemma4_unified"
    model.config = SimpleNamespace(use_bidirectional_attention="vision")
    return model


def _gemma4_model():
    model = _FakeModel()
    model.model_type = "gemma4"
    model.config = SimpleNamespace(use_bidirectional_attention="vision")
    return model


def _gemma4_non_bidir_model():
    model = _FakeModel()
    model.model_type = "gemma4"
    model.config = SimpleNamespace(use_bidirectional_attention=None)
    return model


def test_generation_batch_applies_per_sequence_processors_and_top_logprobs():
    """Processors are per-row, and sampled token metadata follows decode-ahead."""
    model = _FakeModel()
    history_processor = _HistoryProcessor(token=3)
    second_history_processor = _HistoryProcessor(token=4)
    batch = GenerationBatch(
        model=model,
        uids=[10, 11],
        inputs=mx.array([1, 2], dtype=mx.int32),
        prompt_cache=[_FakeBatchCache()],
        samplers=[_argmax_sampler, _argmax_sampler],
        stop_criteria=lambda _token: False,
        max_tokens=[3, 3],
        top_logprobs_k=2,
        all_tokens=[[100], [200]],
        logits_processors=[[history_processor], [second_history_processor]],
        prefix_cache_save_states=_prefix_cache_save_states(2),
    )

    first = batch.next()
    second = batch.next()

    assert [response.token for response in first] == [1, 2]
    assert [response.token for response in second] == [3, 4]
    assert [response.top_logprobs[0][0] for response in second] == [3, 4]
    assert history_processor.calls[0] == [100, 1]
    assert second_history_processor.calls[0] == [200, 2]


def test_int_last_token_processor_uses_full_context_call():
    """Structured processors do not receive MLX arrays via process_last_token."""
    processor = _IntLastTokenProcessor(token=5)

    logits = batcher._apply_logits_processors(
        mx.zeros((1, 8), dtype=mx.float32),
        [[100]],
        [[processor]],
        last_tokens=mx.array([2], dtype=mx.int32),
    )

    assert processor.calls == [[100, 2]]
    assert processor.last_token_calls == []
    assert mx.argmax(logits, axis=-1).tolist() == [5]


def test_generation_batch_finish_returns_cache_tokens_and_rope_delta():
    """A finished row returns the mutable cache state needed by hot restore."""
    prompt_cache = [_FakeBatchCache()]
    batch = GenerationBatch(
        model=_FakeModel(),
        uids=[7],
        inputs=mx.array([9], dtype=mx.int32),
        prompt_cache=prompt_cache,
        samplers=[_argmax_sampler],
        stop_criteria=lambda _token: False,
        max_tokens=[1],
        all_tokens=[[1, 2]],
        rope_deltas=mx.array([5], dtype=mx.int32),
        logits_processors=[[]],
        prefix_cache_save_states=_prefix_cache_save_states(1),
    )

    response = batch.next()[0]

    assert response.finish_reason == "length"
    assert response.all_tokens == [1, 2, 9]
    assert response.prompt_cache[0].name == "cache:0"
    assert response.rope_deltas.tolist() == [[5]]


def test_generation_batch_extends_mixed_rope_rows_without_broadcasting():
    """Appending text-only work to image work gives each row its own RoPE delta."""
    model = _FakeModel()
    batch = GenerationBatch(
        model=model,
        uids=[1],
        inputs=mx.array([5], dtype=mx.int32),
        prompt_cache=[_FakeBatchCache("image")],
        samplers=[_argmax_sampler],
        stop_criteria=lambda _token: False,
        max_tokens=[3],
        all_tokens=[[5]],
        rope_deltas=mx.array([9], dtype=mx.int32),
        logits_processors=[[]],
        prefix_cache_save_states=_prefix_cache_save_states(1),
    )
    text_only = GenerationBatch(
        model=model,
        uids=[2],
        inputs=mx.array([6], dtype=mx.int32),
        prompt_cache=[_FakeBatchCache("text")],
        samplers=[_argmax_sampler],
        stop_criteria=lambda _token: False,
        max_tokens=[3],
        all_tokens=[[6]],
        logits_processors=[[]],
        prefix_cache_save_states=_prefix_cache_save_states(1),
    )

    batch.append_prefilled_sequence(text_only)
    batch.next()

    assert model.calls[-1]["rope_deltas"] == [[9], [0]]


def test_generation_batch_append_materializes_pending_state_before_cache_extend(
    monkeypatch,
):
    calls = []

    class RecordingCache(_FakeBatchCache):
        def extend(self, other):
            calls.append(("extend-cache", self.name, other.name))
            super().extend(other)

    def record_eval(batch):
        calls.append(("eval", tuple(batch.uids)))

    monkeypatch.setattr(GenerationBatch, "_eval_pending_state", record_eval)
    batch = GenerationBatch(
        model=_FakeModel(),
        uids=[1],
        inputs=mx.array([5], dtype=mx.int32),
        prompt_cache=[RecordingCache("active")],
        samplers=[_argmax_sampler],
        stop_criteria=lambda _token: False,
        max_tokens=[3],
        all_tokens=[[5]],
        logits_processors=[[]],
        prefix_cache_save_states=_prefix_cache_save_states(1),
    )
    prefilled = GenerationBatch(
        model=_FakeModel(),
        uids=[2],
        inputs=mx.array([6], dtype=mx.int32),
        prompt_cache=[RecordingCache("prefilled")],
        samplers=[_argmax_sampler],
        stop_criteria=lambda _token: False,
        max_tokens=[3],
        all_tokens=[[6]],
        logits_processors=[[]],
        prefix_cache_save_states=_prefix_cache_save_states(1),
    )

    batch.append_prefilled_sequence(prefilled)

    assert calls == [
        ("eval", (1,)),
        ("eval", (2,)),
        ("extend-cache", "active", "prefilled"),
    ]


def test_generation_batch_filter_materializes_pending_state_before_cache_filter(
    monkeypatch,
):
    calls = []

    class RecordingCache(_FakeBatchCache):
        def filter(self, keep):
            calls.append(("filter-cache", keep.tolist()))
            super().filter(keep)

    def record_eval(batch):
        calls.append(("eval", tuple(batch.uids)))

    monkeypatch.setattr(GenerationBatch, "_eval_pending_state", record_eval)
    batch = GenerationBatch(
        model=_FakeModel(),
        uids=[1, 2],
        inputs=mx.array([5, 6], dtype=mx.int32),
        prompt_cache=[RecordingCache()],
        samplers=[_argmax_sampler, _argmax_sampler],
        stop_criteria=lambda _token: False,
        max_tokens=[3, 3],
        all_tokens=[[5], [6]],
        logits_processors=[[], []],
        prefix_cache_save_states=_prefix_cache_save_states(2),
    )

    batch.filter([0])

    assert calls == [("eval", (1, 2)), ("filter-cache", [0])]


def test_capture_rope_deltas_keeps_qwen3_5_text_only_none():
    """Qwen3.5 text-only decode stays on the fast text RoPE path."""
    qwen3_5_model = SimpleNamespace(
        language_model=SimpleNamespace(model_type="qwen3_5_vl", _rope_deltas=None)
    )
    qwen_model = SimpleNamespace(
        language_model=SimpleNamespace(model_type="qwen2_vl", _rope_deltas=None)
    )

    assert batcher._capture_rope_deltas(qwen3_5_model, rows=2) is None
    assert batcher._capture_rope_deltas(qwen_model, rows=2).tolist() == [[0], [0]]


def test_batch_generator_slices_position_ids_and_saves_prefill_boundaries(
    monkeypatch,
):
    """Chunked prefill keeps Qwen MRoPE positions aligned with sliced embeds."""
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    monkeypatch.setattr(
        batcher,
        "make_prompt_cache",
        lambda _model: [_FakeBatchCache()],
    )
    model = _FakeModel()
    generator = BatchGenerator(
        model=model,
        stop_criteria=lambda _token: False,
        prefill_step_size=256,
    )
    snapshots = []
    prompt = list(range(513))
    position_ids = mx.array(
        [
            [list(range(513))],
            [list(range(1000, 1513))],
            [list(range(2000, 2513))],
        ],
        dtype=mx.int32,
    )

    prefix_chunks = build_prefix_cache_chunks(prompt, [])

    def save_snapshot(
        cache,
        chunks,
        start_chunk_idx,
        end_chunk_idx,
        snapshot_len,
        *,
        is_final_prompt_boundary,
    ):
        snapshots.append(
            (
                cache,
                chunks,
                start_chunk_idx,
                end_chunk_idx,
                snapshot_len,
                is_final_prompt_boundary,
            )
        )

    try:
        generator.insert(
            prompt,
            inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={"position_ids": position_ids},
            prefix_cache_chunks=prefix_chunks,
            all_tokens=[],
            next_prefix_cache_chunk_idx=0,
            image_spans=[],
            prompt_cache_save_callback=save_snapshot,
        )

        generator.next()
        generator.next()
        generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [256, 256, 1]
    assert [len(call["position_ids"][0][0]) for call in model.calls] == [256, 256, 1]
    assert model.calls[0]["position_ids"][0][0][0] == 0
    assert model.calls[0]["position_ids"][0][0][-1] == 255
    assert model.calls[1]["position_ids"][0][0][0] == 256
    assert model.calls[1]["position_ids"][0][0][-1] == 511
    assert model.calls[2]["position_ids"][0][0] == [512]
    assert [
        (start_chunk_idx, end_chunk_idx, snapshot_len, is_final_prompt_boundary)
        for (
            _cache,
            _chunks,
            start_chunk_idx,
            end_chunk_idx,
            snapshot_len,
            is_final_prompt_boundary,
        ) in snapshots
    ] == [
        (0, 1, C, False),
    ]


def test_batch_generator_logs_deep_prefill_timing_when_enabled(monkeypatch):
    """Diagnostic mode emits chunk and final prefill timings from the VLM batcher."""
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    monkeypatch.setattr(
        batcher,
        "make_prompt_cache",
        lambda _model: [_FakeBatchCache()],
    )
    monkeypatch.setenv("MLX_ENGINE_BATCHED_TIMING", "1")
    perf_counter_values = iter([1.0, 1.125, 2.0, 2.25, 3.0, 3.375])
    monkeypatch.setattr(
        batcher.time,
        "perf_counter",
        lambda: next(perf_counter_values),
    )
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    batcher.logger.addHandler(handler)
    original_level = batcher.logger.level
    batcher.logger.setLevel(logging.WARNING)

    generator = BatchGenerator(
        model=_FakeModel(),
        stop_criteria=lambda _token: False,
        prefill_step_size=4,
    )
    try:
        generator.insert(
            list(range(6)),
            inputs_embeds=mx.zeros((1, 6, 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={},
            prefix_cache_chunks=[],
            all_tokens=[],
            next_prefix_cache_chunk_idx=0,
            image_spans=[],
            request_id="req-prefill",
        )

        generator.next()
        generator.next()
        generator.next()
    finally:
        batcher.logger.setLevel(original_level)
        batcher.logger.removeHandler(handler)
        generator.close()

    log_output = stream.getvalue()
    assert '"event": "vlm_prefill_chunk"' in log_output
    assert '"request_id": "req-prefill"' in log_output
    assert '"chunk_tokens": 4' in log_output
    assert '"duration_ms": 125.0' in log_output
    assert '"event": "vlm_prefill_final"' in log_output
    assert '"final_tokens": 2' in log_output
    assert '"duration_ms": 250.0' in log_output
    assert '"event": "vlm_decode_step"' in log_output
    assert '"first_step": true' in log_output
    assert '"request_ids": ["req-prefill"]' in log_output
    assert '"duration_ms": 375.0' in log_output


def test_batch_generator_keeps_gemma4_visual_prefix_together(monkeypatch):
    """Gemma4 visual masks need prompt start through last visual token in one call."""
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    monkeypatch.setattr(
        batcher,
        "make_prompt_cache",
        lambda _model: [_FakeBatchCache()],
    )
    model = _gemma4_unified_model()
    generator = BatchGenerator(
        model=model,
        stop_criteria=lambda _token: False,
        prefill_step_size=4,
    )
    prompt = list(range(14))
    mm_token_type_ids = mx.array(
        [[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]],
        dtype=mx.int32,
    )

    try:
        generator.insert(
            prompt,
            inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={"mm_token_type_ids": mm_token_type_ids},
            prefix_cache_chunks=[],
            all_tokens=[],
            next_prefix_cache_chunk_idx=0,
            image_spans=[],
        )

        generator.next()
        generator.next()
        generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [8, 4, 2]
    assert model.calls[0]["mm_token_type_ids"] == [[0, 0, 0, 0, 0, 1, 1, 1]]
    assert model.calls[1]["mm_token_type_ids"] == [[0, 0, 0, 0]]


def test_batch_generator_chunks_gemma4_text_only_normally(monkeypatch):
    """Gemma4 unified text-only prompts keep the configured prefill size."""
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    monkeypatch.setattr(
        batcher,
        "make_prompt_cache",
        lambda _model: [_FakeBatchCache()],
    )
    model = _gemma4_unified_model()
    generator = BatchGenerator(
        model=model,
        stop_criteria=lambda _token: False,
        prefill_step_size=4,
    )
    prompt = list(range(10))

    try:
        generator.insert(
            prompt,
            inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={
                "mm_token_type_ids": mx.zeros((1, len(prompt)), dtype=mx.int32)
            },
            prefix_cache_chunks=[],
            all_tokens=[],
            next_prefix_cache_chunk_idx=0,
            image_spans=[],
        )

        generator.next()
        generator.next()
        generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [4, 4, 2]


def test_batch_generator_does_not_split_gemma4_visual_prompt_tail(monkeypatch):
    """If the last visual token is also the last prompt token, use final prefill."""
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    monkeypatch.setattr(
        batcher,
        "make_prompt_cache",
        lambda _model: [_FakeBatchCache()],
    )
    model = _gemma4_unified_model()
    generator = BatchGenerator(
        model=model,
        stop_criteria=lambda _token: False,
        prefill_step_size=4,
    )
    prompt = list(range(8))
    mm_token_type_ids = mx.array([[0, 0, 0, 0, 0, 1, 1, 1]], dtype=mx.int32)

    try:
        generator.insert(
            prompt,
            inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={"mm_token_type_ids": mm_token_type_ids},
            prefix_cache_chunks=[],
            all_tokens=[],
            next_prefix_cache_chunk_idx=0,
            image_spans=[],
        )

        generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [8]


def test_batch_generator_uses_image_spans_without_gemma4_token_types(monkeypatch):
    """Image spans provide a fallback boundary if Gemma4 token types are absent."""
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    monkeypatch.setattr(
        batcher,
        "make_prompt_cache",
        lambda _model: [_FakeBatchCache()],
    )
    model = _gemma4_unified_model()
    generator = BatchGenerator(
        model=model,
        stop_criteria=lambda _token: False,
        prefill_step_size=4,
    )
    prompt = list(range(10))

    try:
        generator.insert(
            prompt,
            inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={},
            prefix_cache_chunks=[],
            all_tokens=[],
            next_prefix_cache_chunk_idx=0,
            image_spans=[PromptImageSpan(start=5, end=8, image_hash="image")],
        )

        generator.next()
        generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [8, 2]


def test_batch_generator_pads_gemma4_token_types_after_restore(monkeypatch):
    """A new-image suffix can build masks against restored cached prefix keys."""
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    model = _gemma4_model()
    generator = BatchGenerator(
        model=model,
        stop_criteria=lambda _token: False,
        prefill_step_size=4,
    )
    prompt = list(range(8))

    try:
        generator.insert(
            prompt,
            inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={
                "mm_token_type_ids": mx.array(
                    [[0, 0, 1, 1, 0, 0, 0, 0]],
                    dtype=mx.int32,
                )
            },
            prefix_cache_chunks=[],
            cache=[_FakeScalarCache()],
            all_tokens=[100, 101, 102, 103, 104],
            next_prefix_cache_chunk_idx=0,
            image_spans=[],
        )

        generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [4]
    assert model.calls[0]["mm_token_type_ids"] == [[0, 0, 0, 0, 0, 0, 0, 1, 1]]


def test_batch_generator_pads_gemma4_token_types_for_final_prefill(monkeypatch):
    """Final prefill also needs key-length token types when restored before image."""
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    model = _gemma4_model()
    generator = BatchGenerator(
        model=model,
        stop_criteria=lambda _token: False,
        prefill_step_size=4,
    )
    prompt = list(range(3))

    try:
        generator.insert(
            prompt,
            inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={"mm_token_type_ids": mx.array([[0, 1, 1]], dtype=mx.int32)},
            prefix_cache_chunks=[],
            cache=[_FakeScalarCache()],
            all_tokens=[100, 101, 102, 103, 104],
            next_prefix_cache_chunk_idx=0,
            image_spans=[],
        )

        generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [3]
    assert model.calls[0]["mm_token_type_ids"] == [[0, 0, 0, 0, 0, 0, 1, 1]]


def test_batch_generator_applies_visual_policy_to_bidir_gemma4(monkeypatch):
    """Non-unified bidirectional Gemma4 keeps visual spans in one prefill call."""
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    monkeypatch.setattr(
        batcher,
        "make_prompt_cache",
        lambda _model: [_FakeBatchCache()],
    )
    model = _gemma4_model()
    generator = BatchGenerator(
        model=model,
        stop_criteria=lambda _token: False,
        prefill_step_size=4,
    )
    prompt = list(range(10))
    mm_token_type_ids = mx.array(
        [[0, 0, 0, 0, 0, 1, 1, 1, 0, 0]],
        dtype=mx.int32,
    )

    try:
        generator.insert(
            prompt,
            inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={"mm_token_type_ids": mm_token_type_ids},
            prefix_cache_chunks=[],
            all_tokens=[],
            next_prefix_cache_chunk_idx=0,
            image_spans=[],
        )

        generator.next()
        generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [8, 2]


def test_batch_generator_chunks_non_bidir_gemma4_normally(monkeypatch):
    """Non-bidirectional Gemma4 models keep their existing chunking behavior."""
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    monkeypatch.setattr(
        batcher,
        "make_prompt_cache",
        lambda _model: [_FakeBatchCache()],
    )
    model = _gemma4_non_bidir_model()
    generator = BatchGenerator(
        model=model,
        stop_criteria=lambda _token: False,
        prefill_step_size=4,
    )
    prompt = list(range(10))
    mm_token_type_ids = mx.array(
        [[0, 0, 0, 0, 0, 1, 1, 1, 0, 0]],
        dtype=mx.int32,
    )

    try:
        generator.insert(
            prompt,
            inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={"mm_token_type_ids": mm_token_type_ids},
            prefix_cache_chunks=[],
            all_tokens=[],
            next_prefix_cache_chunk_idx=0,
            image_spans=[],
        )

        generator.next()
        generator.next()
        generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [4, 4, 2]


def test_batch_generator_aligns_restored_prefill_only_for_cache_saves(monkeypatch):
    """Restored prefill alignment is only worth paying for disk snapshots."""
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())

    def call_lengths(prompt_cache_save_callback, steps: int):
        model = _FakeModel()
        generator = BatchGenerator(
            model=model,
            stop_criteria=lambda _token: False,
            prefill_step_size=4,
        )
        prompt = [10, 11, 12, 13, 14, 15, 16]

        try:
            generator.insert(
                prompt,
                inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
                sampler=_argmax_sampler,
                logits_processors=[],
                prompt_kwargs={},
                prefix_cache_chunks=[],
                image_spans=[],
                cache=[_FakeScalarCache()],
                all_tokens=[0, 1],
                next_prefix_cache_chunk_idx=0,
                prompt_cache_save_callback=prompt_cache_save_callback,
            )

            for _ in range(steps):
                generator.next()
        finally:
            generator.close()

        return [len(call["input_ids"][0]) for call in model.calls]

    assert call_lengths(None, steps=2) == [4, 3]
    assert call_lengths(lambda *_args, **_kwargs: None, steps=3) == [2, 4, 1]


def test_batch_generator_state_cache_lands_on_reusable_tail_boundary(monkeypatch):
    """Opaque state caches need an exact checkpoint at the final chunk boundary.

    The final reusable chunk may be shorter than the prefix chunk size. The
    prefill must land on its exact end so the opaque state checkpoint is saved
    at the same boundary as the KV delta. Otherwise the final snapshot saves
    the state one token ahead of the restored prefix, corrupting warm restore
    for stateful models such as LFM2.5-VL.
    """
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    monkeypatch.setattr(
        batcher,
        "make_prompt_cache",
        lambda _model: [ArraysCache()],
    )
    model = _FakeModel()
    generator = BatchGenerator(
        model=model,
        stop_criteria=lambda _token: False,
        prefill_step_size=2048,
    )
    snapshots = []
    prompt = list(range(1795))
    prefix_chunks = build_prefix_cache_chunks(prompt, [])

    def save_snapshot(
        cache,
        chunks,
        start_chunk_idx,
        end_chunk_idx,
        snapshot_len,
        *,
        is_final_prompt_boundary,
    ):
        snapshots.append(
            (
                cache,
                chunks,
                start_chunk_idx,
                end_chunk_idx,
                snapshot_len,
                is_final_prompt_boundary,
            )
        )

    try:
        generator.insert(
            prompt,
            inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={},
            prefix_cache_chunks=prefix_chunks,
            all_tokens=[],
            next_prefix_cache_chunk_idx=0,
            image_spans=[],
            prompt_cache_save_callback=save_snapshot,
        )

        generator.next()
        generator.next()
        generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [3 * C, 258, 1]
    assert [
        (start_chunk_idx, end_chunk_idx, snapshot_len, is_final_prompt_boundary)
        for (
            _cache,
            _chunks,
            start_chunk_idx,
            end_chunk_idx,
            snapshot_len,
            is_final_prompt_boundary,
        ) in snapshots
    ] == [
        (0, 3, 3 * C, False),
        (3, 4, 1794, False),
        (3, 4, 1795, True),
    ]


def test_batch_generator_state_cache_opt_out_keeps_old_alignment(monkeypatch):
    """Setting the alignment env var to 0 restores the old prefill behavior."""
    monkeypatch.setenv("MLX_ENGINE_VLM_FINAL_CHUNK_STATE_ALIGN", "0")
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    monkeypatch.setattr(
        batcher,
        "make_prompt_cache",
        lambda _model: [ArraysCache()],
    )
    model = _FakeModel()
    generator = BatchGenerator(
        model=model,
        stop_criteria=lambda _token: False,
        prefill_step_size=2048,
    )
    snapshots = []
    prompt = list(range(1795))
    prefix_chunks = build_prefix_cache_chunks(prompt, [])

    def save_snapshot(
        cache,
        chunks,
        start_chunk_idx,
        end_chunk_idx,
        snapshot_len,
        *,
        is_final_prompt_boundary,
    ):
        snapshots.append(
            (
                cache,
                chunks,
                start_chunk_idx,
                end_chunk_idx,
                snapshot_len,
                is_final_prompt_boundary,
            )
        )

    try:
        generator.insert(
            prompt,
            inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={},
            prefix_cache_chunks=prefix_chunks,
            all_tokens=[],
            next_prefix_cache_chunk_idx=0,
            image_spans=[],
            prompt_cache_save_callback=save_snapshot,
        )

        generator.next()
        generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [3 * C, 259]
    assert [
        (start_chunk_idx, end_chunk_idx, snapshot_len, is_final_prompt_boundary)
        for (
            _cache,
            _chunks,
            start_chunk_idx,
            end_chunk_idx,
            snapshot_len,
            is_final_prompt_boundary,
        ) in snapshots
    ] == [
        (0, 3, 3 * C, False),
        (3, 4, 1795, True),
    ]


def test_batch_generator_defers_clear_cache_until_delay(monkeypatch):
    """Scheduled Metal cache cleanup should wait for the delay window."""
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    clear_calls = []
    monkeypatch.setattr(batcher, "_sync_and_clear_cache", lambda: clear_calls.append(1))

    generator = BatchGenerator(
        model=_FakeModel(),
        stop_criteria=lambda _token: False,
    )
    try:
        generator._steps_counter = 10
        generator._schedule_deferred_clear()

        assert generator._deferred_clear_at == 10 + batcher.DEFERRED_CLEAR_DELAY_STEPS

        generator._steps_counter = generator._deferred_clear_at - 1
        generator._maybe_clear_cache()
        assert clear_calls == []

        generator._steps_counter += 1
        generator._maybe_clear_cache()
        assert clear_calls == [1]
        assert generator._deferred_clear_at is None
    finally:
        generator.close()


def test_batch_generator_close_drains_deferred_clear(monkeypatch):
    """Closing the generator should not leave deferred Metal cleanup pending."""
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    clear_calls = []
    monkeypatch.setattr(batcher, "_sync_and_clear_cache", lambda: clear_calls.append(1))

    generator = BatchGenerator(
        model=_FakeModel(),
        stop_criteria=lambda _token: False,
    )
    generator._schedule_deferred_clear()

    generator.close()

    assert clear_calls == [1]
    assert generator._deferred_clear_at is None
