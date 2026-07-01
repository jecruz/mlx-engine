import pytest

import mlx.core as mx
import mlx_engine.model_kit.batched_vision.prompt_cache.cache_store as cache_store_module
from mlx_engine.model_kit.batched_vision.prompt_cache.cache_store import (
    VlmPromptCacheStore,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.chunks import (
    build_prefix_cache_chunks,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.types import (
    DEFAULT_PREFIX_CHUNK_SIZE,
    PreparedPromptMetadata,
    PromptImageSpan,
    RECORD_KIND_KV_DELTA,
    RECORD_KIND_ROTATING_DELTA,
    RECORD_KIND_STATE_CHECKPOINT,
    make_record_key,
)
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache


C = DEFAULT_PREFIX_CHUNK_SIZE


@pytest.fixture
def cache_store():
    store = VlmPromptCacheStore()
    yield store
    store.close()


def _kv_cache(prefix_len: int):
    cache = KVCache()
    keys = mx.arange(prefix_len, dtype=mx.float32).reshape(1, 1, prefix_len, 1)
    cache.state = (keys, keys + 1000)
    return cache


def _arrays_cache(prefix_len: int):
    cache = ArraysCache(size=1)
    cache[0] = mx.array([[prefix_len]], dtype=mx.int32)
    return cache


def _rotating_cache(prefix_len: int):
    window_size = 512
    window_start = max(0, prefix_len - window_size)
    keys = mx.arange(window_start, prefix_len, dtype=mx.float32).reshape(
        1,
        1,
        prefix_len - window_start,
        1,
    )
    cache = RotatingKVCache(max_size=window_size, keep=0)
    cache.state = (keys, keys + 2000)
    cache.offset = prefix_len
    cache._idx = keys.shape[2]
    return cache


def _prompt_cache(prefix_len: int):
    return [_kv_cache(prefix_len), _arrays_cache(prefix_len)]


def _rotating_prompt_cache(prefix_len: int):
    return [_kv_cache(prefix_len), _rotating_cache(prefix_len)]


def _mixed_prompt_cache(prefix_len: int):
    return [
        _kv_cache(prefix_len),
        _rotating_cache(prefix_len),
        _arrays_cache(prefix_len),
    ]


def _save_chunk(
    cache_store,
    chunk,
    chunks,
    prompt_cache,
    *,
    save_state_checkpoint=True,
    is_final_prompt_boundary=False,
):
    chunk_idx = chunks.index(chunk)
    # Production does prepare on generation thread, then commit on cache I/O.
    cache_store.commit_pending_save(
        cache_store.prepare_save(
            chunk=chunk,
            prefix_chunks=chunks[: chunk_idx + 1],
            prompt_cache=prompt_cache,
            save_state_checkpoint=save_state_checkpoint,
            is_final_prompt_boundary=is_final_prompt_boundary,
        )
    )


def _assert_two_chunk_restore(loaded):
    kv_keys, kv_values = loaded.prompt_cache[0].state
    boundary_state = loaded.prompt_cache[1][0]
    mx.eval(kv_keys, kv_values, boundary_state)

    expected_prefix_len = 2 * C
    assert loaded.cached_prefix_len == expected_prefix_len
    assert kv_keys.shape[2] == expected_prefix_len
    assert kv_keys[0, 0, 0, 0].item() == 0
    assert kv_keys[0, 0, -1, 0].item() == expected_prefix_len - 1
    assert kv_values[0, 0, -1, 0].item() == expected_prefix_len + 999
    assert boundary_state.item() == expected_prefix_len


def _assert_rotating_restore(loaded):
    kv_keys, _ = loaded.prompt_cache[0].state
    rotating_keys, _ = loaded.prompt_cache[1].state
    mx.eval(kv_keys, rotating_keys)

    expected_prefix_len = 3 * C
    assert loaded.cached_prefix_len == expected_prefix_len
    assert kv_keys.shape[2] == expected_prefix_len
    assert rotating_keys.shape[2] == C
    assert rotating_keys[0, 0, 0, 0].item() == expected_prefix_len - C
    assert rotating_keys[0, 0, -1, 0].item() == expected_prefix_len - 1


def test_cache_store_commits_and_restores_prefix_records(cache_store):
    prompt_input_ids = list(range((2 * C) + 100))
    chunks = build_prefix_cache_chunks(prompt_input_ids, [])

    # Two saved chunks should restore a two-chunk prefix.
    for chunk in chunks[:2]:
        _save_chunk(cache_store, chunk, chunks, _prompt_cache(chunk.end))

    restore_plan = cache_store.plan_longest_prefix_restore(prompt_input_ids, [])
    assert restore_plan is not None
    loaded = cache_store.load_restore_plan(restore_plan)
    _assert_two_chunk_restore(loaded)

    stats = cache_store.snapshot_stats()
    # Evicting one record should not destroy the best available restore.
    cache_store.commit_budget_update(stats.total_bytes - 1)

    restore_plan = cache_store.plan_longest_prefix_restore(prompt_input_ids, [])
    assert restore_plan is not None
    loaded = cache_store.load_restore_plan(restore_plan)
    _assert_two_chunk_restore(loaded)


def test_cache_store_restores_reusable_prefix_tail(cache_store):
    """Saving the final reusable tail should restore the longer prefix."""
    prompt_input_ids = list(range(600))
    chunks = build_prefix_cache_chunks(prompt_input_ids, [])

    for chunk in chunks:
        _save_chunk(cache_store, chunk, chunks, _prompt_cache(chunk.end))

    restore_plan = cache_store.plan_longest_prefix_restore(prompt_input_ids, [])
    assert restore_plan is not None
    loaded = cache_store.load_restore_plan(restore_plan)

    kv_keys, _ = loaded.prompt_cache[0].state
    boundary_state = loaded.prompt_cache[1][0]
    mx.eval(kv_keys, boundary_state)
    assert loaded.cached_prefix_len == 599
    assert kv_keys.shape[2] == 599
    assert boundary_state.item() == 599


def test_cache_store_skips_redundant_current_kv_when_span_exists(cache_store):
    """Two-chunk KV spans should replace redundant current-only KV records."""
    prompt_input_ids = list(range((3 * C) + 100))
    chunks = build_prefix_cache_chunks(prompt_input_ids, [])

    for chunk in chunks[:3]:
        _save_chunk(cache_store, chunk, chunks, _prompt_cache(chunk.end))

    current_only_key = (
        f"{make_record_key(chunks[2].key, RECORD_KIND_KV_DELTA)}"
        f":span:{chunks[2].start}:{chunks[2].end}"
    )
    span_key = (
        f"{make_record_key(chunks[2].key, RECORD_KIND_KV_DELTA)}"
        f":span:{chunks[1].start}:{chunks[2].end}"
    )
    stats = cache_store.snapshot_stats()
    assert current_only_key not in stats.record_sizes_by_key
    assert span_key in stats.record_sizes_by_key

    restore_plan = cache_store.plan_longest_prefix_restore(prompt_input_ids, [])
    assert restore_plan is not None
    assert span_key in restore_plan.record_keys_by_chunk_key[chunks[2].key]
    loaded = cache_store.load_restore_plan(restore_plan)

    kv_keys, _ = loaded.prompt_cache[0].state
    boundary_state = loaded.prompt_cache[1][0]
    mx.eval(kv_keys, boundary_state)
    assert loaded.cached_prefix_len == chunks[2].end
    assert kv_keys.shape[2] == chunks[2].end
    assert boundary_state.item() == chunks[2].end


def test_cache_store_terminal_packed_final_kv_is_default(cache_store):
    """Final-boundary full-prefix KV packing should be enabled by default."""
    prompt_input_ids = list(range((3 * C) + 100))
    chunks = build_prefix_cache_chunks(prompt_input_ids, [])

    for chunk in chunks[:2]:
        _save_chunk(cache_store, chunk, chunks, _prompt_cache(chunk.end))
    _save_chunk(
        cache_store,
        chunks[2],
        chunks,
        _prompt_cache(chunks[2].end),
        is_final_prompt_boundary=True,
    )

    record_key = make_record_key(chunks[2].key, RECORD_KIND_KV_DELTA)
    metadata = cache_store._record_metadata_by_key[record_key]
    assert metadata.chunk_span == [chunks[0].start, chunks[2].end]
    assert metadata.is_terminal_packed is True

    restore_plan = cache_store.plan_longest_prefix_restore(prompt_input_ids, [])
    assert restore_plan is not None
    assert restore_plan.record_keys_by_chunk_key == {
        chunks[0].key: [],
        chunks[1].key: [],
        chunks[2].key: [
            record_key,
            _record_key(chunks[2], RECORD_KIND_STATE_CHECKPOINT),
        ],
    }
    loaded = cache_store.load_restore_plan(restore_plan)
    kv_keys, _ = loaded.prompt_cache[0].state
    boundary_state = loaded.prompt_cache[1][0]
    mx.eval(kv_keys, boundary_state)
    assert loaded.cached_prefix_len == chunks[2].end
    assert kv_keys.shape[2] == chunks[2].end
    assert boundary_state.item() == chunks[2].end


def test_cache_store_terminal_packed_final_kv_can_be_disabled(
    cache_store, monkeypatch
):
    """Operators can disable terminal-packed final KV with an explicit env flag."""
    monkeypatch.setenv("MLX_ENGINE_VLM_TERMINAL_PACKED_FINAL_KV", "0")
    prompt_input_ids = list(range((3 * C) + 100))
    chunks = build_prefix_cache_chunks(prompt_input_ids, [])

    for chunk in chunks[:2]:
        _save_chunk(cache_store, chunk, chunks, _prompt_cache(chunk.end))
    _save_chunk(
        cache_store,
        chunks[2],
        chunks,
        _prompt_cache(chunks[2].end),
        is_final_prompt_boundary=True,
    )

    record_key = (
        f"{make_record_key(chunks[2].key, RECORD_KIND_KV_DELTA)}"
        f":span:{chunks[1].start}:{chunks[2].end}"
    )
    metadata = cache_store._record_metadata_by_key[record_key]
    assert metadata.chunk_span == [chunks[1].start, chunks[2].end]
    assert metadata.is_terminal_packed is False


def _record_key(chunk, record_kind):
    return make_record_key(chunk.key, record_kind)


def test_cache_store_records_save_and_restore_latency_metrics(cache_store):
    """Snapshot stats include cache-store save and restore timing counters."""
    prompt_input_ids = list(range(600))
    chunks = build_prefix_cache_chunks(prompt_input_ids, [])

    initial_stats = cache_store.snapshot_stats()
    assert initial_stats.save_count == 0
    assert initial_stats.save_latency_ms == 0.0
    assert initial_stats.restore_count == 0
    assert initial_stats.restore_latency_ms == 0.0

    _save_chunk(cache_store, chunks[0], chunks, _prompt_cache(chunks[0].end))
    save_stats = cache_store.snapshot_stats()
    assert save_stats.save_count == 1
    assert save_stats.save_latency_ms >= 0.0
    assert save_stats.restore_count == 0

    restore_plan = cache_store.plan_longest_prefix_restore(prompt_input_ids, [])
    assert restore_plan is not None
    cache_store.load_restore_plan(restore_plan)

    restore_stats = cache_store.snapshot_stats()
    assert restore_stats.save_count == 1
    assert restore_stats.restore_count == 1
    assert restore_stats.restore_latency_ms >= 0.0


def test_cache_store_logs_profiled_record_load_timing(
    cache_store,
    monkeypatch,
):
    """Timing diagnostics split record loads into deserialization stages."""
    prompt_input_ids = list(range(600))
    chunks = build_prefix_cache_chunks(prompt_input_ids, [])
    _save_chunk(cache_store, chunks[0], chunks, _prompt_cache(chunks[0].end))

    restore_plan = cache_store.plan_longest_prefix_restore(prompt_input_ids, [])
    assert restore_plan is not None

    events = []
    monkeypatch.setenv("MLX_ENGINE_BATCHED_TIMING", "1")
    monkeypatch.setattr(
        cache_store_module,
        "log_batched_timing",
        lambda logger, event, **fields: events.append(
            {"event": event, **fields}
        ),
    )
    cache_store.load_restore_plan(restore_plan)

    record_loads = [
        event for event in events if event["event"] == "vlm_cache_record_load"
    ]
    assert record_loads
    for event in record_loads:
        assert event["duration_ms"] >= 0.0
        assert event["safetensor_load_ms"] >= 0.0
        assert event["unflatten_ms"] >= 0.0
        assert event["cache_rebuild_ms"] >= 0.0


def test_cache_store_logs_restore_materialization_counters(
    cache_store,
    monkeypatch,
):
    """Restore detail timing includes eval/materialization counters by kind."""
    prompt_input_ids = list(range(600))
    chunks = build_prefix_cache_chunks(prompt_input_ids, [])
    _save_chunk(
        cache_store,
        chunks[0],
        chunks,
        _mixed_prompt_cache(chunks[0].end),
    )

    restore_plan = cache_store.plan_longest_prefix_restore(prompt_input_ids, [])
    assert restore_plan is not None

    events = []
    monkeypatch.setenv("MLX_ENGINE_BATCHED_TIMING", "1")
    monkeypatch.setattr(
        cache_store_module,
        "log_batched_timing",
        lambda logger, event, **fields: events.append(
            {"event": event, **fields}
        ),
    )
    cache_store.load_restore_plan(restore_plan)

    restore_detail = next(
        event for event in events if event["event"] == "vlm_cache_restore_detail"
    )
    assert restore_detail["eval_target_count"] > 0
    assert restore_detail["materialized_bytes"] > 0
    assert restore_detail["eval_target_count"] == sum(
        restore_detail["eval_target_count_by_kind"].values()
    )
    assert restore_detail["materialized_bytes"] == sum(
        restore_detail["materialized_bytes_by_kind"].values()
    )
    assert restore_detail["record_count_by_kind"] == {
        RECORD_KIND_KV_DELTA: 1,
        RECORD_KIND_ROTATING_DELTA: 1,
        RECORD_KIND_STATE_CHECKPOINT: 1,
    }
    assert set(restore_detail["record_bytes_by_kind"]) == {
        RECORD_KIND_KV_DELTA,
        RECORD_KIND_ROTATING_DELTA,
        RECORD_KIND_STATE_CHECKPOINT,
    }
    assert restore_detail["record_bytes"] == sum(
        restore_detail["record_bytes_by_kind"].values()
    )
    for record_kind in (
        RECORD_KIND_KV_DELTA,
        RECORD_KIND_ROTATING_DELTA,
        RECORD_KIND_STATE_CHECKPOINT,
    ):
        assert restore_detail["eval_target_count_by_kind"][record_kind] > 0
        assert restore_detail["materialized_bytes_by_kind"][record_kind] > 0


def test_cache_store_restore_eval_barrier_materializes_disk_restore(
    cache_store,
    monkeypatch,
):
    """Disk restore keeps the mx.eval barrier before returning restored cache."""
    prompt_input_ids = list(range(600))
    chunks = build_prefix_cache_chunks(prompt_input_ids, [])
    _save_chunk(cache_store, chunks[0], chunks, _prompt_cache(chunks[0].end))

    restore_plan = cache_store.plan_longest_prefix_restore(prompt_input_ids, [])
    assert restore_plan is not None

    eval_calls = []
    real_eval = cache_store_module.mx.eval

    def recording_eval(*args, **kwargs):
        eval_calls.append(args)
        return real_eval(*args, **kwargs)

    monkeypatch.setattr(cache_store_module.mx, "eval", recording_eval)
    loaded = cache_store.load_restore_plan(restore_plan)

    restore_eval_calls = [
        args
        for args in eval_calls
        if args and all(isinstance(arg, mx.array) for arg in args)
    ]
    assert restore_eval_calls
    assert restore_eval_calls[-1]
    kv_keys, _ = loaded.prompt_cache[0].state
    boundary_state = loaded.prompt_cache[1][0]
    real_eval(kv_keys, boundary_state)
    assert loaded.cached_prefix_len == chunks[0].end
    assert kv_keys.shape[2] == chunks[0].end
    assert boundary_state.item() == chunks[0].end


def test_cache_store_diagnostic_mixed_restore_materializes_before_handoff(
    cache_store,
    monkeypatch,
):
    """Mixed KV/rotating/state restores cross the restore-time mx.eval barrier."""
    prompt_input_ids = list(range(600))
    chunks = build_prefix_cache_chunks(prompt_input_ids, [])
    _save_chunk(
        cache_store,
        chunks[0],
        chunks,
        _mixed_prompt_cache(chunks[0].end),
    )

    restore_plan = cache_store.plan_longest_prefix_restore(prompt_input_ids, [])
    assert restore_plan is not None

    events = []
    eval_calls = []
    real_eval = cache_store_module.mx.eval

    def recording_eval(*args, **kwargs):
        eval_calls.append(args)
        return real_eval(*args, **kwargs)

    monkeypatch.setenv("MLX_ENGINE_BATCHED_TIMING", "1")
    monkeypatch.setattr(cache_store_module.mx, "eval", recording_eval)
    monkeypatch.setattr(
        cache_store_module,
        "log_batched_timing",
        lambda logger, event, **fields: events.append(
            {"event": event, **fields}
        ),
    )

    loaded = cache_store.load_restore_plan(restore_plan)

    restore_eval_calls = [
        args
        for args in eval_calls
        if args and all(isinstance(arg, mx.array) for arg in args)
    ]
    assert restore_eval_calls
    restore_eval_targets = restore_eval_calls[-1]
    restore_detail = next(
        event for event in events if event["event"] == "vlm_cache_restore_detail"
    )
    assert len(restore_eval_targets) == restore_detail["eval_target_count"]
    assert restore_detail["record_count_by_kind"] == {
        RECORD_KIND_KV_DELTA: 1,
        RECORD_KIND_ROTATING_DELTA: 1,
        RECORD_KIND_STATE_CHECKPOINT: 1,
    }
    for record_kind in (
        RECORD_KIND_KV_DELTA,
        RECORD_KIND_ROTATING_DELTA,
        RECORD_KIND_STATE_CHECKPOINT,
    ):
        assert restore_detail["eval_target_count_by_kind"][record_kind] > 0
        assert restore_detail["materialized_bytes_by_kind"][record_kind] > 0

    kv_keys, _ = loaded.prompt_cache[0].state
    rotating_keys, _ = loaded.prompt_cache[1].state
    state_checkpoint = loaded.prompt_cache[2][0]
    real_eval(kv_keys, rotating_keys, state_checkpoint)
    assert loaded.cached_prefix_len == chunks[0].end
    assert kv_keys.shape[2] == chunks[0].end
    assert rotating_keys.shape[2] == chunks[0].end
    assert state_checkpoint.item() == chunks[0].end


def test_persistent_cache_store_restores_after_reopen(tmp_path):
    """Persistent cache records survive a store reopen."""
    prompt_input_ids = list(range(600))
    chunks = build_prefix_cache_chunks(prompt_input_ids, [])
    storage_root = tmp_path / "prompt-cache"
    cache_store = VlmPromptCacheStore(
        storage_root=storage_root,
        cache_namespace="model-a",
    )

    try:
        _save_chunk(cache_store, chunks[0], chunks, _prompt_cache(chunks[0].end))
        assert cache_store.snapshot_stats().entry_count > 0
    finally:
        cache_store.close()

    reopened = VlmPromptCacheStore(
        storage_root=storage_root,
        cache_namespace="model-a",
    )
    try:
        restore_plan = reopened.plan_longest_prefix_restore(prompt_input_ids, [])
        assert restore_plan is not None
        loaded = reopened.load_restore_plan(restore_plan)
        kv_keys, _ = loaded.prompt_cache[0].state
        mx.eval(kv_keys)
        assert loaded.cached_prefix_len == chunks[0].end
        assert kv_keys.shape[2] == chunks[0].end
    finally:
        reopened.close()


def test_persistent_cache_store_loads_legacy_v1_record_metadata(tmp_path):
    """Format-v1 records remain readable when optional metadata keys are absent."""
    prompt_input_ids = list(range(600))
    chunks = build_prefix_cache_chunks(prompt_input_ids, [])
    storage_root = tmp_path / "prompt-cache"
    cache_store = VlmPromptCacheStore(
        storage_root=storage_root,
        cache_namespace="model-a",
    )

    try:
        _save_chunk(cache_store, chunks[0], chunks, _prompt_cache(chunks[0].end))
    finally:
        cache_store.close()

    [index_path] = storage_root.glob("*/prompt-cache-index.json")
    data = cache_store_module.json.loads(index_path.read_text())
    assert data["format_version"] == 1
    for record_metadata in data["records"].values():
        record_metadata.pop("chunk_span", None)
        record_metadata.pop("is_terminal_packed", None)
    index_path.write_text(cache_store_module.json.dumps(data, sort_keys=True))

    reopened = VlmPromptCacheStore(
        storage_root=storage_root,
        cache_namespace="model-a",
    )
    try:
        restore_plan = reopened.plan_longest_prefix_restore(prompt_input_ids, [])
        assert restore_plan is not None
        loaded = reopened.load_restore_plan(restore_plan)
        kv_keys, _ = loaded.prompt_cache[0].state
        boundary_state = loaded.prompt_cache[1][0]
        mx.eval(kv_keys, boundary_state)
        assert loaded.cached_prefix_len == chunks[0].end
        assert kv_keys.shape[2] == chunks[0].end
        assert boundary_state.item() == chunks[0].end
    finally:
        reopened.close()


def test_persistent_cache_store_namespace_mismatch_is_safe_miss(tmp_path):
    """Persistent cache indexes are isolated by namespace."""
    prompt_input_ids = list(range(600))
    chunks = build_prefix_cache_chunks(prompt_input_ids, [])
    storage_root = tmp_path / "prompt-cache"
    cache_store = VlmPromptCacheStore(
        storage_root=storage_root,
        cache_namespace="model-a",
    )

    try:
        _save_chunk(cache_store, chunks[0], chunks, _prompt_cache(chunks[0].end))
    finally:
        cache_store.close()

    mismatched = VlmPromptCacheStore(
        storage_root=storage_root,
        cache_namespace="model-b",
    )
    try:
        assert mismatched.snapshot_stats().entry_count == 0
        assert mismatched.plan_longest_prefix_restore(prompt_input_ids, []) is None
    finally:
        mismatched.close()


def test_persistent_cache_store_restores_prepared_prompt_metadata(tmp_path):
    """Persistent indexes retain exact prepared-prompt metadata across reopen."""
    prompt_input_ids = list(range(600))
    chunks = build_prefix_cache_chunks(prompt_input_ids, [])
    storage_root = tmp_path / "prompt-cache"
    cache_store = VlmPromptCacheStore(
        storage_root=storage_root,
        cache_namespace="model-a",
    )
    metadata = PreparedPromptMetadata(
        request_key="request-a",
        prompt_input_ids=[1, 20, 20, 2],
        image_spans=[PromptImageSpan(1, 3, "image")],
        vision_cache_key="prepared-images:image",
        image_grid_thw=[[1, 1, 2]],
    )

    try:
        cache_store.remember_prepared_prompt_metadata(
            metadata,
            prefix_chunks=chunks,
        )
    finally:
        cache_store.close()

    reopened = VlmPromptCacheStore(
        storage_root=storage_root,
        cache_namespace="model-a",
    )
    try:
        restored = reopened.lookup_prepared_prompt_metadata("request-a")
        assert restored == metadata
    finally:
        reopened.close()


def test_persistent_cache_store_skips_small_prompts_by_default(tmp_path):
    """Persistent cache admission should avoid disk records for tiny prompts."""
    prompt_input_ids = list(range(300))
    chunks = build_prefix_cache_chunks(prompt_input_ids, [])
    storage_root = tmp_path / "prompt-cache"
    cache_store = VlmPromptCacheStore(
        storage_root=storage_root,
        cache_namespace="small",
    )

    try:
        assert not cache_store.should_save_prompt(chunks)
        assert cache_store.snapshot_stats().entry_count == 0
    finally:
        cache_store.close()


def test_persistent_cache_store_override_allows_small_prompt_save(tmp_path):
    """Persistent cache admission threshold is configurable for experiments."""
    prompt_input_ids = list(range(300))
    chunks = build_prefix_cache_chunks(prompt_input_ids, [])
    cache_store = VlmPromptCacheStore(
        storage_root=tmp_path / "prompt-cache",
        cache_namespace="small",
        min_save_tokens=0,
    )

    try:
        assert cache_store.should_save_prompt(chunks)
        _save_chunk(cache_store, chunks[-1], chunks, _prompt_cache(chunks[-1].end))
        assert cache_store.snapshot_stats().entry_count > 0
    finally:
        cache_store.close()


def test_persistent_cache_store_saves_long_prompts_by_default(tmp_path):
    """Persistent cache admission should keep long-context restore chains."""
    prompt_input_ids = list(range(900))
    chunks = build_prefix_cache_chunks(prompt_input_ids, [])
    cache_store = VlmPromptCacheStore(
        storage_root=tmp_path / "prompt-cache",
        cache_namespace="long",
    )

    try:
        assert cache_store.should_save_prompt(chunks)
        for chunk in chunks:
            _save_chunk(cache_store, chunk, chunks, _prompt_cache(chunk.end))
        restore_plan = cache_store.plan_longest_prefix_restore(prompt_input_ids, [])
        assert restore_plan is not None
    finally:
        cache_store.close()


def test_cache_store_eviction_preserves_shorter_prefix_restore(cache_store):
    prompt_input_ids = list(range(600))
    chunks = build_prefix_cache_chunks(prompt_input_ids, [])
    first_chunk, second_chunk = chunks[:2]

    # Budget enough for the first chunk so adding the second evicts the suffix.
    _save_chunk(cache_store, first_chunk, chunks, [_kv_cache(first_chunk.end)])
    first_size = cache_store.snapshot_stats().chunk_sizes_by_key[first_chunk.key]
    cache_store.commit_budget_update(first_size + 64)

    _save_chunk(cache_store, second_chunk, chunks, [_kv_cache(second_chunk.end)])

    restore_plan = cache_store.plan_longest_prefix_restore(prompt_input_ids, [])
    assert restore_plan is not None
    loaded = cache_store.load_restore_plan(restore_plan)

    assert loaded.cached_prefix_len >= first_chunk.end
    assert cache_store.snapshot_stats().total_bytes <= cache_store.snapshot_stats().max_bytes


def test_cache_store_eviction_preserves_shorter_state_checkpoint_restore(cache_store):
    """Over-budget suffix saves should not destroy the shorter stateful restore."""
    prompt_input_ids = list(range(600))
    chunks = build_prefix_cache_chunks(prompt_input_ids, [])
    first_chunk, second_chunk = chunks[:2]

    # This budget can keep the first chunk's KV plus exact opaque checkpoint.
    _save_chunk(cache_store, first_chunk, chunks, _prompt_cache(first_chunk.end))
    first_size = cache_store.snapshot_stats().total_bytes
    cache_store.commit_budget_update(first_size + 64)

    _save_chunk(cache_store, second_chunk, chunks, _prompt_cache(second_chunk.end))

    restore_plan = cache_store.plan_longest_prefix_restore(prompt_input_ids, [])
    assert restore_plan is not None
    loaded = cache_store.load_restore_plan(restore_plan)

    kv_keys, _ = loaded.prompt_cache[0].state
    boundary_state = loaded.prompt_cache[1][0]
    mx.eval(kv_keys, boundary_state)
    assert loaded.cached_prefix_len >= first_chunk.end
    assert kv_keys.shape[2] == loaded.cached_prefix_len
    assert boundary_state.item() == loaded.cached_prefix_len


def test_cache_store_skips_state_for_backfilled_chunks(cache_store):
    """Backfilled KV chunks do not advertise stale opaque state checkpoints."""
    prompt_input_ids = list(range((2 * C) + 188))
    chunks = build_prefix_cache_chunks(prompt_input_ids, [])

    # A two-chunk model call can backfill the first chunk KV, but its opaque
    # state is exact only at the second chunk boundary.
    _save_chunk(
        cache_store,
        chunks[0],
        chunks,
        _prompt_cache(2 * C),
        save_state_checkpoint=False,
    )
    _save_chunk(cache_store, chunks[1], chunks, _prompt_cache(2 * C))

    stats = cache_store.snapshot_stats()
    old_state_record_key = make_record_key(
        chunks[0].key,
        RECORD_KIND_STATE_CHECKPOINT,
    )
    target_state_record_key = make_record_key(
        chunks[1].key,
        RECORD_KIND_STATE_CHECKPOINT,
    )
    assert old_state_record_key not in stats.record_sizes_by_key
    assert target_state_record_key in stats.record_sizes_by_key

    restore_plan = cache_store.plan_longest_prefix_restore(prompt_input_ids, [])
    assert restore_plan is not None
    loaded = cache_store.load_restore_plan(restore_plan)
    _assert_two_chunk_restore(loaded)


def test_cache_store_does_not_restore_to_prefix_inside_image_span(cache_store):
    """Disk chunks inside images are internal records, not terminal restore points."""
    prompt_input_ids = list(range((3 * C) + 264))
    image_spans = [PromptImageSpan(start=200, end=(2 * C) + 176, image_hash="image")]
    chunks = build_prefix_cache_chunks(prompt_input_ids, image_spans)

    _save_chunk(cache_store, chunks[0], chunks, _prompt_cache(chunks[0].end))
    _save_chunk(cache_store, chunks[1], chunks, _prompt_cache(chunks[1].end))

    assert (
        cache_store.plan_longest_prefix_restore(prompt_input_ids, image_spans) is None
    )

    _save_chunk(cache_store, chunks[2], chunks, _prompt_cache(chunks[2].end))

    restore_plan = cache_store.plan_longest_prefix_restore(
        prompt_input_ids,
        image_spans,
    )

    assert restore_plan is not None
    assert restore_plan.cached_prefix_len == chunks[2].end


def test_cache_store_rotating_restore_uses_target_window(cache_store):
    prompt_input_ids = list(range((3 * C) + 100))
    chunks = build_prefix_cache_chunks(prompt_input_ids, [])

    # Full KV needs every chunk; rotating KV only needs the target window.
    for chunk in chunks[:3]:
        _save_chunk(cache_store, chunk, chunks, _rotating_prompt_cache(chunk.end))

    restore_plan = cache_store.plan_longest_prefix_restore(prompt_input_ids, [])
    assert restore_plan is not None
    loaded = cache_store.load_restore_plan(restore_plan)
    _assert_rotating_restore(loaded)

    old_rotating_record_key = make_record_key(
        chunks[0].key,
        RECORD_KIND_ROTATING_DELTA,
    )
    stats = cache_store.snapshot_stats()
    assert old_rotating_record_key in stats.record_sizes_by_key

    # The first rotating record is outside the target SWA window.
    cache_store.commit_budget_update(stats.total_bytes - 1)
    stats = cache_store.snapshot_stats()
    assert old_rotating_record_key not in stats.record_sizes_by_key

    restore_plan = cache_store.plan_longest_prefix_restore(prompt_input_ids, [])
    assert restore_plan is not None
    loaded = cache_store.load_restore_plan(restore_plan)
    _assert_rotating_restore(loaded)
