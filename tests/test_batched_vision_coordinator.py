import mlx.core as mx
import mlx_engine.model_kit.batched_vision.prompt_cache.coordinator as coordinator_module
from mlx_engine.model_kit.batched_vision.prompt_cache.cache_store import (
    DiskPromptCacheRestorePlan,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.chunks import (
    build_prefix_cache_chunks,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.coordinator import (
    VlmPromptCacheCoordinator,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.types import (
    LoadedDiskPromptCache,
    PromptImageSpan,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache


class _FakeCacheStore:
    def __init__(
        self,
        *,
        disk_prefix_len: int | None = None,
        allow_save: bool = True,
    ):
        self.restore_plan = (
            None
            if disk_prefix_len is None
            else DiskPromptCacheRestorePlan(
                cached_prefix_len=disk_prefix_len,
                chunks=[],
                record_keys_by_chunk_key={},
            )
        )
        self.loaded_state = (
            None
            if disk_prefix_len is None
            else LoadedDiskPromptCache(
                cached_prefix_len=disk_prefix_len,
                prompt_cache=["disk-cache"],
            )
        )
        self.recorded_tokens = []
        self.loaded_plans = []
        self.allow_save = allow_save
        self.prepared_chunks = []

    def plan_longest_prefix_restore(self, prompt_input_ids, image_spans):
        return self.restore_plan

    def load_restore_plan(self, plan):
        self.loaded_plans.append(plan)
        return self.loaded_state

    def record_restore_tokens(self, *, hit_tokens: int, miss_tokens: int):
        self.recorded_tokens.append((hit_tokens, miss_tokens))
        return (
            sum(hit for hit, _miss in self.recorded_tokens),
            sum(miss for _hit, miss in self.recorded_tokens),
        )

    def can_store_records(self) -> bool:
        return True

    def should_save_prompt(self, prefix_chunks) -> bool:
        return self.allow_save

    def prepare_save(
        self,
        *,
        chunk,
        prefix_chunks,
        prompt_cache,
        save_state_checkpoint,
        is_final_prompt_boundary=False,
    ):
        self.prepared_chunks.append(chunk)
        return {
            "chunk": chunk,
            "save_state_checkpoint": save_state_checkpoint,
            "is_final_prompt_boundary": is_final_prompt_boundary,
        }


def _coordinator(cache_store):
    enqueued_saves = []
    coordinator = VlmPromptCacheCoordinator(cache_store, enqueued_saves.append)
    return coordinator, enqueued_saves


def _kv_cache(prefix_len: int):
    cache = KVCache()
    keys = mx.arange(prefix_len, dtype=mx.float32).reshape(1, 1, prefix_len, 1)
    cache.state = (keys, keys + 1000)
    return cache


def _rotating_kv_cache(prefix_len: int, *, max_size: int):
    cache = RotatingKVCache(max_size=max_size, keep=0)
    keys = mx.arange(prefix_len, dtype=mx.float32).reshape(1, 1, prefix_len, 1)
    cache.update_and_fetch(keys, keys + 1000)
    return cache


def test_coordinator_skips_save_when_store_admission_rejects_prompt():
    """Prompt-cache admission should skip prepare/enqueue for small prompts."""
    cache_store = _FakeCacheStore(allow_save=False)
    coordinator, enqueued_saves = _coordinator(cache_store)
    chunks = build_prefix_cache_chunks(list(range(300)), [])

    coordinator.save_prompt_cache_snapshot(
        prompt_cache=[_kv_cache(chunks[-1].end)],
        prefix_chunks=chunks,
        start_chunk_idx=0,
        end_chunk_idx=len(chunks),
        snapshot_len=chunks[-1].end,
    )

    assert cache_store.prepared_chunks == []
    assert enqueued_saves == []


def test_coordinator_marks_only_final_chunk_as_final_prompt_boundary():
    """Final prompt snapshots only mark the terminal crossed chunk."""
    cache_store = _FakeCacheStore()
    coordinator, enqueued_saves = _coordinator(cache_store)
    chunks = build_prefix_cache_chunks(list(range(1600)), [])

    coordinator.save_prompt_cache_snapshot(
        prompt_cache=[_kv_cache(chunks[-1].end)],
        prefix_chunks=chunks,
        start_chunk_idx=0,
        end_chunk_idx=len(chunks),
        snapshot_len=chunks[-1].end,
        is_final_prompt_boundary=True,
    )

    expected_final_flags = [False] * (len(chunks) - 1) + [True]
    assert [
        save["is_final_prompt_boundary"] for save in enqueued_saves
    ] == expected_final_flags
    assert [
        save["save_state_checkpoint"] for save in enqueued_saves
    ] == expected_final_flags


def test_coordinator_skips_one_token_ahead_final_state_checkpoint():
    """Final snapshots keep exact state from the reusable-prefix boundary."""
    cache_store = _FakeCacheStore()
    coordinator, enqueued_saves = _coordinator(cache_store)
    prompt = list(range(1600))
    chunks = build_prefix_cache_chunks(prompt, [])

    coordinator.save_prompt_cache_snapshot(
        prompt_cache=[_kv_cache(len(prompt))],
        prefix_chunks=chunks,
        start_chunk_idx=len(chunks) - 1,
        end_chunk_idx=len(chunks),
        snapshot_len=len(prompt),
        is_final_prompt_boundary=True,
    )

    assert [save["is_final_prompt_boundary"] for save in enqueued_saves] == [True]
    assert [save["save_state_checkpoint"] for save in enqueued_saves] == [False]


def test_coordinator_final_state_checkpoint_opt_out_keeps_old_alignment(monkeypatch):
    """Operators can opt out and keep the old final-state checkpoint behavior."""
    monkeypatch.setenv("MLX_ENGINE_VLM_FINAL_CHUNK_STATE_ALIGN", "0")
    cache_store = _FakeCacheStore()
    coordinator, enqueued_saves = _coordinator(cache_store)
    prompt = list(range(1600))
    chunks = build_prefix_cache_chunks(prompt, [])

    coordinator.save_prompt_cache_snapshot(
        prompt_cache=[_kv_cache(len(prompt))],
        prefix_chunks=chunks,
        start_chunk_idx=len(chunks) - 1,
        end_chunk_idx=len(chunks),
        snapshot_len=len(prompt),
        is_final_prompt_boundary=True,
    )

    assert [save["is_final_prompt_boundary"] for save in enqueued_saves] == [True]
    assert [save["save_state_checkpoint"] for save in enqueued_saves] == [True]


def test_coordinator_restores_exact_hot_prefix():
    """A completed hot cache serves an immediate follow-up and preserves rope."""
    cache_store = _FakeCacheStore()
    coordinator, _ = _coordinator(cache_store)
    prompt_cache = [_kv_cache(512)]
    rope_deltas = object()

    coordinator.store_hot_prompt_cache(
        prompt_input_ids=list(range(512)),
        image_spans=[],
        prompt_cache=prompt_cache,
        rope_deltas=rope_deltas,
    )
    restored = coordinator.restore(
        prompt_input_ids=list(range(512)) + [999],
        image_spans=[],
    )

    assert restored is not None
    assert restored.cached_prefix_len == 512
    assert restored.prompt_cache is prompt_cache
    assert restored.rope_deltas is rope_deltas
    assert cache_store.recorded_tokens == [(512, 0)]


def test_coordinator_trims_hot_branch_and_drops_rope():
    """A branch from hot cache trims reusable KV and recomputes rope state."""
    cache_store = _FakeCacheStore()
    coordinator, _ = _coordinator(cache_store)
    prompt_cache = [_kv_cache(768)]

    coordinator.store_hot_prompt_cache(
        prompt_input_ids=list(range(768)),
        image_spans=[],
        prompt_cache=prompt_cache,
        rope_deltas=object(),
    )
    restored = coordinator.restore(
        prompt_input_ids=list(range(512)) + [-1],
        image_spans=[],
    )
    kv_keys, _ = prompt_cache[0].state
    mx.eval(kv_keys)

    assert restored is not None
    assert restored.cached_prefix_len == 512
    assert restored.prompt_cache is prompt_cache
    assert restored.rope_deltas is None
    assert kv_keys.shape[2] == 512
    assert cache_store.recorded_tokens == [(512, 0)]


def test_coordinator_trims_hidden_reasoning_for_trimmable_image_followup():
    """Trimmable hot caches can restore the prompt boundary after hidden reasoning."""
    cache_store = _FakeCacheStore()
    coordinator, _ = _coordinator(cache_store)
    base_prompt = list(range(300))
    reasoning_tokens = [1001, 1002]
    response_tokens = [2001, 2002]
    followup_tokens = [3001, 3002]
    image_spans = [PromptImageSpan(start=40, end=90, image_hash="image")]
    completed_tokens = base_prompt + reasoning_tokens + response_tokens
    followup_prompt = base_prompt + response_tokens + followup_tokens
    prompt_cache = [_kv_cache(len(completed_tokens))]

    coordinator.store_hot_prompt_cache(
        prompt_input_ids=completed_tokens,
        image_spans=image_spans,
        prompt_cache=prompt_cache,
        rope_deltas=object(),
    )
    restored = coordinator.restore(
        prompt_input_ids=followup_prompt,
        image_spans=image_spans,
    )
    kv_keys, _ = prompt_cache[0].state
    mx.eval(kv_keys)

    assert restored is not None
    assert restored.cached_prefix_len == len(base_prompt)
    assert restored.prompt_cache is prompt_cache
    assert restored.rope_deltas is None
    assert kv_keys.shape[2] == len(base_prompt)
    assert cache_store.recorded_tokens == [
        (len(base_prompt), len(followup_prompt) - 1 - len(base_prompt))
    ]


def test_coordinator_uses_disk_when_hidden_reasoning_hot_cache_cannot_trim():
    """Gemma4-style rotating caches may force fallback to a chunked disk restore."""
    cache_store = _FakeCacheStore(disk_prefix_len=768)
    coordinator, _ = _coordinator(cache_store)
    base_prompt = list(range(1200))
    reasoning_tokens = [2001, 2002]
    response_tokens = [3001, 3002]
    followup_tokens = [4001, 4002]
    image_spans = [PromptImageSpan(start=500, end=760, image_hash="image")]
    completed_tokens = base_prompt + reasoning_tokens + response_tokens
    followup_prompt = base_prompt + response_tokens + followup_tokens
    prompt_cache = [_rotating_kv_cache(len(completed_tokens), max_size=1024)]

    coordinator.store_hot_prompt_cache(
        prompt_input_ids=completed_tokens,
        image_spans=image_spans,
        prompt_cache=prompt_cache,
        rope_deltas=object(),
    )
    restored = coordinator.restore(
        prompt_input_ids=followup_prompt,
        image_spans=image_spans,
    )

    assert restored is not None
    assert restored.cached_prefix_len == 768
    assert restored.prompt_cache == ["disk-cache"]
    assert restored.rope_deltas is None
    assert cache_store.loaded_plans == [cache_store.restore_plan]
    assert cache_store.recorded_tokens == [(768, len(followup_prompt) - 1 - 768)]


def test_coordinator_prefers_longer_disk_restore():
    """Disk wins when it can restore more tokens than the hot cache."""
    cache_store = _FakeCacheStore(disk_prefix_len=512)
    coordinator, _ = _coordinator(cache_store)

    coordinator.store_hot_prompt_cache(
        prompt_input_ids=list(range(256)),
        image_spans=[],
        prompt_cache=["hot-cache"],
        rope_deltas=object(),
    )
    restored = coordinator.restore(
        prompt_input_ids=list(range(700)),
        image_spans=[],
    )

    assert restored is not None
    assert restored.cached_prefix_len == 512
    assert restored.prompt_cache == ["disk-cache"]
    assert restored.rope_deltas is None
    assert cache_store.loaded_plans == [cache_store.restore_plan]
    assert cache_store.recorded_tokens == [(512, 187)]


def test_coordinator_disk_planning_timing_is_opt_in(monkeypatch):
    """Disk restore planning should not log timing unless timing is enabled."""
    cache_store = _FakeCacheStore(disk_prefix_len=512)
    coordinator, _ = _coordinator(cache_store)
    timing_events = []
    monkeypatch.setattr(coordinator_module, "batched_timing_enabled", lambda: False)
    monkeypatch.setattr(
        coordinator_module,
        "log_batched_timing",
        lambda *args, **kwargs: timing_events.append((args, kwargs)),
    )

    restored = coordinator.restore(
        prompt_input_ids=list(range(700)),
        image_spans=[],
    )

    assert restored is not None
    assert timing_events == []


def test_coordinator_logs_disk_planning_timing_when_enabled(monkeypatch):
    """Timing diagnostics should separate disk restore planning from loading."""
    cache_store = _FakeCacheStore(disk_prefix_len=512)
    cache_store.restore_plan.chunks = [object(), object()]
    coordinator, _ = _coordinator(cache_store)
    timing_events = []
    monkeypatch.setattr(coordinator_module, "batched_timing_enabled", lambda: True)
    monkeypatch.setattr(coordinator_module, "elapsed_ms", lambda start: 1.25)
    monkeypatch.setattr(
        coordinator_module,
        "log_batched_timing",
        lambda logger, event, **fields: timing_events.append((event, fields)),
    )

    restored = coordinator.restore(
        prompt_input_ids=list(range(700)),
        image_spans=[PromptImageSpan(start=10, end=20, image_hash="image")],
    )

    assert restored is not None
    assert timing_events == [
        (
            "vlm_cache_restore_plan",
            {
                "prompt_tokens": 700,
                "images": 1,
                "cached_tokens": 512,
                "chunks": 2,
                "outcome": "hit",
                "duration_ms": 1.25,
            },
        )
    ]
