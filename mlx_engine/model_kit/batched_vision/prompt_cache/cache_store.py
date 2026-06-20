from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any

import mlx.core as mx
from mlx_engine.model_kit.batched_vision.prompt_cache.chunks import (
    build_prefix_cache_chunks,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.disk_budget import (
    final_cache_store_budget_bytes,
    provisional_cache_store_budget_bytes,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.records import (
    assemble_prompt_cache_chunks,
    make_prompt_cache_layout,
    prepare_prompt_cache_records_for_chunk,
    _slice_kv_cache,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.restore_planner import (
    PromptCacheRestorePlanner,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.types import (
    DEFAULT_PREFIX_CHUNK_SIZE,
    PromptCacheLayout,
    PromptImageSpan,
    PromptCacheRecordMetadata,
    PromptPrefixChunk,
    PreparedPromptMetadata,
    PreparedPromptRecord,
    PendingPromptCacheSave,
    PromptCacheStoreStats,
    RECORD_KIND_KV_DELTA,
    RECORD_KIND_ROTATING_DELTA,
    RECORD_KIND_STATE_CHECKPOINT,
    RECORD_WRITE_ORDER,
    RecordKind,
    LoadedDiskPromptCache,
    make_record_key,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.blob_store import (
    PersistentSafetensorBlobStore,
    TemporarySafetensorBlobStore,
)
from mlx_engine.utils.batched_timing import (
    batched_timing_enabled,
    elapsed_ms,
    log_batched_timing,
)
from mlx.utils import tree_flatten


logger = logging.getLogger(__name__)


_MIB_BYTES = 1024 * 1024
_CACHE_USAGE_LOG_COOLDOWN_SECONDS = 60.0
_RECORD_RETENTION_PRIORITY: tuple[RecordKind, ...] = (
    RECORD_KIND_STATE_CHECKPOINT,
    RECORD_KIND_KV_DELTA,
    RECORD_KIND_ROTATING_DELTA,
)
_PERSISTENT_CACHE_FORMAT_VERSION = 1
_READABLE_PERSISTENT_CACHE_FORMAT_VERSIONS = frozenset({1})
_PERSISTENT_CACHE_INDEX_FILENAME = "prompt-cache-index.json"
_DEFAULT_PERSISTENT_MIN_SAVE_TOKENS = 512
_MAX_PREPARED_PROMPT_METADATA_ENTRIES = 128
_RECORD_SPAN_KEY = "chunk_span"


@dataclass
class DiskPromptCacheRestorePlan:
    """Same-thread selection of disk records for one prompt-cache restore.

    The caller may compare this with a hot-cache candidate, but any selected
    plan must be loaded without yielding off the cache I/O thread.
    """

    cached_prefix_len: int
    chunks: list[PromptPrefixChunk]
    record_keys_by_chunk_key: dict[str, list[str]]


def _make_record_key(
    chunk_key: str,
    record_kind: RecordKind,
    *,
    chunk_start: int | None = None,
    chunk_end: int | None = None,
) -> str:
    """Return a cache record key, including optional span metadata."""
    base_key = make_record_key(chunk_key, record_kind)
    if chunk_start is None or chunk_end is None:
        return base_key
    if chunk_start <= 0:
        return base_key
    return f"{base_key}:span:{chunk_start}:{chunk_end}"


def _chunk_span_from_metadata(metadata: PromptCacheRecordMetadata) -> tuple[int, int] | None:
    span = metadata.chunk_span
    if not span or len(span) != 2:
        return None
    span_start, span_end = span
    if not isinstance(span_start, int) or not isinstance(span_end, int):
        return None
    return span_start, span_end


def _prefix_len_splits_image_span(
    prefix_len: int,
    image_spans: list[PromptImageSpan],
) -> bool:
    # A chunk ending inside an image span can be an internal record written from
    # a later full visual-prefill snapshot, but it is not a valid terminal restore
    # point because the cached image-token states depended on later image tokens.
    return any(span.start < prefix_len < span.end for span in image_spans)


class VlmPromptCacheStore:
    """Cache-I/O-thread-owned index and temporary safetensor blob store.

    Mutable index/blob-store operations run on the prompt-cache I/O thread. The
    generation thread may prepare immutable cache records and make advisory
    budget checks, but it must not mutate committed cache store state.

    Invariants:
    - A selected restore chain is loaded without interleaved eviction.
    - Selected record keys exist in both metadata and the blob store.
    - Touched LRU keys are committed physical records.
    """

    def __init__(
        self,
        max_kv_size: int | None = None,
        *,
        storage_root: Path | None = None,
        cache_namespace: str = "default",
        min_save_tokens: int | None = None,
        prefix_chunk_size: int = DEFAULT_PREFIX_CHUNK_SIZE,
    ):
        if storage_root is None:
            base_dir = Path("/tmp")
        else:
            namespace_digest = hashlib.sha256(cache_namespace.encode()).hexdigest()
            base_dir = storage_root / namespace_digest[:16]
        base_dir.mkdir(parents=True, exist_ok=True)
        self._base_dir = base_dir
        self._max_kv_size = max_kv_size
        self._persistent_index_path: Path | None = None
        self._cache_namespace = cache_namespace
        if min_save_tokens is None:
            min_save_tokens = (
                _DEFAULT_PERSISTENT_MIN_SAVE_TOKENS
                if storage_root is not None
                else 0
            )
        self._min_save_tokens = max(0, min_save_tokens)
        self._prefix_chunk_size = max(1, prefix_chunk_size)
        if storage_root is None:
            self._blob_store = TemporarySafetensorBlobStore(base_dir)
            storage_lifetime = "temporary"
            storage_mode = "temporary"
        else:
            self._persistent_index_path = base_dir / _PERSISTENT_CACHE_INDEX_FILENAME
            self._blob_store = PersistentSafetensorBlobStore(base_dir / "records")
            storage_lifetime = "persistent"
            storage_mode = str(base_dir)
        self._empirical_budget_set = False
        self._layout: PromptCacheLayout | None = None
        self._record_metadata_by_key: dict[str, PromptCacheRecordMetadata] = {}
        self._prepared_prompt_metadata_by_key: OrderedDict[
            str, PreparedPromptMetadata
        ] = OrderedDict()
        self._key_sizes: dict[str, int] = {}
        self._lru_keys: OrderedDict[str, None] = OrderedDict()
        self._total_bytes = 0
        self._max_cache_store_bytes = provisional_cache_store_budget_bytes(base_dir)
        self._restore_hit_tokens = 0
        self._restore_miss_tokens = 0
        self._restore_count = 0
        self._restore_latency_ms = 0.0
        self._save_count = 0
        self._save_latency_ms = 0.0
        self._cache_evictions = 0
        self._cache_evicted_bytes = 0
        self._last_cache_usage_log_time = 0.0
        self._load_persistent_index()
        logger.info(
            "VLM prompt cache disk store: lifetime=%s storage=%s "
            "cleanup=%s min_save_tokens=%s",
            storage_lifetime,
            storage_mode,
            "manual_or_budget_eviction" if storage_root is not None else (
                "model_unload_or_process_exit"
            ),
            self._min_save_tokens,
        )

    def plan_longest_prefix_restore(
        self,
        prompt_input_ids: list[int],
        image_spans: list[PromptImageSpan],
    ) -> DiskPromptCacheRestorePlan | None:
        """Select the longest matching cacheable prefix without loading blobs.

        The final prompt token stays uncached so generation has a suffix to
        process. Too-short or unchunkable prompts do not produce a disk plan.
        """
        max_reusable_prefix_len = len(prompt_input_ids) - 1
        if max_reusable_prefix_len <= 0:
            return None
        if self._layout is None:
            return None

        eligible_chunks = [
            chunk
            for chunk in build_prefix_cache_chunks(
                prompt_input_ids,
                image_spans,
                prefix_chunk_size=self._prefix_chunk_size,
            )
            if chunk.end <= max_reusable_prefix_len
        ]
        if not eligible_chunks:
            return None

        restore_planner = self._cache_restore_planner()
        # A later SWA boundary may be restorable even if an earlier boundary is
        # missing old rotating records, so scan downward for the best boundary.
        for end_idx in range(len(eligible_chunks), 0, -1):
            chunks = eligible_chunks[:end_idx]
            if _prefix_len_splits_image_span(chunks[-1].end, image_spans):
                continue
            record_keys_by_chunk_key = (
                restore_planner.restore_record_keys_for_chunk_chain(chunks)
            )
            if record_keys_by_chunk_key is not None:
                return DiskPromptCacheRestorePlan(
                    cached_prefix_len=chunks[-1].end,
                    chunks=chunks,
                    record_keys_by_chunk_key=record_keys_by_chunk_key,
                )

        return None

    def set_prefix_chunk_size(self, prefix_chunk_size: int) -> None:
        """Set the cache chunk size used for future restore planning."""
        self._prefix_chunk_size = max(1, prefix_chunk_size)

    def load_restore_plan(
        self,
        plan: DiskPromptCacheRestorePlan,
    ) -> LoadedDiskPromptCache:
        """Load a restore plan selected on this cache I/O thread."""
        start_time = perf_counter()
        try:
            loaded = self._load_restore_plan(plan)
            self._write_persistent_index()
            return loaded
        finally:
            self._restore_count += 1
            self._restore_latency_ms += (perf_counter() - start_time) * 1000.0

    def _load_restore_plan(
        self,
        plan: DiskPromptCacheRestorePlan,
    ) -> LoadedDiskPromptCache:
        """Load a restore plan and let the public wrapper record timing."""
        layout = self._require_layout()
        timing_enabled = batched_timing_enabled()
        restore_start = perf_counter() if timing_enabled else None

        chunk_prompt_caches = []

        # Load each chunk's physical records into sparse per-layer cache lists.
        load_chunks_start = perf_counter() if timing_enabled else None
        record_count = 0
        for chunk in plan.chunks:
            record_count += len(plan.record_keys_by_chunk_key[chunk.key])
            prompt_cache = self._load_one_chunk(
                plan.record_keys_by_chunk_key[chunk.key],
                layout,
                timing_enabled=timing_enabled,
            )
            chunk_prompt_caches.append(prompt_cache)
        load_chunks_ms = elapsed_ms(load_chunks_start) if timing_enabled else 0.0

        assemble_start = perf_counter() if timing_enabled else None
        prompt_cache = assemble_prompt_cache_chunks(
            chunk_prompt_caches,
            plan.chunks,
            layout,
        )
        assemble_ms = elapsed_ms(assemble_start) if timing_enabled else 0.0
        # Disk restores run on the prompt-cache I/O thread; decode consumes the
        # cache on the generation thread. Force assembled arrays now so no lazy
        # graph keeps a thread-local MLX stream from the restore worker.
        eval_start = perf_counter() if timing_enabled else None
        mx.eval(
            [
                value
                for _, value in tree_flatten([cache.state for cache in prompt_cache])
            ]
        )
        eval_ms = elapsed_ms(eval_start) if timing_enabled else 0.0

        # Restore access refreshes exactly the records used by this chain.
        touch_start = perf_counter() if timing_enabled else None
        for record_key in self._ordered_record_keys_for_touch(
            [chunk.key for chunk in plan.chunks],
            plan.record_keys_by_chunk_key,
        ):
            self._touch_cache_entry(record_key)
        touch_ms = elapsed_ms(touch_start) if timing_enabled else 0.0

        if timing_enabled:
            log_batched_timing(
                logger,
                "vlm_cache_restore_detail",
                cached_tokens=plan.cached_prefix_len,
                chunks=len(plan.chunks),
                records=record_count,
                load_chunks_ms=load_chunks_ms,
                assemble_ms=assemble_ms,
                eval_ms=eval_ms,
                touch_ms=touch_ms,
                duration_ms=elapsed_ms(restore_start),
            )

        return LoadedDiskPromptCache(
            cached_prefix_len=plan.cached_prefix_len,
            prompt_cache=prompt_cache,
        )

    def record_restore_tokens(
        self,
        *,
        hit_tokens: int,
        miss_tokens: int,
    ) -> tuple[int, int]:
        """Record one request and return lifetime hit/miss token totals."""
        self._restore_hit_tokens += hit_tokens
        self._restore_miss_tokens += miss_tokens
        return self._restore_hit_tokens, self._restore_miss_tokens

    def _load_one_chunk(
        self,
        record_keys: list[str],
        layout: PromptCacheLayout,
        *,
        timing_enabled: bool = False,
    ) -> list[Any]:
        prompt_cache: list[Any] = [None] * len(layout.layer_kinds)
        for record_key in record_keys:
            record_metadata = self._record_metadata_by_key[record_key]
            record_size = self._key_sizes.get(record_key, 0)
            load_start = perf_counter() if timing_enabled else None
            try:
                if timing_enabled:
                    loaded_record = self._blob_store.load_record_profiled(record_key)
                    record_prompt_cache = loaded_record.prompt_cache
                else:
                    loaded_record = None
                    record_prompt_cache = self._blob_store.load_record(record_key)
            except Exception:
                self._evict_key(record_key)
                raise
            if timing_enabled:
                log_batched_timing(
                    logger,
                    "vlm_cache_record_load",
                    record_kind=record_metadata.record_kind,
                    layers=len(record_metadata.layer_indices),
                    bytes=record_size,
                    duration_ms=elapsed_ms(load_start),
                    safetensor_load_ms=loaded_record.safetensor_load_ms,
                    unflatten_ms=loaded_record.unflatten_ms,
                    cache_rebuild_ms=loaded_record.cache_rebuild_ms,
                )

            for layer_idx, cache in zip(
                record_metadata.layer_indices, record_prompt_cache
            ):
                prompt_cache[layer_idx] = cache

        return prompt_cache

    def can_store_records(self) -> bool:
        """Return False when the cache store is intentionally hot-only."""
        return self._max_cache_store_bytes != 0

    def should_save_prompt(self, prefix_chunks: list[PromptPrefixChunk]) -> bool:
        """Return True when a prompt is large enough to justify disk records."""
        if not self.can_store_records() or not prefix_chunks:
            return False
        return prefix_chunks[-1].end >= self._min_save_tokens

    def lookup_prepared_prompt_metadata(
        self,
        request_key: str | None,
    ) -> PreparedPromptMetadata | None:
        """Return exact-request prompt metadata, if persistent index has it."""
        if request_key is None:
            return None
        metadata = self._prepared_prompt_metadata_by_key.get(request_key)
        if metadata is not None:
            self._prepared_prompt_metadata_by_key.move_to_end(request_key)
        return metadata

    def remember_prepared_prompt_metadata(
        self,
        metadata: PreparedPromptMetadata,
        *,
        prefix_chunks: list[PromptPrefixChunk],
    ) -> None:
        """Remember exact prepared prompt metadata for later process restarts."""
        if self._persistent_index_path is None or not self.should_save_prompt(
            prefix_chunks
        ):
            return
        self._prepared_prompt_metadata_by_key[metadata.request_key] = metadata
        self._prepared_prompt_metadata_by_key.move_to_end(metadata.request_key)
        while (
            len(self._prepared_prompt_metadata_by_key)
            > _MAX_PREPARED_PROMPT_METADATA_ENTRIES
        ):
            self._prepared_prompt_metadata_by_key.popitem(last=False)
        self._write_persistent_index()

    def prepare_save(
        self,
        *,
        chunk: PromptPrefixChunk,
        prefix_chunks: list[PromptPrefixChunk],
        prompt_cache: list[Any],
        save_state_checkpoint: bool = True,
        is_final_prompt_boundary: bool = False,
    ) -> PendingPromptCacheSave:
        """Prepare a cache save for the cache I/O thread."""
        record_caches, record_kinds = prepare_prompt_cache_records_for_chunk(
            prompt_cache,
            chunk.start,
            chunk.end,
        )
        layout = make_prompt_cache_layout(record_caches, record_kinds)
        records = []
        for record_kind in RECORD_WRITE_ORDER:
            if (
                record_kind == RECORD_KIND_STATE_CHECKPOINT
                and not save_state_checkpoint
            ):
                continue
            layer_indices = layout.layer_indices_by_kind.get(record_kind, [])
            if not layer_indices:
                continue
            kv_span_start = (
                self._kv_span_start(
                    chunk=chunk,
                    prefix_chunks=prefix_chunks,
                )
                if record_kind == RECORD_KIND_KV_DELTA
                else None
            )

            if kv_span_start is None:
                chunk_record_caches = [
                    record_caches[idx]
                    for idx in layer_indices
                ]
                records.append(
                    self._prepare_record_save(
                        chunk_key=chunk.key,
                        record_kind=record_kind,
                        layer_indices=layer_indices,
                        record_cache=chunk_record_caches,
                        chunk_start=(
                            chunk.start
                            if record_kind == RECORD_KIND_KV_DELTA
                            else None
                        ),
                        chunk_end=(
                            chunk.end if record_kind == RECORD_KIND_KV_DELTA else None
                        ),
                    )
                )

            if kv_span_start is not None:
                records.append(
                    self._prepare_record_save(
                        chunk_key=chunk.key,
                        record_kind=record_kind,
                        layer_indices=layer_indices,
                        record_cache=[
                            _slice_kv_cache(
                                cache=prompt_cache[layer_idx],
                                chunk_start=kv_span_start,
                                chunk_end=chunk.end,
                            )
                            for layer_idx in layer_indices
                        ],
                        chunk_start=kv_span_start,
                        chunk_end=chunk.end,
                    )
                )

        return PendingPromptCacheSave(
            prefix_chunks=prefix_chunks,
            cache_layout=layout,
            records=records,
            is_final_prompt_boundary=is_final_prompt_boundary,
        )

    def _kv_span_start(
        self,
        *,
        chunk: PromptPrefixChunk,
        prefix_chunks: list[PromptPrefixChunk],
    ) -> int | None:
        """Return the start token for one-step KV coalescing with the prior chunk."""
        if len(prefix_chunks) <= 1:
            return None
        chunk_index = prefix_chunks.index(chunk)
        if chunk_index == 0:
            return None
        prev_chunk = prefix_chunks[chunk_index - 1]
        if prev_chunk.end != chunk.start:
            return None
        return prev_chunk.start

    def budget_update_from_completed_cache(self, prompt_cache: list[Any]) -> int | None:
        """Return the empirical cache store budget from a completed cache."""
        if self._empirical_budget_set:
            return None

        try:
            return final_cache_store_budget_bytes(
                self._base_dir,
                prompt_cache,
                self._max_kv_size,
            )
        except Exception:
            logger.warning(
                "Failed to estimate VLM prompt cache disk budget; "
                "disabling disk records",
                exc_info=True,
            )
            return 0

    def commit_budget_update(self, max_cache_store_bytes: int) -> None:
        """Set the empirical budget and evict records from the cache I/O thread."""
        if self._empirical_budget_set:
            return
        self._max_cache_store_bytes = max_cache_store_bytes
        self._empirical_budget_set = True

        self._evict_if_needed()
        self._write_persistent_index()

    def commit_pending_save(self, pending_save: PendingPromptCacheSave) -> None:
        """Commit a pending save from the cache I/O thread."""
        if not self.can_store_records():
            return
        start_time = perf_counter()
        if self._layout is None:
            self._layout = pending_save.cache_layout

        try:
            for record in pending_save.records:
                if self._blob_store.exists(record.key):
                    self._record_metadata_by_key[record.key] = record.metadata
                    self._touch_cache_entry(record.key)
                    continue

                # The I/O thread waits, writes, then publishes/account each record.
                mx.eval(list(record.snapshot_arrays.values()))
                self._blob_store.put(
                    record.key,
                    record.snapshot_arrays,
                    record.safetensor_metadata,
                )
                self._record_metadata_by_key[record.key] = record.metadata
                self._touch_cache_entry(record.key)

        finally:
            self._save_count += 1
            self._save_latency_ms += (perf_counter() - start_time) * 1000.0
            self._touch_longest_budget_fit_restore_chain(pending_save.prefix_chunks)
            self._evict_if_needed()
            self._write_persistent_index()
            self._maybe_log_cache_usage()

    def snapshot_stats(self) -> PromptCacheStoreStats:
        """Return best-effort diagnostics for smokes/debug output."""
        total_bytes = self._total_bytes
        max_bytes = self._max_cache_store_bytes
        entry_count = len(self._record_metadata_by_key)
        hit_tokens = self._restore_hit_tokens
        miss_tokens = self._restore_miss_tokens
        evictions = self._cache_evictions
        restore_count = self._restore_count
        restore_latency_ms = self._restore_latency_ms
        save_count = self._save_count
        save_latency_ms = self._save_latency_ms
        record_sizes_by_key = dict(self._key_sizes)
        record_metadata_by_key = dict(self._record_metadata_by_key)
        chunk_sizes_by_key = {}
        chunk_records_available_by_key = {}
        chunk_keys = sorted(
            {metadata.chunk_key for metadata in record_metadata_by_key.values()}
        )
        for chunk_key in chunk_keys:
            record_keys = [
                record_key
                for record_key, metadata in record_metadata_by_key.items()
                if metadata.chunk_key == chunk_key
            ]
            chunk_sizes_by_key[chunk_key] = sum(
                record_sizes_by_key.get(record_key, 0) for record_key in record_keys
            )
            chunk_records_available_by_key[chunk_key] = bool(record_keys) and all(
                record_key in record_sizes_by_key
                and self._blob_store.exists(record_key)
                for record_key in record_keys
            )

        return PromptCacheStoreStats(
            total_bytes=total_bytes,
            max_bytes=max_bytes,
            entry_count=entry_count,
            hit_tokens=hit_tokens,
            miss_tokens=miss_tokens,
            evictions=evictions,
            restore_count=restore_count,
            restore_latency_ms=restore_latency_ms,
            save_count=save_count,
            save_latency_ms=save_latency_ms,
            record_sizes=sorted(record_sizes_by_key.values()),
            record_sizes_by_key=record_sizes_by_key,
            chunk_sizes_by_key=chunk_sizes_by_key,
            chunk_records_available_by_key=chunk_records_available_by_key,
        )

    def close(self) -> None:
        """Clear metadata and close the temporary blob store."""
        self._write_persistent_index()
        self._layout = None
        self._record_metadata_by_key.clear()
        self._prepared_prompt_metadata_by_key.clear()
        self._key_sizes.clear()
        self._lru_keys.clear()
        self._total_bytes = 0
        self._restore_count = 0
        self._restore_latency_ms = 0.0
        self._save_count = 0
        self._save_latency_ms = 0.0
        self._blob_store.close()

    def _load_persistent_index(self) -> None:
        if self._persistent_index_path is None or not self._persistent_index_path.exists():
            return

        try:
            data = json.loads(self._persistent_index_path.read_text())
        except Exception:
            logger.warning("Failed to read persistent VLM prompt cache index.", exc_info=True)
            return

        if data.get("format_version") not in _READABLE_PERSISTENT_CACHE_FORMAT_VERSIONS:
            logger.info("Ignoring unreadable VLM prompt cache index format.")
            return
        if data.get("cache_namespace") != self._cache_namespace:
            logger.info("Ignoring VLM prompt cache index for different namespace.")
            return

        layout_data = data.get("layout")
        if layout_data is not None:
            try:
                self._layout = PromptCacheLayout(
                    layer_kinds=list(layout_data["layer_kinds"]),
                    layer_indices_by_kind={
                        kind: list(indices)
                        for kind, indices in layout_data[
                            "layer_indices_by_kind"
                        ].items()
                    },
                    rotating_window_size=layout_data.get("rotating_window_size"),
                )
            except Exception:
                logger.warning(
                    "Failed to restore persistent VLM prompt cache layout.",
                    exc_info=True,
                )
                self._layout = None
                return

        records = data.get("records", {})
        for record_key, metadata in records.items():
            if not self._blob_store.exists(record_key):
                continue
            try:
                self._record_metadata_by_key[record_key] = PromptCacheRecordMetadata(
                    chunk_key=metadata["chunk_key"],
                    record_kind=metadata["record_kind"],
                    layer_indices=list(metadata["layer_indices"]),
                    chunk_span=(
                        metadata.get(_RECORD_SPAN_KEY)
                    ),
                )
                self._key_sizes[record_key] = self._blob_store.size(record_key)
            except Exception:
                logger.debug(
                    "Skipping invalid persistent VLM prompt cache record metadata.",
                    exc_info=True,
                )

        lru_keys = data.get("lru_keys", [])
        for key in lru_keys:
            if key in self._record_metadata_by_key:
                self._lru_keys[key] = None
        for key in self._record_metadata_by_key:
            self._lru_keys.setdefault(key, None)

        self._total_bytes = sum(self._key_sizes.values())
        prepared_prompts = data.get("prepared_prompts", {})
        for request_key, metadata in prepared_prompts.items():
            try:
                image_spans = [
                    PromptImageSpan(
                        start=int(span["start"]),
                        end=int(span["end"]),
                        image_hash=str(span["image_hash"]),
                    )
                    for span in metadata.get("image_spans", [])
                ]
                self._prepared_prompt_metadata_by_key[request_key] = (
                    PreparedPromptMetadata(
                        request_key=request_key,
                        prompt_input_ids=[
                            int(token) for token in metadata["prompt_input_ids"]
                        ],
                        image_spans=image_spans,
                        vision_cache_key=metadata.get("vision_cache_key"),
                        image_grid_thw=metadata.get("image_grid_thw"),
                    )
                )
            except Exception:
                logger.debug(
                    "Skipping invalid prepared VLM prompt metadata.",
                    exc_info=True,
                )

        self._evict_if_needed()

    def _write_persistent_index(self) -> None:
        if self._persistent_index_path is None:
            return

        records = {
            key: asdict(metadata)
            for key, metadata in self._record_metadata_by_key.items()
            if self._blob_store.exists(key)
        }
        data = {
            "format_version": _PERSISTENT_CACHE_FORMAT_VERSION,
            "cache_namespace": self._cache_namespace,
            "layout": None if self._layout is None else asdict(self._layout),
            "records": records,
            "prepared_prompts": {
                key: asdict(metadata)
                for key, metadata in self._prepared_prompt_metadata_by_key.items()
            },
            "lru_keys": [
                key for key in self._lru_keys.keys() if key in records
            ],
            "max_bytes": self._max_cache_store_bytes,
        }
        tmp_path = self._persistent_index_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(data, sort_keys=True))
        tmp_path.replace(self._persistent_index_path)

    def _prepare_record_save(
        self,
        *,
        chunk_key: str,
        record_kind: RecordKind,
        layer_indices: list[int],
        record_cache: list[Any],
        chunk_start: int | None = None,
        chunk_end: int | None = None,
    ) -> PreparedPromptRecord:
        record_key = _make_record_key(
            chunk_key,
            record_kind,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
        )
        cache_data = [cache.state for cache in record_cache]
        cache_meta_states = [cache.meta_state for cache in record_cache]
        cache_arrays = dict(tree_flatten(cache_data))
        cache_class_names = [type(cache).__name__ for cache in record_cache]
        safetensor_metadata = dict(
            tree_flatten(
                [
                    cache_meta_states,
                    cache_class_names,
                ]
            )
        )
        snapshot_arrays = {
            name: mx.contiguous(array) for name, array in cache_arrays.items()
        }

        # Schedule snapshot materialization before handing off disk I/O.
        mx.async_eval(list(snapshot_arrays.values()))

        return PreparedPromptRecord(
            key=record_key,
            metadata=PromptCacheRecordMetadata(
            chunk_key=chunk_key,
            record_kind=record_kind,
            layer_indices=layer_indices,
            chunk_span=[chunk_start, chunk_end] if chunk_start is not None and chunk_end is not None else None,
        ),
            snapshot_arrays=snapshot_arrays,
            safetensor_metadata=safetensor_metadata,
        )

    def _touch_cache_entry(self, key: str) -> None:
        total_size = self._blob_store.size(key)
        previous_size = self._key_sizes.get(key, 0)
        self._key_sizes[key] = total_size
        self._total_bytes += total_size - previous_size
        self._lru_keys.pop(key, None)
        self._lru_keys[key] = None

    def _ordered_record_keys_for_touch(
        self,
        chunk_keys: list[str],
        record_keys_by_chunk_key: dict[str, list[str]],
    ) -> list[str]:
        ordered_record_keys = []
        # vLLM/LMCache order the LRU so suffix blocks evict before prefixes.
        for chunk_key in reversed(chunk_keys):
            record_keys = record_keys_by_chunk_key.get(chunk_key, [])
            if not record_keys:
                continue
            record_keys_by_kind = {
                self._record_metadata_by_key[record_key].record_kind: record_key
                for record_key in record_keys
            }
            # Touch low retention priority first so important records stay newest.
            for record_kind in reversed(_RECORD_RETENTION_PRIORITY):
                record_key = record_keys_by_kind.get(record_kind)
                if record_key is not None:
                    ordered_record_keys.append(record_key)

        return ordered_record_keys

    def _touch_longest_budget_fit_restore_chain(
        self,
        prefix_chunks: list[PromptPrefixChunk],
    ) -> None:
        planner = self._cache_restore_planner()
        for end_idx in range(len(prefix_chunks), 0, -1):
            candidate_chunks = prefix_chunks[:end_idx]
            record_keys_by_chunk_key = planner.restore_record_keys_for_chunk_chain(
                candidate_chunks
            )
            if record_keys_by_chunk_key is None:
                continue

            record_keys = self._ordered_record_keys_for_touch(
                [chunk.key for chunk in candidate_chunks],
                record_keys_by_chunk_key,
            )
            # Preserve the longest complete restore set that can survive eviction.
            if sum(self._key_sizes[key] for key in record_keys) <= (
                self._max_cache_store_bytes
            ):
                for record_key in record_keys:
                    self._touch_cache_entry(record_key)
                return

    def _evict_if_needed(self) -> None:
        while self._total_bytes > self._max_cache_store_bytes:
            key_to_evict = next(iter(self._lru_keys), None)
            if key_to_evict is None:
                break

            self._evict_key(key_to_evict)

    def _maybe_log_cache_usage(self) -> None:
        now = monotonic()
        if (
            self._last_cache_usage_log_time
            and now - self._last_cache_usage_log_time
            < _CACHE_USAGE_LOG_COOLDOWN_SECONDS
        ):
            return

        self._last_cache_usage_log_time = now
        logger.info(
            "VLM prompt cache disk usage: used_mib=%.1f cap_mib=%.1f "
            "lifetime_evicted_mib=%.1f records=%s restores=%s "
            "restore_ms=%.3f saves=%s save_ms=%.3f lifetime=model_load",
            self._total_bytes / _MIB_BYTES,
            self._max_cache_store_bytes / _MIB_BYTES,
            self._cache_evicted_bytes / _MIB_BYTES,
            len(self._record_metadata_by_key),
            self._restore_count,
            self._restore_latency_ms,
            self._save_count,
            self._save_latency_ms,
        )

    def _cache_restore_planner(self) -> PromptCacheRestorePlanner:
        """Return a short-lived read-only view over committed indexes."""
        return PromptCacheRestorePlanner(
            layout=self._require_layout(),
            record_metadata_by_key=self._record_metadata_by_key,
            record_exists=self._blob_store.exists,
        )

    def _require_layout(self) -> PromptCacheLayout:
        if self._layout is None:
            raise RuntimeError("prompt cache layout is not initialized")
        return self._layout

    def _evict_key(self, key: str) -> None:
        evicted_bytes = self._key_sizes.pop(key, 0)
        self._total_bytes -= evicted_bytes
        self._lru_keys.pop(key, None)
        self._record_metadata_by_key.pop(key, None)
        self._cache_evictions += 1
        self._cache_evicted_bytes += evicted_bytes

        self._blob_store.delete(key)
