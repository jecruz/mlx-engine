from collections.abc import Callable

from mlx_engine.model_kit.batched_vision.prompt_cache.types import (
    PromptCacheLayout,
    PromptCacheRecordMetadata,
    PromptPrefixChunk,
    RECORD_KIND_KV_DELTA,
    RECORD_KIND_ROTATING_DELTA,
    RECORD_KIND_STATE_CHECKPOINT,
    RECORD_WRITE_ORDER,
    make_record_key,
)


class PromptCacheRestorePlanner:
    """Read-only planner for records needed to restore a prompt prefix.

    Restore policy:
    - KV deltas are needed for every chunk in the prefix chain.
    - Rotating deltas are needed only inside the target sliding window.
    - Opaque state checkpoints are needed only at the exact target chunk.

    Short-lived by design: callers construct this from the cache-I/O-thread-owned
    physical record index when they need to evaluate restore availability.
    """

    def __init__(
        self,
        *,
        layout: PromptCacheLayout,
        record_metadata_by_key: dict[str, PromptCacheRecordMetadata],
        record_exists: Callable[[str], bool],
    ):
        self._layout = layout
        self._record_metadata_by_key = record_metadata_by_key
        self._record_exists = record_exists
        self._kv_record_keys_by_chunk_key: dict[str, list[str]] = {}
        for record_key, metadata in record_metadata_by_key.items():
            if metadata.record_kind == RECORD_KIND_KV_DELTA:
                self._kv_record_keys_by_chunk_key.setdefault(
                    metadata.chunk_key,
                    [],
                ).append(record_key)

    def restore_record_keys_for_chunk_chain(
        self, chunks: list[PromptPrefixChunk]
    ) -> dict[str, list[str]] | None:
        """Return physical records needed to restore a cached chunk chain.

        Returns None when the index says the chain is not currently restorable:
        required records are not indexed, or blobs were already evicted from the
        blob store.
        """
        if not chunks:
            return {}

        # The last chunk is the restore boundary for SWA windowing/checkpoints.
        target_chunk = chunks[-1]
        target_chunk_end = target_chunk.end
        rotating_window_size = self._layout.rotating_window_size
        chunk_index_by_start = {chunk.start: idx for idx, chunk in enumerate(chunks)}
        chunk_indices = list(enumerate(chunks))
        covered_kv_chunk_indices: set[int] = set()
        kv_record_key_by_chunk_key: dict[str, str] = {}

        for idx, chunk in reversed(chunk_indices):
            if idx in covered_kv_chunk_indices:
                continue

            if not self._layout.layer_indices_by_kind.get(RECORD_KIND_KV_DELTA):
                continue

            min_kv_span_start = (
                chunks[idx - 1].start
                if idx > 0 and chunks[idx - 1].end == chunk.start
                else chunk.start
            )
            kv_record_key = self._select_kv_record_key(
                chunk,
                min_span_start=min_kv_span_start,
                allow_terminal_packed=chunk.key == target_chunk.key,
            )
            if not kv_record_key:
                return None

            kv_record_key_by_chunk_key[chunk.key] = kv_record_key
            metadata = self._record_metadata_by_key[kv_record_key]
            metadata_span_start, _ = self._metadata_span(metadata)
            if (
                metadata_span_start is not None
                and metadata_span_start < chunk.start
            ):
                span_start_index = chunk_index_by_start.get(metadata_span_start)
                if span_start_index is None:
                    return None
                covered_kv_chunk_indices.update(
                    range(span_start_index, idx)
                )

        record_keys_by_chunk_key: dict[str, list[str]] = {}
        for idx, chunk in chunk_indices:
            record_keys: list[str] = []
            for record_kind in RECORD_WRITE_ORDER:
                if not self._layout.layer_indices_by_kind.get(record_kind):
                    continue
                if record_kind == RECORD_KIND_KV_DELTA and idx in covered_kv_chunk_indices:
                    continue
                if record_kind == RECORD_KIND_STATE_CHECKPOINT:
                    # Opaque state caches are exact-boundary checkpoints.
                    if chunk.key != target_chunk.key:
                        continue
                elif record_kind == RECORD_KIND_ROTATING_DELTA:
                    if rotating_window_size is None:
                        return None
                    if not self._rotating_chunk_overlaps_target_window(
                        chunk=chunk,
                        target_chunk_end=target_chunk_end,
                        rotating_window_size=rotating_window_size,
                    ):
                        continue

                record_key = (
                    kv_record_key_by_chunk_key.get(chunk.key)
                    if record_kind == RECORD_KIND_KV_DELTA
                    else make_record_key(chunk.key, record_kind)
                )
                if record_key is None:
                    return None
                if not self._has_record(record_key):
                    return None

                record_keys.append(record_key)
            record_keys_by_chunk_key[chunk.key] = record_keys

        return record_keys_by_chunk_key

    @staticmethod
    def _metadata_span(
        metadata: PromptCacheRecordMetadata,
    ) -> tuple[int | None, int | None]:
        span = metadata.chunk_span
        if not span or len(span) != 2:
            return None, None
        try:
            return int(span[0]), int(span[1])
        except (TypeError, ValueError):
            return None, None

    def _has_record(self, record_key: str) -> bool:
        return record_key in self._record_metadata_by_key and self._record_exists(
            record_key
        )

    def _select_kv_record_key(
        self,
        chunk: PromptPrefixChunk,
        *,
        min_span_start: int,
        allow_terminal_packed: bool,
    ) -> str | None:
        """Return the preferred bounded KV record key for a chunk.

        Production saves only one-step KV spans that cover the current chunk
        plus its immediate predecessor. The one exception is an explicitly
        marked terminal-packed record for the true restore boundary.
        """
        plain_record_key = make_record_key(chunk.key, RECORD_KIND_KV_DELTA)
        span_candidate_keys = []
        for record_key in self._kv_record_keys_by_chunk_key.get(chunk.key, []):
            metadata = self._record_metadata_by_key[record_key]
            if not self._has_record(record_key):
                continue
            chunk_span = self._metadata_span(metadata)
            if chunk_span == (None, None):
                continue
            span_start, span_end = chunk_span
            if span_start is None or span_end is None:
                continue
            if span_start < min_span_start and not (
                allow_terminal_packed and metadata.is_terminal_packed
            ):
                continue
            if span_start <= chunk.start <= span_end and span_end >= chunk.end:
                span_candidate_keys.append((span_start, span_end, record_key))
        if span_candidate_keys:
            span_candidate_keys.sort(key=lambda item: (item[0], -item[1]))
            return span_candidate_keys[0][2]
        if self._has_record(plain_record_key):
            return plain_record_key
        return None

    def _rotating_chunk_overlaps_target_window(
        self,
        *,
        chunk: PromptPrefixChunk,
        target_chunk_end: int,
        rotating_window_size: int,
    ) -> bool:
        # Sliding-window layers only need chunks overlapping the target boundary.
        window_start = target_chunk_end - rotating_window_size
        return chunk.end > window_start
