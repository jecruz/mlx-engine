from mlx_engine.model_kit.batched_vision.prompt_cache.restore_planner import (
    PromptCacheRestorePlanner,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.types import (
    PromptCacheLayout,
    PromptCacheRecordMetadata,
    PromptPrefixChunk,
    RECORD_KIND_KV_DELTA,
    RECORD_KIND_ROTATING_DELTA,
    RECORD_KIND_STATE_CHECKPOINT,
    RECORD_WRITE_ORDER,
    RecordKind,
    make_record_key,
)


def _chunk(start: int, end: int) -> PromptPrefixChunk:
    return PromptPrefixChunk(start=start, end=end, key=f"chunk-{start}-{end}")


def _layout() -> PromptCacheLayout:
    return PromptCacheLayout(
        layer_kinds=[
            RECORD_KIND_KV_DELTA,
            RECORD_KIND_ROTATING_DELTA,
            RECORD_KIND_STATE_CHECKPOINT,
        ],
        layer_indices_by_kind={
            RECORD_KIND_KV_DELTA: [0],
            RECORD_KIND_ROTATING_DELTA: [1],
            RECORD_KIND_STATE_CHECKPOINT: [2],
        },
        rotating_window_size=512,
    )


def _metadata_for(chunk: PromptPrefixChunk, record_kind: RecordKind):
    return PromptCacheRecordMetadata(
        chunk_key=chunk.key,
        record_kind=record_kind,
        layer_indices=[],
    )


def _planner(chunks, existing_records):
    metadata_by_key = {}
    for chunk in chunks:
        for record_kind in RECORD_WRITE_ORDER:
            record_key = _record_key(chunk, record_kind)
            if record_key in existing_records:
                metadata_by_key[record_key] = _metadata_for(chunk, record_kind)

    return PromptCacheRestorePlanner(
        layout=_layout(),
        record_metadata_by_key=metadata_by_key,
        record_exists=existing_records.__contains__,
    )


def _record_key(chunk: PromptPrefixChunk, record_kind: str) -> str:
    return make_record_key(chunk.key, record_kind)


def _record_key_span(
    chunk: PromptPrefixChunk, record_kind: str, span_start: int, span_end: int
) -> str:
    return f"{_record_key(chunk, record_kind)}:span:{span_start}:{span_end}"


def test_restore_planner_selects_records_by_cache_kind():
    """KV needs every chunk, SWA needs the target window, state needs target only."""
    chunks = [_chunk(0, 256), _chunk(256, 512), _chunk(512, 768)]
    existing_records = {
        _record_key(chunks[0], RECORD_KIND_KV_DELTA),
        _record_key(chunks[1], RECORD_KIND_KV_DELTA),
        _record_key(chunks[1], RECORD_KIND_ROTATING_DELTA),
        _record_key(chunks[2], RECORD_KIND_KV_DELTA),
        _record_key(chunks[2], RECORD_KIND_ROTATING_DELTA),
        _record_key(chunks[2], RECORD_KIND_STATE_CHECKPOINT),
    }

    record_keys_by_chunk = _planner(
        chunks,
        existing_records,
    ).restore_record_keys_for_chunk_chain(chunks)

    assert record_keys_by_chunk == {
        chunks[0].key: [_record_key(chunks[0], RECORD_KIND_KV_DELTA)],
        chunks[1].key: [
            _record_key(chunks[1], RECORD_KIND_KV_DELTA),
            _record_key(chunks[1], RECORD_KIND_ROTATING_DELTA),
        ],
        chunks[2].key: [
            _record_key(chunks[2], RECORD_KIND_KV_DELTA),
            _record_key(chunks[2], RECORD_KIND_ROTATING_DELTA),
            _record_key(chunks[2], RECORD_KIND_STATE_CHECKPOINT),
        ],
    }


def test_restore_planner_returns_none_when_required_record_is_missing():
    """A restore plan is all-or-nothing for the selected boundary."""
    chunks = [_chunk(0, 256), _chunk(256, 512)]
    existing_records = {
        _record_key(chunks[0], RECORD_KIND_KV_DELTA),
        _record_key(chunks[1], RECORD_KIND_KV_DELTA),
        _record_key(chunks[1], RECORD_KIND_STATE_CHECKPOINT),
    }

    record_keys_by_chunk = _planner(
        chunks,
        existing_records,
    ).restore_record_keys_for_chunk_chain(chunks)

    assert record_keys_by_chunk is None


def test_restore_planner_prefers_span_kv_record_and_skips_covered_chunks():
    """Span-aware KV records should be used and earlier covered chunks should be skipped."""
    chunks = [_chunk(0, 256), _chunk(256, 512), _chunk(512, 768)]
    coalesced_span = _record_key_span(chunks[1], RECORD_KIND_KV_DELTA, 0, 512)
    existing_records = {
        coalesced_span,
        _record_key(chunks[1], RECORD_KIND_KV_DELTA),
        _record_key(chunks[1], RECORD_KIND_ROTATING_DELTA),
        _record_key(chunks[2], RECORD_KIND_KV_DELTA),
        _record_key(chunks[2], RECORD_KIND_ROTATING_DELTA),
        _record_key(chunks[2], RECORD_KIND_STATE_CHECKPOINT),
    }

    metadata_by_key = {
        coalesced_span: PromptCacheRecordMetadata(
            chunk_key=chunks[1].key,
            record_kind=RECORD_KIND_KV_DELTA,
            layer_indices=[],
            chunk_span=[0, 512],
        ),
        _record_key(chunks[1], RECORD_KIND_KV_DELTA): PromptCacheRecordMetadata(
            chunk_key=chunks[1].key,
            record_kind=RECORD_KIND_KV_DELTA,
            layer_indices=[],
        ),
        _record_key(chunks[1], RECORD_KIND_ROTATING_DELTA): PromptCacheRecordMetadata(
            chunk_key=chunks[1].key,
            record_kind=RECORD_KIND_ROTATING_DELTA,
            layer_indices=[],
        ),
        _record_key(chunks[2], RECORD_KIND_KV_DELTA): PromptCacheRecordMetadata(
            chunk_key=chunks[2].key,
            record_kind=RECORD_KIND_KV_DELTA,
            layer_indices=[],
        ),
        _record_key(chunks[2], RECORD_KIND_ROTATING_DELTA): PromptCacheRecordMetadata(
            chunk_key=chunks[2].key,
            record_kind=RECORD_KIND_ROTATING_DELTA,
            layer_indices=[],
        ),
        _record_key(chunks[2], RECORD_KIND_STATE_CHECKPOINT): PromptCacheRecordMetadata(
            chunk_key=chunks[2].key,
            record_kind=RECORD_KIND_STATE_CHECKPOINT,
            layer_indices=[],
        ),
    }

    planner = PromptCacheRestorePlanner(
        layout=_layout(),
        record_metadata_by_key=metadata_by_key,
        record_exists=existing_records.__contains__,
    )
    record_keys_by_chunk = planner.restore_record_keys_for_chunk_chain(chunks)

    assert record_keys_by_chunk == {
        chunks[0].key: [],
        chunks[1].key: [
            coalesced_span,
            _record_key(chunks[1], RECORD_KIND_ROTATING_DELTA),
        ],
        chunks[2].key: [
            _record_key(chunks[2], RECORD_KIND_KV_DELTA),
            _record_key(chunks[2], RECORD_KIND_ROTATING_DELTA),
            _record_key(chunks[2], RECORD_KIND_STATE_CHECKPOINT),
        ],
    }


def test_restore_planner_ignores_overwide_kv_span_when_local_span_exists():
    """Rejected full-prefix KV spans should not outrank bounded local spans."""
    chunks = [_chunk(0, 256), _chunk(256, 512), _chunk(512, 768)]
    overwide_span = _record_key_span(chunks[2], RECORD_KIND_KV_DELTA, 0, 768)
    local_span = _record_key_span(chunks[2], RECORD_KIND_KV_DELTA, 256, 768)
    prior_span = _record_key_span(chunks[1], RECORD_KIND_KV_DELTA, 0, 512)
    existing_records = {
        _record_key(chunks[0], RECORD_KIND_KV_DELTA),
        prior_span,
        overwide_span,
        local_span,
        _record_key(chunks[1], RECORD_KIND_ROTATING_DELTA),
        _record_key(chunks[2], RECORD_KIND_ROTATING_DELTA),
        _record_key(chunks[2], RECORD_KIND_STATE_CHECKPOINT),
    }
    metadata_by_key = {
        _record_key(chunks[0], RECORD_KIND_KV_DELTA): PromptCacheRecordMetadata(
            chunk_key=chunks[0].key,
            record_kind=RECORD_KIND_KV_DELTA,
            layer_indices=[],
        ),
        prior_span: PromptCacheRecordMetadata(
            chunk_key=chunks[1].key,
            record_kind=RECORD_KIND_KV_DELTA,
            layer_indices=[],
            chunk_span=[0, 512],
        ),
        overwide_span: PromptCacheRecordMetadata(
            chunk_key=chunks[2].key,
            record_kind=RECORD_KIND_KV_DELTA,
            layer_indices=[],
            chunk_span=[0, 768],
        ),
        local_span: PromptCacheRecordMetadata(
            chunk_key=chunks[2].key,
            record_kind=RECORD_KIND_KV_DELTA,
            layer_indices=[],
            chunk_span=[256, 768],
        ),
        _record_key(chunks[1], RECORD_KIND_ROTATING_DELTA): PromptCacheRecordMetadata(
            chunk_key=chunks[1].key,
            record_kind=RECORD_KIND_ROTATING_DELTA,
            layer_indices=[],
        ),
        _record_key(chunks[2], RECORD_KIND_ROTATING_DELTA): PromptCacheRecordMetadata(
            chunk_key=chunks[2].key,
            record_kind=RECORD_KIND_ROTATING_DELTA,
            layer_indices=[],
        ),
        _record_key(chunks[2], RECORD_KIND_STATE_CHECKPOINT): PromptCacheRecordMetadata(
            chunk_key=chunks[2].key,
            record_kind=RECORD_KIND_STATE_CHECKPOINT,
            layer_indices=[],
        ),
    }

    planner = PromptCacheRestorePlanner(
        layout=_layout(),
        record_metadata_by_key=metadata_by_key,
        record_exists=existing_records.__contains__,
    )

    record_keys_by_chunk = planner.restore_record_keys_for_chunk_chain(chunks)

    assert record_keys_by_chunk == {
        chunks[0].key: [_record_key(chunks[0], RECORD_KIND_KV_DELTA)],
        chunks[1].key: [_record_key(chunks[1], RECORD_KIND_ROTATING_DELTA)],
        chunks[2].key: [
            local_span,
            _record_key(chunks[2], RECORD_KIND_ROTATING_DELTA),
            _record_key(chunks[2], RECORD_KIND_STATE_CHECKPOINT),
        ],
    }


def test_restore_planner_prefers_explicit_terminal_packed_kv_on_target_chunk():
    """The final restore chunk may intentionally use one full-prefix KV record."""
    chunks = [_chunk(0, 256), _chunk(256, 512), _chunk(512, 768)]
    terminal_packed_span = _record_key(chunks[2], RECORD_KIND_KV_DELTA)
    prior_span = _record_key_span(chunks[1], RECORD_KIND_KV_DELTA, 0, 512)
    existing_records = {
        terminal_packed_span,
        prior_span,
        _record_key(chunks[1], RECORD_KIND_ROTATING_DELTA),
        _record_key(chunks[2], RECORD_KIND_ROTATING_DELTA),
        _record_key(chunks[2], RECORD_KIND_STATE_CHECKPOINT),
    }
    metadata_by_key = {
        terminal_packed_span: PromptCacheRecordMetadata(
            chunk_key=chunks[2].key,
            record_kind=RECORD_KIND_KV_DELTA,
            layer_indices=[],
            chunk_span=[0, 768],
            is_terminal_packed=True,
        ),
        prior_span: PromptCacheRecordMetadata(
            chunk_key=chunks[1].key,
            record_kind=RECORD_KIND_KV_DELTA,
            layer_indices=[],
            chunk_span=[0, 512],
        ),
        _record_key(chunks[1], RECORD_KIND_ROTATING_DELTA): PromptCacheRecordMetadata(
            chunk_key=chunks[1].key,
            record_kind=RECORD_KIND_ROTATING_DELTA,
            layer_indices=[],
        ),
        _record_key(chunks[2], RECORD_KIND_ROTATING_DELTA): PromptCacheRecordMetadata(
            chunk_key=chunks[2].key,
            record_kind=RECORD_KIND_ROTATING_DELTA,
            layer_indices=[],
        ),
        _record_key(chunks[2], RECORD_KIND_STATE_CHECKPOINT): PromptCacheRecordMetadata(
            chunk_key=chunks[2].key,
            record_kind=RECORD_KIND_STATE_CHECKPOINT,
            layer_indices=[],
        ),
    }

    planner = PromptCacheRestorePlanner(
        layout=_layout(),
        record_metadata_by_key=metadata_by_key,
        record_exists=existing_records.__contains__,
    )

    record_keys_by_chunk = planner.restore_record_keys_for_chunk_chain(chunks)

    assert record_keys_by_chunk == {
        chunks[0].key: [],
        chunks[1].key: [_record_key(chunks[1], RECORD_KIND_ROTATING_DELTA)],
        chunks[2].key: [
            terminal_packed_span,
            _record_key(chunks[2], RECORD_KIND_ROTATING_DELTA),
            _record_key(chunks[2], RECORD_KIND_STATE_CHECKPOINT),
        ],
    }


def test_restore_planner_span_lookup_skips_unrelated_kv_records():
    """KV span selection should only inspect records for the target chunk."""
    chunks = [_chunk(0, 256), _chunk(256, 512)]
    local_span = _record_key_span(chunks[1], RECORD_KIND_KV_DELTA, 0, 512)
    unrelated_chunk = _chunk(1024, 1280)
    unrelated_span = _record_key_span(
        unrelated_chunk,
        RECORD_KIND_KV_DELTA,
        768,
        1280,
    )
    existing_records = {
        local_span,
        unrelated_span,
        _record_key(chunks[0], RECORD_KIND_ROTATING_DELTA),
        _record_key(chunks[1], RECORD_KIND_ROTATING_DELTA),
        _record_key(chunks[1], RECORD_KIND_STATE_CHECKPOINT),
    }
    metadata_by_key = {
        local_span: PromptCacheRecordMetadata(
            chunk_key=chunks[1].key,
            record_kind=RECORD_KIND_KV_DELTA,
            layer_indices=[],
            chunk_span=[0, 512],
        ),
        unrelated_span: PromptCacheRecordMetadata(
            chunk_key=unrelated_chunk.key,
            record_kind=RECORD_KIND_KV_DELTA,
            layer_indices=[],
            chunk_span=[768, 1280],
        ),
        _record_key(chunks[0], RECORD_KIND_ROTATING_DELTA): PromptCacheRecordMetadata(
            chunk_key=chunks[0].key,
            record_kind=RECORD_KIND_ROTATING_DELTA,
            layer_indices=[],
        ),
        _record_key(chunks[1], RECORD_KIND_ROTATING_DELTA): PromptCacheRecordMetadata(
            chunk_key=chunks[1].key,
            record_kind=RECORD_KIND_ROTATING_DELTA,
            layer_indices=[],
        ),
        _record_key(chunks[1], RECORD_KIND_STATE_CHECKPOINT): PromptCacheRecordMetadata(
            chunk_key=chunks[1].key,
            record_kind=RECORD_KIND_STATE_CHECKPOINT,
            layer_indices=[],
        ),
    }
    existence_checks = []

    def record_exists(record_key: str) -> bool:
        existence_checks.append(record_key)
        return record_key in existing_records

    planner = PromptCacheRestorePlanner(
        layout=_layout(),
        record_metadata_by_key=metadata_by_key,
        record_exists=record_exists,
    )

    record_keys_by_chunk = planner.restore_record_keys_for_chunk_chain(chunks)

    assert record_keys_by_chunk == {
        chunks[0].key: [_record_key(chunks[0], RECORD_KIND_ROTATING_DELTA)],
        chunks[1].key: [
            local_span,
            _record_key(chunks[1], RECORD_KIND_ROTATING_DELTA),
            _record_key(chunks[1], RECORD_KIND_STATE_CHECKPOINT),
        ],
    }
    assert unrelated_span not in existence_checks
