"""Benchmark VLM prompt-cache restore-planner lookup scaling.

This benchmark is intentionally CPU-only. It exercises the planner's metadata
selection path without loading model weights, touching Metal, or reading cache
blobs from disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlx_engine.model_kit.batched_vision.prompt_cache.restore_planner import (  # noqa: E402
    PromptCacheRestorePlanner,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.types import (  # noqa: E402
    PromptCacheLayout,
    PromptCacheRecordMetadata,
    PromptPrefixChunk,
    RECORD_KIND_KV_DELTA,
    make_record_key,
)


DEFAULT_CHUNK_SIZE = 512


def chunk_at(index: int, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> PromptPrefixChunk:
    """Return a deterministic prompt-prefix chunk for benchmark metadata."""
    start = index * chunk_size
    end = start + chunk_size
    return PromptPrefixChunk(start=start, end=end, key=f"chunk-{start}-{end}")


def span_record_key(
    chunk: PromptPrefixChunk,
    *,
    span_start: int,
    span_end: int,
) -> str:
    """Return the physical record key for a synthetic span KV record."""
    return (
        f"{make_record_key(chunk.key, RECORD_KIND_KV_DELTA)}"
        f":span:{span_start}:{span_end}"
    )


def kv_only_layout() -> PromptCacheLayout:
    """Return the planner layout used to isolate full-attention KV lookup cost."""
    return PromptCacheLayout(
        layer_kinds=[RECORD_KIND_KV_DELTA],
        layer_indices_by_kind={RECORD_KIND_KV_DELTA: [0]},
        rotating_window_size=None,
    )


def build_metadata(
    *,
    index_chunks: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[list[PromptPrefixChunk], dict[str, PromptCacheRecordMetadata], set[str]]:
    """Build synthetic one-step-span KV metadata for a persistent cache index."""
    chunks = [chunk_at(index, chunk_size=chunk_size) for index in range(index_chunks)]
    metadata_by_key: dict[str, PromptCacheRecordMetadata] = {}
    existing_records: set[str] = set()
    for index, chunk in enumerate(chunks):
        if index == 0:
            record_key = make_record_key(chunk.key, RECORD_KIND_KV_DELTA)
            chunk_span = [chunk.start, chunk.end]
        else:
            prev_chunk = chunks[index - 1]
            record_key = span_record_key(
                chunk,
                span_start=prev_chunk.start,
                span_end=chunk.end,
            )
            chunk_span = [prev_chunk.start, chunk.end]
        metadata_by_key[record_key] = PromptCacheRecordMetadata(
            chunk_key=chunk.key,
            record_kind=RECORD_KIND_KV_DELTA,
            layer_indices=[0],
            chunk_span=chunk_span,
        )
        existing_records.add(record_key)

    return chunks, metadata_by_key, existing_records


def metadata_span(
    metadata: PromptCacheRecordMetadata,
) -> tuple[int | None, int | None]:
    """Return normalized optional span metadata."""
    span = metadata.chunk_span
    if not span or len(span) != 2:
        return None, None
    try:
        return int(span[0]), int(span[1])
    except (TypeError, ValueError):
        return None, None


class LegacyScanPromptCacheRestorePlanner:
    """Pre-index span selector used only to compare planner-scaling cost."""

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

    def restore_record_keys_for_chunk_chain(
        self,
        chunks: list[PromptPrefixChunk],
    ) -> dict[str, list[str]] | None:
        """Return the KV restore chain with legacy full-index span scans."""
        if not chunks:
            return {}
        covered_indices: set[int] = set()
        selected_by_chunk_key: dict[str, str] = {}
        chunk_index_by_start = {chunk.start: idx for idx, chunk in enumerate(chunks)}
        for idx, chunk in reversed(list(enumerate(chunks))):
            if idx in covered_indices:
                continue
            min_span_start = (
                chunks[idx - 1].start
                if idx > 0 and chunks[idx - 1].end == chunk.start
                else chunk.start
            )
            record_key = self._select_kv_record_key(
                chunk,
                min_span_start=min_span_start,
            )
            if record_key is None:
                return None
            selected_by_chunk_key[chunk.key] = record_key
            span_start, _ = metadata_span(self._record_metadata_by_key[record_key])
            if span_start is not None and span_start < chunk.start:
                span_start_index = chunk_index_by_start.get(span_start)
                if span_start_index is None:
                    return None
                covered_indices.update(range(span_start_index, idx))

        return {
            chunk.key: (
                [] if idx in covered_indices else [selected_by_chunk_key[chunk.key]]
            )
            for idx, chunk in enumerate(chunks)
        }

    def _select_kv_record_key(
        self,
        chunk: PromptPrefixChunk,
        *,
        min_span_start: int,
    ) -> str | None:
        """Return the same bounded KV record choice using full metadata scans."""
        plain_record_key = make_record_key(chunk.key, RECORD_KIND_KV_DELTA)
        span_candidates = []
        for record_key, metadata in self._record_metadata_by_key.items():
            if (
                metadata.chunk_key != chunk.key
                or metadata.record_kind != RECORD_KIND_KV_DELTA
                or not self._has_record(record_key)
            ):
                continue
            span_start, span_end = metadata_span(metadata)
            if span_start is None or span_end is None:
                continue
            if span_start < min_span_start:
                continue
            if span_start <= chunk.start <= span_end and span_end >= chunk.end:
                span_candidates.append((span_start, span_end, record_key))
        if span_candidates:
            span_candidates.sort(key=lambda item: (item[0], -item[1]))
            return span_candidates[0][2]
        if self._has_record(plain_record_key):
            return plain_record_key
        return None

    def _has_record(self, record_key: str) -> bool:
        """Return whether a synthetic record is indexed and available."""
        return record_key in self._record_metadata_by_key and self._record_exists(
            record_key
        )


def time_planner(
    planner_type: (
        type[PromptCacheRestorePlanner] | type[LegacyScanPromptCacheRestorePlanner]
    ),
    *,
    iterations: int,
    restore_chunks: list[PromptPrefixChunk],
    layout: PromptCacheLayout,
    metadata_by_key: dict[str, PromptCacheRecordMetadata],
    existing_records: set[str],
) -> tuple[float, dict[str, list[str]]]:
    """Return median construct-and-plan duration plus the last restore chain."""
    durations_ms = []
    last_plan = None
    for _ in range(iterations):
        start = perf_counter()
        planner = planner_type(
            layout=layout,
            record_metadata_by_key=metadata_by_key,
            record_exists=existing_records.__contains__,
        )
        last_plan = planner.restore_record_keys_for_chunk_chain(restore_chunks)
        durations_ms.append((perf_counter() - start) * 1000.0)
    if last_plan is None:
        raise RuntimeError("planner failed to restore synthetic benchmark chain")
    return median(durations_ms), last_plan


def run_benchmark(
    *,
    index_chunks: int,
    restore_chunks: int,
    iterations: int,
) -> dict[str, float | int]:
    """Run indexed and legacy planner benchmarks and return summary metrics."""
    if index_chunks < 1:
        raise ValueError("index_chunks must be positive")
    if restore_chunks < 1 or restore_chunks > index_chunks:
        raise ValueError("restore_chunks must be in [1, index_chunks]")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    chunks, metadata_by_key, existing_records = build_metadata(
        index_chunks=index_chunks,
    )
    restore_chain = chunks[:restore_chunks]
    layout = kv_only_layout()
    indexed_ms, indexed_plan = time_planner(
        PromptCacheRestorePlanner,
        iterations=iterations,
        restore_chunks=restore_chain,
        layout=layout,
        metadata_by_key=metadata_by_key,
        existing_records=existing_records,
    )
    legacy_ms, legacy_plan = time_planner(
        LegacyScanPromptCacheRestorePlanner,
        iterations=iterations,
        restore_chunks=restore_chain,
        layout=layout,
        metadata_by_key=metadata_by_key,
        existing_records=existing_records,
    )
    if indexed_plan != legacy_plan:
        raise RuntimeError("indexed planner result diverged from legacy scan result")

    speedup = legacy_ms / indexed_ms if indexed_ms else float("inf")
    selected_records = sum(len(records) for records in indexed_plan.values())
    return {
        "index_chunks": index_chunks,
        "restore_chunks": restore_chunks,
        "iterations": iterations,
        "records_in_index": len(metadata_by_key),
        "selected_records": selected_records,
        "indexed_median_ms": round(indexed_ms, 6),
        "legacy_scan_median_ms": round(legacy_ms, 6),
        "speedup": round(speedup, 3),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the restore-planner benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark VLM restore-planner KV span lookup scaling."
    )
    parser.add_argument("--index-chunks", type=int, default=4096)
    parser.add_argument("--restore-chunks", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    return parser.parse_args()


def main() -> None:
    """Run the benchmark and print a concise report."""
    args = parse_args()
    result = run_benchmark(
        index_chunks=args.index_chunks,
        restore_chunks=args.restore_chunks,
        iterations=args.iterations,
    )
    if args.json:
        print(json.dumps(result, sort_keys=True))
        return

    print("VLM restore-planner lookup benchmark")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
