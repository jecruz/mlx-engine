import logging
import inspect
import traceback
from dataclasses import dataclass
from itertools import count
from queue import Empty as QueueEmpty
from queue import PriorityQueue
from queue import Queue
from threading import Event, Thread
from typing import Any, Callable

from mlx_engine.model_kit.batched_vision.prompt_cache.cache_store import (
    VlmPromptCacheStore,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.types import (
    PendingPromptCacheSave,
    PromptPrefixChunk,
)
from mlx_engine.model_kit.batched_vision.request_lifecycle import (
    FailedRestore,
    GenerationRequest,
    PreparedInsert,
)
from mlx_engine.utils.mlx_threading import (
    install_mlx_compile_cache_cleanup_for_thread,
)

logger = logging.getLogger(__name__)


_RESTORE_JOB_PRIORITY = 0
_CACHE_STORE_BUDGET_UPDATE_JOB_PRIORITY = 1
_SAVE_JOB_PRIORITY = 2
_SHUTDOWN_JOB_PRIORITY = -1


@dataclass
class RestoreJob:
    request: GenerationRequest


@dataclass
class SaveJob:
    pending_save: PendingPromptCacheSave


@dataclass
class CacheStoreBudgetUpdateJob:
    max_cache_store_bytes: int


class PromptCacheIOThread:
    """Runs background restore prep and blocking cache store commits."""

    def __init__(
        self,
        *,
        cache_store: VlmPromptCacheStore,
        generation_queue: Queue,
        prepare_request: Callable[[GenerationRequest], PreparedInsert],
    ):
        self._cache_store = cache_store
        self._generation_queue = generation_queue
        self._prepare_request = prepare_request
        self._prepare_request_accepts_flush_helper = (
            len(inspect.signature(prepare_request).parameters) >= 2
        )
        self._queue = PriorityQueue()
        self._sequence = count()
        self._thread = None
        self._closed = Event()

    def start(self) -> None:
        self._thread = Thread(
            target=self._run,
            name="mlx-engine-vlm-cache-io",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        if self._thread is None:
            self._close_cache_store()
            return
        self._enqueue(_SHUTDOWN_JOB_PRIORITY, None)
        self._thread.join()
        self._close_cache_store()

    def enqueue_restore(self, request: GenerationRequest) -> None:
        self._enqueue(_RESTORE_JOB_PRIORITY, RestoreJob(request))

    def enqueue_save(self, pending_save: PendingPromptCacheSave) -> None:
        # The generation thread already prepared arrays; this thread does disk I/O.
        self._enqueue(_SAVE_JOB_PRIORITY, SaveJob(pending_save))

    def enqueue_cache_store_budget_update(
        self, max_cache_store_bytes: int | None
    ) -> None:
        if max_cache_store_bytes is None:
            return
        # Budget changes can evict blob-store records, so keep them on this thread.
        self._enqueue(
            _CACHE_STORE_BUDGET_UPDATE_JOB_PRIORITY,
            CacheStoreBudgetUpdateJob(max_cache_store_bytes),
        )

    def _enqueue(self, priority: int, job: Any) -> None:
        self._queue.put((priority, next(self._sequence), job))

    def _close_cache_store(self) -> None:
        self._cache_store.close()

    def _discard_queued_jobs(self) -> None:
        while True:
            try:
                # Dropped cache store jobs only hold immutable arrays or budgets.
                self._queue.get_nowait()
            except QueueEmpty:
                return

    @staticmethod
    def _save_prefix_matches_request(
        pending_save: PendingPromptCacheSave,
        request_prefix_chunks: list[PromptPrefixChunk],
    ) -> bool:
        save_prefix_chunks = pending_save.prefix_chunks
        if not save_prefix_chunks or len(save_prefix_chunks) > len(request_prefix_chunks):
            return False
        return all(
            saved.key == requested.key
            for saved, requested in zip(save_prefix_chunks, request_prefix_chunks)
        )

    def _flush_matching_save_jobs(
        self,
        request_prefix_chunks: list[PromptPrefixChunk],
    ) -> int:
        """Commit an ordered prefix of matching queued save jobs before restore.

        Restore jobs keep higher queue priority overall. This only helps the
        active restore when the next queued save(s) would make its disk prefix
        immediately restorable. Budget updates ahead of those saves are applied
        first so save ordering and eviction semantics stay intact.
        """
        if not request_prefix_chunks:
            return 0

        queued_items = []
        while True:
            try:
                queued_items.append(self._queue.get_nowait())
            except QueueEmpty:
                break

        if not queued_items:
            return 0

        flushed = 0
        flush_prefix_len = 0
        saw_matching_save = False
        for priority, sequence, job in queued_items:
            if isinstance(job, CacheStoreBudgetUpdateJob) and not saw_matching_save:
                flush_prefix_len += 1
                continue
            if isinstance(job, SaveJob) and self._save_prefix_matches_request(
                job.pending_save,
                request_prefix_chunks,
            ):
                saw_matching_save = True
                flush_prefix_len += 1
                continue
            break

        if saw_matching_save:
            for _, _, job in queued_items[:flush_prefix_len]:
                if isinstance(job, CacheStoreBudgetUpdateJob):
                    self._cache_store.commit_budget_update(job.max_cache_store_bytes)
                elif isinstance(job, SaveJob):
                    self._cache_store.commit_pending_save(job.pending_save)
                    flushed += 1

        for queued_item in queued_items[flush_prefix_len if saw_matching_save else 0 :]:
            self._queue.put(queued_item)

        return flushed

    def _run(self) -> None:
        install_mlx_compile_cache_cleanup_for_thread()
        while True:
            _, _, job = self._queue.get()
            if job is None:
                self._discard_queued_jobs()
                return

            if isinstance(job, RestoreJob):
                try:
                    if self._prepare_request_accepts_flush_helper:
                        prepared_insert = self._prepare_request(
                            job.request,
                            self._flush_matching_save_jobs,
                        )
                    else:
                        prepared_insert = self._prepare_request(job.request)
                except Exception as exc:
                    self._generation_queue.put(FailedRestore(job.request, exc))
                    continue

                self._generation_queue.put(prepared_insert)
                continue

            if isinstance(job, SaveJob):
                try:
                    self._cache_store.commit_pending_save(job.pending_save)
                except Exception:
                    logger.error(
                        "Failed to commit pending prompt cache save:\n%s",
                        traceback.format_exc(),
                    )
                continue

            if isinstance(job, CacheStoreBudgetUpdateJob):
                try:
                    self._cache_store.commit_budget_update(job.max_cache_store_bytes)
                except Exception:
                    logger.error(
                        "Failed to commit prompt cache store budget:\n%s",
                        traceback.format_exc(),
                    )
