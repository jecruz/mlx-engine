import copy
import logging
from typing import Any, List, Literal, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import maybe_quantize_kv_cache
from mlx_lm.models.cache import (
    LRUPromptCache,
    KVCache,
    QuantizedKVCache,
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)

from mlx_engine.utils.mlx_lm_stream import prepare_mlx_lm_generation_stream
from mlx_engine.utils.prompt_progress_reporter import (
    PromptProgressReporter,
    StopPromptProcessing,
)
from mlx_engine.utils.specprefill import (
    SpecPrefillOptions,
    cleanup_rope,
    try_sparse_prefill,
)


PROMPT_PROCESSING_CHUNK_SIZE = 2048
DEFERRED_CLEAR_DELAY_STEPS = 64

# Checkpoint N tokens before end of prompt
# This value is at parity with mlx-lm:
# https://github.com/ml-explore/mlx-lm/blob/d9c63f/mlx_lm/server.py#L587
DEFAULT_CHECKPOINT_TAIL_TOKENS = 11

logger = logging.getLogger(__name__)


def _clone_cache_value(value: Any) -> Any:
    if isinstance(value, mx.array):
        return mx.array(value)
    if isinstance(value, list):
        return [_clone_cache_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_cache_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_cache_value(item) for key, item in value.items()}
    return copy.deepcopy(value)


def _clone_kv_cache_entry(entry: Any) -> Any:
    clone = KVCache()
    clone.keys = mx.array(entry.keys) if entry.keys is not None else None
    clone.values = mx.array(entry.values) if entry.values is not None else None
    clone.offset = entry.offset
    return clone


def _clone_quantized_kv_cache_entry(entry: Any) -> Any:
    clone = QuantizedKVCache(group_size=entry.group_size, bits=entry.bits)
    clone.keys = (
        tuple(mx.array(item) for item in entry.keys) if entry.keys is not None else None
    )
    clone.values = (
        tuple(mx.array(item) for item in entry.values)
        if entry.values is not None
        else None
    )
    clone.offset = entry.offset
    return clone


def _clone_cache_entry(entry: Any) -> Any:
    if isinstance(entry, KVCache):
        return _clone_kv_cache_entry(entry)
    if isinstance(entry, QuantizedKVCache):
        return _clone_quantized_kv_cache_entry(entry)

    from_state = getattr(type(entry), "from_state", None)
    if (
        callable(from_state)
        and hasattr(entry, "state")
        and hasattr(entry, "meta_state")
    ):
        try:
            return from_state(
                _clone_cache_value(entry.state),
                _clone_cache_value(entry.meta_state),
            )
        except Exception:
            logger.debug("Cache snapshot clone via from_state failed; falling back.")
    return copy.deepcopy(entry)


class FastLRUPromptCache(LRUPromptCache):
    def fetch_nearest_cache(self, model: Any, tokens: List[int]):
        result = self._trie.search(model, tokens)
        if result.exact is not None:
            cache_entry = self._trie.get(result.model, result.exact)
            return [_clone_cache_entry(entry) for entry in cache_entry.prompt_cache], []

        short_length = len(result.shorter) if result.shorter is not None else 0
        if result.longer is not None and result.common_prefix > short_length:
            cache_entry = self._trie.get(result.model, result.longer)
            if can_trim_prompt_cache(cache_entry.prompt_cache):
                cache = [
                    _clone_cache_entry(entry) for entry in cache_entry.prompt_cache
                ]
                prefix = min(len(tokens) - 1, result.common_prefix)
                num_to_trim = len(result.longer) - prefix
                trim_prompt_cache(cache, num_to_trim)
                return cache, tokens[prefix:]

        if short_length > 0:
            cache_entry = self._trie.get(result.model, result.shorter)
            return [
                _clone_cache_entry(entry) for entry in cache_entry.prompt_cache
            ], tokens[short_length:]

        return None, tokens


def validate_prefill_step_size(prefill_step_size: Optional[int] = None) -> int:
    if prefill_step_size is None:
        return PROMPT_PROCESSING_CHUNK_SIZE
    if (
        isinstance(prefill_step_size, bool)
        or not isinstance(prefill_step_size, int)
        or prefill_step_size < 1
    ):
        raise ValueError("prefill_step_size must be a positive integer")
    return prefill_step_size


class CacheWrapper:
    def __init__(
        self,
        model: nn.Module,
        max_kv_size: Optional[int],
        *,
        kv_bits: Optional[int] = None,
        kv_group_size: Optional[int] = None,
        quantized_kv_start: Optional[int] = None,
        chunk_size: int,
        checkpoint_tail_tokens: int = DEFAULT_CHECKPOINT_TAIL_TOKENS,
        history_capacity: int = 10,
    ):
        self.model = model
        self._draft_model: Optional[nn.Module] = None
        self._max_kv_size = max_kv_size
        self._chunk_size = chunk_size
        self._checkpoint_tail_tokens = checkpoint_tail_tokens
        self._history_capacity = history_capacity
        self._kv_cache_qtn_params = dict(
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
        )

        self._history = self._make_history()
        self._history_key = "session"
        self._generation_step_counter = 0
        self._deferred_clear_at: int | None = None
        self._sparse_cache_active = False
        # Keep token history host-side. Sequential requests can resume on a
        # different native thread, and lazy MLX arrays carry stream ownership.
        self._live_tokens: Optional[List[int]] = None
        self._live_cache: List[Any] = self._make_cache()

    @property
    def cache(self) -> List[Any]:
        return self._live_cache

    def _make_cache(self) -> List[Any]:
        cache = make_prompt_cache(self.model, self._max_kv_size)
        if self._draft_model is not None:
            cache += make_prompt_cache(self._draft_model)
        return cache

    def _make_history(self) -> LRUPromptCache:
        # Store up to N checkpoints. This number can be tuned (or made configurable) if
        # it's too high or low
        return FastLRUPromptCache(max_size=self._history_capacity)

    def _num_tokens_in_cache(self, cache: Optional[List[Any]] = None) -> int | None:
        cache = self._live_cache if cache is None else cache
        if cache:
            head_offset = getattr(cache[0], "offset", None)
            if head_offset is not None:
                return head_offset

        for entry in cache[1:]:
            if hasattr(entry, "offset"):
                return entry.offset
        return None

    def _store_snapshot(
        self,
        tokens: Sequence[int],
        cache: List[Any],
        *,
        cache_type: Literal["user", "assistant"],
    ) -> None:
        if len(tokens) == 0:
            return
        self._history.insert_cache(
            self._history_key,
            list(tokens),
            [_clone_cache_entry(entry) for entry in cache],
            cache_type=cache_type,
        )

    def _clear_cache_now(self) -> None:
        mx.synchronize()
        mx.clear_cache()

    def _schedule_deferred_clear(self) -> None:
        target = self._generation_step_counter + DEFERRED_CLEAR_DELAY_STEPS
        if self._deferred_clear_at is None or target > self._deferred_clear_at:
            self._deferred_clear_at = target

    def _maybe_clear_cache(self) -> None:
        if (
            self._deferred_clear_at is None
            or self._generation_step_counter < self._deferred_clear_at
        ):
            return

        self._deferred_clear_at = None
        self._clear_cache_now()

    def _flush_live_cache(self) -> None:
        if self._live_tokens is None:
            return
        if self._sparse_cache_active:
            self._sparse_cache_active = False
            self._live_tokens = None
            self._live_cache = self._make_cache()
            return

        cache_length = self._num_tokens_in_cache()
        if cache_length is None:
            logger.warning(
                "Could not determine the number of tokens in the live cache. Resetting it."
            )
            self._live_tokens = None
            self._live_cache = self._make_cache()
            return
        if cache_length > len(self._live_tokens):
            logger.warning(
                "The live cache is longer than the tracked token history. Resetting it."
            )
            self._live_tokens = None
            self._live_cache = self._make_cache()
            return
        if cache_length <= 0:
            return

        snapshot_length = min(len(self._live_tokens), cache_length + 1)
        self._store_snapshot(
            self._live_tokens[:snapshot_length],
            self._live_cache,
            cache_type="assistant",
        )

    def _restore_cache(
        self,
        prompt_tokens: mx.array,
        prompt_token_list: List[int],
    ) -> tuple[Optional[List[Any]], mx.array]:
        if len(prompt_token_list) == 0:
            return None, prompt_tokens

        cache, rest = self._history.fetch_nearest_cache(
            self._history_key,
            prompt_token_list,
        )
        if cache is not None:
            if len(rest) > 0:
                return cache, prompt_tokens[len(prompt_token_list) - len(rest) :]

            if can_trim_prompt_cache(cache) and trim_prompt_cache(cache, 1) == 1:
                return cache, prompt_tokens[-1:]

        if len(prompt_token_list) <= 1:
            return None, prompt_tokens

        # Exact hits need one token outside the cache to seed decode. If the
        # exact-hit cache cannot be trimmed, retry with one less prompt token
        # so a stored checkpoint can win.
        truncated_prompt = prompt_token_list[:-1]
        cache, rest = self._history.fetch_nearest_cache(
            self._history_key,
            truncated_prompt,
        )
        if cache is None:
            return None, prompt_tokens

        prefix_length = len(truncated_prompt) - len(rest)
        return cache, prompt_tokens[prefix_length:]

    def _prefill_cache(
        self,
        model: nn.Module,
        cache: List[Any],
        cache_start: int,
        tokens: mx.array,
        reporter: PromptProgressReporter,
        is_draft: bool,
        checkpoint_prefix_len: Optional[int] = None,
    ) -> None:
        remaining_tokens = tokens
        num_processed = 0
        stored_checkpoint = False
        current_cache_size = self._num_tokens_in_cache(cache)

        while remaining_tokens.size > 0:
            current_chunk_size = min(self._chunk_size, remaining_tokens.size)
            if (
                checkpoint_prefix_len is not None
                and current_cache_size is not None
                and current_cache_size < checkpoint_prefix_len
                and current_cache_size + current_chunk_size > checkpoint_prefix_len
            ):
                current_chunk_size = checkpoint_prefix_len - current_cache_size

            current_chunk = remaining_tokens[:current_chunk_size]
            model(current_chunk[None], cache=cache)
            maybe_quantize_kv_cache(prompt_cache=cache, **self._kv_cache_qtn_params)
            self._live_cache[cache_start : cache_start + len(cache)] = cache
            mx.eval([entry.state for entry in cache])
            if current_cache_size is not None:
                current_cache_size += current_chunk_size

            remaining_tokens = remaining_tokens[current_chunk_size:]
            num_processed += current_chunk_size

            if (
                checkpoint_prefix_len is not None
                and not stored_checkpoint
                and current_cache_size == checkpoint_prefix_len
            ):
                self._store_snapshot(
                    self._live_tokens[:checkpoint_prefix_len],
                    self._live_cache,
                    cache_type="user",
                )
                stored_checkpoint = True

            if not reporter.update(is_draft, num_processed):
                logger.info("Prompt processing was cancelled by the user.")
                live_cache_size = current_cache_size
                if live_cache_size is None:
                    live_cache_size = self._num_tokens_in_cache()
                if live_cache_size is None:
                    self._live_tokens = None
                    self._live_cache = self._make_cache()
                else:
                    self._live_tokens = self._live_tokens[:live_cache_size]
                raise StopPromptProcessing

    def update_cache(
        self,
        prompt_tokens: mx.array,
        reporter: PromptProgressReporter,
        draft_model: Optional[nn.Module] = None,
        specprefill_options: Optional[SpecPrefillOptions] = None,
    ) -> mx.array:
        prompt_token_list = prompt_tokens.tolist()
        total_prompt_tokens = len(prompt_tokens)

        self._maybe_clear_cache()
        self._flush_live_cache()

        restored_cache, uncached_tokens = self._restore_cache(
            prompt_tokens, prompt_token_list
        )
        self._live_cache = (
            restored_cache if restored_cache is not None else self._make_cache()
        )
        self._live_tokens = prompt_token_list

        cached_tokens = total_prompt_tokens - len(uncached_tokens)
        logger.info(
            "Prompt cache: using %d/%d tokens from cache",
            cached_tokens,
            total_prompt_tokens,
        )

        reporter.begin(
            is_draft=False,
            cached_tokens=cached_tokens,
            total_prompt_tokens=total_prompt_tokens,
            prefill_tokens_processed=0,
        )

        # Leave one token outside the cache to seed decode.
        prefill_tokens = uncached_tokens[:-1]
        if (
            specprefill_options is not None
            and specprefill_options.enabled
            and draft_model is not None
            and self._kv_cache_qtn_params["kv_bits"] is None
            and self._max_kv_size is None
            and len(uncached_tokens) > specprefill_options.threshold
            and specprefill_options.system_tokens < len(uncached_tokens)
        ):
            try:
                result = try_sparse_prefill(
                    model=self.model,
                    draft_model=draft_model,
                    prompt_tokens=prompt_tokens,
                    uncached_tokens=uncached_tokens,
                    cache=self._live_cache[: len(self.model.layers)],
                    cached_tokens=cached_tokens,
                    options=specprefill_options,
                    chunk_size=self._chunk_size,
                    reporter=reporter,
                )
            except Exception:
                logger.exception("SpecPrefill failed; falling back to full prefill")
            else:
                if result is not None:
                    self._live_cache = result.cache
                    self._live_tokens = list(result.live_tokens)
                    self._sparse_cache_active = True
                    reporter.finish(is_draft=False)
                    self._schedule_deferred_clear()
                    return result.seed_tokens

        checkpoint_prefix_len = None
        # Only checkpoint the main-model path; quantized caches skip checkpointing.
        if self._draft_model is None and self._kv_cache_qtn_params["kv_bits"] is None:
            checkpoint_prefix_len = total_prompt_tokens - self._checkpoint_tail_tokens
            # Skip checkpoints that are already cached or would be empty.
            if checkpoint_prefix_len <= cached_tokens:
                checkpoint_prefix_len = None
            if checkpoint_prefix_len is not None and checkpoint_prefix_len <= 0:
                checkpoint_prefix_len = None

        generation_stream = prepare_mlx_lm_generation_stream(reason="cache-prefill")
        try:
            with mx.stream(generation_stream):
                if self._draft_model is not None:
                    draft_cache = self._live_cache[len(self.model.layers) :]
                    self._prefill_cache(
                        model=self._draft_model,
                        cache=draft_cache,
                        cache_start=len(self.model.layers),
                        tokens=prefill_tokens,
                        reporter=reporter,
                        is_draft=True,
                        checkpoint_prefix_len=None,
                    )

                main_cache = self._live_cache[: len(self.model.layers)]
                self._prefill_cache(
                    model=self.model,
                    cache=main_cache,
                    cache_start=0,
                    tokens=prefill_tokens,
                    reporter=reporter,
                    is_draft=False,
                    checkpoint_prefix_len=checkpoint_prefix_len,
                )
        except StopPromptProcessing:
            self._clear_cache_now()
            raise

        reporter.finish(is_draft=False)
        self._schedule_deferred_clear()
        return uncached_tokens[-1:]

    def record_generated_token(self, token: int) -> None:
        self._generation_step_counter += 1
        self._maybe_clear_cache()
        if self._live_tokens is None:
            self._live_tokens = [token]
            return
        self._live_tokens.append(token)

    def cleanup_specprefill(self) -> None:
        """Restore transient SpecPrefill model state after generation exits."""
        cleanup_rope(self.model)

    def set_draft_model(self, draft_model: nn.Module) -> None:
        if self.model is None:
            raise ValueError("Cannot add a draft model to cache without a main model")
        if self._draft_model is draft_model:
            return
        if self._max_kv_size is not None:
            logger.info("Disabling max_kv_size when setting a draft model for cache")
            self._max_kv_size = None

        self._history = self._make_history()
        self._draft_model = draft_model
        self._deferred_clear_at = None
        self._sparse_cache_active = False
        self._live_tokens = None
        self._live_cache = self._make_cache()

    def unset_draft_model(self) -> None:
        if self._draft_model is None:
            return
        main_cache = self._live_cache[: len(self.model.layers)]
        self._history = self._make_history()
        self._draft_model = None
        self._deferred_clear_at = None
        self._sparse_cache_active = False
        if len(main_cache) == len(self.model.layers):
            self._live_cache = main_cache
            return
        self._live_tokens = None
        self._live_cache = self._make_cache()
