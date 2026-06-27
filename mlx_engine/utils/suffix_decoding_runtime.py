from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from functools import partial
import logging
import os
import time
from typing import Any, Optional, Union

import mlx.core as mx
from mlx import nn
from mlx_lm.generate import (
    GenerationResponse,
    generation_stream,
    maybe_quantize_kv_cache,
    wired_limit,
)
from mlx_lm.models import cache as mlx_cache
from mlx_lm.tokenizer_utils import TokenizerWrapper

from mlx_engine.model_kit.distributed_model_kit import DistributedModelKit
from mlx_engine.utils.suffix_decoding import (
    SuffixDecodingProposal,
    propose_suffix_decoding_tokens,
)

logger = logging.getLogger(__name__)

SUFFIX_DECODING_ENV = "MLX_ENGINE_SUFFIX_DECODING"
SUFFIX_DECODING_MAX_DRAFT_TOKENS_ENV = "MLX_ENGINE_SUFFIX_DECODING_MAX_DRAFT_TOKENS"
DEFAULT_SUFFIX_DECODING_MAX_DRAFT_TOKENS = 4


@dataclass(frozen=True, slots=True)
class SuffixDecodingOptions:
    enabled: bool
    max_draft_tokens: int = DEFAULT_SUFFIX_DECODING_MAX_DRAFT_TOKENS


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "")
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def resolve_suffix_decoding_options(
    suffix_decoding_toggle: bool | None,
    suffix_decoding_max_draft_tokens: int | None,
) -> SuffixDecodingOptions:
    """Resolve the opt-in state for suffix decoding."""

    enabled = (
        _env_flag(SUFFIX_DECODING_ENV)
        if suffix_decoding_toggle is None
        else suffix_decoding_toggle
    )
    if not enabled:
        return SuffixDecodingOptions(enabled=False)

    max_draft_tokens = (
        _env_int(
            SUFFIX_DECODING_MAX_DRAFT_TOKENS_ENV,
            DEFAULT_SUFFIX_DECODING_MAX_DRAFT_TOKENS,
        )
        if suffix_decoding_max_draft_tokens is None
        else suffix_decoding_max_draft_tokens
    )
    if max_draft_tokens < 1:
        raise ValueError("suffix_decoding_max_draft_tokens must be positive")
    return SuffixDecodingOptions(enabled=True, max_draft_tokens=max_draft_tokens)


def validate_suffix_decoding_compatibility(
    *,
    suffix_decoding_enabled: bool,
    model_kit: Any,
    images_b64: Optional[list[str]],
    speculative_decoding_toggle: Optional[bool],
    num_draft_tokens: Optional[int],
    specprefill_toggle: Optional[bool],
) -> None:
    """Fail closed for surfaces that are not in the first slice."""

    if not suffix_decoding_enabled:
        return

    if images_b64 is not None and len(images_b64) > 0:
        raise ValueError("SuffixDecoding is only supported for sequential text generation")

    if specprefill_toggle is True:
        raise ValueError("SuffixDecoding cannot be combined with SpecPrefill yet")

    if speculative_decoding_toggle is True or num_draft_tokens is not None:
        raise ValueError(
            "SuffixDecoding cannot be combined with loaded draft-model speculation yet"
        )

    if getattr(model_kit, "draft_model", None) is not None:
        raise ValueError(
            "SuffixDecoding cannot be combined with a loaded draft model yet"
        )

    if isinstance(model_kit, DistributedModelKit):
        raise ValueError("SuffixDecoding is only supported for sequential text generation")

    if hasattr(model_kit, "uses_distributed_batching") and (
        model_kit.uses_distributed_batching()
    ):
        raise ValueError("SuffixDecoding is only supported for sequential text generation")


def suffix_stream_generate(
    model: nn.Module,
    tokenizer: Union[Any, TokenizerWrapper],
    prompt: Union[str, mx.array, list[int]],
    *,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[list[Callable[[mx.array, mx.array], mx.array]]] = None,
    prompt_cache: Optional[list[Any]] = None,
    prompt_progress_callback: Optional[Callable[[int, int], None]] = None,
    prefill_step_size: int = 512,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    max_draft_tokens: int = DEFAULT_SUFFIX_DECODING_MAX_DRAFT_TOKENS,
    proposal_fn: Callable[
        [Sequence[int]], SuffixDecodingProposal | None
    ] = propose_suffix_decoding_tokens,
) -> Generator[GenerationResponse, None, None]:
    """Stream generation with suffix proposal verification on the target model."""

    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if not isinstance(prompt, mx.array):
        if isinstance(prompt, str):
            add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
                tokenizer.bos_token
            )
            prompt = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        prompt = mx.array(prompt)

    if prompt_cache is None:
        prompt_cache = mlx_cache.make_prompt_cache(model)

    if not mlx_cache.can_trim_prompt_cache(prompt_cache):
        types = {type(c).__name__ for c in prompt_cache if not c.is_trimmable()}
        raise ValueError(
            "SuffixDecoding requires a trimmable prompt cache "
            f"(got {types})."
        )

    detokenizer = tokenizer.detokenizer
    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
    prompt_progress_callback = prompt_progress_callback or (lambda *_: None)
    quantize_cache_fn = partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )
    prompt_tokens = prompt.tolist()
    prev_tokens: mx.array | None = None

    def _process_and_sample(tokens, logits):
        if logits_processors:
            for processor in logits_processors:
                logits = processor(tokens, logits)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        sampled = sampler(logprobs)
        return sampled, logprobs

    def _step(input_tokens: mx.array, n_predict: int = 1):
        nonlocal prev_tokens
        with mx.stream(generation_stream):
            logits = model(input_tokens[None], cache=prompt_cache)
            logits = logits[:, -n_predict:, :]
            quantize_cache_fn(prompt_cache)
            if logits_processors and n_predict > 1:
                if len(input_tokens) > n_predict - 1:
                    prev_tokens = input_tokens[: -(n_predict - 1)]
                out_y, out_logprobs = [], []
                for i in range(n_predict):
                    prev_tokens = (
                        mx.concatenate([prev_tokens, input_tokens[i : i + 1]])
                        if prev_tokens is not None
                        else input_tokens[i : i + 1]
                    )
                    y_i, logprobs_i = _process_and_sample(
                        prev_tokens, logits[:, i, :]
                    )
                    out_y.append(y_i)
                    out_logprobs.append(logprobs_i)
                return mx.concatenate(out_y, axis=0), mx.concatenate(out_logprobs, axis=0)
            if n_predict > 1:
                out_y, out_logprobs = [], []
                for i in range(n_predict):
                    logits_i = logits[:, i, :]
                    logprobs_i = logits_i - mx.logsumexp(logits_i, axis=-1, keepdims=True)
                    out_y.append(sampler(logprobs_i))
                    out_logprobs.append(logprobs_i)
                return mx.concatenate(out_y, axis=0), mx.concatenate(out_logprobs, axis=0)
            if logits_processors:
                prev_tokens = (
                    mx.concatenate([prev_tokens, input_tokens])
                    if prev_tokens is not None
                    else input_tokens
                )
                sampled, logprobs = _process_and_sample(prev_tokens, logits[:, -1, :])
                return sampled, logprobs.squeeze(0)
            logits = logits[:, -1, :]
            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            sampled = sampler(logprobs)
            return sampled, logprobs.squeeze(0)

    def _prefill(y: mx.array):
        while y.size > 1:
            n_to_process = min(prefill_step_size, y.size - 1)
            model(y[:n_to_process][None], cache=prompt_cache)
            quantize_cache_fn(prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            y = y[n_to_process:]
            mx.clear_cache()
        return y

    def _suffix_token_generator():
        total_prompt_tokens = len(prompt)
        prompt_processed_tokens = 0
        prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
        remaining_prompt = prompt
        while total_prompt_tokens - prompt_processed_tokens > 1:
            remaining = (total_prompt_tokens - prompt_processed_tokens) - 1
            n_to_process = min(prefill_step_size, remaining)
            model(remaining_prompt[:n_to_process][None], cache=prompt_cache)
            quantize_cache_fn(prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            prompt_processed_tokens += n_to_process
            prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
            remaining_prompt = remaining_prompt[n_to_process:]
            mx.clear_cache()

        y, logprobs = _step(_prefill(remaining_prompt))
        mx.async_eval(y, logprobs)
        emitted_history = prompt_tokens.copy() + [y.item()]
        n = 0
        prompt_progress_emitted = False
        while True:
            proposal = proposal_fn(
                emitted_history,
                max_draft_tokens=max_draft_tokens,
            )
            num_draft = (
                min(max_tokens - n, len(proposal.draft_tokens), max_draft_tokens)
                if proposal is not None
                else 0
            )
            if num_draft > 0:
                draft_tokens = mx.array(proposal.draft_tokens[:num_draft], mx.uint32)
                tokens, draft_logprobs = _step(
                    mx.concatenate([y, draft_tokens]), num_draft + 1
                )
                mx.eval(tokens, draft_tokens)
                tokens_list = tokens.tolist()
                draft_tokens_list = draft_tokens.tolist()
                accepted = 0
                while accepted < num_draft:
                    tn, dtn, lpn = (
                        tokens_list[accepted],
                        draft_tokens_list[accepted],
                        draft_logprobs[accepted],
                    )
                    if tn != dtn:
                        break
                    emitted_history.append(tn)
                    if not prompt_progress_emitted:
                        prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
                        prompt_progress_emitted = True
                    n += 1
                    yield tn, lpn, True
                    if n == max_tokens:
                        return
                    accepted += 1

                if num_draft - accepted:
                    mlx_cache.trim_prompt_cache(prompt_cache, num_draft - accepted)

                token = tokens_list[accepted]
                logprobs = draft_logprobs[accepted]
                emitted_history.append(token)
                if not prompt_progress_emitted:
                    prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
                    prompt_progress_emitted = True
                n += 1
                yield token, logprobs, False
                if n == max_tokens:
                    return
                y = mx.array([token], mx.uint32)
                continue

            if n != max_tokens:
                next_y, next_logprobs = _step(y)
                mx.async_eval(next_y, next_logprobs)
            if n == max_tokens:
                return
            token = y.item()
            if not prompt_progress_emitted:
                mx.eval(y)
                prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
                prompt_progress_emitted = True
            yield token, logprobs, False
            if n % 256 == 0:
                mx.clear_cache()
            y, logprobs = next_y, next_logprobs
            n += 1

    token_generator = _suffix_token_generator()
    prompt_tps = 0.0
    token = 0
    from_draft = False
    logprobs = mx.array([])
    n = -1
    with wired_limit(model, [generation_stream]):
        tic = time.perf_counter()
        for n, (token, logprobs, from_draft) in enumerate(token_generator):
            if n == 0:
                prompt_time = time.perf_counter() - tic
                prompt_tps = len(prompt) / prompt_time
                tic = time.perf_counter()
            if token in tokenizer.eos_token_ids:
                break

            detokenizer.add_token(token)
            if (n + 1) == max_tokens:
                break

            yield GenerationResponse(
                text=detokenizer.last_segment,
                token=token,
                logprobs=logprobs,
                from_draft=from_draft,
                prompt_tokens=len(prompt),
                prompt_tps=prompt_tps,
                generation_tokens=n + 1,
                generation_tps=(n + 1) / (time.perf_counter() - tic),
                peak_memory=mx.get_peak_memory() / 1e9,
                finish_reason=None,
            )

        detokenizer.finalize()
        yield GenerationResponse(
            text=detokenizer.last_segment,
            token=token,
            logprobs=logprobs,
            from_draft=from_draft,
            prompt_tokens=len(prompt),
            prompt_tps=prompt_tps,
            generation_tokens=n + 1,
            generation_tps=(n + 1) / (time.perf_counter() - tic),
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason="stop" if token in tokenizer.eos_token_ids else "length",
        )
