"""Native DFlash draft/verify runtime scaffold."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
from mlx_vlm.speculative.common import (
    _dflash_block_total,
    _record_speculative_round,
    _speculative_walk,
    generation_stream,
)
from mlx_vlm.speculative.drafters import load_drafter, validate_drafter_compatibility
from mlx_vlm.speculative.drafters.qwen3_dflash.dflash import DFlashDraftModel

from mlx_engine.model_kit.distributed_model_kit import DistributedModelKit
from mlx_engine.model_kit.model_kit import ModelKit
from mlx_engine.utils.dflash_boundary import (
    DFlashBoundaryOptions,
    DFlashUnavailableError,
)
from mlx_engine.utils.generation_helpers import (
    create_stop_string_processor,
    process_stop_string_check,
    setup_repetition_logits_processors,
    should_yield_token,
    validate_top_logprobs,
)
from mlx_engine.utils.generation_result import (
    GenerationResult,
    GenerationStopCondition,
    construct_user_cancelled_result,
)
from mlx_engine.utils.prompt_progress_reporter import (
    LoggerReporter,
    PromptProgressReporter,
    StopPromptProcessing,
)
from mlx_engine.utils.sampling import create_sampler
from mlx_engine.utils.set_seed import set_seed
from mlx_engine.utils.token import Token
from mlx_engine.utils.top_logprobs import summarize_top_logprobs


def load_dflash_drafter_model(
    target_model: Any,
    dflash_drafter_path: str | Path,
) -> DFlashDraftModel:
    """Load and validate a native DFlash drafter snapshot."""

    draft_model, resolved_kind = load_drafter(str(dflash_drafter_path), kind="dflash")
    validate_drafter_compatibility(target_model, draft_model, resolved_kind)
    if not isinstance(draft_model, DFlashDraftModel):
        raise ValueError(
            "DFlash drafter snapshot did not load as DFlashDraftModel"
        )
    return draft_model


def _apply_logits_processors(
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]] | None,
    tokens: mx.array,
    logits: mx.array,
) -> mx.array:
    if logits_processors:
        for processor in logits_processors:
            logits = processor(tokens, logits)
    return logits


def _copy_prompt_cache(model_kit: ModelKit | DistributedModelKit) -> list[Any]:
    prompt_cache = getattr(getattr(model_kit, "cache_wrapper", None), "cache", None)
    if prompt_cache is None:
        prompt_cache = getattr(model_kit, "prompt_cache", None)
    if prompt_cache is None:
        raise ValueError("DFlash requires a prompt cache")
    return prompt_cache


def dflash_stream_generate(
    model_kit: ModelKit | DistributedModelKit,
    prompt_tokens: list[int],
    *,
    prompt_progress_reporter: Optional[PromptProgressReporter] = None,
    images_b64: Optional[list[str]] = None,
    max_image_size: Optional[tuple[int, int]] = None,
    stop_strings: Optional[list[str]] = None,
    top_logprobs: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    temp: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    min_tokens_to_keep: Optional[int] = None,
    seed: Optional[int] = None,
    json_schema: Optional[str] = None,
    max_tokens: Optional[int] = 10000000,
    request_id: Optional[str] = None,
    dflash_options: DFlashBoundaryOptions,
    dflash_draft_model: Optional[DFlashDraftModel] = None,
    proposal_observer: Optional[
        Callable[[Sequence[int], Sequence[int]], None]
    ] = None,
) -> Iterator[GenerationResult]:
    """Stream DFlash draft/verify generation for sequential text."""

    if isinstance(model_kit, DistributedModelKit):
        raise DFlashUnavailableError(
            "DFlash is only supported for sequential text generation"
        )

    if prompt_progress_reporter is None:
        prompt_progress_reporter = LoggerReporter()

    set_seed(seed)

    generate_args: dict[str, Any] = {}
    if getattr(model_kit, "max_kv_size", None) is not None:
        generate_args["max_kv_size"] = getattr(model_kit, "max_kv_size", None)
    if getattr(model_kit, "kv_bits", None) is not None:
        generate_args["kv_bits"] = getattr(model_kit, "kv_bits", None)
    if getattr(model_kit, "kv_group_size", None) is not None:
        generate_args["kv_group_size"] = getattr(model_kit, "kv_group_size", None)
    if getattr(model_kit, "quantized_kv_start", None) is not None:
        generate_args["quantized_kv_start"] = getattr(
            model_kit, "quantized_kv_start", None
        )

    try:
        prompt_tokens_array, _ = model_kit.process_prompt(
            prompt_tokens,
            images_b64,
            prompt_progress_reporter,
            generate_args,
            max_image_size,
            speculative_decoding_toggle=None,
            draft_model_override=None,
            specprefill_options=None,
        )
    except StopPromptProcessing:
        yield construct_user_cancelled_result()
        return

    prompt_cache = generate_args.get("prompt_cache")
    if prompt_cache is None:
        prompt_cache = _copy_prompt_cache(model_kit)

    draft_model = (
        dflash_draft_model
        if dflash_draft_model is not None
        else load_dflash_drafter_model(
            getattr(model_kit, "model", model_kit),
            dflash_options.drafter_model_path,
        )
    )
    target_layer_ids = list(draft_model.config.target_layer_ids)
    draft_cache = draft_model.reset(getattr(model_kit, "model", model_kit))

    tokenizer = model_kit.tokenizer
    input_tokens_list = (
        prompt_tokens_array.tolist()
        if hasattr(prompt_tokens_array, "tolist")
        else list(prompt_tokens_array)
    )
    logits_processors = setup_repetition_logits_processors(
        repetition_penalty,
        repetition_context_size,
        prompt_tokens,
        input_tokens_list,
    )
    sampler = create_sampler(temp, top_p, min_p, min_tokens_to_keep, top_k)
    top_logprobs = validate_top_logprobs(top_logprobs)
    stop_string_processor = create_stop_string_processor(stop_strings, tokenizer)

    token_buffer: list[Token] = []
    top_logprobs_buffer: list[list[Token]] = []
    text = ""
    emitted_history: list[int] = []

    def _emit_token(token: int, logprobs: mx.array, from_draft: bool) -> GenerationResult | None:
        nonlocal text, token_buffer, top_logprobs_buffer
        text += tokenizer.decode(token)
        if getattr(model_kit, "is_cross_prompt_cache_active", lambda: False)():
            getattr(model_kit, "record_token_to_cache")(token)
        token_buffer.append(
            Token(
                token,
                tokenizer.decode(token),
                float(logprobs[token]),
                from_draft=from_draft,
            )
        )
        if top_logprobs:
            top_logprobs_buffer.append(
                summarize_top_logprobs(tokenizer, logprobs, top_logprobs)
            )

        should_stop, should_buffer, stop_result = process_stop_string_check(
            stop_string_processor, token
        )
        if should_stop:
            return GenerationResult(
                text=text,
                tokens=token_buffer,
                stop_condition=GenerationStopCondition(
                    stop_reason="stop_string",
                    stop_string=stop_result.stop_string,
                    stop_tokens=stop_result.stop_tokens,
                ),
                top_logprobs=top_logprobs_buffer,
            )
        if should_buffer:
            return None

        should_yield, stop_condition = should_yield_token(text, token, tokenizer)
        if (
            stop_condition is None
            and token_buffer
            and len(token_buffer) >= max_tokens
        ):
            should_yield = True
            stop_condition = GenerationStopCondition(
                stop_reason="token_limit",
                stop_string="",
                stop_tokens=[],
            )
        if should_yield:
            result = GenerationResult(
                text=text,
                tokens=token_buffer,
                stop_condition=stop_condition,
                top_logprobs=top_logprobs_buffer,
            )
            text = ""
            token_buffer = []
            top_logprobs_buffer = []
            return result
        return None

    try:
        with mx.stream(generation_stream):
            verify_out = getattr(model_kit, "model", model_kit)(
                prompt_tokens_array[None]
                if getattr(prompt_tokens_array, "ndim", 1) == 1
                else prompt_tokens_array,
                cache=prompt_cache,
                capture_layer_ids=target_layer_ids,
                hidden_sink=[],
                gdn_sink=[],
                target_verify=True,
            )
        logits = _apply_logits_processors(
            logits_processors,
            prompt_tokens_array,
            verify_out.logits,
        )
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        first_bonus = sampler(logprobs[:, -1, :])
        mx.async_eval(first_bonus, logprobs)

        first_bonus_token = int(first_bonus.item())
        emitted_history.append(first_bonus_token)

        first_result = _emit_token(
            first_bonus_token,
            logprobs[0, -1],
            from_draft=False,
        )
        if first_result is not None:
            yield first_result
        if len(token_buffer) == 0 and text == "":
            pass
        if len(emitted_history) >= max_tokens:
            if token_buffer:
                yield GenerationResult(
                    text=text,
                    tokens=token_buffer,
                    stop_condition=GenerationStopCondition(
                        stop_reason="token_limit",
                        stop_string="",
                        stop_tokens=[],
                    ),
                    top_logprobs=top_logprobs_buffer,
                )
            return

        hidden = mx.concatenate(verify_out.hidden_states, axis=-1)
        target_model = getattr(model_kit, "model", model_kit)
        lm = target_model.language_model if hasattr(target_model, "language_model") else target_model
        if not hasattr(lm, "rollback_speculative_cache"):
            raise DFlashUnavailableError(
                f"{type(lm).__name__} does not implement rollback_speculative_cache"
            )

        emitted = len(emitted_history)
        while emitted < max_tokens:
            block_total = _dflash_block_total(draft_model, dflash_options.max_draft_tokens)
            bs = min(block_total, dflash_options.max_draft_tokens, max_tokens - emitted + 1)
            if bs <= 1:
                break

            draft_tokens = draft_model.draft_block(
                emitted_history[-1],
                hidden,
                draft_cache,
                bs,
                sampler,
                mx.int32,
            )
            if proposal_observer is not None:
                proposal_observer(
                    tuple(emitted_history), tuple(draft_tokens.reshape(-1).tolist())
                )
            mx.async_eval(draft_tokens)

            with mx.stream(generation_stream):
                verify_input = mx.concatenate(
                    [mx.array([[emitted_history[-1]]], dtype=mx.int32), draft_tokens],
                    axis=1,
                )
                verify_out = target_model(
                    verify_input,
                    cache=prompt_cache,
                    capture_layer_ids=target_layer_ids,
                    hidden_sink=[],
                    gdn_sink=[],
                    target_verify=True,
                )
            logits = _apply_logits_processors(
                logits_processors,
                verify_input,
                verify_out.logits,
            )
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            target_tokens = sampler(logprobs)
            mx.async_eval(target_tokens, logprobs)

            accepted, new_tokens = _speculative_walk(
                draft_tokens,
                target_tokens,
                max_tokens - emitted,
            )
            _record_speculative_round(draft_model, accepted, bs - 1)

            hidden = mx.concatenate(verify_out.hidden_states, axis=-1)
            hidden = hidden[:, : len(new_tokens), :]

            for index, token in enumerate(new_tokens):
                from_draft = index < accepted
                maybe_result = _emit_token(
                    token,
                    logprobs[0, index],
                    from_draft=from_draft,
                )
                emitted_history.append(token)
                emitted += 1
                if maybe_result is not None:
                    yield maybe_result
                if emitted >= max_tokens:
                    break

            if accepted < bs - 1:
                rollback = getattr(lm, "rollback_speculative_cache", None)
                if rollback is None:
                    raise DFlashUnavailableError(
                        f"{type(lm).__name__} does not implement rollback_speculative_cache"
                    )
                with mx.stream(generation_stream):
                    rollback(prompt_cache, verify_out.gdn_states, accepted, bs)

            if emitted % 256 == 0:
                mx.clear_cache()

        if token_buffer:
            yield GenerationResult(
                text=text,
                tokens=token_buffer,
                stop_condition=GenerationStopCondition(
                    stop_reason="token_limit",
                    stop_string="",
                    stop_tokens=[],
                ),
                top_logprobs=top_logprobs_buffer,
            )
        return
    except Exception:
        raise
