"""Native DFlash draft/verify runtime scaffold."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import time

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
    build_dflash_runtime_no_go_message,
    validate_dflash_runtime_compatibility,
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


# Per-round DFlash telemetry kinds. The harness and any consumer that wires
# ``telemetry_collector`` can switch on these values without ambiguity.
DFLASH_TELEMETRY_KIND_INITIAL_BONUS = "initial_bonus"
DFLASH_TELEMETRY_KIND_TARGET_ONLY = "target_only"
DFLASH_TELEMETRY_KIND_DRAFT_ROUND_ACCEPTED = "draft_round_accepted"
DFLASH_TELEMETRY_KIND_DRAFT_ROUND_PARTIAL = "draft_round_partial"


@dataclass(frozen=True, slots=True)
class DFlashRoundTelemetry:
    """Per-round DFlash scheduling and timing telemetry record.

    Each round that consumes scheduling budget emits exactly one record so
    reports can attribute latency to drafter work, target verification,
    rollback, and emission independently of the surrounding token stream.
    The ``kind`` field classifies the round so callers can recognize the
    four documented round shapes (initial bonus sample, target-only
    ``max_draft_tokens=1``, fully accepted draft round, and partially
    rejected draft round that triggers ``rollback_speculative_cache``).

    The dataclass is intentionally side-effect free so callers can store,
    diff, and aggregate records without needing to replay the generator.
    """

    round_index: int
    kind: str
    scheduled_block_size: int
    draft_count: int
    accepted_count: int
    rejected_count: int
    target_verify_input_length: int
    rollback_occurred: bool
    drafter_elapsed_s: float
    target_verify_elapsed_s: float
    rollback_elapsed_s: float
    emission_elapsed_s: float
    from_draft_token_count: int
    from_target_token_count: int


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


def _emit_dflash_round_telemetry(
    telemetry_collector: Optional[Callable[[DFlashRoundTelemetry], None]],
    record: DFlashRoundTelemetry,
) -> None:
    """Invoke ``telemetry_collector`` if it is supplied.

    The collector is purely opt-in. When ``telemetry_collector`` is
    ``None`` the runtime records no overhead beyond the dataclass
    construction itself, preserving the existing default-off behavior
    and the M15 invariant that no scheduling decision depends on
    telemetry.
    """

    if telemetry_collector is not None:
        telemetry_collector(record)


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


def _align_gdn_states_with_prompt_cache(
    prompt_cache: list[Any],
    gdn_states: Optional[Sequence[Any]],
    lm: Any,
) -> Optional[list[Any]]:
    """Return a ``gdn_states`` list aligned 1:1 with ``prompt_cache``.

    The mlx-vlm ``target_verify`` path populates ``gdn_states`` with one
    entry per GDN (linear / gated-delta) layer in iteration order. For
    Qwen3.5/Qwen3.6 the prompt-cache list mixes ``KVCache`` and
    ``ArraysCache`` layers in layer-index order (16 KVCache + 48
    ArraysCache for the proven sequential layout), so the flat
    ``gdn_states`` cannot be zipped directly against ``prompt_cache``.

    This helper walks the patched Qwen3.5 ``lm.layers`` list, identifies
    which layer indices are linear (``is_linear=True``), and rewrites
    ``gdn_states`` so ``aligned[i]`` is the GDN sink tuple captured for
    the linear layer at cache index ``i`` (``None`` for non-linear
    layers). The ``rollback_speculative_cache`` hook can then look up
    the correct per-layer GDN state by cache index.
    """

    if gdn_states is None:
        return None
    layers = getattr(lm, "layers", None)
    if layers is None:
        # No layer info available; pass through the original list so the
        # hook can fall back to its defensive behavior.
        return list(gdn_states)

    aligned: list[Any] = [None] * len(prompt_cache)
    gdn_iter = iter(gdn_states)
    matched = 0
    for layer_index, layer in enumerate(layers):
        if layer_index >= len(aligned):
            break
        if not bool(getattr(layer, "is_linear", False)):
            continue
        try:
            aligned[layer_index] = next(gdn_iter)
            matched += 1
        except StopIteration:
            break

    if matched == 0:
        # No linear layers matched; keep the original list shape so the
        # hook can decide whether to ignore or fail closed.
        return list(gdn_states)
    return aligned


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
    telemetry_collector: Optional[
        Callable[[DFlashRoundTelemetry], None]
    ] = None,
) -> Iterator[GenerationResult]:
    """Stream DFlash draft/verify generation for sequential text.

    ``telemetry_collector`` (optional) is invoked once per scheduling
    round with a ``DFlashRoundTelemetry`` record describing the round's
    scheduled block size, draft/accepted/rejected counts, target verify
    input length, rollback occurrence, and per-stage timings (drafter,
    target verify, rollback, emission). The collector is purely
    opt-in: when ``None`` the runtime records no overhead beyond the
    timing reads, so default-off behavior and scheduling decisions
    remain unchanged.
    """

    if isinstance(model_kit, DistributedModelKit):
        raise DFlashUnavailableError(
            "DFlash is only supported for sequential text generation"
        )

    if prompt_progress_reporter is None:
        prompt_progress_reporter = LoggerReporter()

    runtime_blockers = validate_dflash_runtime_compatibility(model_kit)
    if runtime_blockers:
        raise DFlashUnavailableError(
            build_dflash_runtime_no_go_message(runtime_blockers)
        )

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

    target_model = getattr(model_kit, "model", model_kit)
    lm = target_model.language_model if hasattr(target_model, "language_model") else target_model

    draft_model = (
        dflash_draft_model
        if dflash_draft_model is not None
        else load_dflash_drafter_model(
            target_model,
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
        initial_target_verify_start = time.perf_counter()
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
        initial_target_verify_elapsed_s = time.perf_counter() - initial_target_verify_start
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

        emission_start = time.perf_counter()
        first_result = _emit_token(
            first_bonus_token,
            logprobs[0, -1],
            from_draft=False,
        )
        if first_result is not None:
            yield first_result
        initial_emission_elapsed_s = time.perf_counter() - emission_start
        # Round 0 covers the prompt-processing bonus. The drafter is
        # never invoked here and no rollback is required; the only
        # measurement of consequence is the prompt target-verify call
        # (which scales with prompt length) plus the bonus emission.
        prompt_tokens_length = (
            prompt_tokens_array.shape[-1]
            if hasattr(prompt_tokens_array, "shape")
            else len(prompt_tokens_array)
        )
        _emit_dflash_round_telemetry(
            telemetry_collector,
            DFlashRoundTelemetry(
                round_index=0,
                kind=DFLASH_TELEMETRY_KIND_INITIAL_BONUS,
                scheduled_block_size=1,
                draft_count=0,
                accepted_count=0,
                rejected_count=0,
                target_verify_input_length=int(prompt_tokens_length),
                rollback_occurred=False,
                drafter_elapsed_s=0.0,
                target_verify_elapsed_s=initial_target_verify_elapsed_s,
                rollback_elapsed_s=0.0,
                emission_elapsed_s=initial_emission_elapsed_s,
                from_draft_token_count=0,
                from_target_token_count=1,
            ),
        )
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
        emitted = len(emitted_history)
        round_index = 0
        while emitted < max_tokens:
            round_index += 1
            block_total = _dflash_block_total(draft_model, dflash_options.max_draft_tokens)
            bs = min(block_total, dflash_options.max_draft_tokens, max_tokens - emitted + 1)
            if bs < 1:
                break

            if bs == 1:
                # Target-only round: ``max_draft_tokens=1`` (or any
                # configuration whose per-round block budget collapses
                # to a single bonus verify position). The drafter is
                # bypassed this round; the runtime still calls the
                # target with ``target_verify=True`` so the next bonus
                # is sampled from target logits (no unverified drafter
                # tokens are ever emitted). The cache advances by one
                # entry per round (``[last_bonus]`` is appended by the
                # verify call). This branch is what restores multi-token
                # generation for the conservative
                # ``--dflash-max-draft-tokens 1`` quality-gate retry:
                # the prior ``if bs <= 1: break`` terminated the loop
                # after the first token, which caused
                # ``completion_tokens=1`` and ``finish_reason=null`` on
                # every prompt.
                verify_input = mx.array(
                    [[emitted_history[-1]]], dtype=mx.int32
                )
                target_only_verify_start = time.perf_counter()
                with mx.stream(generation_stream):
                    verify_out = target_model(
                        verify_input,
                        cache=prompt_cache,
                        capture_layer_ids=target_layer_ids,
                        hidden_sink=[],
                        gdn_sink=[],
                        target_verify=True,
                    )
                target_only_verify_elapsed_s = (
                    time.perf_counter() - target_only_verify_start
                )
                logprobs = verify_out.logits - mx.logsumexp(
                    verify_out.logits, axis=-1, keepdims=True
                )
                target_token = sampler(logprobs[:, -1, :])
                mx.async_eval(target_token, logprobs)
                new_bonus = int(target_token.item())
                _record_speculative_round(draft_model, 0, 0)
                emitted_history.append(new_bonus)
                emitted += 1
                emission_start = time.perf_counter()
                maybe_result = _emit_token(
                    new_bonus,
                    logprobs[0, -1],
                    from_draft=False,
                )
                target_only_emission_elapsed_s = time.perf_counter() - emission_start
                if maybe_result is not None:
                    yield maybe_result
                _emit_dflash_round_telemetry(
                    telemetry_collector,
                    DFlashRoundTelemetry(
                        round_index=round_index,
                        kind=DFLASH_TELEMETRY_KIND_TARGET_ONLY,
                        scheduled_block_size=1,
                        draft_count=0,
                        accepted_count=0,
                        rejected_count=0,
                        target_verify_input_length=1,
                        rollback_occurred=False,
                        drafter_elapsed_s=0.0,
                        target_verify_elapsed_s=target_only_verify_elapsed_s,
                        rollback_elapsed_s=0.0,
                        emission_elapsed_s=target_only_emission_elapsed_s,
                        from_draft_token_count=0,
                        from_target_token_count=1,
                    ),
                )
                if emitted % 256 == 0:
                    mx.clear_cache()
                continue

            drafter_start = time.perf_counter()
            draft_tokens = draft_model.draft_block(
                emitted_history[-1],
                hidden,
                draft_cache,
                bs,
                sampler,
                mx.int32,
            )
            drafter_elapsed_s = time.perf_counter() - drafter_start
            if proposal_observer is not None:
                proposal_observer(
                    tuple(emitted_history), tuple(draft_tokens.reshape(-1).tolist())
                )
            mx.async_eval(draft_tokens)

            draft_round_verify_start = time.perf_counter()
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
            draft_round_verify_elapsed_s = (
                time.perf_counter() - draft_round_verify_start
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

            emission_start = time.perf_counter()
            from_draft_count = 0
            from_target_count = 0
            for index, token in enumerate(new_tokens):
                from_draft = index < accepted
                if from_draft:
                    from_draft_count += 1
                else:
                    from_target_count += 1
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
            draft_round_emission_elapsed_s = time.perf_counter() - emission_start

            rollback_occurred = accepted < bs - 1
            rollback_elapsed_s = 0.0
            if rollback_occurred:
                rollback_start = time.perf_counter()
                rollback = getattr(lm, "rollback_speculative_cache", None)
                if rollback is None:
                    raise DFlashUnavailableError(
                        f"{type(lm).__name__} does not implement rollback_speculative_cache"
                    )
                aligned_gdn_states = _align_gdn_states_with_prompt_cache(
                    prompt_cache, verify_out.gdn_states, lm
                )
                with mx.stream(generation_stream):
                    rollback(prompt_cache, aligned_gdn_states, accepted, bs)
                rollback_elapsed_s = time.perf_counter() - rollback_start

            kind = (
                DFLASH_TELEMETRY_KIND_DRAFT_ROUND_ACCEPTED
                if accepted >= bs - 1
                else DFLASH_TELEMETRY_KIND_DRAFT_ROUND_PARTIAL
            )
            _emit_dflash_round_telemetry(
                telemetry_collector,
                DFlashRoundTelemetry(
                    round_index=round_index,
                    kind=kind,
                    scheduled_block_size=bs,
                    draft_count=bs - 1,
                    accepted_count=accepted,
                    rejected_count=max(0, (bs - 1) - accepted),
                    target_verify_input_length=int(bs),
                    rollback_occurred=rollback_occurred,
                    drafter_elapsed_s=drafter_elapsed_s,
                    target_verify_elapsed_s=draft_round_verify_elapsed_s,
                    rollback_elapsed_s=rollback_elapsed_s,
                    emission_elapsed_s=draft_round_emission_elapsed_s,
                    from_draft_token_count=from_draft_count,
                    from_target_token_count=from_target_count,
                ),
            )

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
