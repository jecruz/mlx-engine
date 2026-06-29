"""Native DFlash draft/verify runtime scaffold."""

from __future__ import annotations

from collections import deque
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
# M15 low-acceptance fallback telemetry kinds. These round shapes are
# emitted ONCE each, at the round the fallback detector flips the
# scheduler into fallback mode. Subsequent rounds continue with the
# existing ``target_only`` kind because they share the same
# target-verified emission path; the harness aggregates the fallback
# reason on the per-row metadata block via the fallback state
# snapshot. The fallback kinds are intentionally distinct from
# ``fallback_unsupported_surface`` and ``fallback_preflight`` (which
# describe DFlash opt-in errors) so the harness ``fallback_status``
# field can stay diagnostic without competing with the new lane.
DFLASH_TELEMETRY_KIND_FALLBACK_LOW_ACCEPTANCE = "fallback_low_acceptance"
DFLASH_TELEMETRY_KIND_FALLBACK_PATHOLOGICAL_TARGET_ONLY = (
    "fallback_pathological_target_only"
)


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


# Default tuning for ``DFlashAdaptiveScheduler``. The scheduler starts
# conservative, grows only after a full-acceptance window, and shrinks
# after any rejection so DFlash never pays draft overhead the target
# cannot recover. The values are documented in the per-round telemetry
# profile so future workers can tune them against captured evidence.
DEFAULT_DFLASH_SCHEDULER_INITIAL_BLOCK_SIZE = 2
DEFAULT_DFLASH_SCHEDULER_HISTORY_WINDOW = 4
DEFAULT_DFLASH_SCHEDULER_GROW_THRESHOLD = 1.0
DEFAULT_DFLASH_SCHEDULER_SHRINK_THRESHOLD = 1.0

# Default tuning for the M15 low-acceptance / pathological target-only
# fallback detector. The detector owns a *separate* state from the
# adaptive scheduler: the scheduler tracks recent acceptance ratios to
# grow / shrink the next block size, while the fallback detector
# tracks (a) whether the recent window of draft rounds has shipped a
# mean acceptance ratio below ``LOW_ACCEPTANCE_THRESHOLD`` (covering
# cases such as ``max_draft_tokens=4`` with frequent full rejections),
# and (b) whether the scheduler has been collapsed to ``bs == 1``
# (target-only) for ``PATHO_TARGET_ONLY_ROUNDS`` consecutive rounds
# (covering the conservative ``--dflash-max-draft-tokens 1`` lane and
# any residual-budget case where DFlash pays target verification per
# emitted token without ever shipping a draft). The detector emits a
# fallback telemetry record the first round it engages, then stays
# engaged for the rest of the generation; subsequent rounds continue
# to record ``target_only`` telemetry so the harness can attribute
# per-round cost as usual.
DEFAULT_DFLASH_FALLBACK_LOW_ACCEPTANCE_WINDOW = 4
DEFAULT_DFLASH_FALLBACK_LOW_ACCEPTANCE_THRESHOLD = 0.5
DEFAULT_DFLASH_FALLBACK_LOW_ACCEPTANCE_MIN_DRAFTS = 4
DEFAULT_DFLASH_FALLBACK_PATHO_TARGET_ONLY_ROUNDS = 4

# Fallback reason labels. Stable strings so the harness can switch on
# them when wiring the ``fallback_reason`` field on the per-row
# metadata block. ``DFLASH_FALLBACK_REASON_NONE`` is the sentinel for
# "fallback detector is engaged but the round being recorded is not
# the trigger round".
DFLASH_FALLBACK_REASON_NONE = None
DFLASH_FALLBACK_REASON_LOW_ACCEPTANCE = "low_acceptance"
DFLASH_FALLBACK_REASON_PATHOLOGICAL_TARGET_ONLY = "pathological_target_only"


@dataclass(frozen=True, slots=True)
class DFlashSchedulerDecision:
    """Snapshot of the scheduler state for a single ``next_block_size`` call.

    The dataclass is purely descriptive: it lets tests and telemetry
    consumers verify which bound clipped the requested size, what the
    scheduler believed the next size should be before any cap, and
    which acceptance ratio (if any) drove the decision. The runtime
    itself only consumes ``scheduled_block_size``; the other fields are
    exported so focused tests can prove the scheduler honored all
    three documented caps (DFlash block size, configured
    ``max_draft_tokens``, and remaining token budget) on every call.
    """

    scheduled_block_size: int
    desired_block_size: int
    effective_cap: int
    clip_reason: str
    current_block_size: int
    history_size: int


@dataclass(frozen=True, slots=True)
class DFlashFallbackDecision:
    """Snapshot of the fallback detector state for a single round.

    The fallback detector is purely opt-in: the runtime only
    constructs / advances it when ``dflash_options.adaptive_scheduling``
    is ``True`` (i.e. when the operator explicitly opts in to the
    adaptive scheduler lane that owns the fallback detector). When
    the detector is engaged, the runtime continues with the proven
    ``bs == 1`` target-only path for the remainder of the generation
    and surfaces the trigger reason in the per-row telemetry block.

    The dataclass is purely descriptive. The runtime never reads it
    during generation; it is exported so focused tests can prove the
    detector honors the documented thresholds on every call without
    re-running the real scheduling loop. ``fallback_engaged`` is the
    simple gate; the harness uses it to populate
    ``fallback_status = "fallback_low_acceptance"`` /
    ``"fallback_pathological_target_only"`` once it flips ``True``.
    """

    fallback_engaged: bool
    fallback_reason: Optional[str]
    low_acceptance_active: bool
    low_acceptance_window_remaining: int
    low_acceptance_threshold: float
    low_acceptance_min_drafts: int
    low_acceptance_observed_mean: Optional[float]
    pathological_target_only_active: bool
    pathological_target_only_streak: int
    pathological_target_only_threshold: int


class DFlashAdaptiveScheduler:
    """Adaptive block-size scheduler for the sequential DFlash loop.

    The scheduler chooses the next per-round block size from a bounded
    history of recent acceptance outcomes. It is purely opt-in: the
    runtime only constructs a scheduler when
    ``dflash_options.adaptive_scheduling`` is ``True``, and the
    fixed-size path remains the default-off behavior.

    Policy (deterministic, side-effect free):

    * Start conservative (``initial_block_size``, clamped to the
      configured ``max_draft_tokens`` and at least ``1``). The default
      initial size is ``2`` so the very first round pays a small
      drafter cost and the runtime never races ahead of the proven
      M14 acceptance profile.
    * Grow by exactly one slot when the **most recent** round met
      the ``grow_threshold`` (default ``1.0`` = full acceptance:
      ``accepted == scheduled - 1``). Growth is capped by the lesser
      of ``max_draft_tokens``, the DFlash model ``block_size``, and
      the remaining token budget.
    * Shrink by exactly one slot when the **most recent** round fell
      below the ``shrink_threshold`` (default ``1.0`` = anything
      less than full acceptance, i.e. any rejection). Shrinkage
      floors at ``1`` so the scheduler never drops into a
      zero-drafter regime; the existing ``bs == 1`` target-only
      branch in ``dflash_stream_generate`` still emits every token
      through ``target_verify=True``.
    * Hold the current size when the acceptance ratio sits between
      the two thresholds (only reachable when both thresholds are
      relaxed from the default ``1.0``).
    * Target-only rounds (``scheduled_block_size == 1``) do not
      influence the scheduler's history because they carry no drafts
      to evaluate. The scheduler keeps the most recent draft-round
      decision in force.
    * All three documented caps (``max_draft_tokens``, the DFlash
      model block size returned by ``_dflash_block_total``, and the
      remaining token budget) are applied **after** the grow/shrink
      decision inside ``next_block_size`` so the scheduler can never
      silently widen the runtime surface. The runtime always
      receives the capped size and records the clip reason for
      downstream observability.

    The scheduler does not consult the per-round telemetry collector
    directly. Telemetry remains a pure observability surface; the
    scheduler's state is internal so its decisions are reproducible
    from the sequence of recorded round outcomes alone.
    """

    def __init__(
        self,
        *,
        max_draft_tokens: int,
        initial_block_size: int = DEFAULT_DFLASH_SCHEDULER_INITIAL_BLOCK_SIZE,
        history_window: int = DEFAULT_DFLASH_SCHEDULER_HISTORY_WINDOW,
        grow_threshold: float = DEFAULT_DFLASH_SCHEDULER_GROW_THRESHOLD,
        shrink_threshold: float = DEFAULT_DFLASH_SCHEDULER_SHRINK_THRESHOLD,
        low_acceptance_window: int = (
            DEFAULT_DFLASH_FALLBACK_LOW_ACCEPTANCE_WINDOW
        ),
        low_acceptance_threshold: float = (
            DEFAULT_DFLASH_FALLBACK_LOW_ACCEPTANCE_THRESHOLD
        ),
        low_acceptance_min_drafts: int = (
            DEFAULT_DFLASH_FALLBACK_LOW_ACCEPTANCE_MIN_DRAFTS
        ),
        pathological_target_only_rounds: int = (
            DEFAULT_DFLASH_FALLBACK_PATHO_TARGET_ONLY_ROUNDS
        ),
    ) -> None:
        if max_draft_tokens < 1:
            raise ValueError("max_draft_tokens must be >= 1")
        if initial_block_size < 1:
            raise ValueError("initial_block_size must be >= 1")
        if history_window < 1:
            raise ValueError("history_window must be >= 1")
        if not 0.0 <= shrink_threshold <= 1.0:
            raise ValueError("shrink_threshold must lie in [0, 1]")
        if not 0.0 <= grow_threshold <= 1.0:
            raise ValueError("grow_threshold must lie in [0, 1]")
        if grow_threshold < shrink_threshold:
            raise ValueError("grow_threshold must be >= shrink_threshold")
        if low_acceptance_window < 1:
            raise ValueError("low_acceptance_window must be >= 1")
        if not 0.0 <= low_acceptance_threshold <= 1.0:
            raise ValueError("low_acceptance_threshold must lie in [0, 1]")
        if low_acceptance_min_drafts < 1:
            raise ValueError("low_acceptance_min_drafts must be >= 1")
        if pathological_target_only_rounds < 1:
            raise ValueError("pathological_target_only_rounds must be >= 1")

        self._max_draft_tokens = int(max_draft_tokens)
        self._initial_block_size = min(
            int(initial_block_size), self._max_draft_tokens
        )
        self._history_window = int(history_window)
        self._grow_threshold = float(grow_threshold)
        self._shrink_threshold = float(shrink_threshold)
        self._current_block_size = max(1, self._initial_block_size)
        self._history: deque[tuple[int, int]] = deque(maxlen=self._history_window)

        # Fallback detector state. The fallback is only consulted by
        # the runtime when ``adaptive_scheduling`` is True; the
        # adaptive-off path keeps default-off behavior unchanged.
        # ``_fallback_engaged`` is sticky: once True it never flips
        # back to False so the rest of the generation is guaranteed
        # to stay on the proven ``bs == 1`` target-only path
        # (target_verify=True, no drafter, no rollback).
        self._low_acceptance_window = int(low_acceptance_window)
        self._low_acceptance_threshold = float(low_acceptance_threshold)
        self._low_acceptance_min_drafts = int(low_acceptance_min_drafts)
        self._pathological_target_only_rounds = int(
            pathological_target_only_rounds
        )
        self._fallback_engaged = False
        self._fallback_reason: Optional[str] = DFLASH_FALLBACK_REASON_NONE
        self._recent_drafts: deque[tuple[int, int]] = deque(
            maxlen=self._low_acceptance_window
        )
        self._pathological_target_only_streak = 0

    @property
    def _recent_drafts_count(self) -> int:
        """Total draft-token count across the recent-drafts window.

        The fallback detector requires ``low_acceptance_min_drafts``
        actual drafts (not just rounds) before the mean-acceptance
        threshold can fire; this helper sums ``scheduled - 1`` across
        the configured window so the threshold only trips when the
        recent rounds have actually proposed enough tokens for the
        mean ratio to be statistically meaningful.
        """

        return sum(
            max(0, scheduled - 1) for _accepted, scheduled in self._recent_drafts
        )

    @property
    def current_block_size(self) -> int:
        return self._current_block_size

    @property
    def history(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._history)

    @property
    def history_window(self) -> int:
        return self._history_window

    @property
    def fallback_engaged(self) -> bool:
        return self._fallback_engaged

    @property
    def fallback_reason(self) -> Optional[str]:
        return self._fallback_reason

    def fallback_state(self) -> DFlashFallbackDecision:
        """Return a snapshot of the current fallback-detector state.

        The runtime queries this on every round so it can flip its
        record kind to ``fallback_low_acceptance`` /
        ``fallback_pathological_target_only`` exactly once, then keep
        using the existing ``target_only`` kind on subsequent rounds.
        The harness also reads the snapshot at row close to populate
        the ``fallback_reason`` field on the per-row metadata block.
        """

        observed_mean: Optional[float] = None
        if self._recent_drafts:
            total_drafts = sum(
                max(0, scheduled - 1) for _accepted, scheduled in self._recent_drafts
            )
            total_accepted = sum(
                min(accepted, max(0, scheduled - 1))
                for accepted, scheduled in self._recent_drafts
            )
            if total_drafts > 0:
                observed_mean = total_accepted / total_drafts

        low_acceptance_active = (
            len(self._recent_drafts) >= self._low_acceptance_window
            and self._recent_drafts_count >= self._low_acceptance_min_drafts
            and observed_mean is not None
            and observed_mean < self._low_acceptance_threshold
        )
        pathological_target_only_active = (
            self._pathological_target_only_streak
            >= self._pathological_target_only_rounds
        )

        return DFlashFallbackDecision(
            fallback_engaged=self._fallback_engaged,
            fallback_reason=self._fallback_reason,
            low_acceptance_active=low_acceptance_active,
            low_acceptance_window_remaining=max(
                0, self._low_acceptance_window - len(self._recent_drafts)
            ),
            low_acceptance_threshold=self._low_acceptance_threshold,
            low_acceptance_min_drafts=self._low_acceptance_min_drafts,
            low_acceptance_observed_mean=observed_mean,
            pathological_target_only_active=pathological_target_only_active,
            pathological_target_only_streak=self._pathological_target_only_streak,
            pathological_target_only_threshold=self._pathological_target_only_rounds,
        )

    def record_round(
        self,
        *,
        accepted_count: int,
        scheduled_block_size: int,
    ) -> None:
        """Record the outcome of a draft round.

        Target-only rounds (``scheduled_block_size <= 1``) are ignored
        by the adaptive scheduler's grow/shrink history because they
        carry no drafts to evaluate. The fallback detector instead
        counts them toward the pathological-target-only streak, so
        ``max_draft_tokens=1`` (or any other configuration that
        collapses every round to ``bs == 1``) trips the fallback
        after ``pathological_target_only_rounds`` consecutive rounds.
        Draft rounds (regardless of acceptance) are appended to the
        fallback detector's recent-drafts window so the
        low-acceptance threshold can fire after
        ``low_acceptance_min_drafts`` draft tokens with a mean ratio
        below ``low_acceptance_threshold``.
        """

        if accepted_count < 0:
            accepted_count = 0
        if accepted_count > scheduled_block_size:
            accepted_count = scheduled_block_size

        if scheduled_block_size <= 1:
            self._pathological_target_only_streak += 1
        else:
            self._pathological_target_only_streak = 0
            drafts = scheduled_block_size - 1
            accepted_in_range = min(accepted_count, drafts)
            self._recent_drafts.append(
                (int(accepted_in_range), int(scheduled_block_size))
            )

        drafts = max(0, scheduled_block_size - 1)
        if drafts > 0:
            accepted_in_range = min(accepted_count, drafts)
            self._history.append((int(accepted_in_range), int(scheduled_block_size)))

    def next_block_size(
        self,
        *,
        block_total: int,
        remaining_budget: int,
    ) -> DFlashSchedulerDecision:
        """Return the capped block size for the next round.

        ``block_total`` is the drafter-side ceiling (from
        ``_dflash_block_total``); ``remaining_budget`` is the
        caller-visible budget (typically ``max_tokens - emitted + 1``).
        Both are clipped together with ``max_draft_tokens`` so the
        scheduler can never widen the runtime surface beyond the
        configured caps.

        Policy: shrink on any rejection (the ``accepted < drafts``
        half of the documented contract), grow on acceptance that
        meets the ``grow_threshold`` (default full acceptance, the
        ``accepted >= grow_threshold * drafts`` half). Between the
        two thresholds the scheduler holds the current size so partial
        acceptance neither grows nor shrinks.
        """

        effective_cap = max(
            1,
            min(
                self._max_draft_tokens,
                max(1, int(block_total)),
                max(1, int(remaining_budget)),
            ),
        )

        desired = self._current_block_size
        clip_reason = "within_caps"

        if self._history:
            last_accepted, last_scheduled = self._history[-1]
            last_drafts = last_scheduled - 1
            if last_drafts > 0:
                last_ratio = last_accepted / last_drafts
                # Grow takes precedence when ``last_ratio`` meets the
                # grow threshold; otherwise shrink when it is below
                # the shrink threshold. Holding is the default when
                # neither branch fires (i.e. partial acceptance that
                # does not meet the strict grow threshold and does not
                # fall below the shrink threshold).
                if (
                    last_ratio >= self._grow_threshold
                    and self._current_block_size < effective_cap
                ):
                    desired = self._current_block_size + 1
                elif (
                    last_ratio < self._shrink_threshold
                    and self._current_block_size > 1
                ):
                    desired = self._current_block_size - 1

        if desired > effective_cap:
            desired = effective_cap
            clip_reason = "effective_cap"
        elif desired < 1:
            desired = 1
            clip_reason = "floor_one"

        self._current_block_size = desired
        return DFlashSchedulerDecision(
            scheduled_block_size=desired,
            desired_block_size=desired,
            effective_cap=effective_cap,
            clip_reason=clip_reason,
            current_block_size=desired,
            history_size=len(self._history),
        )

    def evaluate_fallback(self) -> DFlashFallbackDecision:
        """Return the current fallback-detector state.

        The runtime calls this on every post-bonus round (after
        ``record_round`` has advanced the detectors for the round that
        just completed). The returned ``DFlashFallbackDecision`` is a
        pure snapshot: the runtime copies ``fallback_engaged`` /
        ``fallback_reason`` locally so subsequent detector updates
        cannot mutate the live state mid-round.

        Once the detector flips ``fallback_engaged`` to ``True`` the
        runtime drives the rest of the generation through the proven
        ``bs == 1`` target-only path: ``target_verify=True`` for
        every emitted token, no drafter calls, no rollback calls. The
        adaptive scheduler's grow/shrink state is *not* consulted
        after fallback engages because the per-round block size is
        hard-coded to ``1`` by the runtime fallback path; this
        guarantees the runtime cannot accidentally spend draft budget
        while fallback is engaged.
        """

        decision = self.fallback_state()
        if not self._fallback_engaged:
            if decision.low_acceptance_active:
                self._fallback_engaged = True
                self._fallback_reason = DFLASH_FALLBACK_REASON_LOW_ACCEPTANCE
            elif decision.pathological_target_only_active:
                self._fallback_engaged = True
                self._fallback_reason = (
                    DFLASH_FALLBACK_REASON_PATHOLOGICAL_TARGET_ONLY
                )
        if self._fallback_engaged:
            return DFlashFallbackDecision(
                fallback_engaged=True,
                fallback_reason=self._fallback_reason,
                low_acceptance_active=decision.low_acceptance_active,
                low_acceptance_window_remaining=(
                    decision.low_acceptance_window_remaining
                ),
                low_acceptance_threshold=decision.low_acceptance_threshold,
                low_acceptance_min_drafts=decision.low_acceptance_min_drafts,
                low_acceptance_observed_mean=(
                    decision.low_acceptance_observed_mean
                ),
                pathological_target_only_active=(
                    decision.pathological_target_only_active
                ),
                pathological_target_only_streak=(
                    decision.pathological_target_only_streak
                ),
                pathological_target_only_threshold=(
                    decision.pathological_target_only_threshold
                ),
            )
        return decision


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


@dataclass
class _DFlashFallbackTriggerTracker:
    """Mutable per-call trigger tracker for the low-acceptance fallback.

    The runtime constructs one of these per ``dflash_stream_generate``
    invocation and threads it through the bs==1 branch so each
    fallback reason is emitted exactly once per generation. The
    tracker intentionally does not persist across calls so it cannot
    accidentally let a fallback state leak from one request to the
    next.
    """

    low_acceptance_seen: bool = False
    pathological_target_only_seen: bool = False


def _select_round_kind(
    *,
    fallback_decision: Optional[DFlashFallbackDecision],
    fallback_trigger_tracker: _DFlashFallbackTriggerTracker,
) -> str:
    """Pick the per-round telemetry kind for the bs==1 / fallback path.

    * If the fallback detector is not engaged this round the runtime
      emits the existing ``target_only`` kind (it is the same code
      path as the conservative ``--dflash-max-draft-tokens 1`` lane).
    * If fallback engaged during this round and the reason is
      ``low_acceptance`` and no earlier round has already tagged the
      trigger, emit the one-shot ``fallback_low_acceptance`` kind
      and mark the tracker so subsequent rounds fall back to plain
      ``target_only``. The same pattern applies for
      ``pathological_target_only``.
    * Subsequent rounds after the trigger record continue with
      ``target_only`` because the per-round emission cost is
      identical (drafter / rollback bypassed either way); the harness
      consumes the runtime's ``fallback_engaged`` / ``fallback_reason``
      snapshot at row close to populate the per-row metadata.
    """

    if fallback_decision is None or not fallback_decision.fallback_engaged:
        return DFLASH_TELEMETRY_KIND_TARGET_ONLY
    reason = fallback_decision.fallback_reason
    if (
        reason == DFLASH_FALLBACK_REASON_LOW_ACCEPTANCE
        and not fallback_trigger_tracker.low_acceptance_seen
    ):
        fallback_trigger_tracker.low_acceptance_seen = True
        return DFLASH_TELEMETRY_KIND_FALLBACK_LOW_ACCEPTANCE
    if (
        reason == DFLASH_FALLBACK_REASON_PATHOLOGICAL_TARGET_ONLY
        and not fallback_trigger_tracker.pathological_target_only_seen
    ):
        fallback_trigger_tracker.pathological_target_only_seen = True
        return DFLASH_TELEMETRY_KIND_FALLBACK_PATHOLOGICAL_TARGET_ONLY
    return DFLASH_TELEMETRY_KIND_TARGET_ONLY


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

    When ``dflash_options.adaptive_scheduling`` is ``True`` the
    runtime engages an opt-in ``DFlashAdaptiveScheduler`` that picks
    the next per-round block size from a bounded history of recent
    acceptance outcomes. The scheduler starts conservative (default
    ``initial_block_size=2``), grows only after a fully accepted round
    (capped by ``max_draft_tokens``, the DFlash model ``block_size``,
    and the remaining token budget), and shrinks by exactly one slot
    whenever any draft is rejected. The adaptive scheduler is
    default-off; without the opt-in the proven fixed-size path is
    used. Every target call still carries ``target_verify=True``, every
    emitted token is target-verified, and the existing rollback path
    is unchanged.

    The same adaptive scheduler owns the M15 low-acceptance fallback
    detector. When recent rounds show a sustained acceptance collapse
    (mean acceptance ratio below ``low_acceptance_threshold`` across
    ``low_acceptance_window`` rounds with at least
    ``low_acceptance_min_drafts`` total drafts), OR when the
    scheduler has collapsed to ``bs == 1`` for
    ``pathological_target_only_rounds`` consecutive rounds, the
    detector flips its sticky ``fallback_engaged`` flag and the
    runtime forces ``bs == 1`` for every subsequent round. The
    fallback path emits a one-shot telemetry record of kind
    ``fallback_low_acceptance`` or
    ``fallback_pathological_target_only`` so the harness can
    attribute the trigger reason on the per-row metadata block; all
    later rounds continue with the existing ``target_only`` kind
    because the per-round cost is identical. Once fallback is engaged
    the runtime never invokes the drafter or the rollback hook (the
    bs==1 branch bypasses both); every emitted token is still
    target-verified, ``target_verify=True`` is preserved on every
    call, and the default-off baseline (no opt-in) is unchanged.
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
        # Default-off adaptive scheduler: constructed only when the
        # operator explicitly opts in via ``adaptive_scheduling`` (or
        # the matching ``MLX_ENGINE_DFLASH_ADAPTIVE_SCHEDULING`` env
        # var). When ``None`` the loop falls back to the proven fixed
        # ``bs = min(block_total, max_draft_tokens, remaining_budget)``
        # size, preserving M14 invariants and the default-off behavior.
        adaptive_scheduler: Optional[DFlashAdaptiveScheduler] = None
        if getattr(dflash_options, "adaptive_scheduling", False):
            initial_block_size = (
                2
                if dflash_options.max_draft_tokens >= 2
                else 1
            )
            adaptive_scheduler = DFlashAdaptiveScheduler(
                max_draft_tokens=dflash_options.max_draft_tokens,
                initial_block_size=initial_block_size,
            )
        # M15 fallback state. Holds the *current* (round-stable)
        # snapshot of the fallback detector so the loop can emit the
        # correct telemetry kind for the round it is recording. The
        # snapshot is refreshed at the top of every loop iteration by
        # ``adaptive_scheduler.evaluate_fallback()``, which records
        # the current round and then advances the detector. Once the
        # detector engages, this snapshot stays ``fallback_engaged``
        # and the loop forces ``bs == 1`` for every subsequent round.
        fallback_decision: Optional[DFlashFallbackDecision] = None
        # Per-call tracker for one-shot fallback telemetry kinds. The
        # bs==1 branch uses it to tag the trigger round exactly once
        # per fallback reason; the tracker is purely local to this
        # generator call so it cannot leak fallback state between
        # requests.
        fallback_trigger_tracker = _DFlashFallbackTriggerTracker()
        while emitted < max_tokens:
            round_index += 1
            # Always query the fallback detector *first* so the
            # scheduler's grow/shrink history advances with the most
            # recent round outcome. The fallback decision is sticky:
            # once ``fallback_engaged`` flips True the loop forces
            # ``bs == 1`` and never asks the adaptive scheduler for
            # another block-size choice, so the scheduler's
            # ``current_block_size`` becomes irrelevant for the rest
            # of this generation.
            if adaptive_scheduler is not None:
                fallback_decision = adaptive_scheduler.evaluate_fallback()
            else:
                fallback_decision = None
            if fallback_decision is not None and fallback_decision.fallback_engaged:
                # Force the loop into the proven ``bs == 1`` target-only
                # branch every subsequent round. This guarantees:
                #   * No drafter.draft_block calls (the runtime bs==1
                #     branch bypasses the drafter entirely).
                #   * No rollback calls (the bs==1 branch records
                #     ``_record_speculative_round(draft_model, 0, 0)``
                #     and never triggers ``rollback_speculative_cache``).
                #   * No draft cache mutation: the bs==1 branch never
                #     touches the drafter's per-layer state.
                #   * Every emitted token is target-verified
                #     (``target_verify=True`` is hard-coded on the
                #     bs==1 verify call below).
                bs = 1
            else:
                block_total = _dflash_block_total(
                    draft_model, dflash_options.max_draft_tokens
                )
                remaining_budget = max_tokens - emitted + 1
                if adaptive_scheduler is not None:
                    decision = adaptive_scheduler.next_block_size(
                        block_total=block_total,
                        remaining_budget=remaining_budget,
                    )
                    bs = decision.scheduled_block_size
                else:
                    bs = min(
                        block_total,
                        dflash_options.max_draft_tokens,
                        remaining_budget,
                    )
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
                # When the fallback detector flipped on during this
                # round, the loop's first iteration after ``evaluate_fallback``
                # recorded the detector engaged *before* this verify
                # call ran. Emit a one-shot telemetry record tagged with
                # the fallback kind so the harness can attribute the
                # trigger reason; every subsequent round reuses the
                # existing ``target_only`` shape because the per-round
                # emission cost is identical (the drafter / rollback
                # paths are bypassed in both cases).
                fallback_kind = _select_round_kind(
                    fallback_decision=fallback_decision,
                    fallback_trigger_tracker=fallback_trigger_tracker,
                )
                _emit_dflash_round_telemetry(
                    telemetry_collector,
                    DFlashRoundTelemetry(
                        round_index=round_index,
                        kind=fallback_kind,
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
                # Record this round with the adaptive scheduler so the
                # fallback detector's pathological-target-only streak
                # advances. The scheduler ignores bs==1 rounds for its
                # *grow/shrink* history (so no drafter state is
                # perturbed) but still records them toward the
                # pathological-target-only counter that drives the
                # fallback trigger.
                if adaptive_scheduler is not None:
                    adaptive_scheduler.record_round(
                        accepted_count=0,
                        scheduled_block_size=bs,
                    )
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

            if adaptive_scheduler is not None:
                # Feed the round outcome back into the scheduler so
                # the next ``next_block_size`` call can grow (after
                # full acceptance) or shrink (after any rejection).
                # Target-only rounds (``bs == 1``) bypass the
                # scheduler entirely because they carry no drafts to
                # evaluate; the scheduler's internal history already
                # ignores them.
                adaptive_scheduler.record_round(
                    accepted_count=accepted,
                    scheduled_block_size=bs,
                )

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
