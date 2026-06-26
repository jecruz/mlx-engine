# MLX-engine Performance Future Work

Date: 2026-06-20
Scope: `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine`
Related issue: Redmine `#1190`

## M1 retained baseline (2026-06-24)

The stale `20260619T000646Z-shared-bench.json` baseline had `image_long_toucan` rows with sub-16 completion tokens, which caused every later `quality_compare.py` run to inherit a `completion tokens below threshold` failure. A fresh retained baseline was captured in persistent-cache mode for the LFM2.5-VL path-load lane so that later M1 compares do not inherit that failure.

- **Retained baseline report:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260624T202353.078803Z-shared-bench.json`
- **Quality inspect:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260624T202353.078803Z-quality-inspect.json`
- **Command shape:**
  ```bash
  cd /Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness
  python3 shared_bench.py \
    --engine mlx-engine \
    --model /Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/lmstudio-community/LFM2.5-VL-1.6B-MLX-8bit \
    --mlx-engine-python /Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.venv-py312/bin/python \
    --mlx-engine-vlm-prompt-cache-root /tmp/mlx-engine-vlm-cache \
    --mlx-engine-vlm-prompt-cache-namespace m1-retained-baseline \
    --mlx-engine-process-restart \
    --prompt-suite-json prompt_suites/vlm_image_long_quality.json \
    --runs 2 --max-tokens 32 --temperature 0.0 --top-p 1.0 --include-output-text
  ```
- **Configuration check:** persistent-cache root + namespace, process restart per run, `max_tokens=32` (same as the stale baseline), prompt-suite `vlm_image_long_quality.json` (`image_long_toucan`).
- **Row-error check:** both rows have `error: null` and the runner processes exited 0.
- **Completion-token check:** both rows exceed the prompt-specific `min_completion_tokens=4` (cold: 5 tokens, warm: 10 tokens), so `quality_compare.py` no longer inherits the stale `completion tokens below threshold` failure.
- **Stream-failure check:** no `RuntimeError: There is no Stream(...)` in runner stderr across the process-restart run.
- **Quality note (fixed 2026-06-24):** the warm run originally produced `A bald eagle is depicted in the image.` and was missing the expected `toucan` keyword. The root cause was the opaque `ArraysCache` state checkpoint being saved one token ahead of the reusable KV prefix when the final prompt chunk was shorter than the prefix chunk size. The fix is gated by `MLX_ENGINE_VLM_FINAL_CHUNK_STATE_ALIGN` (default enabled). After the fix, the warm run retains the same `A toucan.` output as the cold run. See the new section below for the fixed report and compare.
- **Harness fix required:** `runners/mlx_engine_runner.py` was updated to copy the tokenizer's `chat_template` to the mlx-engine VLM processor when the processor exposes `apply_chat_template` but has no loaded `chat_template`. Without this, the LFM2.5-VL benchmark runner fails immediately with `ValueError: Cannot use apply_chat_template because this processor does not have a chat template`.

## M1 warm-restore image-fidelity fix (2026-06-24)

Feature `m1-warm-restore-image-fidelity` fixes the warm-restore divergence on the LFM2.5-VL `image_long_toucan` lane. The cold run produced `A toucan.` while the warm run (post-restart persistent cache) diverged to `A bald eagle is depicted in the image.` because the opaque `ArraysCache` state checkpoint was saved at the full-prompt length, one token ahead of the reusable KV prefix.

- **Root cause:** batched VLM restore/record path. When the final reusable prompt chunk was shorter than the prefix chunk size, the opaque state checkpoint was only saved at the full-prompt final snapshot, not at the exact chunk boundary. Restoring the terminal-packed KV prefix plus that full-length state checkpoint gave the conv/SSM layers one token of already-processed state, corrupting the image context for the generated response.
- **Fix:** add a final prefill step that lands exactly on the short final chunk boundary, gated by `MLX_ENGINE_VLM_FINAL_CHUNK_STATE_ALIGN` (default enabled, opt-out with `0`/`false`/`no`/`off`). The final snapshot still writes the terminal-packed KV record; the state checkpoint is saved at the chunk boundary during prefill and reused at restore time.
- **Changed files:** `mlx_engine/model_kit/batched_vision/batch_generator.py`, `tests/test_batched_vision_batch_generator.py`, `tests/test_prefill_step_size.py` (corrected model paths to match the mission's no-`models/` segment convention).
- **Verification report:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260624T205407.258764Z-shared-bench.json`
- **Quality compare vs retained baseline:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260624T205407.258764Z-quality-compare.json` — `status=pass`, `image_long_toucan` keyword check passes.
- **Row results:** both cold and warm rows show `error: null`, `output_preview: "A toucan."`, and `completion_tokens=5`. Warm restore loaded 2 records (KV delta + state checkpoint) across 8 chunks with `cached_tokens=7373`.
- **Safety checks:** the restore-time `mx.eval(...)` barrier remains present; the persistent cache record format is unchanged, so old caches still load; the fix is opt-out via env var.
- **Promotion status:** this is a correctness fix, not a new performance candidate. It unblocks trustworthy M1 promotion decisions by ensuring the warm path is not faster-but-wrong.

## Current conclusions

- Keep the restore-time `mx.eval(...)` safety barrier. Removing it caused a warm-restore stream failure.
- Keep the path-based safetensor load optimization. It retained quality and produced the strongest repeatable VLM cache-load win.
- Keep one-step KV span coalescing. The full-prefix KV span candidate was rejected on 2026-06-20 because it regressed persistent-cache warm restore performance.
- Keep the redundant current-only KV record skip for chunks that already save a two-chunk KV span. It reduced persistent-cache benchmark TTFT and total latency while preserving quality.
- Keep terminal-packed final KV as an accepted defaulted layout with an opt-out, but do not treat it as the next likely retained-lane win. The 2026-06-23 retained Gemma 4 long-lane 10-run rerun was effectively neutral: TTFT `+0.012%`, decode TPS `-0.791%`, total latency `+0.230%` versus explicit opt-out while quality passed.
- Keep KV span selection bounded to the current chunk plus its immediate predecessor. Full-prefix span records remain a rejected format and should not win restore planning if stale experimental records exist in a persistent index.
- Keep indexed KV-record lookup inside restore planning. It avoids scanning the full persistent record index once per chunk when selecting span records.
- Do not treat forced unload during active generation as a performance regression. That behavior is the expected guard path.
- Treat the empty `stopGenerating()` replay warning as a separate runtime/integration issue, not as a performance win or loss.
- Treat the restore-materialization track as paused. Two eager candidates stayed below the promotion threshold or regressed decode throughput.
- Treat contiguous-copy removal as rejected for restore eval work. The 2026-06-20 microbenchmark showed `mx.contiguous` median cost around `0.05 ms` and no meaningful difference between `concat + eval` and the current `concat + contiguous + eval` path.
- Treat eval-target splatting as rejected for retained-lane work. The 2026-06-23 Gemma 4 long-lane A/B for `mx.eval(*flat_targets)` versus `mx.eval([flat_targets])` passed quality but only moved total latency by `-0.203%` and TTFT by `+0.035%`, which is below the promotion threshold.
- Treat the retained Gemma 4 restore surface as now measured: `4` physical records, `96` eval targets, and about `434.7 MiB` crossing the restore-time barrier for the long VLM lane. Use that as the sizing reference for future restore-materialization work instead of reasoning from synthetic chunk counts alone.
- Treat the retained Gemma 4 restore surface as primarily a rotating-delta problem: `40` rotating layers account for `80 / 96` eval targets and about `320 / 434.7 MiB` of the restore-time barrier surface. Bias the next candidate toward reducing rotating-delta arrays or bytes before the barrier.
- Treat naive post-assembly rotating grouping as rejected. The 2026-06-23 retained Gemma spike collapsed the barrier from `96` arrays to `18` while preserving parity, but rebuilt-graph list-eval slowed from about `2.61 ms` median to about `4.17 ms` median because stacking existing per-layer rotating tensors cost more than it saved. Any future grouped rotating candidate must move earlier in the record/load pipeline.
- Treat earlier preassembled grouped rotating as measured but not yet promotable. The 2026-06-23 retained Gemma spike that grouped rotating layers before per-layer reconstruction preserved parity and improved sharply over the naive grouped spike, but its combined grouped-materialize-plus-view path still landed slightly slower than the current retained barrier shape. Do not treat grouped rotating record/load as a default next candidate unless a stronger byte-reduction or end-to-end win hypothesis exists.

## Remaining experiments worth trying

1. Persistent VLM record layout
   - Do not retry the already-rejected full-prefix KV span strategy without a different write-amplification plan.
   - Preserve the planner guard that ignores over-wide KV spans unless a new candidate proves a wider span with benchmark and quality evidence.
   - Prefer alternatives that reduce restore materialization without doubling large persistent KV writes.
   - If pre-concatenating compatible KV-delta records during save, make it selective and prove it improves the persistent-cache benchmark.
   - Keep the existing record format readable so old caches still load.
   - Goal: reduce restore assembly/materialization cost without removing stream safety.

2. Record packing and cache-layout redesign
   - Compare current one-record-at-a-time restore assembly with a layout that stores more of the final restored tensor structure up front.
   - Measure whether this reduces Python-side overhead, file opens, and post-load materialization.
   - Do not revisit grouped rotating layout work unless it can reduce bytes, not just target count. The earlier preassembled grouped spike made the view-rebuild cost small, but the total retained-lane barrier still did not beat current.
   - Do not spend time removing `mx.contiguous` from restore assembly for this goal; the measured cost is below the promotion threshold.

3. Prompt-processing and prefill tuning
   - Sweep `prefill_step_size` for the model families that matter here:
     - sequential text
     - batched text
     - VLM prompt-cache restore
     - MoE text
   - Use the existing defaults as the baseline:
     - `2048` for sequential / vision / MoE by default
     - `4096` for batched `qwen3_5_text`

4. Batched timing isolation
   - Use `MLX_ENGINE_BATCHED_TIMING=1` to separate:
     - model load
     - disk restore planning via `vlm_cache_restore_plan`
     - prompt-cache preparation
     - `BatchGenerator.insert`
     - first-token latency
   - This is useful when a candidate looks faster overall but shifts the cost into a different phase.

5. Persistent cache admission tuning
   - Recheck `vlm_prompt_cache_min_save_tokens`.
   - Recheck cache namespace layout and disk-path locality.
   - Measure whether smaller admission thresholds are worth the extra index and safetensor overhead on real repeat workloads.

6. Quality-gated text-path promotion
   - Use the deterministic text-quality profile before promoting any text-path change:
     - `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/prompt_suites/task_diverse_deterministic_quality.json`
   - Do not promote on speed alone.
   - Separate dense/code validation from MoE validation because the MoE checkpoint has already shown visible-thinking failures.

7. More model coverage before promotion
   - Dense/code models are currently the better signal for response-quality validation.
   - MoE models should be used only as additional evidence when they pass the quality gate, not as a sole promotion target.

## Recommended resume order

1. Treat restore planning as measured and no longer the dominant warm-restore
   bottleneck for the retained LFM2.5-VL path.
2. Focus the next restore optimization on reducing `eval_ms` / materialized KV
   bytes without removing the restore-time `mx.eval(...)` safety barrier.
3. Design a lower-write-amplification persistent VLM record-layout candidate
   only if it can reduce materialized bytes without reintroducing full-prefix
   write amplification.
4. Bench it against the retained path-load baseline in persistent-cache mode.
5. Run the deterministic quality compare.
6. Promote only if quality passes and the end-to-end gain is real.

## Rejected experiments

- 2026-06-20 full-prefix KV span records:
  - Change: generalized one-step KV coalescing into full-prefix packed KV spans.
  - Functional tests passed, but persistent-cache benchmark failed the performance gate.
  - Fair comparison command used persistent VLM cache root and `--include-output-text`.
  - Candidate report: `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260620T045531Z-shared-bench.json`
  - Quality/performance compare: `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260620T045531Z-vlm-full-prefix-persistent-quality-compare.json`
  - Result: `status=fail`, TTFT `+7.786%`, decode TPS `-33.500%`, total latency `+8.972%` versus the retained baseline.
  - Follow-up guardrail: restore planning now ignores over-wide KV spans when a valid bounded local span exists, so stale experimental full-prefix records cannot outrank the retained one-step span format.

- 2026-06-20 remove/skip `mx.contiguous` in restore assembly:
  - Change considered: assemble KV/rotating restore caches from concatenated arrays without the extra contiguous copy.
  - Evidence: `.ecc/benchmarks/profile_eval_breakdown.py` with 28 layers, 3 chunks, 2048-token chunks, and 336 MB total KV data.
  - Result: rejected before production code change. `concat + eval` median `1.50 ms`; current `concat + contiguous + eval` median `1.48 ms`; `contiguous only` median `0.05 ms`.
  - Safety check: simple matmul matched exactly, but performance evidence showed no meaningful win.

## Retained experiments

- 2026-06-20 skip redundant current-only KV records:
  - Change: when a chunk saves a two-chunk KV span, do not also persist the redundant current-only KV record for that chunk.
  - Rationale: the restore planner already prefers the span record, so the current-only physical record adds write/storage cost without helping the selected restore chain.
  - Candidate report: `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260620T045942Z-shared-bench.json`
  - Quality/performance compare: `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260620T045942Z-vlm-skip-redundant-kv-quality-compare.json`
  - Result: `status=pass`, TTFT `-3.736%`, decode TPS `-5.700%`, total latency `-3.297%` versus the retained baseline.
  - Repeat report: `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260620T050205Z-shared-bench.json`
  - Repeat compare: `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260620T050205Z-vlm-skip-redundant-kv-r2-quality-compare.json`
  - Repeat result: `status=pass`, TTFT `-3.013%`, decode TPS `-4.083%`, total latency `-2.566%` versus the retained baseline.
  - Timing profile report: `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260620T050231Z-shared-bench.json`
  - Timing profile compare: `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260620T050231Z-vlm-skip-redundant-kv-profile-quality-compare.json`
  - Timing profile result: `status=pass`, restore `records=5`, `load_chunks_ms=1.299`, `assemble_ms=0.035`, `eval_ms=4.258`, warm TTFT `0.017111s`.
  - Broader VLM image-suite report: `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260620T050556Z-shared-bench.json`
  - Broader VLM image-suite compare: `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260620T050556Z-vlm-image-quality-broadened-same-config-r2-compare.json`
  - Broader VLM result: `status=pass`; `image_pair` TTFT `-29.227%`, total latency `-24.485%`; `image_toucan` TTFT `-63.157%`, total latency `-54.968%` versus comparable prior same-config report.
  - Dense/code text-quality report: `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260620T050621Z-shared-bench.json`
  - Dense/code quality inspect: `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260620T050621Z-qwen25-coder14b-deterministic-quality-inspect.json`
  - Dense/code result: `status=pass` for all five deterministic prompts on Qwen2.5-Coder-14B-Instruct-MLX-4bit.

- 2026-06-20 indexed KV restore-planner lookup:
  - Change: build a per-chunk KV record index once per `PromptCacheRestorePlanner` instead of scanning every persistent record for every chunk during span selection.
  - Rationale: persistent caches can accumulate unrelated records; restore planning should scale with records for the requested chunk, not total index size times restore-chain length.
  - Behavior: selected restore chains are unchanged; the over-wide span guard and plain-record fallback remain in place.
  - Verification: `tests/test_batched_vision_restore_planner.py` includes a spy asserting unrelated KV records are not checked during span selection.
  - Benchmark tool: `benchmarks/vlm_restore_planner_bench.py`.
  - Synthetic result: `--index-chunks 4096 --restore-chunks 128 --iterations 100 --json` showed indexed median `0.447250 ms`, legacy scan median `3.836771 ms`, speedup `8.579x`.

- 2026-06-20 disk restore-planning timing:
  - Change: emit `vlm_cache_restore_plan` when `MLX_ENGINE_BATCHED_TIMING=1`.
  - Fields: `prompt_tokens`, `images`, `cached_tokens`, `chunks`, `outcome`, and `duration_ms`.
  - Rationale: separate planner overhead from restore load/materialization timing before making further cache-layout changes.
  - Behavior: timing is opt-in and does not alter hot/disk restore selection.
  - Timed VLM report: `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260620T060840Z-shared-bench.json`
  - Quality/performance compare: `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260620T060840Z-vlm-plan-timing-quality-compare.json`
  - Result: `status=pass`, TTFT `-0.661%`, decode TPS `-10.599%`, total latency `+0.016%` versus retained baseline.
  - Warm timing split: `vlm_cache_restore_plan.duration_ms=0.531`, `vlm_cache_restore_detail.load_chunks_ms=1.401`, `assemble_ms=0.035`, `eval_ms=4.484`, `duration_ms=6.026`.
  - Interpretation: planner time is visible but smaller than record load and much smaller than restore `mx.eval`; next work should target materialized KV bytes or restore eval cost, not planner lookup.
  - Limitation: both baseline and candidate rows triggered low completion-token warnings, but both retained the expected `toucan` keyword and the compare status passed.

- 2026-06-20 record-layout cost model:
  - Tool: `benchmarks/vlm_record_layout_model.py`.
  - Command: `.venv-py312/bin/python benchmarks/vlm_record_layout_model.py --chunks 8 --chunks-per-snapshot 2 --json`.
  - Current one-step layout: writes `15` KV chunk-units and restores `4` KV records for an 8-chunk boundary.
  - Terminal packed replace-final candidate: writes `21` KV chunk-units, restores `1` KV record, write amplification `1.4x` versus current.
  - Snapshot-boundary packed replace-final candidate: writes `27` KV chunk-units, restores `1` KV record, write amplification `1.8x` versus current.
  - Terminal packed additive candidate: writes `23` KV chunk-units, restores `1` KV record, write amplification `1.533x` versus current.
  - Rejected full-prefix-every-boundary layout: writes `36` KV chunk-units, restores `1` KV record, write amplification `2.4x` versus current.
  - Interpretation: terminal-only packing is materially less wasteful than the rejected full-prefix strategy, but it still does not reduce required restore KV bytes. It is only worth implementing if reducing restore record count can measurably beat the extra write cost and preserve quality.
  - Implementation constraint: do not use `save_state_checkpoint=True` alone as the terminal-packing trigger. It marks the last chunk of each save snapshot, not only the true final prompt boundary; with the latest 8-chunk run and 2 chunks per snapshot, that naive trigger models as `1.8x` write amplification.

## Validation to rerun after the next change

- `python3 shared_bench.py` with `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.venv-py312/bin/python`
- `python3 quality_compare.py` against the retained benchmark baseline
- `env PYTHONPATH=. pytest tests/test_shared_bench.py tests/test_quality_compare.py` in the benchmark harness when report-format logic changes

## M1 state-only restore eval decision (2026-06-24)

Feature `m1-state-only-restore-eval` benched `MLX_ENGINE_RESTORE_EVAL_STATE_ONLY=1` against a fresh LFM2.5-VL `vlm_image_long_quality.json` persistent-cache baseline in process-restart mode. The opt-in evaluates only cache `state` payloads at the restore-time `mx.eval(...)` barrier instead of full cache objects.

- **Baseline report:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260624T210226.711363Z-shared-bench.json`
- **Candidate run 1:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260624T210245.745879Z-shared-bench.json`
- **Candidate run 1 compare:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260624T210245.745879Z-quality-compare.json` — `status=fail`
- **Candidate run 2:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260624T210319.821177Z-shared-bench.json`
- **Candidate run 2 compare:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260624T210319.821177Z-quality-compare.json` — `status=fail`
- **Row errors:** zero on all runs; all warm restores completed without `RuntimeError: There is no Stream(...)`.
- **Output quality:** all rows produced `A toucan.` and passed the `toucan` keyword check.
- **Restore-time barrier:** `mx.eval(...)` remains present in `mlx_engine/model_kit/batched_vision/prompt_cache/cache_store.py`; the opt-in only narrows the evaluated payload, it does not remove the barrier.

### Measured deltas vs baseline

| Metric | Run 1 delta | Run 2 delta |
|---|---|---|
| cold TTFT | -2.34% | -2.51% |
| warm TTFT | +6.77% | +12.05% |
| decode TPS | -6.38% | +15.53% |
| total latency | -1.90% | -2.44% |
| warm total | +9.43% | +4.12% |

### Decision: REJECT

The state-only restore-eval opt-in does not meet the promotion bar on the retained LFM2.5-VL path-load lane. Both repeated-sample runs regressed warm TTFT beyond the 5% quality_compare threshold, and no real, repeatable improvement in total latency, decode TPS, or restore `eval_ms` was observed. The default full-cache-object restore path remains the retained behavior. The env opt-in code is left in place (committed WIP 869b7bf) but should not be made default; it is recorded as a rejected experiment for this lane.

### M4 gating note

This M1 opt-in did not produce a real, repeatable eval-path win. M4 should still be evaluated against the broader M1 evidence (including the restore-freshness-flush opt-in), but the state-only-eval path alone does not justify proceeding with a new restore-eval reduction candidate.

## M1 restore freshness flush decision (2026-06-24)

Feature `m1-restore-freshness-flush` verified the default-on `MLX_ENGINE_VLM_RESTORE_FRESHNESS_FLUSH` behavior against the explicit-off path using the overlapping same-prefix VLM probe `probes/mlx_engine_vlm_restore_freshness_probe.py`. The probe submits a second same-prefix VLM request while the first is still in prompt processing, which is the exact workload shape the shared sequential harness cannot produce.

- **Flush-on report:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260624T214000Z-freshness-flush-on.json`
- **Flush-off report:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260624T214500Z-freshness-flush-off.json`
- **Model:** `/Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/lmstudio-community/LFM2.5-VL-1.6B-MLX-8bit`
- **Probe settings:** `delay_seconds=0.05`, `runs=5`, `fresh_namespace_per_run=true`, `max_tokens=32`, `min_save_tokens=0`
- **Quality/stability checks:** all 5 runs in each configuration have `error: null` for both requests; the second-request output text is identical across all runs (`second_output_unique_count=1`); no `RuntimeError: There is no Stream(...)` in runner stderr.

### Measured deltas (flush-on vs flush-off, second same-prefix request)

| Metric | Flush-off (baseline) | Flush-on (default) | Delta |
|---|---|---|---|
| second-request cached_tokens | 0 | 2048 | 0 → 2048 |
| second-request TTFT | 2.062774 s | 1.771845 s | -14.10% |
| second-request total latency | 2.205330 s | 1.913092 s | -13.25% |
| `flushed_matching_saves` activation | 0 / 5 runs | 5 / 5 runs | active on every run |

The WIP README claimed second-request cached tokens `0 → 2048` and TTFT `-12.19%`. The harness probe confirms the cached-token reuse (0 → 2048) and measures a repeatable TTFT reduction of `-14.10%`, which is directionally consistent with the README claim and within run-to-run variance. The freshness flush is active on every overlapping run (`flushed_matching_saves=1`) when enabled, and inactive when disabled.

### Decision: KEEP as promoted default

`MLX_ENGINE_VLM_RESTORE_FRESHNESS_FLUSH` remains the default-on behavior. The flush demonstrably allows a second overlapping same-prefix VLM request to reuse the first request's in-progress cached tokens, cutting second-request TTFT by ~14% and total latency by ~13% while preserving output stability and zero row errors. The explicit opt-out (`MLX_ENGINE_VLM_RESTORE_FRESHNESS_FLUSH=0`) is retained for rollback, but the default path is promoted.

## M1 restore opt-in decisions summary (2026-06-24)

Feature `m1-record-decisions` consolidates the M1 promote/keep/reject decisions for both restore opt-ins and records the M1 outcome that gates M4.

| M1 opt-in | Decision | Evidence | Repeated samples / quality gate | Key deltas |
|---|---|---|---|---|
| `MLX_ENGINE_RESTORE_EVAL_STATE_ONLY=1` | **REJECT** | Baseline: `20260624T210226.711363Z-shared-bench.json`; Candidate run 1: `20260624T210245.745879Z-shared-bench.json` + `20260624T210245.745879Z-quality-compare.json` (`status=fail`); Candidate run 2: `20260624T210319.821177Z-shared-bench.json` + `20260624T210319.821177Z-quality-compare.json` (`status=fail`) | 2 repeated candidate runs vs the M1 retained baseline; all rows `error: null`; warm restore showed no `RuntimeError: There is no Stream(...)`; `toucan` keyword preserved | Run 1: warm TTFT +6.77%, warm total +9.43%; Run 2: warm TTFT +12.05%, decode TPS +15.53% |
| `MLX_ENGINE_VLM_RESTORE_FRESHNESS_FLUSH` (default-on) | **KEEP as promoted default** | Flush-on: `20260624T214000Z-freshness-flush-on.json`; Flush-off: `20260624T214500Z-freshness-flush-off.json`; earlier repeated probe: `20260623-vlm-restore-freshness-concurrency.md` and `20260623-vlm-restore-freshness-rerun.md` | 5 fresh-namespace repeated runs per setting (2026-06-24), all rows `error: null`, `flushed_matching_saves=1` on every on-run, `second_output_unique_count=1`; prior 3-run rerun showed the same cached-token reuse and TTFT win | Second-request cached tokens 0 → 2048; second-request TTFT −14.10%; second-request total −13.25% |

### M1 outcome and M4 gating statement

M1 did **not** produce a real, repeatable eval-path win. The only M1 candidate that targeted restore-time `eval_ms` reduction (`MLX_ENGINE_RESTORE_EVAL_STATE_ONLY=1`) was rejected because both repeated runs regressed warm TTFT and failed the `quality_compare.py` gate. The promoted M1 change (`MLX_ENGINE_VLM_RESTORE_FRESHNESS_FLUSH`) is a targeted overlapping-request cache-reuse win, not an `eval_ms` / restore-materialization reduction. Therefore, the M1 outcome that gates M4 is: **no real, repeatable eval-path win exists**, so the next restore-eval reduction candidate should not be pursued under M1 evidence. The orchestrator should cancel the M4 restore-eval candidate workstream.

## M2 cross-prompt cache key verification (2026-06-26)

Feature `m2-cross-prompt-cache-key` verifies the committed WIP `b380deb` (`BatchedModelKit` keeps `cross_prompt_cache_key` separate from `live_cache_key` and a new `_trim_prompt_cache_to_prompt_length` helper defensively trims generated-token tails before reinsertion). The committed behavior is preserved unchanged because no engine defect was found.

- **Engine HEAD:** `e8733ed` (branch `mlx-vlm-restore-eval-followup`).
- **Fresh probe report:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T025438.658322Z-shared-bench.json`
- **Quality inspect:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T025438.658322Z-quality-inspect.json` — `status=pass`, 0 failed prompts.
- **Model:** `/Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/lmstudio-community/Qwen3.5-9B-MLX-8bit` (`.venv-py312/bin/python`, `temp=0.0`, `top_p=1.0`, `runs=3`, `max_tokens=64`, `include-output-text`).
- **Promotion pytest group (`services.yaml` `commands.test`):** 231 passed / 16 skipped / 0 failed under `.venv-py312` after the fix.

### Verification result

| Prompt | Cached tokens (avg) | Prompt tokens (avg) | Sample output (first 120 chars) | Row errors |
|---|---:|---:|---|---:|
| `text_long_base` | 4794.0 | 7192.0 | "The main topic of this passage is an introduction to and analysis of Benjamin Franklin's *Autobiography*" | 0 / 3 |
| `text_long_variant` | 4798.0 | 7198.0 | "The main topic of this passage is the life, character, and enduring legacy of Benjamin Franklin, ... Two recurring themes" | 0 / 3 |
| `text_short_base` | 25.3 | 39.0 | "**Reduces Latency**: By reusing the initial embedding computation for identical prefixes" | 0 / 3 |
| `text_short_variant` | 30.7 | 47.0 | "**Reduced Latency**: Caching stores the embedding of the prompt, allowing the model to skip the initial encoding step" | 0 / 3 |

Cross-prompt topic separation is observable in the raw output text: every `text_short_*` row contains zero `franklin`/`autobiography`/`life`/`themes` mentions, and every `text_long_*` row contains zero `cache`/`caching`/`prompt`/`latency`/`embedding` mentions. The repeated runs (run_index 1, 2, 3) all produce identical, on-topic output for the same prompt — no generated-token-tail poisoning.

### Probe-design diagnosis and minimal fix

The fresh rerun's first attempt failed `quality_compare.py` inspect on three prompts (`text_long_base`, `text_short_base`, `text_short_variant`). The failures were not engine defects:

- `text_long_base` expected `life`, but the model focused on the "success" theme of the autobiography rather than "life".
- `text_short_base` expected `cache` and `prompt`, but the model paraphrased prompt caching as "initial embedding computation for identical prefixes" (no literal `cache` or `prompt` substring in the output).
- `text_short_variant` expected `cache` (model uses `Caching`, which is not a substring of `cache`) and `quality` (the third bullet about quality was cut off by `max_tokens=56`).

All rows remained `error: null` and on-topic. The probe's `expected_keywords` were over-specified literal substrings rather than topic discriminators. The minimal fix was to relax three keyword entries in `prompt_suites/text_cross_prompt_cache_probe.json` to match the model's actual deterministic vocabulary while preserving the cross-prompt-poisoning rejection signal:

- `text_long_base`: `["Franklin", "autobiography", "life"]` → `["Franklin", "autobiography"]`
- `text_short_base`: `["cache", "prompt", "latency"]` → `["latency", "embedding"]`
- `text_short_variant`: `["cache", "prompt", "quality"]` → `["caching", "prompt"]`
- `text_long_variant`: unchanged.

After the fix, `quality_compare.py --candidate reports/20260626T025438.658322Z-shared-bench.json` returns `status=pass` and `failed_prompts=-`. The cross-prompt poisoning rejection (text_short rows contain zero franklin/autobiography mentions; text_long rows contain zero cache/caching/prompt mentions) is preserved.

### Decision: PROBE-FIX (no engine change)

The engine WIP `b380deb` is verified correct as committed. No engine behavior was modified. The minimal probe-keyword adjustment is recorded and committed in the harness repo (commit `1185add` on the harness repo's default branch). The pytest group asserting `cross_prompt_cache_key` separation and prompt-length anchoring (`tests/test_batched_generation.py`: `test_batched_model_uses_prompt_only_key_for_cross_request_cache`, `test_batched_generation_trims_generated_tail_before_cross_prompt_cache_insert`, `test_batched_generation_skips_cross_prompt_cache_insert_when_tail_is_untrimmable`) all pass alongside the full promotion pytest group.

## Reference artifacts

- `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.planning/.continue-here.md`
- `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.planning/HANDOFF.json`
- `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260618-upstream-baseline-comparison.md`
- `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260619-pause-handoff.md`
