# MLX-engine Performance Future Work

Date: 2026-06-20
Scope: `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine`
Related issue: Redmine `#1190`

## Current conclusions

- Keep the restore-time `mx.eval(...)` safety barrier. Removing it caused a warm-restore stream failure.
- Keep the path-based safetensor load optimization. It retained quality and produced the strongest repeatable VLM cache-load win.
- Keep one-step KV span coalescing. The full-prefix KV span candidate was rejected on 2026-06-20 because it regressed persistent-cache warm restore performance.
- Keep the redundant current-only KV record skip for chunks that already save a two-chunk KV span. It reduced persistent-cache benchmark TTFT and total latency while preserving quality.
- Keep KV span selection bounded to the current chunk plus its immediate predecessor. Full-prefix span records remain a rejected format and should not win restore planning if stale experimental records exist in a persistent index.
- Do not treat forced unload during active generation as a performance regression. That behavior is the expected guard path.
- Treat the empty `stopGenerating()` replay warning as a separate runtime/integration issue, not as a performance win or loss.
- Treat the restore-materialization track as paused. Two eager candidates stayed below the promotion threshold or regressed decode throughput.
- Treat contiguous-copy removal as rejected for restore eval work. The 2026-06-20 microbenchmark showed `mx.contiguous` median cost around `0.05 ms` and no meaningful difference between `concat + eval` and the current `concat + contiguous + eval` path.

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

1. Design a lower-write-amplification persistent VLM record-layout candidate.
2. Bench it against the retained path-load baseline in persistent-cache mode.
3. Run the deterministic quality compare.
4. Promote only if quality passes and the end-to-end gain is real.

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

## Validation to rerun after the next change

- `python3 shared_bench.py` with `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.venv-py312/bin/python`
- `python3 quality_compare.py` against the retained benchmark baseline
- `env PYTHONPATH=. pytest tests/test_shared_bench.py tests/test_quality_compare.py` in the benchmark harness when report-format logic changes

## Reference artifacts

- `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.planning/.continue-here.md`
- `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.planning/HANDOFF.json`
- `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260618-upstream-baseline-comparison.md`
- `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260619-pause-handoff.md`
