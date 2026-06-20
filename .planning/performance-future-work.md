# MLX-engine Performance Future Work

Date: 2026-06-20
Scope: `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine`
Related issue: Redmine `#1190`

## Current conclusions

- Keep the restore-time `mx.eval(...)` safety barrier. Removing it caused a warm-restore stream failure.
- Keep the path-based safetensor load optimization. It retained quality and produced the strongest repeatable VLM cache-load win.
- Do not treat forced unload during active generation as a performance regression. That behavior is the expected guard path.
- Treat the empty `stopGenerating()` replay warning as a separate runtime/integration issue, not as a performance win or loss.
- Treat the restore-materialization track as paused. Two eager candidates stayed below the promotion threshold or regressed decode throughput.

## Remaining experiments worth trying

1. Persistent VLM record layout
   - Pre-concatenate compatible KV-delta records during save.
   - Keep the existing record format readable so old caches still load.
   - Goal: reduce restore assembly/materialization cost without removing stream safety.

2. Record packing and cache-layout redesign
   - Compare current one-record-at-a-time restore assembly with a layout that stores more of the final restored tensor structure up front.
   - Measure whether this reduces Python-side overhead, file opens, and post-load materialization.

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

1. Design the persistent VLM record-layout candidate.
2. Prototype save-time pre-concatenation for compatible KV-delta records.
3. Bench it against the retained path-load baseline.
4. Run the deterministic quality compare.
5. Promote only if quality passes and the end-to-end gain is real.

## Validation to rerun after the next change

- `python3 shared_bench.py` with `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.venv-py312/bin/python`
- `python3 quality_compare.py` against the retained benchmark baseline
- `env PYTHONPATH=. pytest tests/test_shared_bench.py tests/test_quality_compare.py` in the benchmark harness when report-format logic changes

## Reference artifacts

- `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.planning/.continue-here.md`
- `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.planning/HANDOFF.json`
- `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260618-upstream-baseline-comparison.md`
- `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260619-pause-handoff.md`
