# MLX-engine Review Package - 2026-06-20

## Scope

- Branch: `mlx-vlm-prompt-cache-perf`
- Base: `origin/main`
- Head: `f34772e [#1190] Fix batched startup warmup blocking`
- Redmine: `#1190`
- Status: promotion-test pass, pushed for storage/review handoff without an open PR.

This branch packages the retained MLX-engine prompt/inference performance work:

- VLM persistent prompt-cache restore improvements.
- Path-based safetensor restore loading.
- Redundant current-only VLM KV record skip.
- Speculative prefill support and cleanup paths.
- Batched text startup warmup hardening.
- Response-quality and benchmark documentation for promoted and rejected candidates.

## Commit Range

```text
f34772e [#1190] Fix batched startup warmup blocking
d198936 [#1190] refresh mlx-engine performance handoff
3ed3dca [#1190] record broader VLM and text quality validation
6d517f3 [#1190] record repeat VLM KV skip benchmark evidence
859f7c4 [#1190] skip redundant VLM KV cache records
8ad603a [#1190] record rejected full-prefix kv span experiment
2859059 [#1190] harden unload flow and record next mlx-engine work
2399209 [#1190] feat(prompt-cache): add span-aware restore and thinking defaults
c4b56a0 [#1190] docs: record cross-model quality validation
fe7d0d3 [#1190] docs: add deterministic text quality gate
e7c8d73 [#1190] docs: record broader text validation
fb18ff5 [#1190] docs: record post-revert benchmark rerun
6a1fae0 Revert "[#1190] perf: eager-materialize VLM cache records during load"
c0649b4 [#1190] docs: record successful py312 benchmark rerun
3e2484a [#1190] docs: record benchmark blocker
0b648c4 [#1190] docs: update mx.eval investigation with Phase 2 test results
021444b [#1190] docs: add mx.eval materialization investigation plan
64b3d2a [#1190] perf: eager-materialize VLM cache records during load
86d34b4 [#1190] VLM prompt-cache performance: persistent cache, path-load, timing, specprefill
3adc950 [#1190] chore: add MLX performance handoff
```

## Diff Summary

```text
63 files changed, 6924 insertions(+), 407 deletions(-)
```

Primary touched areas:

- `mlx_engine/model_kit/batched_vision/`
- `mlx_engine/model_kit/batched_model_kit.py`
- `mlx_engine/cache_wrapper.py`
- `mlx_engine/generate.py`
- `mlx_engine/utils/specprefill.py`
- `tests/test_batched_vision_*`
- `tests/test_prefill_step_size.py`
- `tests/test_batched_generation.py`
- `.planning/`

## Retained Performance Evidence

- Overall VLM TTFT versus original quality baseline: `-48.650%`.
- Overall VLM total latency versus original quality baseline: `-47.435%`.
- KV-delta record-load time: `21.052 ms` down to `1.546-1.861 ms`, about `91-93%` faster.
- Restore detail: `38.030 ms` down to `19.912-20.104 ms`, about `47%` faster.
- Redundant current-only KV record skip repeat:
  - TTFT `-3.013%`.
  - Total latency `-2.566%`.
  - Quality gate passed.
- Broader two-prompt VLM validation repeat passed:
  - `image_pair` TTFT `-29.227%`, total latency `-24.485%`.
  - `image_toucan` TTFT `-63.157%`, total latency `-54.968%`.
- Dense/code Qwen2.5-Coder-14B deterministic quality inspection passed all five prompts.

## Promotion Verification

Promotion-level pytest group after the startup warmup fix:

```bash
/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.venv-py312/bin/python -m pytest tests/test_load_model_default_seq_nums.py tests/test_batched_vision_model_kit.py tests/test_batched_vision_cache_store.py tests/test_batched_vision_coordinator.py tests/test_batched_vision_cache_io_thread.py tests/test_batched_vision_prompt_inputs.py tests/test_batched_vision_batch_generator.py tests/test_batched_vision_chunks.py tests/test_batched_vision_disk_budget.py tests/test_batched_vision_blob_store.py tests/test_batched_vision_restore_planner.py tests/test_prompt_processing_specprefill.py tests/test_specprefill_helpers.py tests/test_specprefill_options.py tests/test_generate_specprefill_cleanup.py tests/test_model_kit_startup.py tests/test_patched_qwen3_5.py tests/test_request_state.py tests/test_cache_wrapper.py tests/test_batched_generation.py tests/test_prefill_step_size.py tests/test_chat_template_args.py tests/test_mlx_threading.py tests/test_distributed_server.py -q
```

Result:

```text
202 passed, 9 skipped, 2 warnings, 14 subtests passed in 62.68s
```

Focused startup-regression evidence:

```text
tests/test_prefill_step_size.py::test_batched_prefill_step_size
1 passed, 2 warnings in 18.94s
```

## Review Constraints

- Do not remove restore-time `mx.eval(...)`; removing it caused a warm restore stream-lifecycle failure.
- Do not promote faster restore shapes unless response-quality checks pass.
- Do not make the exact 7,162-token batched text startup warmup mandatory. Use `MLX_ENGINE_STARTUP_LONG_WARMUP=1` only for controlled benchmark precompile runs.
- Do not retry full-prefix VLM KV span packing without a different write-amplification plan; the fair persistent-cache benchmark regressed.

## Next Review Actions

1. Keep `mlx-vlm-prompt-cache-perf` pushed to `origin` without an open PR at this stage.
2. Split the branch later if review requires a smaller slice.
3. Re-run VLM and deterministic text quality gates only if the branch changes after this package.
4. Continue restore `eval_ms` investigation in a follow-up branch without removing the restore-time `mx.eval(...)` safety barrier.
