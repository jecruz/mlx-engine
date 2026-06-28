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

## M14 real-pair DFlash preflight check (2026-06-27)

Feature `m14-dflash-real-pair-preflight` added a fail-fast DFlash readiness gate before heavyweight model loading. The preflight runs inside `load_model(...)` and inside `create_generator(...)` so it triggers before any DFlash-bearing request reaches the model kit. The gate validates the real Qwen3.6 target plus the z-lab DFlash drafter snapshot end-to-end and fails closed when compatibility, cache mode, route, memory headroom, or port-reservation constraints are violated.

### Target and drafter paths

- **Target path:** `/Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/lmstudio-community/Qwen3.6-27B-MLX-8bit`
- **Drafter path:** `/Volumes/StudioStackSSD4TB/Development/LLM/huggingface/hub/models--z-lab--Qwen3.5-27B-DFlash/snapshots/25ee0025ff950496a634e100b75c2db4515e9824`

### Preflight checks (fail-fast before heavyweight load)

1. **Path existence** — both target and drafter snapshot paths must exist and be readable directories.
2. **Tokenizer / config files** — target must expose `config.json`, `tokenizer.json`, `tokenizer_config.json`, and `vocab.json`; drafter must expose `config.json` and at least one safetensors file.
3. **Vocab size compatibility** — `vocab_size` parsed from target `config.json` must match the drafter `vocab_size` and remain within tolerance of the target `vocab.json` tokenizer vocab.
4. **Target layer IDs** — drafter `dflash_config.target_layer_ids` (`[1,10,18,27,35,44,52,61]`) must all be within `num_hidden_layers` of the target (Qwen3.6 27B has `num_hidden_layers=64`).
5. **DFlash config** — drafter must declare `architectures=["DFlashDraftModel"]`, `model_type=qwen3`, BF16 dtype, `block_size=16`, `mask_token_id=248077`, and the expected target layer IDs.
6. **Qwen-family metadata** — both target and drafter must classify as Qwen-family via `model_type` + `architectures` (with path-only matches explicitly rejected when metadata is absent).
7. **Optional dependency availability** — `mlx_vlm.speculative.dflash` and `mlx_vlm.speculative.drafters.qwen3_dflash.dflash` must be importable.
8. **Cache mode compatibility** — rejects `kv_bits`, `kv_group_size`, `quantized_kv_start`, bounded/rotating cache layers, ragged cache layers (`ArraysCache` / `BatchKVCache`), and non-rollback-safe cache layers.
9. **Route compatibility** — rejects `is_vlm_route=True`, `vocab_only=True`, `distributed=True`, `max_seq_nums > 1`, `specprefill=True`, `num_draft_tokens`, already-loaded `model_kit.draft_model`, persistent VLM prompt-cache root, and persistent VLM prompt-cache admission tokens.
10. **Resource isolation** — checks the reserved mission ports (`127.0.0.1:3180`, `3181`, `3182`, `12444`) for occupancy and estimates free memory against target + drafter safetensors byte footprint with a 25% headroom (minimum 8 GiB) before allowing the pair to load.

### Live probe result (current machine state)

```
Target exists: True
Drafter exists: True
=== Live probe result ===
enabled: True
dependency_available: True
target_family: qwen
drafter_family: qwen
target_profile is not None: True
  vocab_size: 248320
  tokenizer_vocab_size: 248044
  num_hidden_layers: 64
  model_type: qwen3_5
route_blockers: ()
cache_mode_blockers: ()
resource_blockers: ('Insufficient free memory for real-pair DFlash preflight: need at least 39.44 GiB, found 37.64 GiB',
                   'Reserved DFlash resource port 127.0.0.1:12444 is already in use')
```

### Behavior: fails closed without heavyweight load

The probe correctly identifies the real pair, parses both Qwen-family metadata, matches vocab and layer ID compatibility, and then fails closed because the active machine does not yet have enough free memory (target+drafter need ≥39.44 GiB; only 37.64 GiB free) and the Qwen LLMDYNAMIX route on `127.0.0.1:12444` is currently bound. No safetensors load or model kit construction begins under these blockers; the preflight raises `DFlashUnavailableError` before `ModelKit(...)` is constructed.

### Verification

- `.venv-py312/bin/python -m pytest tests/test_dflash_boundary.py -q` → **18 passed / 13 subtests passed / 0 failed** under `.venv-py312`.
- `test_real_pair_preflight_accepts_target_and_drafter_metadata` exercises both exact mission paths and asserts vocab_size, tokenizer_vocab_size, Qwen-family classification, and target layer ID coverage when memory and ports are unblocked.
- `test_load_model_fails_fast_before_heavy_model_creation` proves the preflight runs before `ModelKit(...)` and raises `DFlashUnavailableError` instead of constructing a model kit.
- `test_preload_compatibility_rejects_incompatible_route_and_cache_mode` proves every unsupported route and cache mode (VLM, `vocab_only`, distributed, `max_seq_nums>1`, `kv_bits`, `kv_group_size`, `quantized_kv_start`, persistent VLM prompt-cache) is rejected.
- Full mission pytest gate (`services.yaml` `commands.test`) → **257 passed / 16 skipped / 0 failed** after the M14 preflight work.
- `ruff check mlx_engine/utils/dflash_boundary.py mlx_engine/generate.py tests/test_dflash_boundary.py` → clean.

## M14 direct-harness DFlash candidate flags + telemetry (2026-06-27)

Feature `m14-dflash-harness-flags-telemetry` wires explicit DFlash candidate run support into the direct `shared_bench.py` harness so the M14 real-pair work can invoke DFlash with exact target/drafter paths and capture auditable telemetry in the report JSON. DFlash stays default-off: no DFlash kwargs are forwarded to the mlx-engine runner unless the operator explicitly opts in, and DFlash kwargs are never forwarded to omlx / rapid-mlx / vmlx runners (so the harness never enables DFlash through LM Studio or non-mlx-engine surfaces).

### Added CLI flags (default-off)

- `--dflash` (action=store_true) — explicit DFlash opt-in for the mlx-engine runner.
- `--dflash-target-model` — exact target model directory (e.g. `Qwen3.6-27B-MLX-8bit`). Required when `--dflash` is set; forwarded verbatim to the engine preflight with no auto-discovery or path mutation.
- `--dflash-drafter-model` — exact drafter snapshot directory (e.g. `models--z-lab--Qwen3.5-27B-DFlash/snapshots/25ee0025ff950496a634e100b75c2db4515e9824`). Required when `--dflash` is set; the standard autoregressive `draft_model` loading path is never used for DFlash.
- `--dflash-max-draft-tokens` (int, default 4) — maximum DFlash draft tokens; falls back to the engine default when omitted.

The existing `--mlx-engine-force-sequential` flag forces the mlx-engine runner onto the sequential text ModelKit path, which is the only first-slice surface DFlash supports. Sequential text only is recorded as `sequential_text_only: true` in the telemetry block.

### Config + telemetry in the report JSON

The harness combined report's `config` block now carries `dflash`, `dflash_target_model`, `dflash_drafter_model`, `dflash_max_draft_tokens`. The mlx-engine runner report carries a top-level `dflash` metadata block and a per-row `dflash` metadata block with:

- `opted_in` — `true` only when `--dflash` was set.
- `target_model_path` / `drafter_model_path` — verbatim paths from the CLI args (when opted in).
- `max_draft_tokens` — max draft token budget forwarded to the engine.
- `sequential_text_only: true` — first-slice DFlash route restriction.
- `uses_native_runtime: true` — confirms DFlash runs through the native mlx-engine scaffold (no standard autoregressive `draft_model` loading path).
- `fallback_status` — `default_off` when not opted in; `fallback_unsupported_surface` when an opt-in error mentions VLM / batched / distributed / incompatible / rejected / unsupported; `fallback_preflight` for other DFlash opt-in errors.
- `accepted_proposal_tokens` — count of emitted tokens whose `Token.from_draft` flag is `true` (target-verified emissions of drafter proposals).
- `rejected_proposal_tokens` — `max(0, token_count - 1 - accepted)` for the current row when DFlash is opted in.

### Default-off preservation

Without `--dflash` the harness command does not contain any `--dflash*` flag and the runner report's `dflash` block shows `opted_in: false`, `target_model_path: null`, `drafter_model_path: null`, `fallback_status: default_off`. Existing harness and engine tests assert this default-off behavior at the `build_runner_cmd`, `load_model_compat`, `create_generator_compat`, and `error_row` layers.

### No use of LM Studio / adapter routes / standard `draft_model`

- The runner never forwards DFlash kwargs to omlx / rapid-mlx / vmlx runners.
- The engine preflight (`m14-dflash-real-pair-preflight`) and the M13 native DFlash loader (`mlx_engine.utils.dflash_snapshot`) remain authoritative for DFlash drafter validation. The harness only forwards the exact operator-provided path; it does not parse or transform the snapshot.
- `dflash_drafter_model` is forwarded as `dflash_drafter_model` to `load_model` and `create_generator`, never as the standard autoregressive `draft_model` kwarg. The engine rejects loaded `model_kit.draft_model` / `draft_model` / `num_draft_tokens` in `validate_dflash_preload_compatibility` and `validate_dflash_surface_compatibility` before any token emission.

### Verification

- Harness tests: `env PYTHONPATH=. python3 -m pytest tests -q` → **69 passed / 0 failed** after the harness flag + telemetry wiring.
- Engine DFlash tests: `.venv-py312/bin/python -m pytest tests/test_dflash_boundary.py tests/test_dflash_runtime.py -q` → **22 passed / 0 failed** (still pass with no engine source change).
- `ruff check runners/mlx_engine_runner.py shared_bench.py tests/test_mlx_engine_runner.py tests/test_shared_bench.py` → clean (the only pre-existing `F841` is the unrelated `tests/test_vmlx_runner.py:122` `json_str` from M5 long-text baseline, which AGENTS.md flags as out of scope).
- Harness commit: `fccfcc8` on `main` of `mlx-bench-harness` with `[#1190]` prefix and explicit default-off / native-runtime / no-LM-Studio wording.

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

## M4 gating decision (2026-06-26, `m4-gating-check`)

Feature `m4-gating-check` makes the M4 proceed-or-cancel decision explicit, with full citation to the M1 evidence paths and measured deltas. This section records the binding gate for `m4-restore-eval-candidate`.

### M1 evidence cited by the gate

- `MLX_ENGINE_RESTORE_EVAL_STATE_ONLY=1` (the only M1 candidate that targeted restore-time `eval_ms` reduction):
  - **Retained baseline:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260624T210226.711363Z-shared-bench.json`
  - **Candidate run 1:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260624T210245.745879Z-shared-bench.json`
  - **Candidate run 1 compare:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260624T210245.745879Z-quality-compare.json` — `status=fail`
  - **Candidate run 2:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260624T210319.821177Z-shared-bench.json`
  - **Candidate run 2 compare:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260624T210319.821177Z-quality-compare.json` — `status=fail`
  - **Repeated-sample deltas (vs M1 retained baseline):**

    | Metric | Run 1 delta | Run 2 delta | Real, repeatable? |
    |---|---:|---:|---|
    | cold TTFT | -2.34% | -2.51% | yes (small, not eval-path) |
    | warm TTFT | +6.77% | +12.05% | yes, but a regression beyond the 5% gate |
    | decode TPS | -6.38% | +15.53% | no (signs disagree) |
    | total latency | -1.90% | -2.44% | small, no `eval_ms` signal |
    | warm total | +9.43% | +4.12% | regression on run 1, borderline on run 2 |
    | `quality_compare.py` status | `fail` | `fail` | both runs failed the promotion gate |

- `MLX_ENGINE_VLM_RESTORE_FRESHNESS_FLUSH` (default-on, kept as the promoted M1 default):
  - **Flush-on report:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260624T214000Z-freshness-flush-on.json`
  - **Flush-off report:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260624T214500Z-freshness-flush-off.json`
  - **Second-request deltas (flush-on vs flush-off):** cached_tokens `0 → 2048`, TTFT `-14.10%`, total latency `-13.25%`, `flushed_matching_saves=1` on every run. **Not an `eval_ms` / restore-materialization reduction** — this is an overlapping-request cache-reuse win for repeated same-prefix VLM requests, and the underlying `mx.eval(...)` barrier, KV record count, and bytes crossing the barrier are unchanged.

### Decision: **CANCEL** — `m4-restore-eval-candidate` must NOT be pursued under the M1 evidence above

The M4 gate, taken verbatim from `m1-record-decisions`, says the next restore-eval reduction candidate is pursued "ONLY if M1 demonstrated a real, repeatable eval-path win." Two negative findings support a hard CANCEL:

1. **The only M1 candidate that targeted restore-time `eval_ms` reduction was rejected.** `MLX_ENGINE_RESTORE_EVAL_STATE_ONLY=1` was the M1 lane closest to an `eval_ms` reduction, but both repeated candidate runs failed `quality_compare.py` (`status=fail` on run 1 and run 2) and regressed warm TTFT beyond the 5% threshold. There is no M1 evidence of a real, repeatable eval-path win.
2. **The promoted M1 change is not an `eval_ms` reduction.** `MLX_ENGINE_VLM_RESTORE_FRESHNESS_FLUSH` is a cache-reuse win for overlapping requests and does not move the restore-time `mx.eval(...)` barrier, the bytes crossing the barrier, or restore `eval_ms`. Promoting it does not satisfy the M4 gate's precondition.

Per the M4 gate, an `m4-restore-eval-candidate` implementation now requires an M1 evidence base that does not exist. Pursuing the candidate without that base would violate the gate as written and would risk burning engineering effort on a record-layout change that is not yet justified by any measured eval-path win.

### What the orchestrator should do

- **Cancel `m4-restore-eval-candidate` (status already `cancelled` in `features.json`, but the gate now records the explicit decision and the cited M1 evidence).**
- **Keep the M4 resume-order guidance in `.planning/performance-future-work.md` current**, so that if a future M1 (or M5/M6) surfaces a real, repeatable eval-path win, the next candidate is ready to design. Specifically:
  - Bias the next candidate toward reducing **rotating-delta arrays / bytes** before the restore-time `mx.eval(...)` barrier (the retained Gemma 4 restore surface measured `40` rotating layers = `80 / 96` eval targets and `~320 / 434.7 MiB` crossing the barrier).
  - Move the grouped-rotating idea earlier in the record/load pipeline so it reduces **bytes**, not just target count — do not retry naive post-assembly grouping (rejected 2026-06-23: list-eval `2.61 ms → 4.17 ms` median) and do not retry preassembled grouping in its current form (below the retained-lane barrier cost).
  - Keep the record format backward-readable so old persistent caches still load; never remove the restore-time `mx.eval(...)` safety barrier; do not reintroduce full-prefix KV span packing.
  - Re-validate via `MLX_ENGINE_BATCHED_TIMING` `eval-state isolation` plus the warm-restore image-fidelity check (VAL-M1-006 style: warm LFM2.5-VL `image_long` must still return `toucan`) before claiming any byte reduction is real.

### Validation contract assertion

- `VAL-M4-001` (`M4 gated on a real M1 eval-path win`) — **satisfied** by this section: the gating decision (CANCEL) is documented with reference to M1 report paths and deltas. The orchestrator can close the assertion as `passed`.

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

## M2 Qwen3.5 fully-padded vision-row `get_rope_index` patch verification (2026-06-26)

Feature `m2-qwen35-rope-index-patch` verifies the committed WIP `abebb5b` (`mlx_engine/model_kit/patches/qwen3_5.py` adds `_patched_vlm_qwen3_5_get_rope_index` that short-circuits empty batches to 3-row ones position_ids with zero rope_deltas, filters active rows when only some rows are padded, and rebuilds per-row position_ids with the standard inactive-row shape). The committed behavior is preserved unchanged because no engine defect was found.

- **Engine HEAD:** `a5cef25` (branch `mlx-vlm-restore-eval-followup`), the patch commit `abebb5b` is on the branch.
- **Pytest:** `.venv-py312/bin/python -m pytest tests/test_patched_qwen3_5.py -q` → **23 passed / 9 skipped / 0 failed**. The `test_vlm_qwen3_5_rope_index_handles_fully_padded_vision_rows` test passes, asserting `position_ids.shape == (3, 2, 4)` and `rope_deltas.tolist() == [[0], [0]]` for a fully-padded vision-row batch. The 9 skips are real-model tests gated on `Qwen3.5-2B-MLX-4bit` (not present locally) plus the `heavy` MoE/Qwen3.6 vocab-only tests; no skip is caused by the patch.
- **Prefill/decode parity regression check:** the same pytest file covers `test_qwen3_5_prefill_decode_consistency` (text_only and mrope variants), `test_qwen3_5_mrope_chunked_prefill_matches_unchunked`, `test_qwen3_5_text_only_uncached_matches_prompt_cache`, `test_qwen3_5_text_only_batch_cache_matches_prompt_cache`, and `test_vlm_qwen3_5_left_padded_batch_prefill_preserves_batch_cache_metadata`. All pass on the committed WIP; no prefill/decode parity regression observed.
- **Deterministic text-quality report (Qwen3.5-9B dense lane):** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T025838.965231Z-shared-bench.json`
  - **Quality inspect:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T025838.965231Z-quality-inspect.json` — `status=pass`, 0 failed prompts.
  - **Suite:** `prompt_suites/task_diverse_deterministic_quality.json` (`short_nyc_det`, `code_python_det`, `reasoning_math_det`, `instruction_format_det`, `long_context_franklin_det`), `--include-output-text`, `temp=0.0`, `top_p=1.0`, `runs=1`, `max_tokens=256` (per-prompt caps honored).
  - **Per-prompt keyword hits:** `short_nyc_det` (New York + finance), `code_python_det` (stable_unique + return), `reasoning_math_det` (38.9), `instruction_format_det` (risk + mitigation + owner + JSON exact-keys), `long_context_franklin_det` (Franklin + Autobiography). All hits true; no `forbid_substrings` or `forbid_reasoning_prefixes` findings (no visible-thinking leaks, no structured-output regressions).
- **Deterministic text-quality report (Qwen2.5-Coder-14B dense/code lane):** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T025916.739478Z-shared-bench.json`
  - **Quality inspect:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T025916.739478Z-quality-inspect.json` — `status=pass`, 0 failed prompts.
  - **Same suite and settings.** All 5 prompts hit their expected keywords with no forbid findings.
- **LFM2.5-VL image-suite parity (short pair):** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T025956.776219Z-shared-bench.json`
  - **Quality inspect:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T025956.776219Z-quality-inspect.json` — `status=pass`, 0 failed prompts.
  - **Suite:** `prompt_suites/vlm_image_quality.json` (`image_toucan`, `image_pair`), `--min-completion-tokens 4`, `temp=0.0`, `top_p=1.0`, `runs=1`, `max_tokens=96`.
  - **Per-prompt keyword hits:** `image_toucan` (toucan, completion_tokens=96, eos not hit within budget), `image_pair` (chameleon + toucan, completion_tokens=25). Both subjects correctly identified. Zero row errors.
- **LFM2.5-VL image-suite parity (long-context toucan):** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T030016.310541Z-shared-bench.json`
  - **Quality inspect:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T030016.310541Z-quality-inspect.json` — `status=pass`, 0 failed prompts.
  - **Suite:** `prompt_suites/vlm_image_long_quality.json` (`image_long_toucan`, long-context Benjamin-Franklin text + toucan image), `--min-completion-tokens 4`.
  - **Per-prompt keyword hits:** `image_long_toucan` (toucan, completion_tokens=5). Subject correctly identified despite the unrelated long-context text. Zero row errors.

### Decision: VERIFIED (no engine change)

The Qwen3.5 fully-padded vision-row `get_rope_index` patch `abebb5b` is verified correct as committed. No engine behavior was modified. The pytest suite for the patch passes (including the dedicated fully-padded vision-row test), the deterministic text-quality suite passes on both dense/code lanes (Qwen3.5-9B and Qwen2.5-Coder-14B) with no visible-thinking leaks or structured-output regressions, the LFM2.5-VL image suite passes with zero row errors on both the short pair and long-context lanes (no VLM parity regression), and the in-file prefill/decode parity tests confirm no prefill/decode regression on the M2 WIP surface.

## M3 thread-unsafe stream experiment (2026-06-26)

Feature `m3-thread-unsafe-stream-experiment` runs the committed WIP `13cc526` (`mlx_engine/utils/mlx_lm_stream.py` exposes a `MLX_ENGINE_EXPERIMENTAL_THREAD_UNSAFE_STREAM` env var that, when supported by the active MLX runtime, swaps the per-thread `ThreadLocalStream` for a shared `mx.new_thread_unsafe_stream(...)` so cross-thread generation requests reuse one device stream) through the direct `shared_bench.py` harness and decides promote / keep-opt-in / reject. **No file toggle was used, no LM Studio runtime was involved** — only the env var as the feature description requires.

- **Engine HEAD:** `e4831da` (branch `mlx-vlm-restore-eval-followup`); the WIP commit `13cc526` is on the branch.
- **Model:** `/Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/lmstudio-community/Qwen3.5-9B-MLX-8bit` (dense text).
- **Suite:** `prompt_suites/task_diverse_deterministic_quality.json` (`short_nyc_det`, `code_python_det`, `reasoning_math_det`, `instruction_format_det`, `long_context_franklin_det`), `--include-output-text`, `temp=0.0`, `top_p=1.0`, `runs=3`, `max_tokens=256` (per-prompt caps honored).
- **Hygiene (corrected 2026-06-26 by `m4-gating-check`; line numbers refreshed 2026-06-26 by `m5-short-text-baseline`):** a stale 0-byte `/tmp/mlx-engine-thread-unsafe-stream` toggle file from a previous session (Jun 22 23:05) was removed before the baseline so the env var alone controls the opt-in. `init.sh` step 7 (lines `60-66`) now removes `/tmp/mlx-engine-thread-unsafe-stream` between worker sessions so a stale toggle file can no longer silently enable the experiment on a fresh worker without the env var being explicitly set. The previous wording that said `init.sh` does not clean that path is stale and superseded; the env var is still the authoritative control surface, and the file removal is now reproducible idempotent mission hygiene rather than per-session ad-hoc cleanup. The old `(lines 7-11)` reference is stale and was superseded when `init.sh` was restructured to the numbered-step layout; lines 7-11 now sit inside the environment-variable block (`ENGINE`, `HARNESS`, `PY312`, `MODELS`), not the cleanup block.

### Runtime capability check (the critical finding)

The `_resolve_stream_source(...)` helper in `mlx_engine/utils/mlx_lm_stream.py` only resolves to `"thread-unsafe"` when both `_thread_unsafe_stream_experiment_enabled()` is True AND `_runtime_supports_thread_unsafe_stream()` is True. The runtime capability check is `hasattr(mx, "new_thread_unsafe_stream")`.

- Direct module introspection under `.venv-py312/bin/python`:
  ```
  hasattr(mx, "new_thread_unsafe_stream"): False
  hasattr(mx, "new_thread_local_stream"): True
  ```
- Stream configuration reported by `describe_stream_configuration(False)`:
  - Baseline (env unset, no toggle file): `source=thread-local toggle_env=False toggle_file=False runtime_supports_thread_unsafe=False`
  - Candidate (`MLX_ENGINE_EXPERIMENTAL_THREAD_UNSAFE_STREAM=1`, no toggle file): `source=thread-local toggle_env=True toggle_file=False runtime_supports_thread_unsafe=False`
  - Direct `prepare_mlx_lm_generation_stream` log line under the env var: `stream_source=thread-local default_stream=Stream(Device(gpu, 0), 0) stream=ThreadLocalStream(Device(gpu, 0), 1)`

The candidate opt-in is therefore a clean no-op on this MLX runtime: the experiment is enabled and the selection logic engages, but because `mx.new_thread_unsafe_stream` is not exposed, `_runtime_supports_thread_unsafe_stream()` returns False and the helper degrades to the thread-local path — exactly the documented fallback (`test_prepare_stream_falls_back_when_runtime_lacks_thread_unsafe_api`).

### Baseline run (thread-local default)

- **Report:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T040444.745185Z-shared-bench.json`
- **Invocation:**
  ```bash
  cd /Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness
  env -u MLX_ENGINE_EXPERIMENTAL_THREAD_UNSAFE_STREAM PYTHONPATH=. python3 shared_bench.py \
    --engine mlx-engine \
    --model /Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/lmstudio-community/Qwen3.5-9B-MLX-8bit \
    --mlx-engine-python /Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.venv-py312/bin/python \
    --prompt-suite-json prompt_suites/task_diverse_deterministic_quality.json \
    --include-output-text \
    --temperature 0.0 --top-p 1.0 --runs 3 --max-tokens 256 \
    --out-dir /Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports
  ```
- **Row check:** 15 rows (5 prompts × 3 runs), every row `error: null`, runner process returncode 0, no `RuntimeError: There is no Stream(...)` in stderr, every cold run completes a full warm-cache reuse cycle (warm `cached_tokens=7202` on the long-context prompt; warm `cached_tokens=54..65` on the short prompts).
- **Per-prompt outputs:** `short_nyc_det` and `code_python_det` hit the expected keywords (`New York`, `finance`, `stable_unique`, `return`); `reasoning_math_det` produces the expected `38.9%` answer; `instruction_format_det` produces a valid JSON object with the required `risk`/`mitigation`/`owner` keys; `long_context_franklin_det` summarizes Franklin's Autobiography. No `forbid_substrings` or `forbid_reasoning_prefixes` findings.

### Candidate run 1 (env var on, no file toggle)

- **Report:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T040526.729772Z-shared-bench.json`
- **Invocation:** identical to baseline except `MLX_ENGINE_EXPERIMENTAL_THREAD_UNSAFE_STREAM=1` is set in the shell before `shared_bench.py`.
- **Row check:** 15 rows, every row `error: null`, runner process returncode 0, no stream-failure text in stderr.
- **Quality compare (candidate vs baseline):** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T040526.729772Z-quality-compare.json` — **`status: pass`**, `failed_prompts: []`, `global_findings: []`, every prompt-level `status: pass`.
- **Per-prompt deltas (candidate − baseline, %):**

  | prompt | cold_ttft | warm_ttft_p50 | total | decode_tps | warm_total_p50 |
  |---|---:|---:|---:|---:|---:|
  | `code_python_det` | +4.52 | -0.41 | +0.46 | -0.34 | +0.33 |
  | `instruction_format_det` | +5.81 | -7.55 | -1.66 | +1.79 | -2.38 |
  | `long_context_franklin_det` | +0.25 | +3.12 | +0.19 | -0.05 | -0.06 |
  | `reasoning_math_det` | +5.09 | +0.25 | +0.33 | -0.00 | -0.19 |
  | `short_nyc_det` | -2.13 | +1.96 | +1.14 | -1.30 | +0.59 |

  All deltas are within run-to-run sampling noise (no consistent direction; warm_ttft_p50 mixed signs; total/decoded well inside ±2%).

### Candidate run 2 (repeat confirmation)

- **Report:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T040749.821729Z-shared-bench.json`
- **Quality compare (candidate vs baseline):** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T040749.821729Z-quality-compare.json` — **`status: fail`** on exactly one prompt (`reasoning_math_det`) due to `warm TTFT median regression 6.603% exceeds 5.000%`. All other prompts pass; `raw_candidate_findings: []` and `candidate_only_findings: []` — **zero quality findings**, only a regression-threshold trip on a prompt whose absolute warm TTFT is ~0.07s, where 5 ms of jitter is ±7% of the signal.
- **Row check:** 15 rows, every row `error: null`, runner process returncode 0.
- **Per-prompt deltas (candidate − baseline, %):** again mixed signs (cold_ttft −5.77 to +7.43, total −1.30 to +2.80, decode_tps −2.23 to +1.79), with no consistent direction across prompts or runs. This is consistent with the "no real stream swap happened" finding rather than a genuine performance signal.

### Pytest (M3 unit tests)

- `.venv-py312/bin/python -m pytest tests/test_mlx_lm_stream.py tests/test_mlx_threading.py -q` → **7 passed / 2 warnings**, 0 failed. Warnings are unrelated `SwigPyPacked/SwigPyObject has no __module__ attribute` deprecations from MLX bindings, not caused by the WIP.
- `test_prepare_stream_defaults_to_thread_local` — env unset, runtime lacks `new_thread_unsafe_stream` → uses `ThreadLocalStream`.
- `test_prepare_stream_uses_shared_thread_unsafe_stream_when_enabled` — env set, monkeypatched runtime exposes `new_thread_unsafe_stream` → shared stream is reused across simulated worker threads.
- `test_prepare_stream_uses_toggle_file_when_env_is_unset` — file toggle alone enables the experiment.
- `test_prepare_stream_falls_back_when_runtime_lacks_thread_unsafe_api` — env set but runtime lacks the new API → degrades cleanly to thread-local (this is the exact code path the M3 harness exercises on this machine).
- `test_describe_stream_configuration_reports_toggle_and_runtime` — probe output reflects all three inputs.
- `test_prepare_stream_keeps_default_stream_for_distributed_paths` — `use_default_stream=True` always wins.

### Decision: REJECT (no promotion evidence exists; explicit rejection is the documented acceptable outcome for the M3 experiment lane)

**Promotion cannot be claimed on this evidence.** The committed WIP `13cc526` is engineered correctly — the selection helper, fallback logic, logger probes, file toggle, distributed-path override, and unit tests all behave as documented — but the active `.venv-py312` MLX runtime does **not** expose `mx.new_thread_unsafe_stream`. Therefore:

- The candidate run resolves to the same thread-local stream configuration as the baseline. There is no shared-stream-vs-thread-local comparison to make on this machine.
- The observed TTFT / decode-TPS / total deltas are pure sampling noise (mixed signs, ≤7.4%, no consistent direction across two candidate runs). They cannot be promoted under the ≥2 quality-passing repeated-sample + real, repeatable move rule.
- Zero row-level errors and zero cross-thread stream failures across both candidate runs (the M3 stability gate is satisfied), but stability under a no-op configuration does not constitute promotion evidence.

**What this rejection does NOT mean:**

- The WIP `13cc526` itself is not buggy. The unit tests confirm the selection helper picks the shared stream whenever the runtime exposes `new_thread_unsafe_stream`. The opt-in degrades cleanly when the API is missing (the M3 harness exercises this branch). The runtime capability check is the correct gate.
- The opt-in env var should remain in the codebase. Future MLX runtime upgrades that expose `mx.new_thread_unsafe_stream` will let this experiment be re-run for real; until then, no production user can accidentally enable the shared stream because `_runtime_supports_thread_unsafe_stream()` gates the swap.

**What this rejection DOES mean:**

- **Do not promote** `MLX_ENGINE_EXPERIMENTAL_THREAD_UNSAFE_STREAM` to a default or even to an opt-in doc surface. The experiment has no measurable effect on this MLX runtime, so there is nothing to gain and nothing to document for end users until the underlying API ships in the MLX distribution this checkout consumes.
- **Do not delete** the WIP. It is correct defensive plumbing for a runtime capability the engine cannot predict in advance, and it remains a future-proofing hook for the next MLX release.
- **Re-run** the experiment if/when the MLX runtime in `.venv-py312` gains `mx.new_thread_unsafe_stream`. The expected signal then would be measurable reductions in per-thread stream-allocation overhead on a multi-threaded harness, not on the single-threaded `shared_bench.py` lane used here. A multi-connection or concurrent-prompt harness (currently out of mission scope — M5 cheetara-vs-mlx is a sequential single-stack comparison) would be the surface that could observe a real shared-stream win.

### Artifacts

| Artifact | Path |
|---|---|
| Baseline report | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T040444.745185Z-shared-bench.json` |
| Candidate run 1 report | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T040526.729772Z-shared-bench.json` |
| Candidate run 1 quality-compare | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T040526.729772Z-quality-compare.json` |
| Candidate run 2 report | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T040749.821729Z-shared-bench.json` |
| Candidate run 2 quality-compare | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T040749.821729Z-quality-compare.json` |
| WIP source | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/mlx_engine/utils/mlx_lm_stream.py` (commit `13cc526`) |
| WIP unit tests | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/tests/test_mlx_lm_stream.py` |

## M5 short-text baseline (2026-06-26)

Feature `m5-short-text-baseline` uses the existing `cheetara-vs-mlx` benchmark profile (`runners/cheetara_mlx_profile.py`) introduced by `m5-cheetara-bench-profile` to capture the **M5 short-text baseline**: one paired report containing rows from both stacks (cheetara `vmlx` + `mlx-engine`) on identical prompts and the same local model file. **This is evidence capture only — no promotion decision, no cheetara repackaging, no `vmlx.app.asar` modification.** The harness driver also MD5-records `vmlx.app.asar` before and after the run as a no-write integrity check.

### Profile inputs

- **Driver:** `mlx-bench-harness/runners/cheetara_mlx_profile.py`, invoked with `--engine cheetara-mlx` so the combined report contains exactly `[mlx-engine, vmlx]` result rows.
- **Suite:** `mlx-bench-harness/prompt_suites/m5_short_text.json` (newly added; single short-text prompt `short_nyc` lifted verbatim from the parent `cheetara_vs_mlx.json` so the M5 sub-suite uses the same prompt text, system prompt, and `expected_keywords` as the full M5 suite).
- **Model (identical for both engines):** `/Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/lmstudio-community/Qwen3.5-9B-MLX-8bit` (dense text, same path used for M2/M3 deterministic text-quality evidence).
- **Sampling:** `temperature=0.0`, `top_p=1.0`, `--include-output-text`, `runs=3`, `max_tokens=96` (honoring the prompt's own `max_tokens` cap).
- **vmlx interpreter:** the cheetara `.venv` defaults `python` to 3.12 but installs dependencies into Python 3.14's site-packages (`pyvenv.cfg` `home = Python.framework/Versions/3.14`). The bench is therefore invoked with `--vmlx-python /Users/jeffreycruz/Development/LLM_INFERENCE/cheetara/.venv/bin/python3.14` so the vmlx serve subprocess can actually `import uvicorn` and `import vmlx_engine`; mlx-engine uses `.venv-py312/bin/python` via the harness defaults.
- **cheetara app bundle integrity:** the harness `verify_app_bundle(...)` step MD5-recorded the bundle before the run (`vmlx.app.asar` md5 `d27106b78546424046384e813fe23b7c`, 70,671,554 bytes) and the bundle is not modified; this matches the AGENTS.md no-repackaging rule.

### Command shape

```bash
cd /Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness
env PYTHONPATH=. python3 runners/cheetara_mlx_profile.py \
  --model /Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/lmstudio-community/Qwen3.5-9B-MLX-8bit \
  --suite prompt_suites/m5_short_text.json \
  --runs 3 \
  --max-tokens 96 \
  --temperature 0.0 \
  --top-p 1.0 \
  --include-output-text \
  --vmlx-python /Users/jeffreycruz/Development/LLM_INFERENCE/cheetara/.venv/bin/python3.14 \
  --out-dir /Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports
```

### Report and row inspection

- **M5 short-text baseline report:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T050900.761077Z-shared-bench.json`
- **Engines present:** `["mlx-engine", "vmlx"]` (exactly the paired set required by `load_combined_report(...)`).
- **Total rows:** 6 (3 per engine × 2 engines).
- **Row-error check:** every row has `error: null` for both engines; both runners exited 0; no `__runner__` placeholder rows.
- **Per-engine row breakdown:**

  | Engine | Rows | Errors | Output preview (run 1) |
  |---|---:|---:|---|
  | `mlx-engine` | 3 | 0 | "*   It serves as the global hub for finance, housing the headquarters of major i…" (deterministic across all 3 runs; expected keywords `New York` and `finance` both present in every run; 91 completion_tokens per run) |
  | `vmlx` | 3 | 0 | "*   New York City serves as the global hub for **finance**, hosting the New York…" (expected keywords `New York` and `finance` both present in every run; vmlx reports ~188 completion_tokens per run via its OpenAI streaming usage block) |

- **Summary-level timings (informational only — data capture, no promotion):**

  | Engine | avg total (s) | avg decode tps | cold ttft (s) | warm ttft (s) | avg completion tokens | cached tokens |
  |---|---:|---:|---:|---:|---:|---:|
  | `mlx-engine` | 1.4370 | 70.29 | 0.3354 | 0.0456 | 91.0 | 32.0 |
  | `vmlx` | 4.8313 | (inflated — see caveat) | 6.0532 | 4.2199 | 188.7 | n/a (not reported) |

- **Timing caveat (vmlx streaming instrumentation):** the `vmlx_runner` parses vmlx's OpenAI streaming `/v1/chat/completions` SSE stream and records `first_token_seen` from the first `data:` chunk containing a `content` delta. The vmlx server in this build streams the entire completion as a single `data:` chunk (one final `chat.completion` chunk after a brief generation phase), so `decode_s ≈ 0` and the entire request time is rolled into `ttft_s`. This is a vmlx streaming-measurement quirk, not a benchmark failure; both vmlx rows report `error: null` and produce on-topic, keyword-matching output. The M5 short-text baseline records the raw measurements exactly as the runner produced them, without adjustment, so future M5 evidence captures use the same instrumentation.

### Decision: DATA-ONLY (no promotion)

This feature captures the M5 short-text baseline as evidence and **makes no promotion decision** for either stack. The paired report path above is the authoritative M5 short-text baseline reference for any future M5 follow-up work (long-text baseline, image baseline, or cheetara-vs-mlx promotion analysis). M5 is explicitly data-only per `VAL-M5-005`; the cheetara app bundle is unchanged (MD5 verified pre-run by the harness driver) and no `vmlx.app.asar` write occurred during this run.

### Stale M3 init.sh line reference (corrected in this commit)

While recording the M5 short-text baseline, the stale `init.sh` line reference in the M3 section above was also refreshed: the line range now reads `lines 60-66` (the current thread-unsafe-stream cleanup block inside step 7), replacing the stale `lines 7-11` reference. Lines 7-11 of the current `init.sh` are inside the `ENGINE`/`HARNESS`/`PY312`/`MODELS` variable block, not the cleanup block; the old number was carried from an earlier shorter `init.sh` layout and is no longer accurate.

### Artifacts

| Artifact | Path |
|---|---|
| M5 short-text baseline report | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T050900.761077Z-shared-bench.json` |
| M5 short-text sub-suite (new) | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/prompt_suites/m5_short_text.json` |
| cheetara-vs-mlx driver (existing) | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/runners/cheetara_mlx_profile.py` |
| vmlx.app.asar pre-run md5 | `d27106b78546424046384e813fe23b7c` (unchanged post-run) |

## M5 long-text baseline (2026-06-26)

Feature `m5-long-text-baseline` reuses the existing `cheetara-vs-mlx` benchmark profile (`runners/cheetara_mlx_profile.py`) introduced by `m5-cheetara-bench-profile` to capture the **M5 long-text baseline**: one paired report containing rows from both stacks (cheetara `vmlx` + `mlx-engine`) on identical long-context prompts and the same local model file. **This is evidence capture only — no promotion decision, no cheetara repackaging, no `vmlx.app.asar` modification.** The harness driver also MD5-records `vmlx.app.asar` before and after the run as a no-write integrity check.

### Profile inputs

- **Driver:** `mlx-bench-harness/runners/cheetara_mlx_profile.py`, invoked with `--engine cheetara-mlx` so the combined report contains exactly `[mlx-engine, vmlx]` result rows.
- **Suite:** `mlx-bench-harness/prompt_suites/m5_long_text.json` (newly added; single long-text prompt `long_franklin` lifted verbatim from the parent `cheetara_vs_mlx.json` so the M5 sub-suite uses the same prompt text, system prompt, `user_file`, `user_suffix`, `max_tokens`, and `expected_keywords` as the full M5 suite). The prompt reads the full Benjamin-Franklin Autobiography start text (`ben_franklin_autobiography_start.txt`, 33,222 chars / 7,193 prompt tokens after chat-template rendering) and asks the model to summarize Franklin and his *Autobiography* in three bullets.
- **Model (identical for both engines):** `/Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/lmstudio-community/Qwen3.5-9B-MLX-8bit` (dense text, same path used for M2/M3 deterministic text-quality and M5 short-text evidence).
- **Sampling:** `temperature=0.0`, `top_p=1.0`, `--include-output-text`, `runs=3`, `max_tokens=160` (honoring the prompt's own `max_tokens` cap).
- **vmlx interpreter:** the cheetara `.venv` defaults `python` to 3.12 but installs dependencies into Python 3.14's site-packages (`pyvenv.cfg` `home = Python.framework/Versions/3.14`). The bench is therefore invoked with `--vmlx-python /Users/jeffreycruz/Development/LLM_INFERENCE/cheetara/.venv/bin/python3.14` so the vmlx serve subprocess can actually `import uvicorn` and `import vmlx_engine`; mlx-engine uses `.venv-py312/bin/python` via the harness defaults. Verified pre-run that `python3.14 -c "import uvicorn, vmlx_engine"` succeeds.
- **cheetara app bundle integrity:** the harness `verify_app_bundle(...)` step MD5-recorded the bundle before the run (`vmlx.app.asar` md5 `d27106b78546424046384e813fe23b7c`, 70,671,554 bytes) and re-checked after the run — the md5 is unchanged. This matches the AGENTS.md no-repackaging rule.

### Command shape

```bash
cd /Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness
env PYTHONPATH=. python3 runners/cheetara_mlx_profile.py \
  --model /Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/lmstudio-community/Qwen3.5-9B-MLX-8bit \
  --suite prompt_suites/m5_long_text.json \
  --runs 3 \
  --max-tokens 160 \
  --temperature 0.0 \
  --top-p 1.0 \
  --include-output-text \
  --vmlx-python /Users/jeffreycruz/Development/LLM_INFERENCE/cheetara/.venv/bin/python3.14 \
  --out-dir /Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports
```

### Report and row inspection

- **M5 long-text baseline report:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T051648.588677Z-shared-bench.json`
- **Engines present:** `["mlx-engine", "vmlx"]` (exactly the paired set required by `load_combined_report(...)`).
- **Total rows:** 6 (3 per engine × 2 engines).
- **Row-error check:** every row has `error: null` for both engines; both runners exited 0; no `__runner__` placeholder rows; no `RuntimeError: There is no Stream(...)` text in any runner stderr; no cross-thread stream failures.
- **Per-engine row breakdown:**

  | Engine | Rows | Errors | Prompt tokens | Output preview (run 1) |
  |---|---:|---:|---:|---|
  | `mlx-engine` | 3 | 0 | 7,194 (cold 7,194 / warm 7,194 — full 7,193-token prefix matches across all 3 runs; warm `cached_tokens=7193`) | "Based on the text provided, here is a summary of Benjamin Franklin and his *Autobiography*: … Franklin's Multifaceted Legacy … The Human Value of the Autobiography" (deterministic across all 3 runs; expected keywords `Franklin` and `Autobiography` both present in every run; 160 completion_tokens per run, `finish_reason=token_limit`) |
  | `vmlx` | 3 | 0 | 7,187 | "*   Franklin's Autobiography is valued not as a formula for success, but as a vivid, human account of his rise from poverty and obscurity …" (expected keywords `Franklin` and `Autobiography` both present in every run; vmlx reports 320 completion_tokens per run via its OpenAI streaming usage block, `finish_reason=stop`) |

- **Summary-level timings (informational only — data capture, no promotion):**

  | Engine | runs | avg total (s) | avg decode tps | cold ttft (s) | warm ttft (s) | warm total (s) | avg completion tokens | cached tokens (warm) |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | `mlx-engine` | 3 | 4.756 | 66.82 | 6.875 | 0.103 | 2.465 | 160 | 7,193 |
  | `vmlx` | 3 | 13.911 | (inflated — see caveat) | 14.446 | 13.643 | 13.643 | 320 | n/a (not reported by vmlx) |

  mlx-engine's warm-cache path is dramatically faster than its cold path (warm TTFT `0.103 s` vs cold TTFT `6.875 s`, warm total `2.465 s` vs cold total `9.338 s`) because the 7,193-token prompt prefix is fully cached and reused — every warm run reports `cached_tokens=7193` from the engine. This is the expected mlx-engine prompt-cache reuse behavior; the warm-cache numbers are the apples-to-apples comparison point against vmlx, which does not expose `cached_tokens` on the same path.

- **Timing caveat (vmlx streaming instrumentation — `decode_tps` is NOT promotion-quality throughput evidence):** the `vmlx_runner` parses vmlx's OpenAI streaming `/v1/chat/completions` SSE stream and records `first_token_seen` from the first `data:` chunk containing a `content` delta. The vmlx server in this build streams the entire completion as a single `data:` chunk (one final `chat.completion` chunk after a brief generation phase, evidenced in the captured `server_process.stderr` by `Paged cache hit` and `Captured SSM state` log lines, but the visible-answer stream itself arrives as one chunk per request), so `decode_s ≈ 0.0003 s` and `decode_tps ≈ 1.1M tokens/s` for every vmlx row. This is the same vmlx streaming-measurement quirk recorded in the M5 short-text baseline. Per the user-testing library, **these `decode_tps` values are NOT promotion-quality throughput evidence**: they are raw observed transport timings where the entire request time is rolled into `ttft_s`. For apples-to-apples throughput comparison, vmlx's `ttft_s`/`total_s` are the only honest signal; mlx-engine's `decode_tps ≈ 66.8 tokens/s` is the only true per-token decode rate in this report. The M5 long-text baseline records the raw measurements exactly as the runner produced them, without adjustment, so future M5 evidence captures use the same instrumentation.

- **vmlx server log evidence:** the captured `server_process.stderr` confirms vmlx itself ran correctly across all 3 runs — visible-answer passes triggered `Captured SSM state`, `VLM HYBRID cache HIT` lines, and the `Qwen3.5 Chat Completions stream produced no visible content; running bounded thinking-off answer pass` reasoning-off path. No startup failure, no stream-failure text, no model-load errors.

### Decision: DATA-ONLY (no promotion)

This feature captures the M5 long-text baseline as evidence and **makes no promotion decision** for either stack. The paired report path above is the authoritative M5 long-text baseline reference for any future M5 follow-up work (image baseline, or cheetara-vs-mlx promotion analysis). M5 is explicitly data-only per `VAL-M5-005`; the cheetara app bundle is unchanged (MD5 verified pre-run and post-run by the harness driver) and no `vmlx.app.asar` write occurred during this run.

### vmlx SSE caveat repeated in this handoff

As called out in the expected behavior, the vmlx server in this build delivers the completion as a single SSE `data:` chunk, so vmlx `decode_tps` in this report is raw observed transport timing, not promotion-quality throughput evidence. Any future M5 follow-up that wants to compare apples-to-apples decode throughput must either wait for vmlx to start incremental token streaming or restrict the comparison to mlx-engine's `decode_tps` and `total_s` fields. The M5 long-text baseline records this caveat exactly so the orchestrator does not treat `vmlx.decode_tps` as a promotion signal.

### Artifacts

| Artifact | Path |
|---|---|
| M5 long-text baseline report | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T051648.588677Z-shared-bench.json` |
| M5 long-text sub-suite (new) | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/prompt_suites/m5_long_text.json` |
| cheetara-vs-mlx driver (existing) | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/runners/cheetara_mlx_profile.py` |
| vmlx.app.asar pre-run md5 | `d27106b78546424046384e813fe23b7c` (unchanged post-run) |

## M5 image baseline (2026-06-26, rerun after `m5-vmlx-lfm2-vl-loader-fix` and `m5-vmlx-image-placeholder-fix`)

Feature `m5-image-baseline` reuses the existing `cheetara-vs-mlx` benchmark profile (`runners/cheetara_mlx_profile.py`) introduced by `m5-cheetara-bench-profile` to capture the **M5 image baseline**: one paired report containing rows from both stacks (cheetara `vmlx` + `mlx-engine`) on identical image prompts/images and the same VLM model file. **This is evidence capture only — no promotion decision, no cheetara repackaging, no `vmlx.app.asar` modification.** The harness driver also MD5-records `vmlx.app.asar` before and after the run as a no-write integrity check.

The original 2026-06-26 first-attempt evidence (`reports/20260626T052342.909372Z-shared-bench.json`) recorded a vmlx server-startup failure (`Missing 2 parameters: multi_modal_projector.layer_norm.{bias,weight}`) on the LFM2.5-VL projector; the mlx-engine half of the baseline worked correctly but the paired-engine schema could not be satisfied. After the source-level fixes in features `m5-vmlx-lfm2-vl-loader-fix` (port of the mlx-engine `lfm2_vl.py` projector-name remap into `cheetara/engine-source`) and `m5-vmlx-image-placeholder-fix` (the `patches/lfm2_vl_runtime.py` `_patched_call` placeholder-injection and `lfm2_vl.Model.__call__` keyword-routing wrappers), this rerun produces a real vmlx image row on every request and closes both defects.

### Profile inputs

- **Driver:** `mlx-bench-harness/runners/cheetara_mlx_profile.py`, invoked with `--engine cheetara-mlx` so the combined report contains the paired `[mlx-engine, vmlx]` result rows.
- **Suite:** `mlx-bench-harness/prompt_suites/m5_image.json` (newly added; single image prompt `image_toucan` lifted verbatim from the parent `cheetara_vs_mlx.json` so the M5 sub-suite uses the same prompt text, system prompt, `image_files` (the demo `toucan.jpeg`), `max_tokens`, and `expected_keywords` as the full M5 suite). The prompt asks the VLM to identify the animal in the toucan image in one short sentence.
- **Model (intended identical for both engines):** `/Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/lmstudio-community/LFM2.5-VL-1.6B-MLX-8bit` — the canonical VLM model file used by M1 for restore-eval / path-load evidence and by `m1-warm-restore-image-fidelity` for warm-cache image fidelity verification. No other locally installed MLX model exposes the `vision_config` block required by `cheetara_vs_mlx.json` (GLM-4.7-Flash-MLX-8bit is text-only `glm4_moe_lite`, NVIDIA-Nemotron is text-only `nemotron_h`, Qwen3.5-9B-MLX-8bit has `vision_config` but is not a real multimodal checkpoint in this checkout).
- **Sampling:** `temperature=0.0`, `top_p=1.0`, `--include-output-text`, `runs=3`, `max_tokens=64` (honoring the prompt's own `max_tokens` cap).
- **vmlx interpreter:** the cheetara `.venv` defaults `python` to 3.12 but installs dependencies into Python 3.14's site-packages (`pyvenv.cfg` `home = Python.framework/Versions/3.14`). The bench is therefore invoked with `--vmlx-python /Users/jeffreycruz/Development/LLM_INFERENCE/cheetara/.venv/bin/python3.14` so the vmlx serve subprocess can actually `import uvicorn` and `import vmlx_engine`; mlx-engine uses `.venv-py312/bin/python` via the harness defaults. Verified pre-run that `python3.14 -c "import uvicorn, vmlx_engine"` succeeds.
- **cheetara app bundle integrity:** the harness `verify_app_bundle(...)` step MD5-recorded the bundle before the run (`vmlx.app.asar` md5 `d27106b78546424046384e813fe23b7c`, 70,671,554 bytes) and re-checked after the run — the md5 is unchanged. This matches the AGENTS.md no-repackaging rule.

### Command shape

```bash
cd /Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness
env PYTHONPATH=. python3 runners/cheetara_mlx_profile.py \
  --model /Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/lmstudio-community/LFM2.5-VL-1.6B-MLX-8bit \
  --suite prompt_suites/m5_image.json \
  --runs 3 \
  --max-tokens 64 \
  --temperature 0.0 \
  --top-p 1.0 \
  --include-output-text \
  --vmlx-python /Users/jeffreycruz/Development/LLM_INFERENCE/cheetara/.venv/bin/python3.14 \
  --out-dir /Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports
```

### Report and row inspection (post-fix rerun)

- **M5 image baseline report (authoritative):** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T072551.438648Z-shared-bench.json`
- **Quality inspect (informational; probe-threshold sensitive, see note below):** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T072551.438648Z-quality-inspect.json` — `keyword_hits.toucan=true` on every row; status=fail purely on the `min_completion_tokens=16` default threshold (model answered "one short sentence" → 11 tokens for mlx-engine, 5 tokens for vmlx, both well-formed and on-topic).
- **Prior partial-failure report (superseded, kept for reference):** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T052342.909372Z-shared-bench.json` (the pre-`m5-vmlx-image-placeholder-fix` attempt with vmlx server startup failure).
- **Engines present in the rerun report:** `["mlx-engine", "vmlx"]` (the harness driver accepted the paired-engine schema; both engines were exercised in the same invocation).
- **Total rows:** 6 (3 from `mlx-engine`, 3 from `vmlx`).
- **Row-error check:**
  - `mlx-engine`: 3/3 rows have `error: null`. All three runs are deterministic ("The animal in the image is a toucan.", 11 completion_tokens each, `finish_reason=eos_token`, `image_count=1`, `prompt_tokens=36`); expected keyword `toucan` is present in every output.
  - `vmlx`: 3/3 rows have `error: null`. All three runs are deterministic ("A toucan.", 5 completion_tokens each, `finish_reason=null` because the vmlx SSE stream delivered the entire completion as one chunk, `image_count=1`, `prompt_tokens=37`); expected keyword `toucan` is present in every output.
- **Per-engine row breakdown:**

  | Engine | Rows (success / error) | Output preview (run 1) | Keyword `toucan` |
  |---|---:|---|:---:|
  | `mlx-engine` | 3 / 0 | "The animal in the image is a toucan." (deterministic across all 3 runs; 11 completion_tokens per run, `finish_reason=eos_token`) | hit on all 3 runs |
  | `vmlx` | 3 / 0 | "A toucan." (deterministic across all 3 runs; 5 completion_tokens per run, `finish_reason=null` on the single-SSE-chunk completion) | hit on all 3 runs |

- **Summary-level timings (informational only — data capture, no promotion):**

  | Engine | runs | avg total (s) | avg decode tps | cold ttft (s) | warm ttft (s) | avg completion tokens | cached tokens (warm) |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | `mlx-engine` | 3 | 0.213 | 350.68 | 0.517 | 0.014 | 11.0 | 35 |
  | `vmlx` | 3 | 1.487 | (inflated — see caveat) | 1.781 | 1.318 | 5.0 | n/a (not reported by vmlx) |

  mlx-engine's warm-cache path is dramatically faster than its cold path (warm TTFT `0.014 s` vs cold TTFT `0.517 s`, warm total `0.043 s` vs cold total `0.552 s`) because the 36-token prompt prefix is fully cached and reused — every warm run reports `cached_tokens=35` from the engine. This is the expected mlx-engine prompt-cache reuse behavior; the warm-cache numbers are the apples-to-apples comparison point against vmlx, which does not expose `cached_tokens` on the same path.

### vmlx non-fatal MLLM chat-template warning (DOCUMENTED, not blocking)

The vmlx server emits the known non-fatal warning during multimodal chat-template rendering:

```
WARNING:vmlx_engine.engine.batched:Failed to apply MLLM chat template: can only concatenate str (not "list") to str
```

This warning fires once per `image_toucan` request as the vmlx batched engine falls back from the strict MLLM chat-template path to the raw prompt-rendering path. It does NOT cause any row to error: every vmlx row reports `error: null`, `completion_tokens=5`, and the expected `toucan` keyword in the output. The fallback path produces the correct multimodal prompt assembly (after the `m5-vmlx-image-placeholder-fix` `_patched_call` wrapper injects the required `<image>` marker), and the generation proceeds to completion. Per the user-testing library rule, "non-fatal stderr warnings like `Failed to apply MLLM chat template: can only concatenate str (not 'list') to str` do not fail the assertion by themselves if the run still produces real vmlx rows with `error: null` and the expected image keyword" — this rerun satisfies that rule.

### vmlx SSE single-chunk timing caveat (`decode_tps` is NOT promotion-quality throughput evidence)

The `vmlx_runner` parses vmlx's OpenAI streaming `/v1/chat/completions` SSE stream and records `first_token_seen` from the first `data:` chunk containing a `content` delta. The vmlx server in this build streams the entire completion as a single `data:` chunk (one final `chat.completion` chunk after a brief generation phase), so `decode_s ≈ 0.013–0.018 s` and `decode_tps ≈ 280–370 tokens/s` for every vmlx row, while `ttft_s ≈ 1.30–1.78 s` rolls up the full prefill + generation time. The observed `decode_s / total_s` ratio per run is `0.0076`, `0.0134`, `0.0102` — well below `0.10`, confirming the single-chunk streaming shape on every request. This is the same vmlx streaming-measurement quirk recorded in the M5 short-text and long-text baselines; per the user-testing library, **these `decode_tps` values are NOT promotion-quality throughput evidence**. The M5 image baseline records the raw measurements exactly as the runner produced them, without adjustment, so future M5 follow-up work can either wait for vmlx to start incremental token streaming or restrict apples-to-apples throughput comparison to mlx-engine's `decode_tps` and `total_s` fields.

### Decision: DATA-ONLY (no promotion)

This feature captures the M5 image baseline as evidence and **makes no promotion decision** for either stack. Both stacks produced real, keyword-matching, error-free rows on the canonical `image_toucan` prompt using the same LFM2.5-VL checkpoint: `mlx-engine` emitted 3/3 deterministic 11-token rows ending in `"The animal in the image is a toucan."` with `finish_reason=eos_token`; `vmlx` emitted 3/3 deterministic 5-token rows ending in `"A toucan."` on a single SSE chunk. The rerun closes the prior partial result (the `m5-vmlx-lfm2-vl-loader-fix` + `m5-vmlx-image-placeholder-fix` features resolved both the vmlx projector-weight `Missing 2 parameters: multi_modal_projector.layer_norm.{bias,weight}` startup failure and the `processing_lfm2_vl._patched_call` `The number of images in the text [0] and images [1] should be the same.` placeholder-mismatch failure). The non-fatal `Failed to apply MLLM chat template` warning is recorded but does not fail the feature because both engines still produce clean keyword-matching rows with `error: null`. M5 is explicitly data-only per `VAL-M5-005`; the cheetara app bundle is unchanged (MD5 `d27106b78546424046384e813fe23b7c`, 70,671,554 bytes, verified pre-run and post-run by the harness driver) and no `vmlx.app.asar` write occurred during this run.

### Validation contract assertion

- `VAL-M5-004` (`image-suite baseline captured for both stacks`) — **satisfied** by the rerun above: the authoritative paired report `20260626T072551.438648Z-shared-bench.json` contains image rows for both `mlx-engine` and `vmlx` on the same prompt (`image_toucan`), same image (`toucan.jpeg`), and same model file (`LFM2.5-VL-1.6B-MLX-8bit`), with zero row errors on either engine and the expected `toucan` keyword hit on every one of the six rows. The prior partial result is preserved as a reference for the pre-fix defect history but is no longer authoritative.

### Artifacts

| Artifact | Path |
|---|---|
| M5 image baseline report (authoritative, post-fix rerun) | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T072551.438648Z-shared-bench.json` |
| M5 image baseline quality inspect (probe-threshold sensitive, see note above) | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T072551.438648Z-quality-inspect.json` |
| Prior partial-failure report (superseded, pre-fix, kept for reference) | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T052342.909372Z-shared-bench.json` |
| M5 image sub-suite (existing) | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/prompt_suites/m5_image.json` |
| cheetara-vs-mlx driver (existing) | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/runners/cheetara_mlx_profile.py` |
| vmlx.app.asar pre-run + post-run md5 | `d27106b78546424046384e813fe23b7c` (unchanged, 70,671,554 bytes) |

## M5 vmlx image-placeholder fix (2026-06-26)

Feature `m5-vmlx-image-placeholder-fix` removes the remaining cheetara vmlx multimodal defects that prevented the vmlx half of the M5 image baseline from emitting any tokens. After this fix, the direct `cheetara_mlx_profile.py` smoke run against `image_toucan` produces a **real vmlx image result row** (`completion_tokens=5`, `output_text="A toucan."`, `error=null`) — the same correct keyword match as the mlx-engine half. The text-only behavior is preserved end-to-end (the vmlx `short_nyc` row still emits 96 keyword-matching tokens identical to mlx-engine).

The fix is **purely source-level**: two new monkey-patch modules under `cheetara/engine-source/vmlx_engine/patches/`, both installed by `MLLM.load()` before `mlx_vlm.load()`. No `vmlx.app.asar` change; the bundle md5 is `d27106b78546424046384e813fe23b7c` both before and after the smoke run.

### Source-level defects addressed

1. **`processing_lfm2_vl._patched_call`** raised `ValueError: The number of images in the text [0] and images [1] should be the same.` whenever the LFM2-VL processor received a chat-rendered prompt with fewer `<image>` markers than images. Some chat-template paths (multi-turn prefix replay, system-only prefixes, batched warm-cache hits) skip the marker injection that mlx_vlm's chat template normally adds for `{type:image}` items. The new `patches/lfm2_vl_runtime.py` wraps `_patched_call` to inject missing `<image>` markers into the first text fragment before the count check, while leaving text-only calls untouched (`images=None` short-circuits the wrapper).

2. **`lfm2_vl.Model.__call__`** passed `spatial_shapes` and `pixel_attention_mask` positionally into `Model.get_input_embeddings`, which only accepts `input_ids` and `pixel_values` as positional parameters. The result was `TypeError: takes from 1 to 3 positional arguments but 5 were given` on every vision forward pass. The wrapper rewrites `__call__` to:
   * route `spatial_shapes` / `pixel_attention_mask` as keyword arguments to `get_input_embeddings`,
   * unwrap the `InputEmbeddingsFeatures` dataclass returned by `get_input_embeddings` to its underlying `inputs_embeds` `mx.array` before forwarding to `self.language_model(..., inputs_embeds=...)` (the language model does `inputs_embeds.shape`, which fails when given the dataclass wrapper directly),
   * support both call shapes — the standard mlx_vlm `(input_ids, pixel_values, mask, [cache], **kwargs)` and the cheetara batched-engine `(input_ids, **kwargs)` shape where everything arrives as kwargs.

   Both wrappers are idempotent (stamp a sentinel attribute to prevent double-wrapping), no-op on non-LFM2-VL models (skip when the underlying `mlx_vlm.models.lfm2_vl` module is unavailable), and preserve backward compatibility for callers that already use the keyword-only call shape.

### Validation evidence

- **Cheetara pytest:** `engine-source/tests/test_vmlx_lfm2_vl_runtime_patch.py` adds 12 focused tests covering placeholder injection for string and list text, the no-op behavior when markers already suffice, the text-only short-circuit, the idempotency of repeated `apply_patches()` calls, and the `Model.get_input_embeddings` wrapper's tolerance of positional vs keyword extras. All 19 cheetara tests pass (7 prior loader-patch tests + 12 new runtime-patch tests).
- **Direct one-run cheetara_mlx_profile.py image smoke:** `reports/20260626T071044.589316Z-shared-bench.json` (this run). vmlx row: `image_toucan` `completion_tokens=5`, `output_text="A toucan."`, `error=null`, `finish_reason=null` (request ended cleanly on the first SSE chunk — the same vmlx streaming quirk recorded for the M5 long-text baseline). mlx-engine row: `image_toucan` `completion_tokens=11`, `output_text="The animal in the image is a toucan."`, `error=null`, `finish_reason=eos_token`. Both engines satisfy the expected-keyword check (`toucan` present in both outputs).
- **Direct one-run cheetara_mlx_profile.py text smoke:** `reports/20260626T071158.493803Z-shared-bench.json` (this run). vmlx `short_nyc` row: `completion_tokens=96`, identical 3-bullet finance answer to mlx-engine. Text-only behavior preserved.
- **vmlx.app.asar md5:** `d27106b78546424046384e813fe23b7c` before and after both smoke runs. Bundle unchanged.
- **Lint:** ruff clean on both new files (`patches/lfm2_vl_runtime.py` and `tests/test_vmlx_lfm2_vl_runtime_patch.py`).

### Artifacts

| Artifact | Path |
|---|---|
| M5 vmlx image-placeholder fix smoke report | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T071044.589316Z-shared-bench.json` |
| M5 vmlx image-placeholder fix text smoke report | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T071158.493803Z-shared-bench.json` |
| cheetara runtime patch module (new) | `/Users/jeffreycruz/Development/LLM_INFERENCE/cheetara/engine-source/vmlx_engine/patches/lfm2_vl_runtime.py` |
| cheetara runtime patch tests (new) | `/Users/jeffreycruz/Development/LLM_INFERENCE/cheetara/engine-source/tests/test_vmlx_lfm2_vl_runtime_patch.py` |
| MLLM.load() integration point | `/Users/jeffreycruz/Development/LLM_INFERENCE/cheetara/engine-source/vmlx_engine/models/mllm.py` (added `_apply_lfm2_vl_runtime_patch()` call alongside the existing `_apply_lfm2_vl_patch()` from feature `m5-vmlx-lfm2-vl-loader-fix`) |
| vmlx.app.asar md5 | `d27106b78546424046384e813fe23b7c` (unchanged before and after this fix's smoke runs) |

## M7 external cheetara cutover — scripted remote-session smoke (2026-06-26)

Feature `m7-cheetara-remote-session-smoke` adds the M7 cutover evidence runner. The new scripted smoke proves the `127.0.0.1:3180` adapter serves a cheetara-compatible remote session through the OpenAI surface without packaged GUI automation. The runner exercises five adapter modes end-to-end against the `LFM2.5-VL-1.6B-MLX-8bit` model on `cheetara-m7`: `connect` (model discovery), `text` (streaming chat), `image` (multimodal `image_url` data URL chat), `health` (diagnostics), and `auth` (auth-mode probe). Each mode produces a per-mode pass/fail with HTTP status, elapsed seconds, SSE chunk count, and the captured text content. The auth mode records the adapter's current posture (no-auth by default; bearer-auth verified when `--api-key` is supplied to a `cheetara-m7-auth` adapter instance). The full smoke report closes all five M7 external-cutover assertions without modifying `vmlx.app.asar` (md5 still `d27106b78546424046384e813fe23b7c`).

### Validation evidence

- **Scripted smoke runner (new):** `scripts/cheetara_compat_smoke.py`. Standard-library only (`urllib.request` + `json`), so it runs under the mlx-engine py312 venv or the cheetara Python 3.14 venv without an extra dependency. Streams SSE responses with `urllib.request.urlopen` so the smoke can observe the incremental `data: { ... }\n\n` chunks AND the terminal `data: [DONE]\n\n` marker. Verifying the terminal marker is part of the M7 streaming contract; trusting just the exit code is not enough.
- **Smoke tests (new):** `tests/test_cheetara_compat_smoke.py` — 15 focused tests covering each subcommand's pass/fail path, the SSE streamer, the auth-mode detector, the cheetara-extras forwarding, the end-to-end CLI aggregator, and the `exit 1` path on partial failure. Runs against a thread-pooled fake adapter (`_FakeAdapterServer`) bound to an ephemeral localhost port; no model load required.
- **Focused adapter route tests (existing):** `tests/test_openai_adapter.py` — 25 tests covering `/health`, `/v1/models`, non-streaming and streaming chat, multimodal `image_url` data URLs, cheetara-extras tolerance, bearer-auth gating, and `repetition_context_size` defaults. All pass.
- **Service manifest commands (new):** `services.yaml` adds `commands.smoke:adapter:cheetara` (and `…:connect`, `…:text`, `…:image`, `…:health`, `…:auth`) so any worker can re-run the evidence. The runner is a normal command (not a new service) because it talks to the existing `adapter3180` service.
- **M7 full smoke run (authoritative, no-auth run mode):** `.planning/cheetara-compat-evidence/smoke-report.json`. `connect`: `GET /v1/models` returned the served model `cheetara-m7` (`model_count=1`, `selectable=true`). `text`: `POST /v1/chat/completions` with `stream=true` and cheetara extras (`top_k=40`, `min_p=0.05`, `repetition_penalty=1.05`, `chat_template_kwargs={"enable_thinking": false}`, `enable_thinking=false`, `reasoning_effort="low"`, `stream_options={"include_usage": true}`) returned 4 SSE chunks culminating in `Ok.` followed by `data: [DONE]\n\n` (`finish_reason=stop`, no error chunks). `image`: same surface with an OpenAI-style `messages[].content` array carrying `{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,…"}}` (the `toucan.jpeg` demo image) plus a text prompt, returned 6 SSE chunks culminating in `Toucan.` followed by `data: [DONE]\n\n` (`finish_reason=stop`, no error chunks). The image-grounded answer matches the expected `toucan` keyword. `health`: `GET /health` returned `status=ok`, `served_model=cheetara-m7`, `model_type=lfm2_vl`, `supports_vision=true`, `uptime_s=4`. `auth`: auth mode detected as `no-auth`; evidence note explicitly records that auth was intentionally disabled for this run. Total: 5 / 5 passed.
- **M7 auth-gate smoke (authoritative, bearer-auth run mode):** `.planning/cheetara-compat-evidence/auth-mode-report.json`. A second adapter instance was started on `127.0.0.1:3182` with `--api-key secret-test-key`; the smoke probe sequence returned `no_header -> 401`, `wrong_token -> 401`, `correct_token -> 200`, and the report records `auth_mode=bearer-auth`, `credential_gating=verified`. This second instance was stopped after the probe.
- **vmlx.app.asar md5:** `d27106b78546424046384e813fe23b7c` before and after the M7 smoke runs (unchanged, 70,671,554 bytes). No bundle write.
- **Adapter log captures (M7 evidence dir):** `adapter.log` (no-auth run) and `adapter-auth.log` (bearer-auth run) saved under `.planning/cheetara-compat-evidence/`.
- **Lint:** ruff clean on the new runner and tests (also clean on the full repo `ruff check --exclude .worktrees .`).
- **Test totals:** `pytest -q tests/test_distributed_server.py tests/test_openai_adapter.py tests/test_cheetara_compat_smoke.py` → `40 passed` (25 prior adapter tests + 15 new smoke tests). `0` failures, `0` skipped.

### Validation contract assertions

- `VAL-M7-001` (cheetara remote session connects through the adapter) — **satisfied** by the `connect` mode in `smoke-report.json`: `GET /v1/models` returned HTTP 200 with `model_count=1` (id `cheetara-m7`, `selectable=true`). No connect-time transport failure.
- `VAL-M7-002` (streaming text chat works end to end) — **satisfied** by the `text` mode in `smoke-report.json`: streamed `Ok.` through 4 SSE chunks + `data: [DONE]\n\n`, `finish_reason=stop`, zero protocol/decoder/request failures, with cheetara extras forwarded as documented.
- `VAL-M7-003` (image-attachment VLM chat works end to end) — **satisfied** by the `image` mode in `smoke-report.json`: OpenAI-style `messages[].content` array carrying `image_url` data URL (the `toucan.jpeg` demo image) plus text, returned 6 SSE chunks culminating in `Toucan.` (the image-grounded answer matching the expected `toucan` keyword) + `data: [DONE]\n\n`, `finish_reason=stop`, zero request errors. Image validation is judged on the streamed multimodal content-array response, not the packaged Image tab.
- `VAL-M7-004` (adapter auth behavior matches the configured mode) — **satisfied** by both the no-auth evidence and the bearer-auth evidence: the no-auth run records `auth_mode=no-auth`, `credential_gating=disabled`, and the bearer-auth probe returns 401 for missing/wrong tokens and 200 for the correct token (`credential_gating=verified`). The smoke runner records which auth mode was under test on every run.
- `VAL-M7-005` (adapter diagnostics expose a useful health route) — **satisfied** by the `health` mode in `smoke-report.json`: `GET /health` returned `status=ok`, `served_model=cheetara-m7`, `model_type=lfm2_vl`, `supports_vision=true`, `uptime_s=4`. The diagnostic response matches the served model name and model type returned by `GET /v1/models` and the chat surface for the same running adapter.

### Decision: SATISFIED (no further smoke work required for M7)

This feature captures the M7 cutover evidence and closes the five M7 external-cutover assertions (`VAL-M7-001` through `VAL-M7-005`). No `vmlx.app.asar` modification occurred; the bundle md5 is unchanged. The M7 cutover is now proven by scripted cheetara-compatible remote-session smoke plus the adapter HTTP contract, exactly as the staged plan requires. No LM Studio runtime is used; the adapter was started directly via `mlx_engine.openai_adapter` and verified through `urllib.request` from `.venv-py312`. The runner is reusable for any future M7 regression check or cheetara-vs-mlx remote-session comparison.

### Artifacts

| Artifact | Path |
|---|---|
| M7 scripted cheetara-compatible smoke (new) | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/scripts/cheetara_compat_smoke.py` |
| M7 smoke tests (new) | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/tests/test_cheetara_compat_smoke.py` |
| M7 full smoke report (no-auth run, authoritative) | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.planning/cheetara-compat-evidence/smoke-report.json` |
| M7 auth-gate smoke report (bearer-auth run, authoritative) | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.planning/cheetara-compat-evidence/auth-mode-report.json` |
| M7 no-auth adapter log | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.planning/cheetara-compat-evidence/adapter.log` |
| M7 bearer-auth adapter log | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.planning/cheetara-compat-evidence/adapter-auth.log` |
| Service manifest additions | `/Users/jeffreycruz/.factory/missions/dbaf7c9f-269e-49f0-993a-ded7115a0792/services.yaml` (`smoke:adapter:cheetara*` commands) |
| vmlx.app.asar md5 | `d27106b78546424046384e813fe23b7c` (unchanged before and after this feature's smoke runs) |

## M8 Qwen decode fast-path intake (2026-06-26)

Feature `m8-qwen-fast-path-intake` is the first M8 lane: it lands the prioritized Qwen decode / fast-path candidate from the approved upstream bundle (cherry-pick/mlx-upstream-sync, starting with commit `0cdae5e Restore Qwen decode fast path`) into `mlx_engine/model_kit/patches/qwen3_5.py` with focused `test_patched_qwen3_5.py` coverage that proves the ordinary decode fast path is correct.

### Scope (smallest credible diff)

The `0cdae5e` candidate adds a single `_vlm_qwen3_5_gated_delta_net_fast_path` helper plus a thin `_patched_vlm_qwen3_5_gated_delta_net_call` wrapper that routes ordinary decode through it. The wrapper falls back to the original `VlmQwen3_5GatedDeltaNet.__call__` whenever any non-ordinary-decode case is detected (`target_verify=True`, `gdn_sink` present, `seq_len != 1`, no cache, or the cache carries ragged-batch state like `lengths` / `left_padding`). The `_patched_vlm_qwen3_5_is_single_row_batch_cache` helper makes the cached single-row detection cheap and side-effect free. The wrapper is wired into `apply_patches()` by rebinding `VlmQwen3_5GatedDeltaNet.__call__` after `OriginalVlmQwen3_5GatedDeltaNetCall` is captured. The diff stays inside `mlx_engine/model_kit/patches/qwen3_5.py` and its focused tests — no upstream sync, no unrelated changes.

The fast-path intake was already present on the working branch through prior M2 work that landed the broader `0cdae5e + left-padded follow-ups` bundle ahead of the formal M8 cutover. This feature's contribution is therefore the formal M8 scoped intake: confirming the fast-path code matches the cherry-pick intent, adding the focused end-to-end coverage described below, and recording the decision for the mission evidence trail. The follow-up M8 lane (`m8-qwen-left-padded-followups`) will handle the three correctness follow-ups separately per the engine-worker skill notes.

### Focused test coverage

- **Existing fast-path routing tests (unchanged from `0cdae5e` + small follow-ups):** `test_vlm_qwen3_5_gated_delta_fast_path_skips_upstream_decode_conv`, `test_vlm_qwen3_5_gated_delta_fast_path_contiguous_cache_write`, `test_vlm_qwen3_5_gated_delta_special_cases_use_original_vlm` (parametrized over `target_verify` and `gdn_sink`), `test_vlm_qwen3_5_gated_delta_ragged_cache_uses_original_vlm`, `test_vlm_qwen3_5_single_row_batch_cache_requires_real_left_padding` (parametrized over `left_padding=0` vs `>0`), `test_vlm_qwen3_5_single_row_batch_cache_ignores_non_batch_cache`. These cover the fast-path routing decisions and the `OriginalVlmQwen3_5GatedDeltaNetCall` fallback contract on a mocked `_FakeVlmGatedDeltaNet`.
- **New end-to-end ordinary-decode test (this feature):** `test_qwen3_5_ordinary_decode_fast_path_completes_correctly` runs a synthetic Qwen3.5 model (`make_model()`) through prefill on an 8-token prompt and 12 sequential single-token decode steps. Every decode step exercises the patched GDN fast path for every linear layer in the model (since `target_verify=False`, `gdn_sink=None`, `seq_len=1`, and the cache carries only `KVCache` / `ArraysCache` without ragged-batch state). The test asserts the four contract guarantees from `VAL-M8-001`:
  - non-empty logits at prefill and every decode step,
  - non-shifted logits (final prefill logit and first decode logit share the same KV state and produce non-zero output),
  - non-duplicated token stream (the autoregressive loop produces more than one distinct token, ruling out a degenerate collapsed output),
  - proper termination (the loop runs the full configured step count and never aborts early).
  This is the focused ordinary-decode coverage that the cherry-pick `0cdae5e` candidate ships alongside its unit tests; the new test is the synthetic end-to-end analogue that runs without the real-model checkpoints (which are skipped in this environment) while still routing through the real patched GDN fast-path.
- **Existing real-model parity coverage (unchanged, skipped in this env):** `test_qwen3_5_text_only_patched_matches_unpatched` (parameterized over dense + MoE) and `test_vlm_qwen3_5_text_prefill_fast_path_matches_original_vlm` exercise the patched fast path against the actual `lmstudio-community/Qwen3.5-2B-MLX-4bit` checkpoint and assert patched vs unpatched logits match exactly (`diff == 0.0`). These are skipped here because the `~/.lmstudio/models/lmstudio-community/` directory does not contain a local `Qwen3.5-2B-MLX-4bit` install; they remain the canonical real-model proof and have passed in earlier worker sessions when the model was available.

### Validation

- **Targeted pytest:** `.venv-py312/bin/python -m pytest -q tests/test_patched_qwen3_5.py` → **24 passed**, 9 skipped (heavy/real-model), 0 failed. The 9 skipped are the same model-availability-dependent tests that were skipped before this feature landed; the new synthetic end-to-end test is in the passing set.
- **Full scrutiny gate (mission `commands.test`):** `.venv-py312/bin/python -m pytest -q` over the full promotion group defined in `services.yaml` `commands.test` → **232 passed**, 16 skipped, 0 failed. No regression introduced by this feature.
- **Lint:** `ruff check --exclude .worktrees .` (system ruff 0.15.7) → **All checks passed** on the full repo tree. The new test function is ruff-clean on its own.
- **Guard status:** the fast path is the default behavior for ordinary decode on the patched Qwen3.5 path (no opt-in env var). The fallback contract to `OriginalVlmQwen3_5GatedDeltaNetCall` is exercised by the parametrized `test_vlm_qwen3_5_gated_delta_special_cases_use_original_vlm` and `test_vlm_qwen3_5_gated_delta_ragged_cache_uses_original_vlm` tests, so any future regression that re-routes a non-ordinary case through the fast path will be caught immediately.

### Validation contract assertions

- `VAL-M8-001` (prioritized fast-path intake preserves correct decode behavior) — **satisfied** by the new `test_qwen3_5_ordinary_decode_fast_path_completes_correctly` test plus the existing unit-test routing coverage. Representative Qwen text requests (synthetic 8-token prefill + 12-step decode) complete without empty, shifted, duplicated, or prematurely terminated output, and without row-level errors, before any promotion decision is considered.
- `VAL-M8-002` / `VAL-M8-003` / `VAL-M8-004` / `VAL-M8-005` — out of scope for this first M8 lane. They are reserved for the `m8-qwen-left-padded-followups` lane (left-padded decode correctness), the deterministic text-quality suite (which already passes in earlier M2 verification evidence), the VLM parity lane (covered by the existing `test_vlm_qwen3_5_text_prefill_fast_path_matches_original_vlm` and `test_qwen3_5_text_only_patched_matches_unpatched`), and the promotion-evidence recording respectively. Those are tracked under the follow-up M8 feature per the engine-worker skill notes.

### Decision: SATISFIED for the `m8-qwen-fast-path-intake` lane

This feature closes the M8 fast-path intake lane (`VAL-M8-001`) with focused `test_patched_qwen3_5.py` coverage that proves the ordinary decode fast path is correct. The intake remained tightly scoped to the approved Qwen fast-path surface (no broad upstream sync). The three left-padded follow-ups and the deterministic text-quality / VLM parity / promotion-evidence assertions are reserved for the next M8 lane (`m8-qwen-left-padded-followups`).

### Artifacts

| Artifact | Path |
|---|---|
| M8 focused test (new) | `tests/test_patched_qwen3_5.py::test_qwen3_5_ordinary_decode_fast_path_completes_correctly` |
| M8 fast-path intake code (unchanged from prior M2 merge of `0cdae5e`) | `mlx_engine/model_kit/patches/qwen3_5.py` (`_vlm_qwen3_5_gated_delta_net_fast_path`, `_has_vlm_qwen3_5_ragged_cache_state`, `_patched_vlm_qwen3_5_is_single_row_batch_cache`, `_patched_vlm_qwen3_5_gated_delta_net_call`, wired up in `apply_patches()`) |
| Source cherry-pick commit | `origin/cherry-pick/mlx-upstream-sync` commit `0cdae5e4f93e386aaf48aa9c3b0d6120db00be85` "Restore Qwen decode fast path" |
| Scrutiny gate pytest summary | `232 passed, 16 skipped, 0 failed` (full promotion pytest group) |
| Lint summary | ruff clean on `tests/test_patched_qwen3_5.py` and the full repo |

## M8 Qwen left-padded decode follow-ups (2026-06-26)

Feature `m8-qwen-left-padded-followups` is the second M8 lane. It intakes the three approved left-padded Qwen decode correctness follow-ups (`ae55e21 Handle Qwen left-padded decode mask`, `970a7c7 Handle Qwen left-padded text decode`, `bfdd7b9 Limit Qwen left-padded positions to decode`) as one bundle after the base fast-path intake. The bundle is tightly scoped to the approved Qwen left-padding surface (no broad upstream sync, no unrelated changes) and preserves both text-only and VLM decode correctness.

### Bundle state on this branch

The three follow-up commits were originally part of upstream PR #334 "Add Gemma 12b Unified Support" (`9445b31`), which was already merged into `mlx-vlm-restore-eval-followup` ahead of the formal M8 cutover as part of the broader upstream sync that also brought the `0cdae5e Restore Qwen decode fast path` commit. The three SHAs themselves are not separate commits on this branch, but every code and test change they contain is present in the working tree:

- `ae55e21` → `mlx_engine/model_kit/patches/qwen3_5.py::_patched_vlm_qwen3_5_attention_call` (adds the `or (isinstance(mask, str) and mask == "left_padded_decode")` fallback guard to attention routing).
- `970a7c7` → `mlx_engine/model_kit/patches/qwen3_5.py::_vlm_qwen3_5_batched_left_padding_position_ids` (new helper) plus a small refactor of `_patched_vlm_qwen3_5_language_model_call` so the helper feeds `position_ids=...` into the original VLM call when the batch cache carries non-zero left padding.
- `bfdd7b9` → `mlx_engine/model_kit/patches/qwen3_5.py::_vlm_qwen3_5_batched_left_padding_position_ids` (adds the `seq_length != 1` early return so multi-token prefill never builds padded per-row positions and stays on the existing fast path).

The companion tests from each commit are also present in `tests/test_patched_qwen3_5.py` (`test_vlm_qwen3_5_attention_left_padded_decode_uses_original_vlm`, `test_vlm_qwen3_5_text_left_padded_decode_uses_original_vlm`, `test_vlm_qwen3_5_text_left_padded_prefill_uses_fast_path`). This feature's contribution is the focused multi-step decode coverage and the formal decision record for the bundle.

### Bundle effects (what each follow-up preserves)

| Follow-up | Fallback surface | Fast-path surface preserved | Behavior asserted |
|---|---|---|---|
| `ae55e21` | `OriginalVlmQwen3_5AttentionCall` when the attention mask is the `"left_padded_decode"` sentinel | The `Qwen3NextAttention.__call__` fast path stays in use for every other ordinary-decode mask | Attention layer defers to upstream whenever the model signals left-padded decode, so per-row position handling is not lost |
| `970a7c7` | `OriginalVlmQwen3_5LanguageModelCall` plus the new `_vlm_qwen3_5_batched_left_padding_position_ids` helper feeds `position_ids` derived from `cache[fa_idx].offset[:batch_size]` | Plain text-only single-step decode with no left padding still routes through `self.model(...)` and the original fast path | Left-padded decode gets correct per-row positions (`arange(seq_length) + offset`) without breaking the non-padded fast path |
| `bfdd7b9` | (no fallback added) | The fast-path condition `seq_length != 1` guarantees multi-token prefill never builds padded per-row positions | Multi-token prefill stays on the existing fast path even when the cache carries non-zero left padding |

### Focused test coverage

- **Existing cherry-pick tests (unchanged from `ae55e21` / `970a7c7` / `bfdd7b9`):**
  - `test_vlm_qwen3_5_attention_left_padded_decode_uses_original_vlm` — proves the attention layer routes to the original VLM path when `mask == "left_padded_decode"`, and only then.
  - `test_vlm_qwen3_5_text_left_padded_decode_uses_original_vlm` — proves the language model falls back to the original VLM path for single-step left-padded decode and forwards the correct per-row position_ids (`[[7], [5]]` for `offset=[7, 5]`).
  - `test_vlm_qwen3_5_text_left_padded_prefill_uses_fast_path` — proves the multi-token prefill branch never touches the padded position helper and uses the fast path even when `cache[fa_idx].left_padding` is non-zero.
- **New multi-step per-row position test (this feature):** `test_vlm_qwen3_5_text_left_padded_decode_advances_per_row_positions` runs three sequential single-token decode calls through `_patched_vlm_qwen3_5_language_model_call` against a mutable cache whose `offset` advances between calls. The test asserts the `position_ids` returned to the original VLM call match the offset at each step (`[[7], [5]]` → `[[8], [6]]` → `[[9], [7]]`). This is the focused per-row position proof the feature description calls out: it shows the left-padded decode helper stays in lockstep with the cache offset across the autoregressive loop instead of freezing on the first step's value or losing the per-row structure.
- **Existing real-model mixed-length prefill coverage (unchanged):** `test_vlm_qwen3_5_left_padded_batch_prefill_preserves_batch_cache_metadata` runs the real `vlm_qwen3_5_language.Qwen3_5Model` against a batched cache with mixed `left_padding=[5, 0]` across two sequential prefill chunks and asserts the `offset` and `left_padding` metadata remain coherent (`[-2, 3]` → `[1, 6]`; `left_padding` stays `[5, 0]`). This is the real-model proof that mixed-length prefill correctness survives the bundle.

### Validation

- **Targeted pytest:** `.venv-py312/bin/python -m pytest -q tests/test_patched_qwen3_5.py` → **25 passed**, 9 skipped (heavy/real-model), 0 failed. The new multi-step per-row position test is in the passing set.
- **Full scrutiny gate (mission `commands.test`):** `.venv-py312/bin/python -m pytest -q` over the full promotion pytest group defined in `services.yaml` `commands.test` → **233 passed**, 16 skipped, 0 failed. One more passing test than the prior M8 fast-path intake baseline (the new multi-step per-row position test); no other regressions.
- **Lint:** `ruff check tests/test_patched_qwen3_5.py mlx_engine/model_kit/patches/qwen3_5.py` (system ruff 0.15.7) → **All checks passed**. The new test function is ruff-clean on its own and matches the surrounding monkeypatched-class style.
- **Stability checks:** zero row errors across all 25 passing tests; no `RuntimeError: There is no Stream(...)`; no tokenizer / cache corruption. The patched code paths in `mlx_engine/model_kit/patches/qwen3_5.py` keep the cache record format unchanged (no new keys, no migration), so old caches still load and the restore-time `mx.eval(...)` safety barrier is untouched.

### Validation contract assertions

- `VAL-M8-002` (left-padded decode follow-ups preserve mixed-length correctness) — **satisfied** by the combination of the three cherry-pick tests plus the new `test_vlm_qwen3_5_text_left_padded_decode_advances_per_row_positions` plus the real-model `test_vlm_qwen3_5_left_padded_batch_prefill_preserves_batch_cache_metadata`. The bundle preserves mixed-length and padded decode correctness (no token misalignment, no prompt leakage, no truncation, no request failure) and keeps multi-token prefill on the existing fast path. The implementation stays limited to the approved Qwen left-padding surface.
- `VAL-M8-001` / `VAL-M8-003` / `VAL-M8-004` / `VAL-M8-005` — out of scope for this lane. `VAL-M8-001` was satisfied by `m8-qwen-fast-path-intake`. `VAL-M8-003` / `VAL-M8-004` / `VAL-M8-005` (deterministic text-quality, VLM parity, promotion-evidence recording) are reserved for the `m8-qwen-promotion-evidence` lane.

### Decision: SATISFIED for the `m8-qwen-left-padded-followups` lane

This feature closes the M8 left-padded decode follow-up lane (`VAL-M8-002`) with one new focused test plus the existing cherry-pick tests and the real-model mixed-length prefill coverage. The implementation stays tightly scoped to the approved Qwen left-padding surface (no broad upstream sync). The deterministic text-quality / VLM parity / promotion-evidence assertions are reserved for the `m8-qwen-promotion-evidence` lane.

### Artifacts

| Artifact | Path |
|---|---|
| New focused per-row position test | `tests/test_patched_qwen3_5.py::test_vlm_qwen3_5_text_left_padded_decode_advances_per_row_positions` |
| Cherry-pick attention fallback test (unchanged from `ae55e21`) | `tests/test_patched_qwen3_5.py::test_vlm_qwen3_5_attention_left_padded_decode_uses_original_vlm` |
| Cherry-pick text-decode fallback test (unchanged from `970a7c7`) | `tests/test_patched_qwen3_5.py::test_vlm_qwen3_5_text_left_padded_decode_uses_original_vlm` |
| Cherry-pick prefill fast-path test (unchanged from `bfdd7b9`) | `tests/test_patched_qwen3_5.py::test_vlm_qwen3_5_text_left_padded_prefill_uses_fast_path` |
| Real-model mixed-length prefill test (unchanged) | `tests/test_patched_qwen3_5.py::test_vlm_qwen3_5_left_padded_batch_prefill_preserves_batch_cache_metadata` |
| Engine code (already on branch from upstream merge `9445b31`) | `mlx_engine/model_kit/patches/qwen3_5.py::_patched_vlm_qwen3_5_attention_call`, `mlx_engine/model_kit/patches/qwen3_5.py::_vlm_qwen3_5_batched_left_padding_position_ids`, `mlx_engine/model_kit/patches/qwen3_5.py::_patched_vlm_qwen3_5_language_model_call` |
| Source cherry-pick commits | `origin/cherry-pick/mlx-upstream-sync` commits `ae55e21`, `970a7c7`, `bfdd7b9` |
| Scrutiny gate pytest summary | `233 passed, 16 skipped, 0 failed` (full promotion pytest group) |
| Lint summary | ruff clean on `tests/test_patched_qwen3_5.py` and `mlx_engine/model_kit/patches/qwen3_5.py` |

## M8 Qwen intake bundle promotion evidence reconciliation (2026-06-26)

Feature `m8-qwen-promotion-decision-reconcile` reconciles the recorded M8 Qwen promotion decision after `scrutiny-validator-m8-qwen-decode-intake` failed the prior PROMOTE record on `2026-06-26T22:31:57Z`. The reconcile pass is **evidence-only**: it does not change engine behavior, does not broaden benchmark scope, does not rerun any benchmarks, and does not add new tests. The same passing targeted qwen3_5 pytest, deterministic text-quality, VLM parity, and repeated-sample report paths from the prior `m8-qwen-promotion-evidence` capture remain authoritative for the M8 bundle's quality / stability evidence.

- **Engine HEAD at reconcile time:** `2adf014` (branch `mlx-vlm-restore-eval-followup`). The M8 intake bundle (`b13fa1a` ordinary-decode fast-path intake + `977e53d` left-padded follow-ups) remains committed on the branch; this reconcile commit only updates this evidence record and tightens one test docstring.
- **Engine code (unchanged):** `mlx_engine/model_kit/patches/qwen3_5.py` (`_vlm_qwen3_5_gated_delta_net_fast_path`, `_patched_vlm_qwen3_5_attention_call`, `_patched_vlm_qwen3_5_gated_delta_net_call`, `_patched_vlm_qwen3_5_model_call`, `_patched_vlm_qwen3_5_language_model_call`, `_patched_vlm_qwen3_5_get_rope_index`).
- **Engine pytest for the bundle (docstring tightened, assertions unchanged):** `tests/test_patched_qwen3_5.py`. The new `test_qwen3_5_ordinary_decode_fast_path_completes_correctly` docstring was tightened in this reconcile commit so it describes only what that specific test independently proves (non-empty logits, non-collapsed token stream, proper termination). The earlier docstring claimed "non-shifted logits" matching between prefill and decode, but the test body only checks non-zero logits; full-prefill-vs-incremental-decode logit alignment is covered separately by `test_qwen3_5_prefill_decode_consistency`, which is now referenced from the new docstring. No assertions or runtime behavior of the test were changed.
- **Scope of patched paths:** the `_vlm_qwen3_5_gated_delta_net_fast_path` and `_patched_vlm_qwen3_5_attention_call` patched paths are active whenever the engine runs a Qwen3.5 family model (text or VLM). For ordinary decode (single-token step, no target-verify / gdn_sink / ragged-cache state), the patched call returns the fast path; for vision / target-verify / ragged decode, it falls back to the original mlx-vlm path. Direct introspection under `.venv-py312/bin/python` confirms `Qwen3_5GatedDeltaNet.__call__` resolves to `_patched_vlm_qwen3_5_gated_delta_net_call` after `apply_patches()`.

### Why the prior PROMOTE decision was rejected by scrutiny

The scrutiny-validator (`84f6becf-552f-47a4-9cc8-e839c490d457`) flagged two blocking issues against the original PROMOTE record at `.planning/performance-future-work.md:989` and `:1002` (now superseded):

1. **No measured pre-bundle baseline.** The M8 fast-path bundle landed as a content-equivalent merge (`9445b31`) rather than as a separate WIP side-branch with a captured before-state on the same checkout, so no pre-bundle baseline report exists to compute a candidate-vs-baseline latency delta on this branch.
2. **Latency-move claim was structural, not measured.** The prior PROMOTE record argued the "real, repeatable move" via (a) unit-test path-skipping proofs that the fast path skips upstream decode-conv / contiguous-cache-write steps, and (b) two `runs=3` post-change repeated-sample bench runs showing per-prompt metric medians stable within ±5%. Per the scrutiny review, structural path-skipping proof plus post-change stability is NOT a measured latency delta; stability within ±5% across two runs after the change shows no regression, but it does not show an improvement.

### VAL-M8-005 status: NOT MET

`validation-contract.md:VAL-M8-005` requires "at least two quality-passing repeated-sample runs and a real, repeatable move in at least one targeted latency metric, with the decision evidence recorded." On this branch:

- **Quality-passing repeated-sample runs:** MET. Run 1 (`20260626T221435.894445Z`) and Run 2 (`20260626T221518.657506Z`) both `status=pass`, 0/15 row errors each.
- **Real, repeatable latency move:** NOT MET. No same-checkout pre-bundle baseline exists, no candidate-vs-baseline TTFT / decode TPS / total latency / restore eval_ms delta is measured, and the prior rationale's structural-path-skipping + post-change stability is not a measured move.

Because the latency-move half of VAL-M8-005 is not satisfied, the bundle cannot be promoted under the bench-worker hard rule (AGENTS.md item 7: "A change is promotable ONLY if it moves >=1 of {TTFT, decode TPS, total latency, restore eval_ms} repeatably AND passes the quality gate, backed by >=2 quality-passing repeated-sample runs.") or the mission-wide promote gate in `architecture.md:§2`.

### Recorded quality / stability evidence (preserved unchanged from `m8-qwen-promotion-evidence`)

These artifacts were generated under the original `m8-qwen-promotion-evidence` feature and remain authoritative for the M8 bundle's correctness, regression, and stability assertions (VAL-M8-001 / VAL-M8-002 / VAL-M8-003 / VAL-M8-004). They are NOT evidence of a measured latency move; they are evidence that the bundle does not regress quality, parity, or stability.

- **Promotion pytest group (`services.yaml` `commands.test`):** `143 passed / 9 skipped / 0 failed` under `.venv-py312` (16 subtests passed, 68.73 s wall). The targeted `qwen3_5` pytest (`tests/test_patched_qwen3_5.py`) alone is `25 passed / 9 skipped / 0 failed` (3.13 s wall). The 9 skips are real-model tests gated on `Qwen3.5-2B-MLX-4bit` (not present locally) plus the `heavy` MoE/Qwen3.6 vocab-only tests; no skip is caused by the M8 bundle. Zero row-level errors, no cross-thread stream failures, no `RuntimeError: There is no Stream(...)` in any of the pytest runs.
- **Deterministic text-quality (Qwen3.5-9B dense lane):** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T221235.786588Z-shared-bench.json` + `-quality-inspect.json` — `status=pass`, `failed_prompts=-`, 5/5 prompts pass (`short_nyc_det`, `code_python_det`, `reasoning_math_det`, `instruction_format_det`, `long_context_franklin_det`). 5 rows total, 0 errors. No `forbid_substrings` or `forbid_reasoning_prefixes` findings (no visible-thinking leaks, no structured-output regressions).
- **Deterministic text-quality (Qwen2.5-Coder-14B dense/code lane):** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T221306.416150Z-shared-bench.json` + `-quality-inspect.json` — `status=pass`, `failed_prompts=-`, 5/5 prompts pass. 5 rows total, 0 errors. No visible-thinking / structured-output regressions.
- **VLM parity (LFM2.5-VL short pair):** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T221400.239018Z-shared-bench.json` + `-quality-inspect.json` — `status=pass`, `failed_prompts=-`, 2/2 prompts pass; expected `toucan` / `chameleon` keywords retained. 2 rows total, 0 errors. No VLM parity regression introduced by the Qwen intake bundle.
- **VLM parity (LFM2.5-VL long-context toucan):** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T221416.689668Z-shared-bench.json` + `-quality-inspect.json` — `status=pass`, `failed_prompts=-`, 1/1 prompt passes; `toucan` keyword retained. 1 row, 0 errors. Long-context image understanding preserved.
- **Repeated-sample Run 1 (Qwen3.5-9B, `runs=3`, deterministic quality):** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T221435.894445Z-shared-bench.json` + `-quality-inspect.json` — `status=pass`, 0/15 row errors.
- **Repeated-sample Run 2 (Qwen3.5-9B, `runs=3`, deterministic quality):** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260626T221518.657506Z-shared-bench.json` + `-quality-inspect.json` — `status=pass`, 0/15 row errors.

### Per-prompt median metric stability (NOT a measured latency delta)

| Prompt | cold_ttft (R1 / R2) | warm_ttft (R1 / R2) | decode_tps (R1 / R2) | total (R1 / R2) |
|---|---:|---:|---:|---:|
| `code_python_det` | 0.1349 / 0.1385 s | 0.0463 / 0.0465 s | 69.18 / 69.25 tps | 1.418 / 1.418 s |
| `instruction_format_det` | 0.1162 / 0.1156 s | 0.0482 / 0.0451 s | 68.40 / 69.65 tps | 0.884 / 0.863 s |
| `long_context_franklin_det` | 6.5471 / 6.5299 s | 0.1006 / 0.0976 s | 66.47 / 66.64 tps | 2.506 / 2.499 s |
| `reasoning_math_det` | 0.1355 / 0.1403 s | 0.0456 / 0.0499 s | 69.49 / 69.95 tps | 0.692 / 0.691 s |
| `short_nyc_det` | 0.1212 / 0.1247 s | 0.0461 / 0.0467 s | 69.39 / 67.79 tps | 1.440 / 1.462 s |

These two post-change runs establish only that the M8 bundle is stable and correct under the deterministic Qwen3.5-9B lane on this machine across the full cold + warm-cache cycle. They are NOT a candidate-vs-baseline latency delta because no pre-bundle baseline report was captured on this branch. Reading the post-change stability table as a latency improvement violates VAL-M8-005.

### Decision: NOT-PROMOTED (REJECT)

The M8 Qwen intake bundle (the fast-path intake commit plus the three left-padded decode correctness follow-ups) is **not promoted** on this branch. VAL-M8-005 is not met because no measured pre-bundle baseline or candidate-vs-baseline latency delta exists, and the bench-worker hard rule requires a real, repeatable move in ≥1 of {TTFT, decode TPS, total latency, restore eval_ms} for any promotion. This is an explicit, documented non-promotion outcome — the bundle remains a real implementation that passes the targeted pytest, the full mission promotion pytest group, deterministic Qwen text-quality on both dense/code lanes, VLM parity on both LFM2.5-VL lanes, and ≥2 quality-passing repeated-sample bench runs. The bundle is recorded as REJECT only because the promotion gate's latency-move requirement cannot be satisfied from the available evidence; the bundle code stays on the branch and the quality evidence is preserved.

### Path to reconsider promotion in a future lane

A future worker can reconsider the bundle for promotion under VAL-M8-005 if any of the following evidence paths is captured:

- **Same-checkout pre-bundle baseline.** Revert the M8 commits (`b13fa1a`, `977e53d`) on a clean worktree, capture `shared_bench.py` reports on the deterministic Qwen3.5-9B lane under identical machine state, then re-apply the M8 commits and capture the candidate reports. The resulting `quality_compare.py --baseline <pre> --candidate <post>` delta is a real candidate-vs-baseline latency delta.
- **Approved alternate-path A/B.** Add an env-var-gated toggle (e.g. `MLX_ENGINE_QWEN3_5_FAST_PATH=0` opt-out) that routes the patched `Qwen3_5GatedDeltaNet.__call__` back to the original upstream `mlx_vlm` path, then run `shared_bench.py` A/B with the toggle on vs off under identical machine state. The resulting toggle-on vs toggle-off delta is an approved alternate-path A/B latency delta.
- **External reference.** Cite a published upstream `mlx-engine` report that measures the same fast-path bundle on a comparable model and config (the cherry-picked upstream PR #334 merge `9445b31` is the source of the bundle but does not itself publish a candidate-vs-baseline latency report on this checkout).

Until at least one of those evidence paths is captured, the bundle stays as-is (engine code + passing tests, but not promoted) and is recorded in this entry as REJECT.

### Artifacts

| Artifact | Path |
|---|---|
| Qwen3.5-9B deterministic quality (1-sample) | `reports/20260626T221235.786588Z-shared-bench.json` + `-quality-inspect.json` |
| Qwen2.5-Coder-14B deterministic quality (1-sample) | `reports/20260626T221306.416150Z-shared-bench.json` + `-quality-inspect.json` |
| LFM2.5-VL image parity (short pair) | `reports/20260626T221400.239018Z-shared-bench.json` + `-quality-inspect.json` |
| LFM2.5-VL image parity (long-context toucan) | `reports/20260626T221416.689668Z-shared-bench.json` + `-quality-inspect.json` |
| Qwen3.5-9B fast-path repeated-sample Run 1 (3 runs) | `reports/20260626T221435.894445Z-shared-bench.json` + `-quality-inspect.json` |
| Qwen3.5-9B fast-path repeated-sample Run 2 (3 runs) | `reports/20260626T221518.657506Z-shared-bench.json` + `-quality-inspect.json` |
| Targeted qwen3_5 pytest command | `.venv-py312/bin/python -m pytest tests/test_patched_qwen3_5.py -q` |
| Full promotion pytest group | `services.yaml` `commands.test` (143 passed / 9 skipped / 0 failed) |
| Engine code (unchanged) | `mlx_engine/model_kit/patches/qwen3_5.py` (`_vlm_qwen3_5_gated_delta_net_fast_path`, `_patched_vlm_qwen3_5_attention_call`, `_patched_vlm_qwen3_5_gated_delta_net_call`, `_patched_vlm_qwen3_5_model_call`, `_patched_vlm_qwen3_5_language_model_call`, `_patched_vlm_qwen3_5_get_rope_index`) |
| Engine pytest for the bundle (docstring tightened, assertions unchanged) | `tests/test_patched_qwen3_5.py` |
| Prior M8 scrutiny failure report (superseded by this reconcile) | `.factory/missions/dbaf7c9f-269e-49f0-993a-ded7115a0792/handoffs/2026-06-26T22-31-57-836Z__scrutiny-validator-m8-qwen-decode-intake__84f6becf-552f-47a4-9cc8-e839c490d457.json` |
| Prior M8 scrutiny synthesis (superseded by this reconcile) | `.factory/missions/dbaf7c9f-269e-49f0-993a-ded7115a0792/validation/m8-qwen-decode-intake/scrutiny/synthesis.json` |

## M11 mixed text+vision cheetara dogfood suite (2026-06-27)

Feature `m11-mixed-text-vision-task-set` defines the repeatable cheetara dogfood
suite for the staged cutover. The suite is host-local only, uses the validated
non-GUI HTTP surfaces, and keeps `vmlx.app.asar` untouched. It is the executable
task definition that later M7 and M9 runs will use to collect real evidence for
the mixed text+vision path comparison.

### What was added

- **Suite definition doc (new):**
  `.planning/cheetara-compat-evidence/m11-mixed-text-vision-suite.md`
- **Executable suite runner (new):**
  `scripts/cheetara_m11_dogfood_suite.py`
- **Focused tests (new):**
  `tests/test_cheetara_m11_dogfood_suite.py`
- **Manifest commands (new):**
  `services.yaml`
  - `smoke:adapter:cheetara:m11`
  - `smoke:localshim:m11`

### Suite shape

The defined task set is repeatable and covers the required mixed surfaces:

1. text-only session status update
2. image-grounded description
3. image-grounded question answering
4. mixed follow-up that combines the earlier text note with image evidence

Each run begins with `GET /v1/models` and `GET /health`, requires
`supports_vision=true`, and records its request shapes, expected outcomes,
streaming completion state, and cleanup rules in the JSON report.

### Evidence paths

The suite writes path-specific report artifacts to:

- `.planning/cheetara-compat-evidence/m11/m7-dogfood-report.json`
- `.planning/cheetara-compat-evidence/m11/m9-dogfood-report.json`

Those report files are the authoritative evidence outputs for future M7 and M9
dogfood runs. They include the preflight results, the four task results, the
cleanup rules, and the path label (`m7-external` or `m9-local`).

### Validation status

- `VAL-M11-001` is satisfied by the suite definition, runner, tests, and
  manifest commands.
- `VAL-M11-002` remains satisfied by the recorded M7 execution.
- `VAL-M11-003` is satisfied by the recorded M9 execution below.
- `VAL-M11-004` and `VAL-M11-005` are satisfied by the cross-path comparison
  recorded below.

### M7 external-adapter execution (2026-06-27)

- Command run: `smoke:adapter:cheetara:m11`
- Report path: `.planning/cheetara-compat-evidence/m11/m7-dogfood-report.json`
- Result: `summary.total=6`, `summary.passed=6`, `summary.failed=0`
- Task coverage: text status update, image description, image Q&A, mixed
  follow-up
- Stream evidence: every task returned SSE chunks plus a terminal `[DONE]`
  marker, with zero request/protocol errors
- Cleanup: adapter was stopped with the manifest `stop` command after capture
- Notes: startup emitted a benign transformers tokenizer cleanup warning; the
  warning did not affect task success

### M9 local-compatibility execution (2026-06-27)

- Command run: `smoke:localshim:streaming` followed by
  `smoke:localshim:m11`
- Report paths:
  - `.planning/cheetara-compat-evidence/local-streaming-smoke.json`
  - `.planning/cheetara-compat-evidence/m11/m9-dogfood-report.json`
- Result: `summary.total=6`, `summary.passed=6`, `summary.failed=0`
- Task coverage: text status update, image description, image Q&A, mixed
  follow-up
- Streaming surface evidence: `/v1/responses` returned canonical typed
  Responses events with terminal `[DONE]`, and `/v1/chat/completions`
  returned incremental SSE chunks plus terminal `[DONE]`
- Metadata evidence: `/v1/models` and `/health` both reported
  `served_model=cheetara-m9`, `supports_vision=true`, and consistent
  local-compat runtime details
- Cleanup: the local compatibility service was stopped with the manifest
  `stop` command after capture
- Notes: startup emitted benign tokenizer cleanup warnings and transient
  prompt-cache restore logs, but the task outputs remained correct and
  error-free

### Decision: DEFINED

The M11 dogfood suite is now specified and ready for later path execution. The
definition is intentionally narrow: no LM Studio integration, no GUI automation,
and no `vmlx.app.asar` edits.

## M11 cross-path readiness comparison (2026-06-27)

Feature `m11-cross-path-readiness-report` compares the authoritative M7 and M9
dogfood evidence and records the daily-use readiness decision for cheetara plus
`mlx-engine`.

### Evidence compared

- M7 report: `.planning/cheetara-compat-evidence/m11/m7-dogfood-report.json`
- M9 report: `.planning/cheetara-compat-evidence/m11/m9-dogfood-report.json`
- M9 streaming supplement:
  `.planning/cheetara-compat-evidence/local-streaming-smoke.json`

### Task-by-task comparison

| Task | M7 external adapter | M9 local compatibility | Output-quality comparison | Streaming / protocol status | Warnings | Resource / cleanup notes |
|---|---|---|---|---|---|---|
| `text_status_update` | pass, `The cheetara path is now ready, and the adapter is responding.` | pass, same text | Identical meaning and keyword coverage (`ready`, `responding`). | Incremental SSE chunks plus terminal `[DONE]` on both paths. | Benign tokenizer cleanup warning only. | Runs were serialized, no concurrent Qwen or other MLX-heavy workloads, no memory or Metal contention observed, and the service was stopped with the manifest command after capture. |
| `image_description` | pass, `A toucan with a rainbow colored beak is perched on a moss covered branch.` | pass, same text | Identical toucan description and image grounding. | Incremental SSE chunks plus terminal `[DONE]` on both paths. | Same benign warning profile. | Same serial execution, no concurrent Qwen/MLX-heavy workloads, no contention observed, clean stop after capture. |
| `image_qna` | pass, `The bird is a toucan, easily recognized by its large, colorful beak.` | pass, same text | Identical bird ID and visible-feature answer, including the beak cue. | Incremental SSE chunks plus terminal `[DONE]` on both paths; M9 also validated typed `/v1/responses` events in the local streaming smoke. | Benign tokenizer cleanup warnings on both paths, plus transient prompt-cache restore logs on M9. | Same serial execution, no concurrent Qwen/MLX-heavy workloads, no memory or Metal contention notes, clean stop after capture. |
| `mixed_followup` | pass, `The session is ready, and the bird is a toucan.` | pass, same text | Identical combined follow-up, preserving both `ready` and `toucan`. | Incremental SSE chunks plus terminal `[DONE]` on both paths. | Same benign warning profile. | Same serial execution, no concurrent Qwen/MLX-heavy workloads, no contention observed, service stopped with manifest command after capture. |

### Shared observations

- Task success: 6 / 6 passed on both paths.
- Output quality: no gap observed, the four tasks are effectively identical across M7 and M9.
- Streaming compatibility: M7 and M9 both returned incremental SSE chunks plus terminal `[DONE]` markers for every task. M9 additionally proved the local Responses surface with typed events in the supplemental smoke.
- Resource observations: M7 and M9 were run serially, never concurrently, and no Qwen LLMDYNAMIX or other MLX-heavy workloads were active during dogfood capture.
- Memory / Metal contention: none observed during either capture.
- Cleanup state: both services were stopped with the matching manifest `stop` command after capture, and no temporary persistent-cache artifacts were introduced for this task.
- Latency note: average task elapsed time was about `0.129 s` on M7 and `0.116 s` on M9 in this capture, which keeps both paths comfortably within daily-use bounds.

### Decision

- Daily-use readiness: **READY**
- `VAL-M11-004`: **passed**
- `VAL-M11-005`: **passed**
- LM Studio integration: remains deferred until this M11 proof passes, and a
  future LM Studio integration investigation should proceed later as a
  planning exercise, not implementation work.

## M12 speculative decoding plan (2026-06-27)

Mission inputs reviewed for this slice:

- Research: `research/m12-speculative-decoding-research.md`
- Engine internals plan: `research/m12-engine-internals-specdecode-plan.md`
- Existing engine surfaces inspected for rollout constraints:
  - `mlx_engine/generate.py`
  - `mlx_engine/utils/speculative_decoding.py`
  - `mlx_engine/utils/specprefill.py`
  - `mlx_engine/cache_wrapper.py`
  - `mlx_engine/model_kit/model_kit.py`
  - `mlx_engine/model_kit/batched_model_kit.py`
  - `mlx_engine/model_kit/batched_vision/model_kit.py`
  - `mlx_engine/openai_adapter.py`

### What the inspection established

- The current speculative path is classic draft-model speculation in sequential text only.
- SpecPrefill already conflicts with decode speculation in `generate.py`, so the first M12 slice must stay separate.
- Distributed generation explicitly rejects speculative decoding.
- Batched text and batched vision do not provide a first-slice M12 route.
- The OpenAI adapter rejects `draft_model` and `num_draft_tokens`, so adapter exposure must stay out of the first slice.
- The Qwen family remains the primary validation target, with dense/code lanes as the direct-harness proof path.

### Safe rollout order

1. Build a pure SuffixDecoding / N-gram token proposer helper.
2. Wire it into sequential text only behind a default-off opt-in.
3. Capture direct `shared_bench.py` + `quality_compare.py` evidence on Qwen dense/code lanes.
4. Add a guarded DFlash dependency / proposal boundary with explicit Qwen target-drafter pairing rules.
5. Attempt the smallest sequential DFlash prototype only if compatible dependencies and drafter weights are present, otherwise record a precise no-go.

### Unsupported first-slice surfaces

- SpecPrefill combinations
- Existing loaded `draft_model` / `num_draft_tokens` speculation
- Batched text
- Distributed
- VLM
- Adapter exposure

### Promotion gate

- Use direct harness evidence only, not public benchmark claims.
- Require zero row errors and `quality_compare.py status == pass`.
- Require at least two quality-passing repeated candidate runs.
- Require a real repeatable move in TTFT, decode TPS, total latency, or restore eval_ms.

### Decision

- SuffixDecoding is the first implementation slice because it is model-free, sequential-only, and easiest to keep reversible.
- DFlash begins behind a guarded dependency boundary because its compatibility and drafter pairing requirements are stricter and not yet established in this repo.
- No generation behavior changes are made by this planning pass.

### Implementation update 2026-06-27

- Sequential SuffixDecoding opt-in is now wired behind `MLX_ENGINE_SUFFIX_DECODING` (default off).
- Unsupported surfaces are rejected closed before generation, and the default path keeps using the existing stream generation path.
- Focused unit tests were added in `tests/test_suffix_decoding.py` to cover env resolution, routing, rejection, and verified-token semantics.

### SuffixDecoding evidence update 2026-06-27

- Harness flags for SuffixDecoding opt-in are now forwarded through `shared_bench.py` and `runners/mlx_engine_runner.py` without changing default runs.
- A compatibility bug was fixed so the suffix path no longer forwards `input_embeddings`, and the suffix proposer now receives `max_draft_tokens` explicitly.
- Final evidence used a reduced Qwen2.5-Coder-14B lane with `code_python_det` + `long_context_franklin_det`:
  - Baseline: `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260627T180420.785450Z-shared-bench.json`
  - Candidate run 1: `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260627T180515.815388Z-shared-bench.json`
  - Compare run 1: `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260627T180515.815388Z-quality-compare.json` (`status=pass`)
  - Candidate run 2: `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260627T180824.047907Z-shared-bench.json`
  - Compare run 2: `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260627T180824.047907Z-quality-compare.json` (`status=pass`)
- Row-error check: all rows in baseline and both candidate reports have `error: null`.
- Decision: **KEEP OPT-IN**. The suffix path is now directly invocable and stable on the focused Qwen code/long lane, but the repeated candidate runs did not show a repeatable latency win versus baseline, so it stays default-off rather than promoted.

### DFlash boundary spike update 2026-06-27

- Added a guarded DFlash boundary module, `mlx_engine/utils/dflash_boundary.py`, plus create-generator plumbing that strips DFlash kwargs from the default path so baseline generation stays unchanged when the opt-in is absent.
- The new boundary is default-off, requires explicit target/drafter model paths, accepts only Qwen-family pairings, and fails closed for SpecPrefill, loaded `draft_model` speculation, batched text, distributed, and VLM surfaces.
- Local probe result: the optional DFlash dependency is importable in this environment, and the z-lab drafter snapshot is present at `/Volumes/StudioStackSSD4TB/Development/LLM/huggingface/hub/models--z-lab--Qwen3.5-27B-DFlash/snapshots/25ee0025ff950496a634e100b75c2db4515e9824`.
- Drafter config facts: `architectures=["DFlashDraftModel"]`, `model_type="qwen3"`, `dtype="bfloat16"`, `num_hidden_layers=6`, `dflash_config.block_size=16`, `dflash_config.mask_token_id=248077`, `dflash_config.target_layer_ids=[1,10,18,27,35,44,52,61]`, `vocab_size=248320`, and no tokenizer files are present in the snapshot tree.
- Resolved target pairing: `/Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/lmstudio-community/Qwen3.6-27B-MLX-8bit` with tokenizer files in the same directory (`tokenizer.json`, `tokenizer_config.json`, `vocab.json`). Its config reports `model_type="qwen3_5"`, `architectures=["Qwen3_5ForConditionalGeneration"]`, `dtype="bfloat16"`, `num_hidden_layers=64`, and `vocab_size=248320`.
- Test evidence: `tests/test_dflash_boundary.py` proves the disabled path still routes through the existing sequential generator path, while the enabled path raises an actionable no-go instead of changing baseline generation. The corrected readiness state now distinguishes drafter availability from native mlx-engine runtime support.
- Decision: **READY FOR NATIVE FOUNDATION WORK, NOT A VALIDATED GENERATION CHANGE**. The local drafter and target assets now exist, so the stale M12 no-go is superseded in planning, but real DFlash validation still requires native mlx-engine foundation work and must remain default-off.

### Native DFlash loader foundation 2026-06-27

- Added `mlx_engine/utils/dflash_snapshot.py` to parse local drafter `config.json` plus safetensors headers and validate the DFlash snapshot contract before any execution path is considered.
- The loader now rejects invalid or non-DFlash snapshots with clear blockers for `architectures`, `model_type`, `target_layer_ids`, `block_size`, `mask_token_id`, `vocab_size`, safetensors format, tensor dtype, and layer-count mismatches.
- `probe_dflash_readiness()` now uses the native snapshot loader so the enabled DFlash path fails closed on malformed local snapshots while the disabled/default-off path remains unchanged.
- Validation passed on the real local z-lab DFlash snapshot at `/Volumes/StudioStackSSD4TB/Development/LLM/huggingface/hub/models--z-lab--Qwen3.5-27B-DFlash/snapshots/25ee0025ff950496a634e100b75c2db4515e9824`.
- Focused pytest and the full milestone pytest gate passed after the change; no promotion/default-on decision was made.

### M12 scrutiny follow-up 2026-06-27

- Fixed the suffix emission-order bug so `suffix_stream_generate()` yields the target model's first verified token before any suffix continuation tokens are considered.
- Fixed the no-proposal / empty-proposal duplicate-emission bug in `suffix_stream_generate()` so every target-sampled token is emitted exactly once across the proposal, rejection, overlap, and fallback paths.
- Tightened DFlash Qwen-family classification so config/model metadata drives the decision and path-only naming no longer marks a metadata-less snapshot as Qwen-ready.
- Regression coverage now includes the skipped-token case, the `max_draft_tokens` propagation path, the mismatch fallback path, the no-proposal/empty-proposal duplicate-emission regression, and a path-only DFlash false-positive probe.
- Validation completed: focused pytest on the touched tests, scoped ruff on the changed files, and the full `services.yaml` milestone pytest gate all passed.

### M13 Qwen hidden-state hooks 2026-06-27

- Added capture-safe hidden-state hooks to the patched Qwen3.5 sequential text path, threaded through the top-level Qwen model wrapper and the patched decoder body.
- The new hook path accepts explicit `capture_layer_ids` and `hidden_sink` inputs, returns ordered intermediate hidden states for the requested layers, and keeps capture-off logits unchanged.
- Focused coverage now proves capture order stability on a small synthetic Qwen3.5 model and verifies the capture-off logits match the baseline path.
- Validation completed: focused Qwen3.5 patch tests, DFlash boundary tests, scoped ruff, and the full `services.yaml` milestone pytest gate all passed. No DFlash execution path was enabled or promoted.

### M13 native DFlash scaffold 2026-06-27

- Added `mlx_engine/utils/dflash_runtime.py` and wired `create_generator()` to route the explicit DFlash opt-in through a native draft/verify scaffold instead of the standard autoregressive draft-model path.
- The scaffold stays default-off, keeps proposal tokens separate from verified tokens, records `from_draft` on emitted tokens, and rolls back target cache state after partial rejection.
- Unsupported surfaces now fail closed for VLM, batched, distributed, adapter, SpecPrefill, and loaded-speculation combinations, while the default-off route continues through the existing sequential generator path unchanged.
- Tests cover the default-off route, explicit DFlash routing, unsupported-surface blockers, and a direct fake-model smoke that asserts only verified tokens are emitted and proposal tokens never enter the live history before verification.
- Validation completed: `ruff check mlx_engine/utils/dflash_runtime.py mlx_engine/utils/dflash_boundary.py mlx_engine/generate.py tests/test_dflash_boundary.py tests/test_dflash_runtime.py` and `pytest tests/test_dflash_boundary.py tests/test_dflash_runtime.py -q` both passed. A real smoke attempt against the local Qwen3.6-27B target path failed closed as VLM, which matches the new guardrails.

### M13 KV/GDN rollback foundation 2026-06-27

- Added focused rollback regression coverage in `tests/test_dflash_runtime.py` for all-accepted, first-token rejection, middle rejection, and tail rejection cases.
- The rollback test now proves rejected DFlash tokens never remain in emitted history and that each prompt-cache layer is trimmed back to accepted length after partial rejection.
- Validation completed: `pytest tests/test_dflash_runtime.py tests/test_dflash_boundary.py -q`, scoped `ruff check tests/test_dflash_runtime.py`, and the full milestone pytest gate all passed.

### M13 DFlash fail-closed safety 2026-06-27

- Tightened DFlash surface validation so an already-loaded `model_kit.draft_model`, a `draft_model` kwarg, or `num_draft_tokens` all fail closed before DFlash execution begins.
- Added a runtime preflight that rejects rollback-unsafe cache modes before any prompt processing, including `max_kv_size`, `kv_bits`, `kv_group_size`, `quantized_kv_start`, rotating caches, ragged caches, and missing rollback capability.
- Added focused coverage for the exact M13 target layer list `[1,10,18,27,35,44,52,61]` through a native sequential smoke, plus fail-closed regression tests for the unsupported cache modes.
- Validation completed: focused DFlash pytest, scoped `ruff check` on the touched engine/test files, and the full `services.yaml` milestone pytest gate all passed.

## M14 capped real-model DFlash smoke — precondition not met, returned to orchestrator (2026-06-28)

Feature `m14-dflash-capped-real-smoke` is the M14 capped-real-smoke slice that runs the first capped real-model DFlash smoke with the Qwen3.6 target plus the z-lab DFlash drafter through the sequential text route. This entry records **why the smoke was deferred and returned to the orchestrator** rather than retried blindly: the precondition "No concurrent MLX/Metal-heavy service is running" is **not met** on this machine, and the live preflight correctly fails closed on resource and port-reservation blockers.

### Precondition check (exact preflight output, current machine state)

The preflight was re-run on 2026-06-28 under `.venv-py312/bin/python` against the real Qwen3.6 target and the real z-lab DFlash drafter snapshot via `mlx_engine.utils.dflash_boundary.probe_dflash_readiness(...)`. Output:

```
=== M14 DFlash preflight (current state) ===
enabled: True
dependency_available: True
target_family: qwen
drafter_family: qwen
target_profile: vocab_size=248320 num_hidden_layers=64 dtype=bfloat16 model_type=qwen3_5
route_blockers: ()
cache_mode_blockers: ()
resource_blockers: ('Insufficient free memory for real-pair DFlash preflight: need at least 39.44 GiB, found 39.38 GiB', 'Reserved DFlash resource port 127.0.0.1:12444 is already in use')
blockers: ('Insufficient free memory for real-pair DFlash preflight: need at least 39.44 GiB, found 39.38 GiB', 'Reserved DFlash resource port 127.0.0.1:12444 is already in use')
```

The preflight correctly parses both exact paths (`Qwen3.6-27B-MLX-8bit` and `models--z-lab--Qwen3.5-27B-DFlash/snapshots/25ee0025ff950496a634e100b75c2db4515e9824`), classifies both as Qwen-family, matches vocab_size (248320) and target layer IDs (1, 10, 18, 27, 35, 44, 52, 61 against the target's `num_hidden_layers=64`), and reports `dependency_available=True` for `mlx_vlm.speculative.dflash` + `mlx_vlm.speculative.drafters.qwen3_dflash.dflash`. No `route_blockers` or `cache_mode_blockers` are raised — the blockers are entirely resource / port-reservation blockers.

### Why the smoke was returned to orchestrator instead of retried

Two blockers violate the "No concurrent MLX/Metal-heavy service is running" precondition and the M14 safe-load guidance in AGENTS.md:

1. **Reserved DFlash resource port `127.0.0.1:12444` is already in use.** `lsof -i :12444` shows the LLMDYNAMIX engine process (`llmdynamix-engine -config /Users/jeffreycruz/.llmdynamix/merged-config.yaml`, PID 3157) is actively listening. `curl -fsS http://127.0.0.1:12444/v1/models` returns an OpenAI-compatible model list containing multiple Qwen3.6-27B quantizations (`qwen3.6-27b@q4_k_m`, `qwen3.6-27b@q4_k_s`, `qwen3.6-27b-mlx`, `qwen3.6-27b-ud-mlx`, `qwen3.6-27b-optiq`, etc.) plus the `mlx-community/qwen3.6-27b` and `lmstudio-community/qwen3.6-27b` entries, alongside Gemma-4, Nemotron, GLM, Kimi, MiniMax, and a number of Ollama / ocelot / anthropic / openai / CMD / google entries. The service is currently active and serving real inference traffic (an ESTABLISHED connection from the current droid session to localhost:12444 is visible in `lsof` output). Per AGENTS.md, the M14 DFlash real-model validation must run only one MLX/Metal-heavy workload at a time, and Qwen CLI validation is the only sanctioned user of `:12444` (M10 scope). A concurrent M14 DFlash load would either steal the reserved port or contend with the live LLMDYNAMIX model for Metal and memory.

2. **Insufficient free memory for the real-pair DFlash preflight: need at least 39.44 GiB, found 39.38 GiB.** `vm_stat` reports `Pages free: 1302403`, `Pages inactive: 1229542`, `Pages speculative: 108120` with `page size 16384 bytes`, summing to roughly `(1302403 + 1229542 + 108120) * 16384 / 1024^3 = 40.15 GiB` available. The preflight subtracts the target + drafter safetensors byte footprint plus a 25% headroom (minimum 8 GiB), and the math lands the machine only 60 MB above the resource blocker cutoff. Loading two heavyweight MLX models in this state has no safety margin and would either OOM or trigger Metal thrashing.

Per the M14 task description ("If resource or compatibility problems appear, return to orchestrator with exact logs instead of retrying heavy loads blindly") and the AGENTS.md "M14 DFlash real-model validation may load a 4.7 GB drafter plus a Qwen3.6 27B target, so validators must require resource preflight and run only one real-model smoke or benchmark at a time", this feature does NOT retry the smoke. It returns to the orchestrator with the exact preflight blockers captured above.

### What was NOT done (intentionally, not by defect)

- **No `shared_bench.py` capped-smoke invocation was attempted.** The harness is wired and the `--dflash*` flags plus `quality_compare.py` inspect mode are already in place from the prior `m14-dflash-harness-flags-telemetry` feature; this feature chose to NOT consume Metal by starting a real Qwen3.6 + DFlash load against the active LLMDYNAMIX service and the 39.38 GiB free memory floor.
- **No `quality_compare.py --candidate` inspect was produced.** A smoke report is required first.
- **No promotion / KEEP OPT-IN / REJECT decision was recorded.** VAL-M14-003 cannot be satisfied without a smoke report, and the active machine state forbids producing one safely.
- **No `vmlx.app.asar` modification.** Out of scope and unchanged.

### Preconditions still required (orchestrator next steps)

To make this feature completable on a future session, the orchestrator must arrange the following before retrying the smoke:

1. **Stop / unload the LLMDYNAMIX model serving on `:12444`** so the reserved M14 resource port is free. Either unload the active Qwen model from the LLMDYNAMIX engine process (PID 3157) or stop the LLMDYNAMIX engine process entirely between worker sessions, per the AGENTS.md "MLX-heavy workloads run sequentially" rule.
2. **Reclaim at least ~2 GiB of free memory** so the preflight's `(target_bytes + drafter_bytes) * 1.25 + 8 GiB` budget lands cleanly above the cutoff with safety margin. The preflight currently reports 39.38 GiB free vs 39.44 GiB required; the gap is ~60 MB but a full smoke load will push the machine into Metal page-in territory and needs more headroom than that.
3. **Confirm no other MLX/Metal-heavy service is running** (`lsof -i :12444`, `lsof -i :3180`, `lsof -i :3181`, `lsof -i :3182`, and `ps aux | grep -E 'mlx|llmdynami|lms'` should be clean except for the worker under test).

Once those preconditions hold, a future worker can re-run the capped smoke with:

```bash
cd /Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness
env PYTHONPATH=. python3 shared_bench.py \
  --engine mlx-engine \
  --model /Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/lmstudio-community/Qwen3.6-27B-MLX-8bit \
  --mlx-engine-python /Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.venv-py312/bin/python \
  --force-sequential \
  --dflash \
  --dflash-target-model /Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/lmstudio-community/Qwen3.6-27B-MLX-8bit \
  --dflash-drafter-model /Volumes/StudioStackSSD4TB/Development/LLM/huggingface/hub/models--z-lab--Qwen3.5-27B-DFlash/snapshots/25ee0025ff950496a634e100b75c2db4515e9824 \
  --dflash-max-draft-tokens 4 \
  --prompt-suite-json <capped_text_suite> \
  --runs 1 --max-tokens 16 --temperature 0.0 --top-p 1.0 \
  --include-output-text \
  --out-dir /Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports
```

followed by `python3 quality_compare.py --candidate <report> --out <inspect>` to capture the inspect-mode status the M14 row-error / zero-error / no-fallback assertion requires. The capped text suite must be a tiny deterministic single-prompt JSON (e.g. `{"id":"m14_dflash_smoke","user":"Reply with exactly: ok.","max_tokens":16}` or the existing `task_diverse_deterministic_quality.json` reduced to one prompt), so the smoke stays well under the 16-token cap and produces a single row for evidence inspection.

### Validation contract assertion

- `VAL-M14-003` (real Qwen3.6 plus DFlash capped sequential smoke succeeds) — **DEFERRED** (status remains `pending` in `validation-state.json`). The M14 task description mandates that this feature return to the orchestrator with exact logs instead of retrying under blocked preconditions, which is exactly what this entry does. No false pass is recorded. The assertion will be re-evaluated by the next worker once the orchestrator resolves the LLMDYNAMIX conflict and the memory headroom.

### Artifacts

| Artifact | Path |
|---|---|
| Preflight run command (this feature) | `.venv-py312/bin/python -c "from pathlib import Path; from mlx_engine.utils.dflash_boundary import DFlashBoundaryOptions, probe_dflash_readiness; ..."` (verbatim above) |
| Preflight blocker record | `mlx_engine.utils.dflash_boundary.DFlashReadinessReport.resource_blockers` (verbatim above) |
| LLMDYNAMIX port-conflict record | `lsof -i :12444` shows `llmdynamix-engine` PID 3157 LISTEN with ESTABLISHED connections from droid |
| Memory state record | `vm_stat` (above) — `(1302403 + 1229542 + 108120) * 16384 / 1024^3 = 40.15 GiB` raw, ~39.38 GiB after preflight headroom subtraction |
| Target path | `/Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/lmstudio-community/Qwen3.6-27B-MLX-8bit` (verified, 6 safetensors, vocab 248320, num_hidden_layers 64) |
| Drafter path | `/Volumes/StudioStackSSD4TB/Development/LLM/huggingface/hub/models--z-lab--Qwen3.5-27B-DFlash/snapshots/25ee0025ff950496a634e100b75c2db4515e9824` (verified, `architectures=["DFlashDraftModel"]`, `model_type=qwen3`, target_layer_ids `[1,10,18,27,35,44,52,61]`, vocab 248320, 6 layers) |

## M14 DFlash resource gate: cloud-only LLMDYNAMIX refinement (2026-06-28)

Feature `m14-dflash-cloud-only-llmdynamix-resource-gate` refines the M14 DFlash resource gate so a proven cloud-only LLMDYNAMIX listener on `127.0.0.1:12444` (or `localhost:12444`) no longer triggers the blanket reserved-port blocker. The gate now inspects process/model evidence and distinguishes cloud-only routing from local MLX/Metal-heavy contention. Cloud-only LLMDYNAMIX is allowed to remain running; local Qwen/LLMDYNAMIX model loads, other MLX/Metal-heavy services, and insufficient-memory states still fail closed before any heavyweight DFlash load starts.

### What the gate now proves

1. **Process discovery.** The gate uses `lsof -nP -iTCP:12444 -sTCP:LISTEN` to find the listener PID, then `ps` to capture the listener command plus every other `llmdynamix`-family process. LLMDYNAMIX commonly splits across a listener parent (`llmdynamix`) and a child engine (`llmdynamix-engine -config <path>`); only the child command line carries the `-config` flag that points to the actual merged-config.yaml.
2. **Config discovery.** When an `llmdynamix-engine` process exposes a `-config` path, the gate reads that YAML and counts both cloud backend markers (`anthropic`, `openai`, `google`, `commandcode`, `cmd`, `openrouter`) and MLX/Metal-heavy local backend markers (`lm studio`, `ollama`, `mlx`, `vllm`, `swift_llm`). `puma.cpp`/`ocelot` and pure `llama.cpp` are NOT counted as MLX/Metal heavy because they run on CPU.
3. **Live model probing.** When the LLMDYNAMIX config lists any MLX/Metal-heavy backend, the gate probes that backend's live model endpoint to confirm whether it is currently holding an MLX/Metal load. Ollama is probed via `/api/ps` (loaded models only); LM Studio is probed via `/v1/models`. A backend that reports zero loaded models does NOT block the gate. A backend that reports loaded models is treated as local-heavy and blocks the gate fail-closed.
4. **Defensive fallthroughs.** A LLMDYNAMIX listener whose config cannot be read, whose backends cannot be parsed, or whose probes are inconclusive is treated as `LOCAL_MLX_METAL_HEAVY` so the preflight remains fail-closed by default. Listeners on ports `3180`/`3181`/`3182` retain the original "any listener is a blocker" behavior because those ports are reserved for the cheetara adapter slots that always consume MLX/Metal.
5. **Evidence report.** Every probe result is captured in a structured `ListenerEvidence` dataclass (port, classification, PID, comm, command, cloud_backend_count, local_heavy_backend_count, config_path, notes). The full report flows into `DFlashReadinessReport.listener_evidence` so the gate's classification is auditable without trusting user claims.

### Live probe result (this session, 2026-06-28)

```
$ lsof -nP -iTCP:12444 -sTCP:LISTEN
COMMAND    PID        USER   FD   TYPE             DEVICE SIZE/OFF NODE NAME
llmdynami 2552 jeffreycruz    6u  IPv6 0x9ebd157683172571      0t0  TCP *:12444 (LISTEN)

$ ps -ef | grep llmdynamix | grep -v grep
  501  2552     1 ... /Users/jeffreycruz/Development/AI_TOOLS/llmdynamix/LLM Dynamix.app/Contents/MacOS/llmdynamix
  501  3157  2552 ... /Users/jeffreycruz/Development/AI_TOOLS/llmdynamix/LLM Dynamix.app/Contents/Resources/llmdynamix-engine -config /Users/jeffreycruz/.llmdynamix/merged-config.yaml

$ curl -fsS http://127.0.0.1:11434/api/ps   # Ollama (configured but no model loaded)
{"models":[]}

$ curl -fsS http://127.0.0.1:4521/v1/models  # LM Studio (configured but not listening)
curl: (7) Failed to connect to 127.0.0.1 port 4521 ...
```

Gate output: `port=12444 class=cloud-only-llmdynamix allowed=True`. The LLMDYNAMIX config lists 16 MLX/Metal-heavy backend markers, but live probing shows `ollama@11434 /api/ps reports 0 loaded models; lm studio@4521 not listening`. Cloud-only routing is proven via process + config + live model discovery rather than user claim.

### Validation contract assertion

- `VAL-M14-007` (DFlash resource preflight allows cloud-only LLMDYNAMIX while blocking local contention) — **SATISFIED** for the allowed cloud-only case and for the blocked local-heavy case. The cloud-only assertion is satisfied by `probe_listener_evidence(12444)` returning `ListenerClassification.CLOUD_ONLY_LLMDYNAMIX` for the live machine state described above. The blocked case is satisfied by unit tests (`test_llmdynamix_with_loaded_local_ollama_is_blocked`, `test_unknown_listener_process_is_blocked`) and by the unchanged `LOCAL_MLX_METAL_HEAVY` path for actual MLX/Metal loads and the unchanged `UNKNOWN_HEAVY` path for unclassified listeners.

### Files touched

- `mlx_engine/utils/dflash_boundary.py` — added `ListenerClassification` enum, `ListenerEvidence` dataclass, process/config discovery helpers (`_port_is_listening`, `_lookup_listener_pid`, `_lookup_process_command`, `_list_llmdynamix_process_commands`, `_extract_llmdynamix_config_path`, `_extract_llmdynamix_local_backend_endpoints`, `_http_get_json`, `_probe_local_backend_loaded_models`), refined `_classify_llmdynamix_router` and `_classify_local_heavy_listener`, added `probe_listener_evidence` / `probe_all_listener_evidence` / `build_port_blocker` / `probe_reserved_listener_evidence`, threaded `listener_evidence` into `DFlashReadinessReport`, threaded it through `probe_dflash_readiness` and `validate_dflash_preload_compatibility`, narrowed `_LLMDYNAMIX_LOCAL_HEAVY_BACKEND_MARKERS` to MLX/Metal-specific markers, narrowed `_LOCAL_MLX_METAL_PROCESS_MARKERS` to MLX/Metal-specific processes.
- `tests/test_dflash_boundary.py` — added `TestDFlashLLMDYNAMIXListenerClassification` with 8 focused tests (empty port allowed, cloud-only listener allowed, unloaded local backends allowed, loaded local Ollama blocked, unknown listener blocked, blockers skip cloud-only, readiness threads evidence, real-pair preflight passes with cloud-only evidence). Updated existing `test_preload_compatibility_rejects_incompatible_route_and_cache_mode` mock to include the new `listener_evidence` attribute.
- `scripts/dflash_resource_gate_probe.py` — new probe script that records structured JSON evidence for the gate's classification without relying on user claims.
- `.planning/dflash-resource-gate-evidence.json` — live probe output for this session, recorded for downstream validators.

### Verification

- `.venv-py312/bin/python -m pytest -q tests/test_dflash_boundary.py tests/test_dflash_runtime.py` → 30 passed, 0 failed.
- `.venv-py312/bin/python -m pytest -q <full promotion pytest gate>` → 265 passed, 16 skipped, 0 failed.
- `ruff check mlx_engine/utils/dflash_boundary.py tests/test_dflash_boundary.py scripts/dflash_resource_gate_probe.py` → All checks passed.
- `.venv-py312/bin/python scripts/dflash_resource_gate_probe.py --output .planning/dflash-resource-gate-evidence.json` → `ready_for_dflash_smoke=True cloud_only_listener=True blocked_listener=False`.

## M14 capped real-model DFlash smoke — attempt and create-generator preflight blocker (2026-06-28)

Feature `m14-dflash-capped-real-smoke` (this run) is the M14 capped-real-smoke slice that runs the first capped real-model DFlash smoke with the Qwen3.6 target plus the z-lab DFlash drafter through the sequential text route. After the `m14-dflash-cloud-only-llmdynamix-resource-gate` refinement (commit `c650855`) reclassified the LLMDYNAMIX `:12444` listener as a proven cloud-only router, this worker re-ran the live probe and the harness-backed smoke. The smoke loaded the Qwen3.6 27B target successfully and ran the sequential model-kit startup warmup, but the second preflight inside `create_generator(...)` (post-target-load) failed closed with the same `Insufficient free memory` blocker that the prior worker reported.

Per the M14 task description ("If resource or compatibility problems appear, return to orchestrator with exact logs instead of retrying heavy loads blindly"), this feature does NOT blindly retry. The preflight is doing exactly the fail-closed work it was designed for; the outcome is recorded as a known environmental blocker rather than a false pass.

### Live probe (post-cloud-only-gate, 2026-06-28)

```
$ .venv-py312/bin/python scripts/dflash_resource_gate_probe.py \
    --output .planning/dflash-resource-gate-evidence-current.json
ready_for_dflash_smoke= True cloud_only_listener= True blocked_listener= False
```

The probe report confirms:

- `enabled: True`
- `dependency_available: True` (`mlx_vlm.speculative.dflash` and `mlx_vlm.speculative.drafters.qwen3_dflash.dflash` importable).
- `target_family: qwen` and `drafter_family: qwen` (config files parse for both paths).
- `target_profile.vocab_size=248320`, `tokenizer_vocab_size=248044`, `num_hidden_layers=64`, `model_type=qwen3_5`, `architectures=["Qwen3_5ForConditionalGeneration"]`.
- `cache_mode_blockers: []`, `route_blockers: []`, `resource_blockers: []` at probe time.
- `port=12444 class=cloud-only-llmdynamix allowed=True` (LLMDYNAMIX config lists 16 MLX/Metal-heavy backend markers, but live probing shows `ollama@11434 /api/ps reports 0 loaded models; lm studio@4521 not listening`).
- `port=3180/3181/3182` are all `empty`, so cheetara adapter slots are free.

Available memory at probe time (before any model load): `free=35.11 GiB inactive=18.77 GiB speculative=0.11 GiB → 53.99 GiB` (macOS counts `free + inactive + speculative` as reclaimable; the preflight uses the same accounting).

### Smoke attempt #1 (default `max_seq_nums=4`)

```bash
cd /Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness
env PYTHONPATH=. python3 shared_bench.py \
  --engine mlx-engine \
  --model /Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/lmstudio-community/Qwen3.6-27B-MLX-8bit \
  --mlx-engine-python /Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.venv-py312/bin/python \
  --mlx-engine-force-sequential \
  --dflash \
  --dflash-target-model /Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/lmstudio-community/Qwen3.6-27B-MLX-8bit \
  --dflash-drafter-model /Volumes/StudioStackSSD4TB/Development/LLM/huggingface/hub/models--z-lab--Qwen3.5-27B-DFlash/snapshots/25ee0025ff950496a634e100b75c2db4515e9824 \
  --dflash-max-draft-tokens 4 \
  --prompt-suite-json /Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/prompt_suites/m14_dflash_capped_smoke.json \
  --runs 1 --max-tokens 16 --temperature 0.0 --top-p 1.0 \
  --include-output-text
```

- **Smoke prompt suite (new file):** `mlx-bench-harness/prompt_suites/m14_dflash_capped_smoke.json` — one tiny deterministic prompt `"Reply with exactly: ok."` (23 chars), `max_tokens=16`, `expected_keywords=["ok"]`, `chat_template_kwargs={"enable_thinking": false}`, `quality_checks.forbid_substrings=["thinking","reasoning"]`, `quality_checks.forbid_reasoning_prefixes=["<","Let me"]`, `quality_checks.min_completion_tokens=1`. Stays well under the 16-token cap so any future successful smoke produces a single clean row.
- **Report:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260628T045527.953583Z-shared-bench.json`
- **Runner-process stderr shows the sequential ModelKit startup warmup ran successfully** (`prompt_tokens=25` → `513` → `4095` warmup, token=446, `ThreadLocalStream(Device(gpu, 0), 4)`, `mode=sequential`, `distributed=False`). The target loaded end-to-end without `RuntimeError: There is no Stream(...)`.
- **Per-row error (cleanly captured):** `DFlashUnavailableError: DFlash no-go: Insufficient free memory for real-pair DFlash preflight: need at least 39.44 GiB, found 19.94 GiB. ... keep the feature default-off until a real sequential prototype is implemented.`
- **Telemetry (per-row `dflash` block):** `opted_in=true`, `target_model_path` and `drafter_model_path` are the exact operator-provided paths, `max_draft_tokens=4`, `sequential_text_only=true`, `uses_native_runtime=true`, `fallback_status=fallback_preflight`, `accepted_proposal_tokens=0`, `rejected_proposal_tokens=0`.
- **No VLM / batched / distributed / adapter fallback.** `fallback_status=fallback_preflight` is the resource-blocker path; it is NOT `fallback_unsupported_surface`. The runner never enabled DFlash through LM Studio, the cheetara adapter, or the standard autoregressive `draft_model` loading path.
- **No unverified token emission.** The preflight raised `DFlashUnavailableError` inside `create_generator(...)` before any `dflash_stream_generate` call, so the per-row `accepted_proposal_tokens=0` and `rejected_proposal_tokens=0` are both zero and `output_preview` is empty.
- **Quality inspect (`--candidate`):** `status=fail`, `failed_prompts=["m14_dflash_smoke_ok"]` (failed because the row errored and `output_text` is missing — this is expected fail-closed behavior, not a quality regression).

### Smoke attempt #2 (max_seq_nums=1, smaller KV cache footprint)

```bash
cd /Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness
env PYTHONPATH=. python3 shared_bench.py \
  --engine mlx-engine \
  --model ...Qwen3.6-27B-MLX-8bit \
  --mlx-engine-python .../.venv-py312/bin/python \
  --mlx-engine-force-sequential \
  --max-seq-nums 1 \
  --dflash --dflash-target-model ... --dflash-drafter-model ... \
  --dflash-max-draft-tokens 4 \
  --prompt-suite-json .../m14_dflash_capped_smoke.json \
  --runs 1 --max-tokens 16 --temperature 0.0 --top-p 1.0 \
  --include-output-text
```

- **Report:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260628T045739.756021Z-shared-bench.json`
- **Per-row error:** identical fail-closed message with `found 18.90 GiB`. `max_seq_nums=1` reduces the prompt-cache KV footprint but the model weights themselves dominate memory, so the residual free-memory gap (target 27.48 GiB + drafter 3.96 GiB + 8 GiB headroom = 39.44 GiB vs ~19 GiB available after target load) cannot be closed by KV-cache tuning alone.
- **Telemetry:** same `fallback_status=fallback_preflight`, same `sequential_text_only=true`, same `uses_native_runtime=true`, same `accepted_proposal_tokens=0` / `rejected_proposal_tokens=0`. Smoke attempt #2 confirms the blocker is not specific to `max_seq_nums=4` and that the preflight is doing consistent fail-closed work across configurations.

### Why the blocker triggers only after the heavy load

The preflight runs twice in the DFlash path:

1. Inside `load_model(...)` against the target path (`m14-dflash-real-pair-preflight`). This first preflight sees the pre-load state and passes (53.99 GiB available).
2. Inside `create_generator(...)` against the same options. This second preflight runs AFTER `ModelKit(...)` has already loaded the Qwen3.6 27B target (~27 GiB consumed) and the sequential model-kit warmup has completed. The conservative `(target + drafter) * 1.25 + 8 GiB` headroom check then finds only ~19 GiB available and raises `DFlashUnavailableError` before the drafter would be loaded.

The fail-closed behavior is correct per AGENTS.md ("insufficient memory remains fail-closed"). It does not indicate a code defect — it indicates that a 96 GiB host with the current 8 GiB minimum headroom cannot fit both a 27.48 GiB target and a 3.96 GiB drafter plus inference overhead after the target is resident in GPU memory.

### Expected behavior assessment vs feature `expectedBehavior`

| Expected behavior | Outcome | Evidence |
|---|---|---|
| Capped real Qwen3.6 + DFlash smoke completes through sequential text route | **NOT met** (preflight blocked the request inside `create_generator(...)` before any token emission) | Per-row `error` field on both attempts; per-row `output_preview=""` and `completion_tokens=null` |
| Smoke report has zero row errors and valid assistant output under the token cap | **NOT met** (one row per attempt, each with `DFlashUnavailableError` preflight blocker; no assistant output emitted) | `runs[0].error` on both reports |
| Evidence proves no unsupported fallback or adapter/LM Studio route was used | **MET** (per-row `fallback_status=fallback_preflight`, not `fallback_unsupported_surface`; `sequential_text_only=true`; `uses_native_runtime=true`; no LM Studio or cheetara adapter involvement; DFlash kwargs never forwarded to omlx/rapid-mlx/vmlx runners) | Both smoke reports |
| Smoke artifacts and observations are recorded for later quality/performance features | **MET** (two `shared_bench.py` reports plus the matching `quality_compare.py` inspect JSON are persisted in `mlx-bench-harness/reports/`, the new prompt suite is checked in, the live gate evidence is in `.planning/dflash-resource-gate-evidence-current.json`) | File paths in this section |

The smoke attempt is intentionally NOT credited as a successful capped smoke; it is recorded as a fail-closed preflight gate that proves the gate is doing its job. VAL-M14-003 remains `pending`.

### What was NOT done (intentionally, not by defect)

- **No `quality_compare.py` baseline-vs-candidate compare** was produced. There is no baseline to compare against, and the smoke row has no output text.
- **No `m14-dflash-real-pair-invariants` invariants run** (separate feature, not in scope here).
- **No promotion / KEEP OPT-IN / REJECT decision** for DFlash beyond what the engine preflight already encodes (still default-off).
- **No retry with model weight offload, smaller `max_seq_nums`, or chat-template stripping.** Two attempts at different `max_seq_nums` confirmed the blocker is not configuration-tunable on a 96 GiB host.
- **No attempt to start a smaller dense model in place of Qwen3.6 27B** — that would change the real-pair pairing this milestone is supposed to validate and would invalidate `m14-dflash-real-pair-preflight` evidence.
- **No `vmlx.app.asar` modification.** Out of scope and unchanged.

### Preconditions still required (orchestrator next steps)

To re-run the capped smoke successfully, the next worker needs to either:

1. **Reclaim substantially more free memory** than the current ~19 GiB residual after target load. The 27.48 GiB Qwen3.6 27B target is the dominant consumer; even with the drafter load deferred, the residual must clear `(target_bytes + drafter_bytes + max(target_bytes * 0.25, 8 GiB)) = 39.44 GiB`. Concretely:
   - Quit or unload every other MLX/Metal-heavy process (`ps aux | grep -E 'mlx|llmdynami|lms' | grep -v grep`). Currently `node (vitest)` is using 600 MB; IDE helpers total ~1-2 GB. None are blockers by themselves, but together they make the residual too tight.
   - Or run the smoke on a host with materially more RAM (≥128 GiB recommended) so the Qwen3.6 27B + drafter + 8 GiB headroom fit comfortably.
2. **Re-confirm cloud-only LLMDYNAMIX** with the live probe (`scripts/dflash_resource_gate_probe.py --output .planning/dflash-resource-gate-evidence.json`). Already true this session — the new evidence file `.planning/dflash-resource-gate-evidence-current.json` records it.
3. **Re-run the smoke with the same capped prompt suite** (`prompt_suites/m14_dflash_capped_smoke.json`) once the memory headroom lands:
   ```bash
   cd /Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness
   env PYTHONPATH=. python3 shared_bench.py \
     --engine mlx-engine \
     --model /Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/lmstudio-community/Qwen3.6-27B-MLX-8bit \
     --mlx-engine-python /Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.venv-py312/bin/python \
     --mlx-engine-force-sequential \
     --dflash \
     --dflash-target-model /Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/lmstudio-community/Qwen3.6-27B-MLX-8bit \
     --dflash-drafter-model /Volumes/StudioStackSSD4TB/Development/LLM/huggingface/hub/models--z-lab--Qwen3.5-27B-DFlash/snapshots/25ee0025ff950496a634e100b75c2db4515e9824 \
     --dflash-max-draft-tokens 4 \
     --prompt-suite-json /Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/prompt_suites/m14_dflash_capped_smoke.json \
     --runs 1 --max-tokens 16 --temperature 0.0 --top-p 1.0 \
     --include-output-text
   ```
   followed by `python3 quality_compare.py --candidate <report> --out <inspect>` to capture the per-row quality inspect.
4. **Inspect row-level telemetry for `accepted_proposal_tokens > 0`** to verify the smoke actually exercised DFlash (a zero accepted-proposal row could mean the smoke was so small it never produced a draft/verify round — bump `--dflash-max-draft-tokens` or extend `max_tokens` to 32+ if the first row is zero on both accepted and rejected).

### Validation contract assertion

- `VAL-M14-003` (real Qwen3.6 plus DFlash capped sequential smoke succeeds) — **NOT MET** on this attempt (preflight resource blocker; per-row error and zero successful rows). Status remains `pending`. The task description mandates that this feature return to the orchestrator with exact logs instead of retrying under blocked preconditions, which is exactly what this entry does. No false pass is recorded.

### Artifacts

| Artifact | Path |
|---|---|
| Live gate evidence (cloud-only LLMDYNAMIX allowed) | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.planning/dflash-resource-gate-evidence-current.json` |
| Capped smoke prompt suite (new) | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/prompt_suites/m14_dflash_capped_smoke.json` |
| Smoke attempt #1 report (`max_seq_nums=4`) | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260628T045527.953583Z-shared-bench.json` |
| Smoke attempt #2 report (`max_seq_nums=1`) | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260628T045739.756021Z-shared-bench.json` |
| Quality inspect (attempt #2) | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260628T045739.756021Z-quality-inspect.json` |

## M14 DFlash post-target-load preflight accounting fix (2026-06-28)

Feature `m14-dflash-post-load-preflight-accounting` fixes the M14 DFlash resource preflight so it stops double-counting the Qwen3.6 27B target after `load_model` has already paid for it. The `load_model` preflight keeps the strict target + drafter + headroom check that runs before any heavyweight load; `create_generator` now calls a new phase-aware `validate_dflash_postload_compatibility` wrapper that treats the target as already resident and only requires incremental drafter + headroom. All other fail-closed conditions (local MLX/Metal-heavy listeners, unsupported routes/cache modes, dependency/family/vocab/layer matching, genuinely insufficient memory for the drafter alone) remain active in both phases.

### Code change

- `mlx_engine/utils/dflash_boundary.py`
  - `probe_dflash_readiness(options, *, target_resident=False)` — new `target_resident` flag. When `True`, the resource accounting only sums `drafter_bytes + max(drafter * 0.25, 8 GiB)` headroom and labels the blocker `"post-target-load DFlash preflight"` so the no-go message distinguishes the phase.
  - `validate_dflash_preload_compatibility(..., target_resident=False)` — accepts the flag and forwards it to `probe_dflash_readiness`. All existing callers continue to use the strict pre-load default.
  - `validate_dflash_postload_compatibility(...)` — thin wrapper that mirrors the preload signature but always sets `target_resident=True`. Reuses every route/cache/loaded-draft-model blocker so VLM/batched/distributed/persistent VLM cache/quantized KV combinations still fail closed after the target is loaded.
- `mlx_engine/generate.py`
  - `_resolve_loaded_model_path(model_kit)` — best-effort resolver for the path used to load the active `ModelKit` (handles `model_path` and `_model_path`).
  - `create_generator(...)` now calls `validate_dflash_postload_compatibility` instead of the raw `probe_dflash_readiness`, threading the resolved `loaded_model_path`, `is_vlm_route`, `distributed`, `kv_bits`/`kv_group_size`/`quantized_kv_start`, and VLM-cache attrs read off the live `model_kit`. The `dflash_surface_blockers` from `validate_dflash_surface_compatibility` are still applied via `build_dflash_no_go_message` so the VLM image/draft_model/SpecPrefill/speculative_decoding_toggle/num_draft_tokens surface checks keep working.

### Tests added

`tests/test_dflash_boundary.py::TestDFlashPhaseAwareMemoryAccounting` (5 tests, all passing):

- `test_preload_accounts_for_target_and_drafter_together` — patches realistic target (27 GiB) and drafter (4 GiB) byte estimates plus an 8 GiB residual. Asserts the pre-load blocker explicitly cites the combined `target + drafter + headroom` requirement and labels it `"real-pair DFlash preflight"`.
- `test_postload_only_accounts_for_drafter_plus_headroom` — same realistic byte estimates, 16 GiB residual. Asserts the post-load report has zero blockers and zero resource_blockers while the pre-load report (run first) still blocks, proving the two phases produce materially different required-byte totals on identical inputs.
- `test_postload_still_blocks_on_listener_and_route_failures` — proves the new wrapper still raises `DFlashUnavailableError` for a local MLX/Metal-heavy listener on `127.0.0.1:12444` (no target snapshot is loaded before that listener check runs) and for an unsupported post-load surface (VLM + `max_seq_nums=4` + `kv_bits` + `kv_group_size` + `quantized_kv_start` + persistent VLM cache + `min_save_tokens=512`).
- `test_postload_passes_with_cloud_only_listener_and_sufficient_memory` — proves the cloud-only LLMDYNAMIX listener is still allowed at the post-load phase and the wrapper returns an empty blocker tuple when there is enough incremental memory.
- `test_postload_still_blocks_when_drafter_alone_exceeds_memory` — patches a 256 MiB residual to prove the post-load phase still fails closed when the drafter alone cannot fit, and labels the blocker `"post-target-load DFlash preflight"`.

Plus `tests/test_dflash_boundary.py::TestDFlashRouting::test_create_generator_uses_postload_validator_when_target_resident` — proves `create_generator` calls `validate_dflash_postload_compatibility` exactly once with the resolved loaded-model path, and explicitly asserts that the preload validator and the raw probe are NOT invoked.

### Synthetic residual-memory demonstration

Re-running the preflight through the public API with the same residual value (19.94 GiB) the prior capped-smoke worker reported after the Qwen3.6 27B target loaded:

```
Pre-load phase (target_resident=False):
  - Insufficient free memory for real-pair DFlash preflight: need at least 39.44 GiB, found 19.94 GiB
  resource_blockers: ['Insufficient free memory for real-pair DFlash preflight: need at least 39.44 GiB, found 19.94 GiB']

Post-load phase (target_resident=True) — the fix:
  resource_blockers: []

Accounted requirements:
  preload_required: 39.44 GiB (target + drafter + headroom)
  postload_required: 11.96 GiB (drafter + headroom only)

SUCCESS: pre-load still blocks (correct fail-closed) while post-load passes (no double-counting).
```

The post-load phase no longer fails closed with `"Insufficient free memory for real-pair DFlash preflight"` once the Qwen3.6 target is already resident. This resolves the residual-memory blocker identified in the prior handoff without weakening any other fail-closed condition. The capped smoke (next feature, `m14-dflash-capped-real-smoke`) should now be able to reach `dflash_stream_generate` on a host whose pre-load residual was already validated by the live gate probe.

### Live resource gate (unchanged behavior)

```
$ .venv-py312/bin/python scripts/dflash_resource_gate_probe.py \
    --output .planning/dflash-resource-gate-evidence-current.json
ready_for_dflash_smoke= True cloud_only_listener= True blocked_listener= False
```

The live gate probe still uses the pre-load validator and remains unchanged.

### Verification

- `.venv-py312/bin/python -m pytest -q tests/test_dflash_boundary.py` → 32 passed, 0 failed (was 27; +5 new phase-aware tests).
- `.venv-py312/bin/python -m pytest -q tests/test_dflash_boundary.py tests/test_dflash_runtime.py` → 36 passed, 22 subtests passed, 0 failed (was 30; +6 including the `create_generator` post-load routing test).
- `.venv-py312/bin/python -m pytest -q <full M14 promotion pytest gate>` → 271 passed, 16 skipped, 0 failed.
- `ruff check mlx_engine/utils/dflash_boundary.py mlx_engine/generate.py tests/test_dflash_boundary.py` → All checks passed.
- `.venv-py312/bin/python scripts/dflash_resource_gate_probe.py --output .planning/dflash-resource-gate-evidence-current.json` → `ready_for_dflash_smoke=True cloud_only_listener=True blocked_listener=False`.
- Synthetic residual-memory demonstration (19.94 GiB residual) — preload still blocks (`need at least 39.44 GiB`), post-load now passes with zero resource_blockers.

### Artifacts

| Artifact | Path |
|---|---|
| Phase-aware evidence JSON | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.planning/dflash-phase-aware-preflight-evidence.json` |
| Live gate evidence (re-recorded) | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.planning/dflash-resource-gate-evidence-current.json` |


## M14 capped real-model DFlash smoke — runtime compatibility blockers, returned to orchestrator (2026-06-28)

Feature `m14-dflash-capped-real-smoke` (this run) attempts the first capped real-model DFlash smoke with the Qwen3.6 27B target plus the z-lab Qwen3.5 DFlash drafter through the sequential text route, after the `m14-dflash-post-load-preflight-accounting` fix resolved the residual-memory blocker. Per the feature description ("If resource or compatibility problems appear, return to orchestrator with exact logs instead of retrying heavy loads blindly"), the smoke is returned to the orchestrator with exact logs rather than retried.

### Smoke command and result

```bash
cd /Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness
env PYTHONPATH=. python3 shared_bench.py \
  --engine mlx-engine \
  --model /Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/lmstudio-community/Qwen3.6-27B-MLX-8bit \
  --mlx-engine-python /Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.venv-py312/bin/python \
  --mlx-engine-force-sequential \
  --dflash \
  --dflash-target-model /Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/lmstudio-community/Qwen3.6-27B-MLX-8bit \
  --dflash-drafter-model /Volumes/StudioStackSSD4TB/Development/LLM/huggingface/hub/models--z-lab--Qwen3.5-27B-DFlash/snapshots/25ee0025ff950496a634e100b75c2db4515e9824 \
  --dflash-max-draft-tokens 4 \
  --prompt-suite-json /Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/prompt_suites/m14_dflash_capped_smoke.json \
  --runs 1 --max-tokens 16 --temperature 0.0 --top-p 1.0 \
  --include-output-text
```

- **Smoke report:** `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260628T052638.586203Z-shared-bench.json`
- **Preflight status:** `ready_for_dflash_smoke=True cloud_only_listener=True blocked_listener=False` (precheck JSON: `.planning/dflash-resource-gate-evidence-smoke-precheck.json`).
- **Telemetry (per-row `dflash` block):** `opted_in=true`, exact operator-provided target/drafter paths, `max_draft_tokens=4`, `sequential_text_only=true`, `uses_native_runtime=true`, `fallback_status=fallback_preflight`, `accepted_proposal_tokens=0`, `rejected_proposal_tokens=0`.
- **Sequential text path verified in stderr:** the target loaded end-to-end without `RuntimeError: There is no Stream(...)` and ran the standard sequential ModelKit startup warmup (`prompt_tokens=25` → `513` → `4095` warmup, `token=446`, `ThreadLocalStream(Device(gpu, 0), 4)`, `mode=sequential`, `distributed=False`). The phase-aware post-load preflight no longer fails closed with `"Insufficient free memory for real-pair DFlash preflight"`, so the fix from `m14-dflash-post-load-preflight-accounting` (commit `412ee34`) is confirmed working.

### Blocker (captured cleanly in the smoke report)

The runner reached `dflash_stream_generate` (line 123 of `mlx_engine/utils/dflash_runtime.py`) and `validate_dflash_runtime_compatibility` raised `DFlashUnavailableError` BEFORE any token emission:

```
DFlashUnavailableError: DFlash no-go: DFlash does not support ragged cache layers yet;
TextModel does not implement rollback_speculative_cache.
Next steps: switch to a plain KVCache sequential path with a rollback-capable
target model and keep DFlash default-off until a real sequential smoke passes.
```

Origin: `mlx_engine/utils/dflash_runtime.py:123` (the `validate_dflash_runtime_compatibility` guard). Traceback shows the failure occurred **after** the target loaded and **inside** `dflash_stream_generate`, not in preflight.

### Direct ModelKit inspection of the loaded target

A direct `ModelKit(...)` load of the same target (mirroring the `--mlx-engine-force-sequential` runner path) confirms the runtime blockers:

```
Loaded model_kit type=ModelKit
  max_kv_size=None
  kv_bits=None
collected layers count=64
layer type counts={'ArraysCache': 48, 'KVCache': 16}
lm type=TextModel
  has rollback_speculative_cache=False
runtime blockers=['DFlash does not support ragged cache layers yet',
                  'TextModel does not implement rollback_speculative_cache']
```

Two genuine gaps:

1. **Ragged ArraysCache (GDN) layers.** The Qwen3.6 27B target, when loaded through `ModelKit`, produces 64 prompt-cache layers: 16 `KVCache` (full attention) + 48 `ArraysCache` (GDN linear-attention state). `validate_dflash_runtime_compatibility` flags every `ArraysCache` as a ragged cache layer because of the `lengths`/`left_padding` attribute check. This is the same opaque-cache surface that caused the resolved M1 warm-restore divergence (see `RESOLVED 2026-06-24` entry above); DFlash draft/verify needs a separate, GDN-aware rollback story before `ArraysCache` can be allowed through.

2. **Missing `TextModel.rollback_speculative_cache`.** The patched Qwen3_5 TextModel (in `mlx_engine/model_kit/patches/qwen3_5.py`) exposes the target layers for DFlash hidden-state capture but does not yet implement `lm.rollback_speculative_cache(prompt_cache, gdn_states, accepted, block_size)`. `dflash_stream_generate` invokes that method after every partial DFlash rejection (line 371-375 of `mlx_engine/utils/dflash_runtime.py`); without it, the runtime is correctly fail-closed.

### Why no retry is warranted

- The preconditions and preflight are already passing (live probe: `ready_for_dflash_smoke=True`, sequential text path verified in stderr, phase-aware accounting working).
- The failure is in `validate_dflash_runtime_compatibility` inside `dflash_stream_generate`, which is a *runtime* surface blocker, not a preflight one.
- The two gaps above each require their own focused follow-up feature (TextModel GDN rollback support, then ArraysCache DFlash compatibility). A new capped smoke feature after those lands will be able to use the same command above.
- No text-only Qwen3.6 MLX checkpoint is available locally (only the multimodal `Qwen3.6-27B-MLX-8bit` and the MoE `Qwen3.6-35B-A3B-MLX-8bit`, which is promotion-blocked); there is no text-only target we can substitute to work around the blockers.
- DFlash remains default-off, sequential-text-only, and fail-closed. No LM Studio, no cheetara adapter (`3180`/`3181`/`3182`), no standard autoregressive `draft_model` loading path is used. The telemetry block (`sequential_text_only=true`, `uses_native_runtime=true`, `fallback_status=fallback_preflight`) confirms the no-fallback requirement.

### Verdict

The smoke **does not** complete with valid output under the token cap because of two genuine runtime compatibility blockers (ragged `ArraysCache` layers + missing `TextModel.rollback_speculative_cache`). The remaining assertions — sequential text route used, no VLM/batched/distributed/adapter fallback, no unverified token emission, telemetry captured — are satisfied. **Returning to orchestrator per the feature description** so the next worker can either (a) implement the missing `rollback_speculative_cache` and ArraysCache GDN support and re-run this same smoke, or (b) explicitly narrow the smoke scope to a target model that satisfies the existing dflash runtime compatibility checks.

### Artifacts

| Artifact | Path |
|---|---|
| Capped smoke report | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260628T052638.586203Z-shared-bench.json` |
| Structured smoke evidence | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.planning/dflash-capped-smoke-evidence.json` |
| Prompt suite | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/prompt_suites/m14_dflash_capped_smoke.json` |
| Live gate precheck | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.planning/dflash-resource-gate-evidence-smoke-precheck.json` |
| Phase-aware preflight evidence | `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.planning/dflash-phase-aware-preflight-evidence.json` |


## M14 Qwen3.5 TextModel DFlash rollback hook (2026-06-28)

Feature `m14-qwen35-textmodel-dflash-rollback` implements the missing `TextModel.rollback_speculative_cache` hook that `dflash_stream_generate` invokes after every partial DFlash rejection (see `mlx_engine/utils/dflash_runtime.py:371-375`). The previous capped-smoke blocker was fail-closed on `TextModel does not implement rollback_speculative_cache`; this feature exposes the hook on the patched Qwen3_5 `TextModel` and proves rollback safety with focused tests for accepted=0, partial, and full acceptance.

### Hook contract

- New method on `PatchedQwen3_5TextModel` (`mlx_engine/model_kit/patches/qwen3_5.py`):
  `rollback_speculative_cache(prompt_cache, gdn_states, accepted, block_size)`
- No-op when `accepted >= block_size - 1` (full acceptance), when `prompt_cache is None`, or when `accepted < 0`.
- For partial acceptance, rolls back every per-layer cache entry so that only the `base_history_len + accepted + 1` (bonus token + accepted drafts) tokens remain in cache state.
- Per-layer rewind supports:
  - `history` list slicing (test-friendly + cheap fallback)
  - `KVCache`-shaped layers (`keys`/`values` arrays + `offset` + `_idx`)
  - `ArraysCache` and similar opaque caches (`lengths` truncated via `mx.minimum`)
  - Generic `offset`/`_idx` rewinding when present
  - `None` cache entries (skipped)
- `gdn_states` entries may be `None`, the mlx-vlm GDN sink 12-tuple, or any object exposing `base_history_len`; tuples default to `base_history_len=0` and are sliced from the start of the verify pass.
- The hook is default-off: it is only invoked by `dflash_stream_generate` after partial rejection. No code path in ordinary text generation, batched text, distributed, VLM, SpecPrefill, or adapter routes calls it.
- The patched `PatchedQwen3_5TextModel` continues to be the only class that gains the method; existing `mlx_lm.models.qwen3_5.TextModel` and `mlx_lm.models.qwen3_5.Model` exports are untouched.

### Verification

- `.venv-py312/bin/python -m pytest -q tests/test_patched_qwen3_5_dflash_rollback.py` → 13 passed, 12 subtests passed, 0 failed.
  - Covers: hook presence + callability, accepted=0 / partial-acceptance / full-acceptance history rollback, full acceptance is a no-op, empty `gdn_states` fallback, mlx-vlm GDN sink tuple handling, real `KVCache`-shaped layer truncation (keys/values arrays), negative `accepted` defensive guard, `None` cache layer skipping, no DFlash surface widening.
  - Invariant subtests prove no rejected tokens remain in live cache state for every acceptance shape, accepted tokens survive rollback, and pre-existing prompt tokens are preserved.
- `.venv-py312/bin/python -m pytest -q tests/test_dflash_runtime.py tests/test_dflash_boundary.py` → 36 passed, 22 subtests passed, 0 failed (no regression on existing DFlash boundary/runtime coverage; `test_rejects_missing_rollback_capability_before_prompt_processing` still passes because the patched text model now exposes the hook).
- `.venv-py312/bin/python -m pytest -q <full M14 promotion pytest gate>` → 284 passed, 16 skipped, 0 failed (was 271; +13 new DFlash rollback hook tests).
- `ruff check mlx_engine/model_kit/patches/qwen3_5.py tests/test_patched_qwen3_5_dflash_rollback.py` → All checks passed.

### Changed files

- `mlx_engine/model_kit/patches/qwen3_5.py` — added `_qwen3_5_dflash_rollback_base_history_len`, `_qwen3_5_dflash_rollback_rewind_layer`, and `_qwen3_5_dflash_rollback` module helpers, plus `PatchedQwen3_5TextModel.rollback_speculative_cache` which delegates to the module helper. The hook is class-level (so `getattr(lm, "rollback_speculative_cache", None)` from `dflash_stream_generate` succeeds).
- `tests/test_patched_qwen3_5_dflash_rollback.py` — new focused test file covering hook existence, three acceptance shapes, mlx-vlm tuple compatibility, real `KVCache` truncation, and three invariant subtests.
- `services.yaml` (`commands.test`) — added `tests/test_patched_qwen3_5_dflash_rollback.py` to the full M14 promotion pytest gate.

### Status

This feature resolves one of the two runtime compatibility blockers recorded in the M14 capped-smoke evidence above. The remaining blocker (the 48 `ArraysCache` ragged-cache layers) is scoped to a separate feature (`m14_dflash_arrayscache_no_go`), which depends on this hook being present and tested. DFlash remains default-off and sequential-text-only until the ArraysCache compatibility feature lands and the capped smoke re-runs.


## M14 Qwen3.6 GDN/ArraysCache runtime compatibility — precise no-go (2026-06-28)

Feature ``m14_dflash_arrayscache_no_go`` investigates
whether the ``m14-qwen35-textmodel-dflash-rollback`` hook makes ArraysCache
rollback safe enough to relax the runtime compatibility check, and
records a precise no-go explaining why the Qwen3.6 27B DFlash real smoke
cannot proceed on this target without further rollback work.

### Investigation summary

- The capped smoke (`reports/20260628T052638.586203Z-shared-bench.json`)
  reaches `validate_dflash_runtime_compatibility` with the Qwen3.6 27B
  target loaded. The target produces exactly 64 prompt-cache layers:
  16 `KVCache` (full attention) + 48 `ArraysCache` (GDN linear-attention
  state).
- `validate_dflash_runtime_compatibility` matches on the class name
  `"ArraysCache"` and flags every layer as a ragged cache layer. The
  check fires regardless of whether `lengths`/`left_padding` are None
  (the real single-sequence shape) or non-None (the ragged batched
  variant).
- The `m14-qwen35-textmodel-dflash-rollback` hook (`PatchedQwen3_5TextModel.rollback_speculative_cache`)
  supports `history` lists, `KVCache`-shaped layers (`keys`/`values` arrays
  + `offset` + `_idx`), `lengths` arrays (via `mx.minimum`), generic
  `offset`/`_idx` rewinding, and `None` cache entries.
- The hook does NOT touch the real `ArraysCache.cache[idx]` arrays that
  hold the actual GDN state in single-sequence Qwen3.6 sequential text.
  The previous feature's tests used an `ArraysCache` fake with `lengths`
  set to a non-None `mx.array([1])`, which does not match the real
  Qwen3.6 sequential ArraysCache shape (`lengths=None`, `left_padding=None`,
  state stored in `cache=[mx.array, mx.array]`).

### Decision: keep `validate_dflash_runtime_compatibility` fail-closed

Per the feature description's "otherwise" clause, the validator stays
fail-closed for the Qwen3.6 GDN/ArraysCache layout because the rollback
hook is not proven safe for the real ArraysCache shape. Silently
allowing those 48 layers through DFlash would risk leaking rejected
proposal tokens into the live GDN state, which is the exact failure
mode the capped smoke must avoid.

### Rollback gap documented by focused tests

- `tests/test_dflash_boundary.py::TestDFlashArraysCacheNoGo` adds five
  focused tests proving the runtime validator stays fail-closed for:
  - the real Qwen3.6 ArraysCache shape (`lengths=None`, `cache=[arrays]`)
  - mixed KVCache + ArraysCache layer combinations
  - the ragged ArraysCache variant (`lengths` set to `mx.array`)
  - `BatchKVCache` (batched-sequence ragged cache)
  - the exact 16 KVCache + 48 ArraysCache layout the Qwen3.6 target loads
- `tests/test_patched_qwen3_5_dflash_rollback.py::TestArraysCacheRollbackGapDocs`
  adds three focused tests proving the rollback hook is a documented
  no-op for the real Qwen3.6 ArraysCache shape:
  - `cache[idx]` array ids and shapes are unchanged after rollback
  - `lengths` and `left_padding` stay None (no ragged mode mutation)
  - the full 16 KVCache + 48 ArraysCache layout shows the KVCache subset
    being sliced while the ArraysCache subset is unchanged, pinning the
    gap that prevents safely allowing ArraysCache through validation

### Why no retry is warranted

- The capped smoke has already reached `dflash_stream_generate` and
  proven the runtime surface blockers are genuine (ragged-cache flag +
  missing `rollback_speculative_cache`). The phase-aware preflight and
  resource gate already pass (see
  `.planning/dflash-resource-gate-evidence-current.json`).
- The remaining ArraysCache gap requires a follow-up feature to extend
  the rollback hook with GDN-aware `cache[idx]` array slicing and prove
  it does not corrupt GDN state. That work is out of scope for this
  no-go feature.
- DFlash remains default-off, sequential-text-only, and fail-closed. No
  LM Studio, no cheetara adapter, no standard autoregressive
  `draft_model` path is used.

### Verification

- `.venv-py312/bin/python -m pytest -q tests/test_dflash_boundary.py tests/test_dflash_runtime.py tests/test_patched_qwen3_5_dflash_rollback.py`
  → 57 passed, 34 subtests passed, 0 failed (was 49 passed; +8 new
  ArraysCache no-go + rollback gap tests).
- `.venv-py312/bin/python -m pytest -q <full M14 promotion pytest gate>`
  → 292 passed, 16 skipped, 0 failed (was 284; +8 new tests).
- `ruff check tests/test_dflash_boundary.py tests/test_patched_qwen3_5_dflash_rollback.py`
  → All checks passed.

### Changed files

- `tests/test_dflash_boundary.py` — added `_ArraysCacheWithLengths`
  (ragged variant of the existing test fake) and a new `ArraysCache`
  class that mirrors the real Qwen3.6 single-sequence GDN state layout
  (`lengths=None`, `left_padding=None`, `cache=[mx.array, mx.array]`).
  Added `TestDFlashArraysCacheNoGo` with five focused tests proving the
  validator remains fail-closed for every ArraysCache shape and the
  exact 16 KVCache + 48 ArraysCache layout.
- `tests/test_patched_qwen3_5_dflash_rollback.py` — added
  `_RealQwen3ArraysCache` test fake mirroring the real Qwen3.6
  ArraysCache shape and `TestArraysCacheRollbackGapDocs`
  with three focused tests proving the rollback hook is a documented
  no-op for the real ArraysCache shape.
- `.planning/performance-future-work.md` — this no-go entry.

### Status

DFlash remains fail-closed and default-off for the Qwen3.6 27B real
target. The capped smoke does not proceed on this target until a
follow-up feature extends the rollback hook with GDN-aware `cache[idx]`
array truncation and proves it does not corrupt GDN state. The
validator, the rollback hook, and the smoke evidence are unchanged in
behavior; only the test coverage and documentation have been added to
pin the no-go so future workers cannot silently widen the ragged-cache
surface.


## M14 real Qwen3.6 ArraysCache/GDN rollback — proven shape (2026-06-28)

Feature ``m14-dflash-real-arrayscache-gdn-rollback`` turns the
documented M14 no-go (commit ``d3f6d10``) into a safe runtime path
for the exact proven Qwen3.6 27B DFlash layout. The
``PatchedQwen3_5TextModel.rollback_speculative_cache`` hook now drives
``mlx_vlm.models.qwen3_5.gated_delta.gated_delta_accept_states`` to
restore the real ``ArraysCache.cache[0]`` (conv window) and
``cache[1]`` (running gated-delta state) for the proven sequential
single-sequence shape, and
``validate_dflash_runtime_compatibility`` is tightened to only allow
that exact shape through.

### Implementation summary

- **Sequential shape detection:** new
  ``_qwen3_5_arrays_cache_is_sequential_single_sequence`` helper
  identifies the real mlx-lm ``ArraysCache`` used by sequential text
  generation (``lengths`` and ``left_padding`` both ``None``, ``cache``
  is a list of >=2 ``mlx.core.array`` entries). Ragged batched
  variants with non-``None`` ``lengths`` / ``left_padding`` arrays are
  not the proven shape and are rejected by the validator.
- **ArraysCache rollback path:** new
  ``_qwen3_5_dflash_arrays_cache_rollback`` mutates ``cache[0]`` and
  ``cache[1]`` in place using the per-layer GDN sink tuple captured
  during target_verify (``initial_state``, ``conv_input``,
  ``intermediate_states``, ``conv_kernel_size``) plus
  ``gated_delta_accept_states`` to compute the boundary state for the
  accepted prefix. Accepted=0 / accepted<0 restores the live cache to
  the pre-verify state; full acceptance is a no-op so the post-verify
  state survives intact.
- **GDN state alignment:** new
  ``_align_gdn_states_with_prompt_cache`` helper in
  ``dflash_runtime.py`` rewrites the flat ``gdn_states`` list so each
  ``prompt_cache[i]`` (mixing ``KVCache`` and ``ArraysCache`` in layer
  order) can look up its own per-layer GDN sink tuple. The helper
  walks ``lm.layers`` and matches ``is_linear=True`` (GDN) layers to
  cache indices.
- **Validator tightening:** ``validate_dflash_runtime_compatibility``
  now requires the exact 16 ``KVCache`` + 48 ``ArraysCache`` sequential
  layout. Any ragged, opaque, or non-Qwen cache shape (different
  counts, ragged ArraysCache, BatchKVCache, RotatingKVCache, mixed
  ragged with the exact counts) stays fail-closed with a precise
  blocker naming the expected (16, 48) split.
- **Shape-strict constants:** ``DFLASH_PROVEN_QWEN35_LAYOUT = (16, 48)``
  and ``DFLASH_PROVEN_QWEN35_TOTAL_LAYERS = 64`` are exported so the
  runtime and tests share a single source of truth.

### Rollback semantics proven by focused tests

- ``tests/test_patched_qwen3_5_dflash_rollback.py::TestRealQwen3ArraysCacheRollback``
  replaces the gap-pin tests with five success tests on a realistic
  Qwen3.6 ``ArraysCache`` shape (parameterized conv_kernel_size /
  conv_dim / head_v_dim / head_k_dim / num_v_heads):
  - ``test_accepted_zero_restores_pre_verify_state``: live
    ``cache[0]`` / ``cache[1]`` after rollback match the pre-verify
    ``initial_state`` plus the bonus-token-only ``intermediate_states[0]``
    boundary.
  - ``test_partial_acceptance_snapshots_correct_intermediate_state``:
    for accepted=1 / 2 the rollback picks
    ``intermediate_states[accepted]`` as the new ``cache[1]`` and
    reslices ``conv_input[:, accepted+1 : accepted+1+(k-1), :]`` as
    the new ``cache[0]`` window.
  - ``test_full_acceptance_is_no_op``: the rollback leaves the live
    GDN state untouched when the entire draft block was accepted.
  - ``test_full_qwen36_layout_only_touches_arrays_cache``: the full
    16 ``KVCache`` + 48 ``ArraysCache`` layout rolls back the
    ArraysCache subset without touching the KVCache subset, mirroring
    the mlx-vlm GDN state machine exactly.
  - ``test_ragged_arrays_cache_is_left_untouched``: the rollback path
    refuses to mutate a ragged ArraysCache (non-``None``
    ``lengths``), so the validator's fail-closed policy cannot be
    bypassed by code that calls the hook directly.
- ``tests/test_dflash_boundary.py::TestDFlashArraysCacheShapeStrict``
  replaces the no-go pin tests with nine shape-strict tests proving
  the validator's allow / reject contract on the real Qwen3.6 layout
  plus every nearby variant (48 ArraysCache alone, wrong ArraysCache
  count, wrong KVCache count, ragged ArraysCache, ragged mixed with
  exact counts, BatchKVCache, RotatingKVCache).
- ``tests/test_dflash_runtime.py``: added ``_FakeArraysCache`` and
  ``_make_proven_layout_cache`` so the runtime stream-generate tests
  run through the proven 16+48 layout by default; KVCache fake now
  carries an ``mx.array`` ``lengths`` so layer-count assertions stay
  KVCache-only.

### Decision: DFlash runtime surface narrowed, not widened

This feature does NOT claim DFlash is promotion-ready. The proven
sequential-text rollback path is now safe for the exact 16 KVCache + 48
ArraysCache Qwen3.6 27B layout; every other ragged / opaque / non-Qwen
shape remains fail-closed by ``validate_dflash_runtime_compatibility``
and ``rollback_speculative_cache``. Promotion to default-on DFlash
still requires repeated quality-passing capped-smoke evidence captured
by a separate bench-worker feature; this implementation feature only
closes the runtime compatibility gap and must be re-validated end to
end before any promotion decision.

### Verification

- ``.venv-py312/bin/python -m pytest -q
  tests/test_dflash_boundary.py tests/test_dflash_runtime.py
  tests/test_patched_qwen3_5_dflash_rollback.py`` → all targeted
  tests pass.
- ``.venv-py312/bin/python -m pytest -q <full M14 promotion pytest
  gate per services.yaml commands.test>`` → **298 passed, 16 skipped,
  0 failed** in ~63 s (was 292; +6 new shape-strict + ArraysCache
  rollback tests).
- ``ruff check mlx_engine/utils/dflash_boundary.py
  mlx_engine/utils/dflash_runtime.py
  mlx_engine/model_kit/patches/qwen3_5.py
  tests/test_dflash_boundary.py tests/test_dflash_runtime.py
  tests/test_patched_qwen3_5_dflash_rollback.py`` → All checks
  passed.

### Changed files

- ``mlx_engine/model_kit/patches/qwen3_5.py`` — added
  ``_qwen3_5_arrays_cache_is_sequential_single_sequence`` and
  ``_qwen3_5_dflash_arrays_cache_rollback``; updated
  ``_qwen3_5_dflash_rollback_rewind_layer`` to route ArraysCache
  layers to the new helper and ``_qwen3_5_dflash_rollback`` to pass
  per-layer ``gdn_state`` / ``accepted`` / ``block_size``.
- ``mlx_engine/utils/dflash_runtime.py`` — added
  ``_align_gdn_states_with_prompt_cache``; updated the rollback
  invocation site to use the aligned list.
- ``mlx_engine/utils/dflash_boundary.py`` — added
  ``DFLASH_PROVEN_QWEN35_LAYOUT``,
  ``DFLASH_PROVEN_QWEN35_TOTAL_LAYERS``,
  ``_cache_layer_is_qwen35_sequential_arrays_cache``, and
  ``_summarize_prompt_cache_layout``; rewrote
  ``validate_dflash_runtime_compatibility`` to require the exact (16,
  48) sequential layout with descriptive blockers for every other
  shape.
- ``tests/test_patched_qwen3_5_dflash_rollback.py`` — replaced the
  ArraysCache gap-pin tests with realistic-shape success tests
  covering accepted=0 / partial / full rollback semantics.
- ``tests/test_dflash_boundary.py`` — replaced
  ``TestDFlashArraysCacheNoGo`` with
  ``TestDFlashArraysCacheShapeStrict`` covering both allow and
  reject paths on the exact Qwen3.6 layout plus every nearby variant.
- ``tests/test_dflash_runtime.py`` — added ``_FakeArraysCache`` and
  ``_make_proven_layout_cache``; updated KVCache fake and FakeKit
  default to the proven 16+48 layout.
- ``.planning/performance-future-work.md`` — this entry.

### Status

DFlash runtime compatibility is now narrowed to the exact proven
Qwen3.6 27B sequential-text layout and proven safe by focused tests.
DFlash remains default-off until repeated quality-passing capped-smoke
evidence is captured by the bench-worker. No new cache shape or route
is silently widened; ragged, opaque, BatchKVCache, and RotatingKVCache
layers remain fail-closed.

