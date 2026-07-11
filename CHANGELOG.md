# Changelog

## [Unreleased]

### Added

- Add an internal macOS distribution path with Python package metadata,
  idempotent install/update, launchd registration, revision verification,
  rollback snapshots, and uninstall support.
- Add macOS CI gates for the hermetic pytest suite and deployment contract.
- Add explicit missing-model test controls so automated runs skip unavailable
  model fixtures by default, fail under `--require-models`, and download only
  under `--download-models`.

- Add `scripts/vlm_restore_eval_split_report.py` to summarize
  `vlm_cache_restore_detail` split timing across repeated `shared_bench.py`
  reports with row-error/cache/output evidence.
- Include by-kind eval-target, materialized-byte, and record-byte aggregates in
  `scripts/vlm_restore_eval_split_report.py` output so restore barrier
  investigations can identify the dominant materialization surface.
- Add `scripts/lfm25_text_cache_bench.py` to produce repeated JSON evidence
  for the retained LFM2.5-VL text-only generated-token cache workload.
- Report follow-up cache-reuse and prefill ratios from
  `scripts/lfm25_text_cache_bench.py` so future candidate comparisons do not
  need to recompute them from token counts.
- Add `scripts/lfm25_text_cache_compare.py` to compare retained LFM2.5-VL
  text-cache reports and fail candidates with cache, prefill, row-error, or
  name-fidelity regressions.
- Add `scripts/lfm25_text_cache_report.py` to render retained LFM2.5-VL
  text-cache benchmark and comparison JSON as a readable Markdown evidence
  report.
- Add `scripts/lfm25_text_cache_promotion_gate.py` to fail closed unless
  retained LFM2.5-VL benchmark, comparison, readable report, preflight, and
  live LM Studio validation evidence all pass.
- Add `scripts/upstream_candidate_scan.py` to produce factual JSON evidence for
  repeat upstream branch triage before any cherry-pick or runtime candidate.
- Add `scripts/upstream_candidate_report.py` to render upstream scan JSON as a
  readable Markdown review report without making a promotion decision.
- Add `scripts/upstream_candidate_scan_diff.py` to compare two upstream scan
  JSON reports and render readable branch-level deltas for repeat triage.
- Add `scripts/upstream_candidate_history.py` to render a readable history from
  multiple upstream scan JSON reports for repeat triage.
- Add `eval_collect_ms` and `eval_barrier_ms` to VLM
  `vlm_cache_restore_detail` / `vlm_cache_restore_cost_model` timing events so
  restore-eval investigations can separate target discovery from the mandatory
  `mx.eval(...)` barrier.
- Add `scripts/lmstudio_vlm_download_probe.py` to run the supported LM Studio
  VLM `lms get` registration path with bounded timeout and JSON evidence.
- Add `scripts/lmstudio_vlm_live_validation_preflight.py` to record whether
  the retained LFM2.5-VL model is visible to `lms load` before live LM Studio
  VLM validation.
- Allow the shared thread-unsafe MLX stream experiment to be enabled through
  `/tmp/mlx-engine-thread-unsafe-stream` so the path can be probed inside the
  real LM Studio runtime when backend env vars are not preserved.
- Allow `scripts/lmstudio-register-engine.sh` to register against a cloned LM
  Studio MLX/app runtime pair through `LMSTUDIO_MLX_RUNTIME_NAME` and
  `LMSTUDIO_CPYTHON_RUNTIME_NAME`.

### Changed

- Record current-manifest paired Gemma4 12B and Qwen3.6 27B promotion evidence;
  Qwen3.6 passes while Gemma4 remains blocked by vMLX image-route row errors.

- Clarify that terminal-packed final KV is backed by direct retained-workload
  evidence and still requires the LM Studio VLM preflight/live-validation gate
  before broader LM Studio packaging or promotion.
- Report copied `~/.lmstudio/models` VLM directories in
  `scripts/lmstudio_vlm_live_validation_preflight.py` as diagnostic-only
  evidence while keeping `lms ls --json` visibility as the live-validation
  readiness gate.
- Reject non-semver backend versions in
  `scripts/lmstudio-register-engine.sh` and harden registry writes with backup
  and entry-count validation.
- Make `scripts/lmstudio-register-engine.sh` default to the vendor runtime pair
  recorded in the installed official LM Studio MLX backend manifest instead of
  blindly selecting the newest Amphibian app/Python runtimes on disk.

### Fixed

- Preserve home-owned staged models when updating the internal runtime so
  `rsync --delete` cannot remove the model selected by the launchd service.

- Keep final-boundary VLM terminal KV saves from overwriting exact
  reusable-prefix opaque state checkpoints when final-chunk state alignment is
  enabled.
- Fail fast when `scripts/lmstudio-register-engine.sh` targets an MLX runtime
  whose `mlx/core.cpython-311-darwin.so` code signature is incompatible with
  the active LM Studio worker process.
- Write the selected vendor runtime pair into the generated LM Studio backend
  `backend-manifest.json` so cloned-runtime registrations do not silently fall
  back to the stock bundled MLX runtime.
- Preserve older custom backend versions in LM Studio's internal engine index
  when registering a new version with the same backend name.
- Keep batched cross-request prompt-cache keys anchored to the original prompt so generated tokens do not poison subsequent prompt-cache reuse for GPT-OSS style batched text models.
