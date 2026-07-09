# Changelog

## [Unreleased]

### Added

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
