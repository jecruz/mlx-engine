# Changelog

## [Unreleased]

### Added

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
