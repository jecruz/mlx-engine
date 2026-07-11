# Internal Production Readiness Closeout

Date: 2026-07-10

## Test and Deployment Gates

- Model-backed tests no longer prompt during automated pytest runs.
- Default behavior skips missing local model fixtures.
- `--require-models` turns missing fixtures into hard failures.
- `--download-models` is the only automated path that permits downloads.
- macOS CI runs the complete hermetic suite and validates package/service
  artifacts.
- The supported runtime installs under `~/.local/share/mlx-engine`, outside the
  source checkout, with launchd, verification, rollback, and uninstall paths.

## Promotion-Classified Evidence

All four paired reports use
`route_type=approved-direct-cheetara-vmlx`, current comparison manifests, three
runs, deterministic sampling, output text, and the shared short-VLM quality
suite.

Qwen3.6 27B 4-bit:

- Baseline: `reports/20260711T005745.389781Z-shared-bench.json`
- Candidate: `reports/20260711T010028.751915Z-shared-bench.json`
- Compare: `reports/20260711T010028.751915Z-qwen36-promotion-compare.json`
- Result: prompt quality `pass`; promotion gate `pass`; both paired runners
  completed all rows without errors.

Gemma4 12B:

- Baseline: `reports/20260711T005616.535231Z-shared-bench.json`
- Candidate: `reports/20260711T010340.072827Z-shared-bench.json`
- Compare: `reports/20260711T010340.072827Z-gemma4-promotion-compare.json`
- Result: mlx-engine prompt quality `pass`; promotion gate `fail`.
- Blocker: vMLX classifies this checkpoint as `mllm=False` and returns HTTP 400
  for all six image rows in both baseline and candidate reports.

The compare gate was hardened in the paired harness to inspect row errors from
every required result, preventing a clean primary runner from masking a failed
secondary runner.
