# Cheetara Replacement Plan (vmlx_engine vs mlx-engine)

## Objective
Upgrade `mlx-engine` so it can replace the current `cheetara` default engine only when end-to-end quality and performance are better than `vmlx_engine` for your target workloads.

## Hard Facts from Current Code Base
- `cheetara` is a de-packaged app payload with its own `vmlx_engine` server and does **not** expose a generic backend-plug-in flag to swap to another engine at runtime.  
  - Source: [`/Users/jeffreycruz/Development/LLM_INFERENCE/cheetara/README.md`](../../cheetara/README.md)
- `cheetara`’s Python entry path is `vmlx_engine` and currently includes features not yet fully matched by `mlx-engine` (notably richer multimodal and cache stack components).
- `mlx-engine` has a mature benchmark harness and quality gates (`shared_bench.py`) that already support deterministic quality checks for speed regression control.

## Non-Negotiable Promotion Gates (before replacement)
1. **Quality gate first**: all deterministic suites pass, including no visible reasoning leak regressions.
2. **Performance gate**:
   - text workloads: `ttft`, `decode_tps`, and `total_s` improve in direction and magnitude over baseline noise floor.
   - VLM workloads: persistent-cache warm path and long-prompt cold prefill are not worse than baseline by default thresholds.
3. **Stability gate**: zero row-level benchmark errors (`row_error`), no crash/restart regressions, no startup hangs.
4. **Scope gate**: must preserve current supported template/model family behavior in user-critical models.

## Why direct replacement is non-trivial
- `cheetara` is not a pure engine module swap in-app (no runtime backend dropdown for this in source).
- A true replacement means either:
  - repackaging `vmlx.app.asar` and changing app wiring, or
  - keeping UI as-is and routing LM Studio/clients through an external `mlx-engine` integration path.
- This should be done only after we have clear benchmark evidence and a migration script.

## Proposed Work Phases

### Phase 1 — Evidence capture
- Add a dedicated cheetara-vs-mlx benchmark profile in `mlx-bench-harness` that can run against both stacks with identical prompts/model files.
- Establish baseline reports for:
  - short prompt (text)
  - long prompt
  - image prompt suites (if applicable)
- Save all baselines in `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/`.

### Phase 2 — Low-risk `mlx-engine` gains
- Prioritize features with high ROI and isolated risk:
  - prompt cache lookup path enhancements
  - vision cache fast-path / embedding cache parity where relevant
  - restore-materialization tuning that preserves `mx.eval(...)` stream safety
- Keep every change behind benchmark + quality evidence.

### Phase 3 — Cutover decision
- If `mlx-engine` meets gates above:
  - package a custom `mlx-engine` backend for LM Studio with `scripts/lmstudio-register-engine.sh` and validate in real workflow.
  - then evaluate whether to keep cheetara as-is or repurpose it as UI client pointing to the external backend.

### Phase 4 — Cheetara swap (only if needed)
- If you need cheetara replacement instead of LM Studio:
  - modify `vmlx.app.asar` wiring or packaged startup path to launch `mlx-engine`-compatible runtime.
  - remove incompatible assumptions in the app payload around `vmlx_engine` modules one-by-one, with smoke tests at each step.
  - repack/re-sign with explicit install verification.

## Immediate Next Milestone
- Implement the benchmark profile and first evidence baseline for Phase 1 so we can compare apples-to-apples before more code work.

## References
- current branch context and constraints: `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.planning/.continue-here.md`
- existing baseline summary: `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness/reports/20260618-upstream-baseline-comparison.md`
- LM Studio backend registration helper (if going via LM Studio): `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/scripts/lmstudio-register-engine.sh`
