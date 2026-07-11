# M87 Upstream Candidate Triage

This report is evidence only. It records a focused review of the current
upstream candidate set and does not authorize a cherry-pick or runtime
promotion.

## Reviewed Candidates

- `upstream/neil/gemma4-tool-context`
- `upstream/neil/img-caching`
- `upstream/will/lfm-2.5-unified`
- `upstream/yagil/mlx-dist-non-batched`
- `upstream/neil/vlm-parity-ci`

## Triage Result

No candidate was promoted for further benchmark work in this pass.

## Why

- `gemma4-tool-context` is a broad Gemma4 tool/runtime refactor with 7 changed
  files and 868 lines of diffstat. The commit trail is primarily grammar and
  runtime simplification, not an isolated latency win.
- `img-caching` is a wide VLM rewrite with 94 changed files and 15k+ lines of
  deletions. The branch reads as a large architecture reset rather than a small,
  reversible cache improvement.
- `lfm-2.5-unified` is a single caching test commit with no broader upstream
  performance signal in this branch snapshot.
- `mlx-dist-non-batched` is a distributed-stream/model-thread refactor with 84
  changed files; the commit trail centers on threading and stream bridge
  behavior, not a directly measurable retained-workload latency delta.
- `vlm-parity-ci` is parity/validation oriented; it improves test coverage and
  concurrency checks rather than a concrete runtime speedup.

## Evidence

- `git log --oneline --max-count=8 upstream/main..upstream/neil/gemma4-tool-context`
- `git diff --stat upstream/main..upstream/neil/gemma4-tool-context`
- `git log --oneline --max-count=8 upstream/main..upstream/neil/img-caching`
- `git diff --stat upstream/main..upstream/neil/img-caching`
- `git log --oneline --max-count=8 upstream/main..upstream/will/lfm-2.5-unified`
- `git diff --stat upstream/main..upstream/will/lfm-2.5-unified`
- `git log --oneline --max-count=8 upstream/main..upstream/yagil/mlx-dist-non-batched`
- `git diff --stat upstream/main..upstream/yagil/mlx-dist-non-batched`
- `git log --oneline --max-count=8 upstream/main..upstream/neil/vlm-parity-ci`
- `git diff --stat upstream/main..upstream/neil/vlm-parity-ci`

## Next Step

Resume the upstream scan loop on the next head change or move back to
retained-workload benchmarking if a smaller candidate emerges.
