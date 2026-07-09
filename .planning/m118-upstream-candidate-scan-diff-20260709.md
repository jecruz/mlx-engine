# M118 Upstream Candidate Scan Diff

This report is evidence only. It highlights scan deltas; it does not
authorize a cherry-pick or runtime promotion.

## Summary

- Baseline head: `7c28d6e`
- Candidate head: `7c28d6e`
- Baseline upstream/main: `8ae2610`
- Candidate upstream/main: `8ae2610`
- Baseline origin branch: `8f0fa26`
- Candidate origin branch: `8f0fa26`
- Baseline candidate count: `6`
- Candidate candidate count: `6`

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| head_vs_upstream_main.left | 244 | 244 | 0 |
| head_vs_upstream_main.right | 1 | 1 | 0 |
| head_vs_origin_branch.left | 49 | 49 | 0 |
| head_vs_origin_branch.right | 0 | 0 | 0 |

## Branch Deltas

| Branch | Baseline head | Candidate head | Status | Surface | Baseline files | Candidate files | Baseline unmatched | Candidate unmatched |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `upstream/neil/gemma4-tool-context` | `9aa3db2` | `9aa3db2` | unchanged | broad | 7 | 7 | 18 | 18 |
| `upstream/neil/img-caching` | `7dfe3cd` | `7dfe3cd` | unchanged | broad | 94 | 94 | 13 | 13 |
| `upstream/neil/vlm-parity-ci` | `ea1a6bb` | `ea1a6bb` | unchanged | broad | 44 | 44 | 82 | 82 |
| `upstream/will/lfm-2.5-unified` | `461015c` | `461015c` | unchanged | broad | 83 | 83 | 1 | 1 |
| `upstream/yagil/dist` | `366ebd4` | `366ebd4` | unchanged | broad | 84 | 84 | 0 | 0 |
| `upstream/yagil/mlx-dist-non-batched` | `c86c23a` | `c86c23a` | unchanged | broad | 84 | 84 | 6 | 6 |

## Scan Summary Deltas

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| head_vs_upstream_main.left | 244 | 244 | 0 |
| head_vs_upstream_main.right | 1 | 1 | 0 |
| head_vs_origin_branch.left | 49 | 49 | 0 |
| head_vs_origin_branch.right | 0 | 0 | 0 |

## Candidate Notes

- This report is factual only; it does not classify promotion readiness.
- A cherry-pick still requires human triage, retained benchmarks, quality gates, and live LM Studio validation before promotion.
