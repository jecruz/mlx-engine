# M63 LFM2.5 Text-Cache Evidence

This report is evidence only. Runtime promotion still requires repeated
retained-workload wins, passing quality gates, candidate-vs-baseline
deltas, and live LM Studio validation.

## Benchmark `.planning/m53-lfm25-text-cache-ratio-bench-20260709.json`

- Model path: `/Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/lmstudio-community/LFM2.5-VL-1.6B-MLX-8bit`
- Samples: `3`
- Row errors: `0`
- All followups cached: `true`
- All followups small prefill: `true`
- All outputs preserve name: `true`
- Prefill step size: `512`
- Story tokens: `512`
- Followup tokens: `64`

| Metric | Average |
| --- | ---: |
| First-turn TTFT seconds | 0.060995 |
| Follow-up TTFT seconds | 0.018306 |
| Follow-up total seconds | 0.027171 |
| Follow-up cached tokens | 542.000000 |
| Follow-up total prompt tokens | 565.000000 |
| Follow-up prefill tokens | 23.000000 |
| Follow-up cache reuse ratio | 0.959292 |
| Follow-up prefill ratio | 0.040708 |

| Sample | Follow-up cached tokens | Follow-up prefill tokens | Follow-up TTFT seconds | Follow-up output | Error |
| ---: | ---: | ---: | ---: | --- | --- |
| 1 | 542 | 23 | 0.018492 | Silas. |  |
| 2 | 542 | 23 | 0.018460 | Silas. |  |
| 3 | 542 | 23 | 0.017965 | Silas. |  |

## Comparison `.planning/m54-lfm25-text-cache-m53-vs-m52-20260709.json`

- Status: `pass`
- Baseline: `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.planning/m52-lfm25-text-cache-bench-20260709.json`
- Candidate: `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.planning/m53-lfm25-text-cache-ratio-bench-20260709.json`

| Check | Status | Value | Threshold |
| --- | --- | ---: | ---: |
| candidate_row_errors | pass | 0 | 0 |
| candidate_followups_cached | pass | true | true |
| candidate_followups_small_prefill | pass | true | true |
| candidate_outputs_preserve_name | pass | true | true |
| cache_reuse_ratio_regression | pass | 0.000000 | 0.010000 |
| prefill_ratio_regression | pass | 0.000000 | 0.010000 |

| Metric | Baseline | Candidate | Delta | Delta percent |
| --- | ---: | ---: | ---: | ---: |
| sample_count | 2 | 3 |  |  |
| followup_cache_reuse_ratio_avg | 0.959292 | 0.959292 | 0.000000 | 0.000000 |
| followup_prefill_ratio_avg | 0.040708 | 0.040708 | 0.000000 | 0.000000 |
| followup_ttft_s_avg | 0.018740 | 0.018306 | -0.000434 | -2.317615 |
| followup_total_s_avg | 0.027828 | 0.027171 | -0.000657 | -2.360200 |
