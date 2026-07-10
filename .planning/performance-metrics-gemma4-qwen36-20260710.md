# Gemma4 12B and Qwen3.6 27B Fresh Metrics Report

Fresh benchmark evidence was collected on 2026-07-10 with the same short VLM
prompt suite for all three checkpoints.

Method:

- Suite: `prompt_suites/vlm_image_quality.json`
- Runs: `3`
- Max tokens: `96`
- Temperature: `0.0`
- Top-p: `1.0`
- Engine: `mlx-engine`

Artifacts:

- Gemma4 12B: `reports/20260710T045748.986785Z-shared-bench.json`
- Qwen3.6 27B 4-bit: `reports/20260710T045849.381011Z-shared-bench.json`
- Qwen3.6 27B 8-bit: `reports/20260710T045927.257035Z-shared-bench.json`

## Gemma4 12B

- Model path: `/Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/mlx-community/gemma-4-12B-it-8bit`
- Row errors: `0`

| Prompt | Avg TTFT s | Avg Decode TPS | Avg Total s | Avg Prompt Tokens | Avg Cached Tokens | Avg Completion Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `image_pair` | 0.430173 | 43.934 | 2.615305 | 45.0 | 388.67 | 96.0 |
| `image_toucan` | 0.780321 | 44.488 | 2.713456 | 38.0 | 201.33 | 86.0 |

Average across prompts:

- TTFT: `0.605247 s`
- Decode TPS: `44.211`
- Total time: `2.664380 s`

## Qwen3.6 27B 4-bit

- Model path: `/Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/mlx-community/Qwen3.6-27B-4bit`
- Row errors: `0`

| Prompt | Avg TTFT s | Avg Decode TPS | Avg Total s | Avg Prompt Tokens | Avg Cached Tokens | Avg Completion Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `image_pair` | 0.663855 | 38.610 | 3.150469 | 47.0 | 306.67 | 96.0 |
| `image_toucan` | 0.830573 | 38.957 | 3.063984 | 38.0 | 68.0 | 87.0 |

Average across prompts:

- TTFT: `0.747214 s`
- Decode TPS: `38.784`
- Total time: `3.107226 s`

## Qwen3.6 27B 8-bit

- Model path: `/Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/lmstudio-community/Qwen3.6-27B-MLX-8bit`
- Row errors: `0`

| Prompt | Avg TTFT s | Avg Decode TPS | Avg Total s | Avg Prompt Tokens | Avg Cached Tokens | Avg Completion Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `image_pair` | 0.720189 | 22.760 | 4.938213 | 47.0 | 306.67 | 96.0 |
| `image_toucan` | 1.920201 | 22.807 | 6.129361 | 38.0 | 68.0 | 96.0 |

Average across prompts:

- TTFT: `1.320195 s`
- Decode TPS: `22.784`
- Total time: `5.533787 s`

## Comparison

- Gemma4 12B was the fastest checkpoint on this short VLM suite.
- Compared with Qwen3.6 27B 4-bit, Gemma4 improved average TTFT by about
  `23.5%`, improved decode TPS by about `14.0%`, and reduced total time by
  about `16.6%`.
- Qwen3.6 27B 4-bit was materially faster than Qwen3.6 27B 8-bit on the same
  suite:
  - TTFT was about `1.77x` lower.
  - Decode TPS was about `1.70x` higher.
  - Total time was about `1.78x` faster.
- All three runs completed with zero row errors, so the numbers above are from
  clean benchmark rows rather than partial failures.

## Interpretation

- For this machine and this prompt suite, Gemma4 12B is the better latency and
  throughput choice for short image prompts.
- If the Qwen3.6 family is required, the 4-bit checkpoint is the safer
  performance pick over 8-bit because it preserves the same row-level quality
  and cuts latency substantially.
