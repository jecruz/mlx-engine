# Gemma4 12B and Qwen3.6 27B Metrics Report

This report is evidence only. The numbers below come from retained benchmark,
inspection, and closeout artifacts already checked into the repo. The Gemma4 and
Qwen3.6 workloads are not identical, so the figures are best read as separate
model-family metrics rather than a single apples-to-apples ranking.

## Gemma4 12B

Source:

- `README.md` M20 and M22 closeout notes

Retained short VLM evidence:

- Zero row errors: `true`
- Quality fidelity: `chameleon` and `toucan` keywords passed
- Image prompt shape: short VLM image path

Metrics:

| Workload | TTFT seconds | Decode TPS | Total seconds |
| --- | ---: | ---: | ---: |
| `image_toucan` | 0.996581 | 44.371 | 2.934772 |
| `image_pair` | 1.047883 | 43.854 | 3.236964 |

Additional retained long-pair stress evidence:

- Warm persistent-cache rows passed until a `Stream(gpu, 3)` failure on the warm path
- Cached tokens before failure: `7619`
- Materialized bytes before failure: `460,374,016`
- Eval targets before failure: `48`

Interpretation:

- Gemma4 12B is the faster of the two families on the short VLM image lane in the retained evidence.
- The long-pair lane is useful as a stability signal, but it is not a promotable speed result because it ended in a runtime failure.

## Qwen3.6 27B

Source:

- `README.md` M23 and M29 closeout notes

Retained direct VLM 4-bit evidence:

- Zero row errors: `true`
- Quality inspect pass: `true`
- Route: direct VLM / batched vision

Metrics:

| Workload | TTFT seconds | Decode TPS | Total seconds | Completion |
| --- | ---: | ---: | ---: | ---: |
| `image_toucan` | 5.245937 | 42.678 | 7.495293 | 96 |
| `image_pair` | 5.418849 | 42.595 | 7.672431 | 96 |

Balanced sweep evidence:

- Quality score pass rate: `1.000`
- Mean quality score: `1.000`
- 4-bit retained model was `1.69x` faster on decode TPS than 8-bit
- 4-bit retained model was `1.85x` faster end-to-end than 8-bit

Interpretation:

- Qwen3.6 27B has substantially higher TTFT than Gemma4 12B in the retained direct VLM evidence, but the direct workloads differ in shape and are not directly comparable.
- On the balanced sweep, the retained 4-bit variant is the safer throughput choice over the 8-bit variant because it preserved quality and won on throughput.

## Practical Takeaway

- If you care about short VLM response latency on this machine, the retained Gemma4 12B short lane is the fastest recorded evidence here.
- If you care about a larger Qwen-family model, the retained Qwen3.6 27B 4-bit lane is quality-safe and materially faster than its 8-bit sibling.
- If you want a fresh, apples-to-apples comparison, the next step is to run the same prompt suite against both models under the same harness and cache state.

