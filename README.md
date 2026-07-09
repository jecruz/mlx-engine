<p align="center">
  <picture> 
    <img alt="lmstudio + MLX" src="https://github.com/user-attachments/assets/128bf3ba-d8d6-4fc8-85c9-4d0113ba5499">
  </picture>
</p>

<p align="center"><bold><code>mlx-engine</code> - <a href="https://github.com/ml-explore/mlx">Apple MLX</a> LLM Engine for <a href="https://lmstudio.ai/">LM Studio</a></bold></p>
<br/>
<p align="center"><a href="https://discord.gg/aPQfnNkxGC"><img alt="Discord" src="https://img.shields.io/discord/1110598183144399058?logo=discord&style=flat&logoColor=white"></a></p>

# mlx-engine
MLX engine for LM Studio

<br/>

## Built with
- [mlx-lm](https://github.com/ml-explore/mlx-lm) - Apple MLX inference engine (MIT)
- [Outlines](https://github.com/dottxt-ai/outlines) - Structured output for LLMs (Apache 2.0)
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) - Vision model inferencing for MLX (MIT)

<br/>

## How to use in LM Studio
LM Studio 0.3.4 and newer for Mac ships pre-bundled with mlx-engine.
Download LM Studio from [here](https://lmstudio.ai/download?os=mac)

<br/>

## Standalone Demo

### Prerequisites

- macOS 14.0 (Sonoma) or greater.
- python3.11
  - The requirements.txt file is compiled specifically for python3.11. python3.11 is the python version bundled within the LM Studio MLX runtime
  - `brew install python@3.11` is a quick way to add python3.11 to your path that doesn't break your default python setup

### Install Steps
To run a demo of model load and inference:
1. Clone the repository
```
git clone https://github.com/lmstudio-ai/mlx-engine.git
cd mlx-engine
```
2. Create a virtual environment (optional)
```
 python3.11 -m venv .venv
 source .venv/bin/activate
```
3. Install the required dependency packages
```
pip install -U -r requirements.txt
```

### Text Model Demo
Download models with the `lms` CLI tool. The `lms` CLI documentation can be found here: https://lmstudio.ai/docs/cli
Run the `demo.py` script with an MLX text generation model:
```bash
lms get mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
python demo.py --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit 
```
[mlx-community/Meta-Llama-3.1-8B-Instruct-4bit](https://model.lmstudio.ai/download/mlx-community/Meta-Llama-3.1-8B-Instruct-4bit) - 4.53 GB

This command will use a default prompt. For a different prompt, add a custom `--prompt` argument like:
```bash
lms get mlx-community/Mistral-Small-Instruct-2409-4bit
python demo.py --model mlx-community/Mistral-Small-Instruct-2409-4bit --prompt "How long will it take for an apple to fall from a 10m tree?"
```
[mlx-community/Mistral-Small-Instruct-2409-4bit](https://model.lmstudio.ai/download/mlx-community/Mistral-Small-Instruct-2409-4bit) - 12.52 GB

`demo.py` loads text models with the low-latency sequential path by default. Use `batched_demo.py` and pass `max_seq_nums=4` if you want the continuous-batching path instead.

### Vision Model Demo
Run the `demo.py` script with an MLX vision model:
```bash
lms get mlx-community/pixtral-12b-4bit
python demo.py --model mlx-community/pixtral-12b-4bit --prompt "Compare these images" --images demo-data/chameleon.webp demo-data/toucan.jpeg
```
Currently supported vision models include:
 - [Llama-3.2-Vision](https://model.lmstudio.ai/download/mlx-community/Llama-3.2-11B-Vision-Instruct-4bit)
   - `lms get mlx-community/Llama-3.2-11B-Vision-Instruct-4bit`
 - [Pixtral](https://model.lmstudio.ai/download/mlx-community/pixtral-12b-4bit)
   - `lms get mlx-community/pixtral-12b-4bit`
 - [Qwen2-VL](https://model.lmstudio.ai/download/mlx-community/Qwen2-VL-7B-Instruct-4bit)
   - `lms get mlx-community/Qwen2-VL-7B-Instruct-4bit`
 - [Llava-v1.6](https://model.lmstudio.ai/download/mlx-community/llava-v1.6-mistral-7b-4bit)
   - `lms get mlx-community/llava-v1.6-mistral-7b-4bit`

### Speculative Decoding Demo
Run the `demo.py` script with an MLX text generation model and a compatible `--draft-model`
```bash
lms get mlx-community/Qwen2.5-7B-Instruct-4bit
lms get lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit
python demo.py \
    --model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --draft-model lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit \
    --prompt "<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Write a quick sort algorithm in C++<|im_end|>
<|im_start|>assistant
"
```

Classic speculative decoding with `--draft-model` is the standard
autoregressive draft-model path.

SuffixDecoding/N-gram speculation is an experimental sequential-text opt-in.
It is default-off because repeated Qwen benchmark samples passed quality checks
but did not prove a repeatable latency win.

Native DFlash is also a default-off, fail-closed foundation for Qwen-family
sequential text only. It wires local `DFlashDraftModel` drafter
metadata/readiness, hidden-state hooks, a draft/verify scaffold, and KV/GDN
rollback safety tests, but it is not a default runtime path or promotion-ready
feature. DFlash is separate from standard autoregressive draft models and is
not supported for VLM, batched, distributed, adapter, SpecPrefill, loaded
`draft_model`, or unsupported cache-mode paths in this foundation stage.

M14 closeout kept DFlash default-off and opt-in only. The native runtime path
now has evidence for preflight, harness telemetry, capped smoke, verified-token
emission, rejected-token cleanup, and KV/GDN rollback safety, but promotion was
rejected after real Qwen3.6 + z-lab DFlash validation: row errors stayed at
zero, yet quality/performance promotion gates failed because the DFlash path
regressed TTFT, decode TPS, and total latency. See commits `9582970`
(runtime-loop fix), `8f98e8a` (quality negative evidence), and `fd32c99`
(performance REJECT) for the final evidence chain.

M15 closeout also keeps DFlash default-off and explicit opt-in only. Adaptive
scheduling and low-acceptance/pathological target-only fallback ran against the
real Qwen3.6 target plus z-lab Qwen3.5 DFlash drafter with zero row errors, but
every row fell back to `fallback_pathological_target_only` and the
quality/performance gates failed. The final M15 decision is REJECT / KEEP
OPT-IN / NO PROMOTION; see the 20260629T205452 shared-bench, quality-compare,
and `.planning/dflash-adaptive-quality-performance-decision-20260629T205452Z.json`
artifacts for the closeout evidence.

M16 closeout tested original/reference DFlash locally through the `mlx_vlm`
reference implementation, not the native `mlx-engine` DFlash runtime. The
load-once A/B found baseline effective generation TPS `25.06` versus reference
DFlash `20.54`, a `-18.04%` regression, with `1/3` prompt wins and `2/3`
losses. Outputs matched and quality was preserved, but reference DFlash failed
as a general optimization, so further DFlash porting/optimization is NO-GO
unless the user explicitly overrides. Evidence lives under
`.planning/m16-reference-dflash-benchmark/`; see commits `8cf32d6` and
`1229cc5` for the M16 closeout and lint-fix evidence chain.

M17 closeout validated upstream Qwen/VLM cherry-pick candidates as a focused
audit, not a broad merge. No new Qwen/VLM cherry-pick is needed because the
relevant upstream/cherry-pick content is already present by ancestry or content
equivalence, and the focused Qwen/VLM tests pass. Gemma4-only upstream #340
(`8ae2610`) remains deferred unless scope expands. DFlash remains closed/no-go
and default-off. See commits `a876def` and `3f9481b` for the audit decision and
focused validation evidence; scrutiny passed with `404` passed / `16` skipped
and ruff clean, and user-testing passed VAL-M17-001 through VAL-M17-004.

M18 closeout manually applied the focused Gemma4 upstream #340 change
(`8ae2610`) because direct cherry-pick conflicted in batched-vision
`model_kit.py`. Gemma4-family configs/models with
`use_bidirectional_attention == "vision"` now receive the visual-prefill policy
previously limited to `gemma4_unified*`, while non-bidirectional Gemma4 and
non-Gemma models remain unchanged. See commits `3c0a0ae`, `5b4243a`, and
`4ec4345`; focused validation passed `19` tests with `20` deselected, scrutiny
passed `411` passed / `16` skipped / `52` subtests and ruff clean, and
user-testing passed VAL-M18-001 through VAL-M18-004. No broad upstream
merge/cherry-pick occurred; Qwen/VLM remains stable from M17, and DFlash remains
closed/no-go/default-off.

M19 closeout refreshed a data-only baseline matrix and regression radar, not a
promotion lane. Retained evidence covers Qwen3.5-9B dense default and
forced-sequential, Qwen2.5-Coder-14B sequential, and LFM2.5-VL short plus
persistent-long runs under `reports/20260630T130730.730931Z-*`,
`reports/20260630T130858.441935Z-*`, `reports/20260630T131852.120849Z-*`,
`reports/20260630T132919.996156Z-*`, and
`reports/20260630T133031.773011Z-*`; the optional VLM long-pair was not retained
after quality failed/missed toucan with `cached_tokens` still `0`. Gemma4 was
blocked by no usable local MLX safetensors checkpoint, Qwen3.6 27B by
memory/headroom and the active LLMDYNAMIX listener, DFlash remains
no-go/default-off, MoE remains excluded from promotion evidence, and no LM
Studio runtime was used. See commits `a46dedb`, `5829553`, `7e95692`,
`264888a`, and `8b79c3a`; scrutiny passed `411` passed / `16` skipped / `52`
subtests and ruff clean, and user-testing passed VAL-M19-001 through
VAL-M19-006.

M20 closeout added direct Gemma4 VLM data-only/no-promotion evidence from the
local MLX checkpoint
`/Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/mlx-community/gemma-4-12B-it-8bit`,
with no LM Studio runtime, no forced sequential text substitute, no DFlash, and
DFlash still no-go/default-off. Retained short evidence is
`reports/20260630T161943.247230Z-shared-bench.json` with inspect
`reports/20260630T161943.247230Z-gemma4-12b-vlm-short-quality-inspect.json`;
quality passed with zero row errors and chameleon/toucan keywords. Short metrics
were image_toucan TTFT `0.996581s`, decode TPS `44.371`, total `2.934772s`,
and image_pair TTFT `1.047883s`, decode TPS `43.854`, total `3.236964s`.
Optional long-pair stress evidence is
`reports/20260630T162050.589588Z-shared-bench.json` with inspect
`reports/20260630T162050.589588Z-gemma4-12b-vlm-long-pair-quality-inspect.json`;
quality passed with TTFT `12.350718s`, decode TPS `34.796`, total
`12.810545s`. Scrutiny passed `411` passed / `16` skipped / `52` subtests and
ruff clean, and user-testing passed VAL-M20-001 through VAL-M20-004.

M21 closeout ran prefill step-size optimization as direct
`shared_bench.py --prefill-step-size` sweeps. Evidence covered Qwen3.5 dense
default, Qwen3.5 forced sequential, Qwen2.5-Coder forced sequential,
LFM2.5-VL short, LFM2.5-VL persistent long, and Gemma4 12B short VLM, with
omitted/default plus explicit `1024`, `2048`, `4096`, and `8192` candidates
where feasible. All retained reports passed row-error and quality inspection,
but the final decision is REJECT / no default change because apparent wins were
quality-failing, small/noisy, route-local, or lacked at least two repeated
quality-passing samples. Notable evidence: Qwen dense `2048` failed compare on
warm TTFT regression; Qwen sequential `1024` and Qwen2.5-Coder `8192` passed
compare but had small single-sample margins; LFM2.5-VL `8192` and Gemma4
`2048` looked promising but were single-sample only. VLM manifests are
`reports/20260630T181722Z-m21-vlm-gemma4-sweep-manifest.json` and
`reports/20260630T181722Z-m21-vlm-gemma4-sweep-manifest-analysis.json`; Qwen
summary is `reports/20260630T180105Z-m21-qwen-text-step-sweep-summary.json`.
Scrutiny passed `411` passed / `16` skipped / `52` subtests and ruff clean,
and user-testing passed VAL-M21-001 through VAL-M21-005. See commits `ecd7f13`,
`347b259`, `ec1549d`, and `756e065`. No LM Studio runtime, DFlash, adapter
route, or MoE promotion evidence was used, and explicit `--prefill-step-size`
overrides remain preserved.

M22 closeout completed the persistent VLM cache materialization reduction lane
as instrumentation-only REJECT / no promotion. Added materialization counters
cover record counts, eval target counts, materialized bytes by cache kind,
record bytes, and restore timing fields while preserving the restore-time
`mx.eval(...)` barrier and old-cache readability. LFM2.5-VL persistent long
evidence passed quality but was not repeatably faster: reports
`reports/20260701T011556.206278Z-shared-bench.json` and
`reports/20260701T012718.987501Z-shared-bench.json` both inspect/compare pass,
with warm `cached_tokens` `7373`, materialized bytes `90,681,344`, and eval
targets `16`. Gemma4 12B persistent long-pair was retained only as
rejection/blocker evidence: reports
`reports/20260701T012908.173985Z-shared-bench.json` and
`reports/20260701T013133.799506Z-shared-bench.json`; cold rows passed but warm
rows failed with `RuntimeError: There is no Stream(gpu, 3) in current thread`.
Gemma4 materialized bytes reached `460,374,016` with eval targets `48` before
failure. Scrutiny passed `414` passed / `16` skipped / `52` subtests and ruff
clean, and user-testing passed VAL-M22-001 through VAL-M22-005.

M23 closeout completed the user-requested direct test of
`/Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/mlx-community/Qwen3.6-27B-4bit`
as data-only PASS / no promotion / no default change. The checkpoint was
classified as Qwen3.5-family VLM/image-text, 4-bit affine, about `14.952 GiB`
safetensors. Report:
`reports/20260701T021407.298827Z-shared-bench.json`; inspect:
`reports/20260701T021407.298827Z-m23-qwen36-27b-4bit-quality-inspect.json`.
The run used the direct VLM/batched-vision route, with no forced sequential
text, no LM Studio runtime, no LLMDYNAMIX route, no adapter, no DFlash, and no
MoE. The full `vlm_image_quality.json` suite had zero row errors and quality
inspect pass. Metrics: `image_toucan` TTFT `5.245937s`, decode TPS `42.678`,
total `7.495293s`, completion `96`; `image_pair` TTFT `5.418849s`, decode TPS
`42.595`, total `7.672431s`, completion `96`. Scrutiny and user-testing passed
VAL-M23-001 through VAL-M23-005.

M24 closeout fixed the Gemma4 12B persistent-cache warm restore
`Stream(gpu, 3)` failure. The repro report
`reports/20260701T041539.771194Z-shared-bench.json` reached warm
`cached_tokens=7619` before failing; the final decision report
`reports/20260701T052904.546758Z-shared-bench.json` and inspect
`reports/20260701T052904.546758Z-m24-gemma4-decision-quality-inspect.json`
passed with zero row errors, warm cached-token reuse, and chameleon/toucan
fidelity. Final decision: FIXED for the current checkout, with the restore-time
`mx.eval(...)` barrier and backward-readable cache records preserved. See
commits `805d86a`, `63c0b36`, `20d1cb9`, and `30065ef`; user-testing passed
VAL-M24-001 through VAL-M24-005.

M25 closeout completed the Qwen3.6 27B 4-bit direct VLM sweep as data-only PASS
/ no promotion / no default change. The retained baseline is
`reports/20260701T130407.875982Z-shared-bench.json` with inspect
`reports/20260701T130407.875982Z-quality-inspect.json`; sweep cells are
summarized in `reports/m25-qwen36-sweeps/summary.json`; persistent-cache long
lanes are retained under `reports/m25-qwen36-long/single/` and
`reports/m25-qwen36-long/pair/`. All prefill-step, `max_seq_nums`, and long
lanes passed quality with zero row errors, but the apparent wins were
single-sample and prompt-local, so no default changed. User-testing passed
VAL-M25-001 through VAL-M25-005.

M26-M28 closeout completed the non-OS-restart enhancement mission as evidence
and diagnostics only. M26 retained Qwen3.6 `max_seq_nums=2` controlled evidence
from same-session and fresh-anchor runs without an OS restart; the limitation is
explicitly preserved, and the final decision is keep/no-default-change with no
promotion. M27 closed restore-layout and rotating-delta diagnostics as
diagnostics-only REJECT/no-default-change: rotating-delta remained irreducible
final-state materialization, and restore safety invariants were preserved. M28
added scoped typecheck infrastructure with `pyrightconfig.json`;
`npx --yes pyright@1.1.403 --project pyrightconfig.json` passes with zero
diagnostics, and the ruff and pytest gates pass. Mission validation passed all
M26-M28 user-testing assertions.

M29 closeout completed Qwen3.6 27B deeper quality scoring and balanced sweeps.
A new deterministic reference/rubric scorer `mlx-bench-harness/quality_score.py`
consumes existing `shared_bench.py` JSON reports plus prompt metadata and an
optional rubric (`prompt_suites/m29_reference_rubric.json`); it emits per-row,
per-prompt, and aggregate quality scores plus blocking findings without altering
the benchmark inference route. The scorer is paired with an opt-in secondary
LLM judge invoked through the primary M29 route:

```bash
python3 quality_score.py \
  --candidate reports/<ts>-shared-bench.json \
  --out reports/<ts>-quality-score.json \
  --rubric prompt_suites/m29_reference_rubric.json \
  --judge-command "pi --provider ollama --model 'glm-5.2:cloud' --print --no-tools --no-session --thinking off"
```

Judge mode is opt-in, requires `STRICT JSON ONLY` output containing `score`,
`rationale`, and `flags`, and the judge block is always labeled
`authoritative: "deterministic"` and `score_kind: "secondary"`. OpenRouter and
LLMDYNAMIX judge fallback routes returned `401` during readiness and are not
part of the M29 contract; the deterministic score alone drives the top-level
status when the judge route is unreachable or returns non-JSON.

The balanced sweep used the curated suite `prompt_suites/m29_balanced_text_vlm.json`
(four text deterministic prompts plus two VLM prompts) with `--runs 3`, deterministic
sampling, the direct `shared_bench.py --engine mlx-engine` route through
`.venv-py312`, and serial concurrency. Retained evidence paths:

- 4-bit Qwen3.6 27B baseline:
  `reports/20260702T035155.543416Z-shared-bench.json` +
  `reports/20260702T035155.543416Z-qwen36-4bit-quality-inspect.json` +
  `reports/20260702T035155.543416Z-qwen36-4bit-quality-score.json`.
- 8-bit Qwen3.6 27B baseline:
  `reports/20260702T035313.633242Z-shared-bench.json` +
  `reports/20260702T035313.633242Z-qwen36-8bit-quality-inspect.json` +
  `reports/20260702T035313.633242Z-qwen36-8bit-quality-score.json`.
- Balanced matrix manifest and per-cell summary:
  `.planning/m29-balanced-matrix-manifest.json` and
  `.planning/m29-balanced-matrix-summary.json`.
- Cell B sampling (4-bit, temp=0.7 / top-p=0.9), Cell C prefill=4096, and
  Cell D max_seq_nums=2: `reports/20260702T040622.549703Z-*`,
  `reports/20260702T040747.091671Z-*`, and `reports/20260702T040852.322437Z-*`.
- Pi/Ollama judge evidence under `.planning/m29-pi-glm-judge/`.

All retained cells passed `quality_compare.py --candidate` inspect (`status=pass`)
and `quality_score.py` deterministic scoring (`status=pass`), with `mean_score=1.000`
and `pass_rate=1.000` across all six prompts. The deterministic tie cannot
distinguish 4-bit from 8-bit on the curated suite, but the 4-bit is
`1.69x` faster on decode TPS and `1.85x` faster end-to-end than the 8-bit, so
the retained safe-per-workload pick is the 4-bit Qwen3.6 27B; the 8-bit is
retained as a quality-ceiling reference only.

The final M29 decision is **data-only / no-default-change / no promotion**:

- The 4-bit-vs-8-bit comparison is safe-to-prefer but not a default change.
- Sampling at temp=0.7/top-p=0.9, `--prefill-step-size 4096`, and
  `--max-seq-nums 2` are recorded as single-sample data-capture-only cells;
  the existing `temperature=0.0`, `top_p=1.0`, default prefill-step-size, and
  `max_seq_nums=4` defaults remain in force, and explicit overrides remain
  available for callers that need them.
- Single-sample cold-TTFT spread and per-cell data-capture-only scope are
  recorded as non-promotion caveats.
- The Pi/Ollama `glm-5.2:cloud` judge route is safe as an opt-in secondary
  signal; OpenRouter and LLMDYNAMIX fallback routes are not contractual.
- DFlash, LM Studio runtime, adapter inference on `:3180/:3181/:3182`,
  MoE (`Qwen3.6-35B-A3B`), and `--mlx-engine-force-sequential` were not used
  for M29 inference-under-test.
- See commits `eda017a`, `30044ca`, `576fceb`, and `42d6da0`; the synthesis
  decision lives in `.planning/performance-future-work.md` under the M29
  heading. User-testing passed VAL-M29-001 through VAL-M29-006.

## Development Setup

### Pre-commit Hooks

We use pre-commit hooks to maintain code quality. Before contributing, please:

1. Install pre-commit:
   ```bash
   pip install pre-commit && pre-commit install
    ```
2. Run pre-commit:
   ```bash
   pre-commit run --all-files
   ```
3. Fix any issues before submitting your PR

## Testing

To run tests, run the following from the root of this repo:
```bash
python -m pip install pytest
python -m pytest tests/
```

For the current evidence-gated optimization paths, also run lint before
promotion or closeout:
```bash
ruff check --exclude .worktrees .
```

To test specific vision models:
```bash
python -m pytest tests/test_vision_models.py -k pixtral
```

## Prefill Tuning

Batched `qwen3_5_text` inference uses a default prompt-processing chunk size of `4096`
tokens. That setting is tuned for better time-to-first-token on the current batched Qwen
benchmarks, especially long prompts.

Sequential loads, vision paths, `qwen3_5_moe_text`, and other batched text families keep the
standard `2048` default unless you override `prefill_step_size` at load time to experiment
with a different tradeoff for a specific model or deployment.

### Startup Warmup

Batched text inference runs bounded startup warmup cases to prime common short and
two-chunk prompt shapes. The exact long-prompt benchmark warmup is disabled by
default because it can make large-model startup look hung while MLX compiles a
large synthetic prompt.

Set `MLX_ENGINE_STARTUP_LONG_WARMUP=1` only for controlled benchmark runs that
need to precompile the 7k-token benchmark shape before the first request:

```bash
MLX_ENGINE_STARTUP_LONG_WARMUP=1 python batched_demo.py --model /path/to/model
```

## Batched Timing Diagnostics

Set `MLX_ENGINE_BATCHED_TIMING=1` to emit structured log events for the batched text
and batched vision inference paths. The diagnostic events cover model load,
generation-stream preparation, startup warmup cases, prompt-cache preparation,
`BatchGenerator.insert`, and first-token latency.
Events are emitted at warning level so they are visible in default benchmark and demo
logging configurations.

For VLM persistent-cache restores, `vlm_cache_restore_detail` keeps the
historical aggregate `eval_ms` field and also reports `eval_collect_ms` and
`eval_barrier_ms`. `eval_collect_ms` covers restore-target discovery and
materialization counter collection; `eval_barrier_ms` covers the mandatory
restore-time `mx.eval(...)` barrier that prevents lazy disk-loaded arrays from
crossing cache-I/O-thread stream state into generation.

To summarize restore eval split evidence across one or more `shared_bench.py`
JSON reports, run:

```bash
python scripts/vlm_restore_eval_split_report.py reports/run-a.json reports/run-b.json \
  --output .planning/vlm-restore-eval-split-summary.json
```

The report extracts `vlm_cache_restore_detail` events from runner stderr,
computes `eval_barrier_ms / eval_ms`, preserves by-kind eval-target,
materialized-byte, and record-byte counters, and keeps row-error/cache/output
evidence so repeated-sample barrier decisions are auditable.

To collect repeated evidence for LFM2.5-VL text-only generated-token cache
reuse, run:

```bash
python scripts/lfm25_text_cache_bench.py \
  --samples 2 \
  --output .planning/lfm25-text-cache-bench.json
```

The script never prompts for downloads. Pass `--model /path/to/model` or set
`MLX_ENGINE_LFM25_VL_MODEL_PATH` when the retained local model path is not
available.

Example:

```bash
MLX_ENGINE_BATCHED_TIMING=1 python batched_demo.py --model /path/to/model
```

The switch is disabled by default so normal benchmark runs are not affected by diagnostic
logging overhead.

To isolate VLM persistent-cache restore-planner overhead without loading a model
or touching Metal, run the synthetic planner benchmark:

```bash
python benchmarks/vlm_restore_planner_bench.py --index-chunks 4096 --restore-chunks 128 --iterations 100
```

Use `--json` when collecting machine-readable benchmark output.

To compare candidate VLM KV record layouts before changing persistent-cache
semantics, run the token-normalized record-layout model:

```bash
python benchmarks/vlm_record_layout_model.py --chunks 8 --chunks-per-snapshot 2
```

## VLM Prompt Cache Persistence

Vision-model prompt-cache records are temporary by default and are cleaned up
when the model unloads or the process exits. For benchmark experiments, the VLM
backend can opt into a persistent prompt-cache store:

```python
from mlx_engine import load_model

model_kit = load_model(
    "/path/to/vlm-model",
    max_seq_nums=4,
    vlm_prompt_cache_storage_root="/tmp/mlx-engine-vlm-cache",
    vlm_prompt_cache_namespace="qwen-vl-benchmark",
    vlm_prompt_cache_min_save_tokens=512,
)
```

`vlm_prompt_cache_storage_root` is only valid for VLM models routed to
`BatchedVisionModelKit`. `vlm_prompt_cache_namespace` isolates cache records
within that root; when omitted, the resolved model path is used.
`vlm_prompt_cache_min_save_tokens` controls persistent-cache admission. Persistent
stores default to `512` reusable prompt tokens so tiny prompts do not pay
safetensor/index overhead; set it to `0` for experiments that need to persist
every cacheable prompt. Persistent mode is opt-in so normal LM Studio and
benchmark runs keep the existing temporary cache behavior.

## Project Objective And Promotion Policy

The current objective for `mlx-engine` is to deliver a Mac-first MLX inference
engine that is measurably faster on prompt processing and generation without
degrading response quality or runtime stability. Speed wins do not count unless
the same candidate also passes the deterministic quality harness and remains
stable under repeated warm-run measurements.

### Promotion Acceptance Criteria

An optimization is eligible for promotion to the default path only when all of
the following are true:

1. The candidate passes the response-quality regression harness with no new
   prompt failures.
2. Warm median TTFT does not regress beyond the configured
   `--max-warm-ttft-p50-regression-pct` threshold in
   `mlx-bench-harness/quality_compare.py`.
3. Warm median total latency does not regress beyond the configured
   `--max-warm-total-p50-regression-pct` threshold in
   `mlx-bench-harness/quality_compare.py`.
4. Any throughput or latency gain survives repeated-run variance review instead
   of depending on a single outlier.
5. The change does not introduce unload/reload instability, cache corruption,
   or model compatibility regressions in LM Studio or direct Python use.

### Optimization States

Use these states consistently when landing or discussing performance work:

- `default`: enabled by default and safe for normal users.
- `candidate`: implemented and benchmarkable, but still under active quality or
  variance review.
- `experiment`: off by default, intended for isolated benchmarking only, and
  not promotable until the acceptance criteria above are met.

### Terminal-Packed Final KV

The retained VLM prompt-cache layout saves one-step KV spans by default. For
true final prompt-boundary saves, the promoted default now replaces the
terminal chunk's one-step KV span with one full-prefix KV record when the
prefix chunks are contiguous.

```bash
MLX_ENGINE_VLM_TERMINAL_PACKED_FINAL_KV=0 python demo.py --model /path/to/vlm-model
```

State: `default`.

The switch now acts as an explicit rollback toggle. It only affects true final
prompt-boundary VLM cache saves. Based on repeated-sample direct benchmark
evidence on retained long-VLM profiles, this path is the default
final-boundary save layout. Broader LM Studio packaging or promotion still
requires the retained VLM model to pass
`scripts/lmstudio_vlm_live_validation_preflight.py` and then live LM Studio
`/v1/chat/completions` validation.

When `MLX_ENGINE_VLM_FINAL_CHUNK_STATE_ALIGN` is enabled, final prompt-boundary
saves still write terminal-packed KV but do not overwrite opaque state
checkpoints whose exact reusable-prefix boundary was saved by the aligned
prefill step. Set `MLX_ENGINE_VLM_FINAL_CHUNK_STATE_ALIGN=0` only to return to
the older alignment behavior for diagnostics.

### Shared Thread-Unsafe Stream

Set `MLX_ENGINE_EXPERIMENTAL_THREAD_UNSAFE_STREAM=1` to opt into the
candidate shared-stream path for text generation when the active MLX runtime
exposes `mx.new_thread_unsafe_stream`.

```bash
MLX_ENGINE_EXPERIMENTAL_THREAD_UNSAFE_STREAM=1 python demo.py --model /path/to/model
```

State: `experiment`.

For live LM Studio validation, the same experiment can also be enabled by
creating the file `/tmp/mlx-engine-thread-unsafe-stream`. This exists because
LM Studio does not reliably preserve custom backend env vars for Python-side
experiments, so a file toggle is the reversible validation path on the real
server runtime.

The current default remains per-thread `mx.new_thread_local_stream` for
non-distributed generation and the device default stream for distributed
paths. The shared thread-unsafe stream path is wired only as an opt-in
benchmark candidate. It should not be promoted until it is tested against the
retained prompt-processing workloads, repeated-sample latency analysis, the
response-quality regression harness, and live LM Studio validation.

### Registering a Custom LM Studio Runtime Pair

Use `scripts/lmstudio-register-engine.sh` to register a custom backend from the
current checkout. For isolated LM Studio validation, the script can target a
cloned vendor runtime pair instead of the default shipped one:

```bash
LMSTUDIO_MLX_RUNTIME_NAME=app-mlx-generate-mac14-arm64@31 \
LMSTUDIO_CPYTHON_RUNTIME_NAME=cpython3.11-mac-arm64@11 \
./scripts/lmstudio-register-engine.sh cheetara-mlx-thread-unsafe-runtime 2026.6.23
```

The version must be semver-compatible, for example `2026.6.23`.

The registration script now writes the selected vendor runtime pair into both
the LM Studio engine registry entry and the backend package
`backend-manifest.json`. That manifest update is required when validating
against a cloned runtime pair; otherwise LM Studio can appear to select the
custom backend while still launching against the stock vendor MLX runtime.

When no override env vars are provided, the script now defaults to the vendor
runtime pair recorded in the installed official LM Studio MLX backend manifest
instead of blindly picking the newest Amphibian app/Python copies on disk.
That keeps custom registrations aligned with the runtime pair LM Studio itself
currently ships and avoids drifting onto newer unsigned vendor runtimes.

The script also preflights code-signing compatibility between the selected MLX
runtime binary and the active LM Studio worker binary. A cloned runtime whose
`mlx/core.cpython-311-darwin.so` does not carry the LM Studio worker Team ID
will be rejected before registration because LM Studio will fail that runtime
at model-load time under hardened runtime library validation. Use
`LMSTUDIO_SKIP_RUNTIME_CODESIGN_CHECK=1` only for deliberate low-level signing
experiments.

If you do override the runtime pair, provide both
`LMSTUDIO_MLX_RUNTIME_NAME` and `LMSTUDIO_CPYTHON_RUNTIME_NAME` together so the
custom backend does not mix vendor app and Python runtimes from different
LM Studio builds.

### LM Studio VLM Live-Validation Preflight

Before registering a custom backend for retained VLM cache validation, verify
that LM Studio can actually load the retained model key:

```bash
.venv-py312/bin/python scripts/lmstudio_vlm_live_validation_preflight.py \
  --output .planning/lmstudio-vlm-live-validation-preflight.json
```

The preflight is read-only. It checks `lms runtime ls`, `lms server status`,
`lms ps`, `lms ls --json`, the retained local LFM2.5-VL directory, and
`~/.lmstudio/.internal/model-data.json`. It also reports whether a complete
copy exists under `~/.lmstudio/models`, but that is diagnostic only. It exits
non-zero until the retained model appears in `lms ls --json`.

If the model directory exists but `lms ls --json` does not expose the model key,
use the supported LM Studio download/registration path instead of copying files
into `~/.lmstudio/models` or editing LM Studio cache files by hand:

```bash
.venv-py312/bin/python scripts/lmstudio_vlm_download_probe.py \
  --output .planning/lmstudio-vlm-download-probe.json \
  --timeout 300
```

The download probe runs the official `lms get
https://huggingface.co/lmstudio-community/LFM2.5-VL-1.6B-MLX-8bit --mlx -y`
path with a bounded timeout and records progress, timeout state, exit status,
and sanitized output tail as JSON. It does not edit LM Studio indexes or cache
files.

Proceed to backend registration and live `/v1/chat/completions` validation only
after the preflight reports `ready_for_live_validation=true`.

### VLM Restore Freshness Flush

The VLM cache I/O thread now flushes an ordered prefix of immediately matching
queued prompt-cache saves before preparing one restore.

```bash
MLX_ENGINE_VLM_RESTORE_FRESHNESS_FLUSH=0 python demo.py --model /path/to/vlm-model
```

State: `promoted default`.

Set `MLX_ENGINE_VLM_RESTORE_FRESHNESS_FLUSH=0` to disable it.

This path does not reprioritize arbitrary disk writes; it only allows the
active restore to commit queued save jobs whose prefix chunk chain exactly
matches the restore candidate, while preserving queued budget updates ahead of
those saves.

Promotion basis:

- repeated cold-sample direct runtime probe: second-request cached tokens
  `0 -> 2048`, TTFT `-12.19%`, stable output
- live LM Studio validation on overlapping same-prefix VLM requests:
  - with a fresh unique prompt prefix and `0.01s` overlap, the baseline second
    request stayed a full miss
  - with the flush enabled, the live second request restored `2048` cached
    tokens with `flushed_matching_saves=1` across `0.02s` to `0.05s` overlap
    windows

## Attribution

Ernie 4.5 modeling code is sourced from [Baidu](https://huggingface.co/baidu/ERNIE-4.5-0.3B-PT/tree/da6f3b1158d5d0d2bbf552bfc3364c9ec64e8aa5)
