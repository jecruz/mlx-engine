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
prompt-boundary VLM cache saves. Based on repeated-sample benchmark evidence on
retained long-VLM profiles plus LM Studio validation of the current worktree,
this path is now the promoted default for final-boundary saves.

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
