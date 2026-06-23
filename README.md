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

## Attribution

Ernie 4.5 modeling code is sourced from [Baidu](https://huggingface.co/baidu/ERNIE-4.5-0.3B-PT/tree/da6f3b1158d5d0d2bbf552bfc3364c9ec64e8aa5)
