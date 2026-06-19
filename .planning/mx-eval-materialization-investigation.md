# mx.eval Restore Materialization Investigation

Status: **active** | Started: 2026-06-19 | Branch: `mlx-vlm-prompt-cache-perf`

## Problem Statement

After path-based safetensor loading reduced KV-delta record-load time from 21ms to ~1.5ms,
the remaining restore cost is dominated by `mx.eval` materialization at `cache_store.py:276`.
Restore detail time is ~20ms, and the `mx.eval` call is the largest component.

The `mx.eval` cannot simply be removed — the no-restore-eval candidate (2026-06-19) failed
with `RuntimeError: There is no Stream(gpu, 3) in current thread.` The restore runs on the
cache I/O thread, but decode runs on the generation thread. Without materialization, lazy
arrays carry the I/O thread's MLX stream into the generation thread.

## What mx.eval Actually Materializes

The `mx.load(str(path))` call loads safetensor arrays that are already materialized on the GPU.
The `mx.eval` at line 276 materializes only the post-load operations:

1. `mx.concatenate([cache.state[0] for cache in caches], axis=2)` — lazy concat of keys
2. `mx.concatenate([cache.state[1] for cache in caches], axis=2)` — lazy concat of values
3. `mx.contiguous(keys)` / `mx.contiguous(values)` — lazy contiguous memory copy
4. For rotating layers: `keys[..., -max_size:, :]` — lazy slice

The loaded arrays themselves are already materialized. The eval cost is:
- GPU memory reorganization (concat)
- GPU memory copy (contiguous)
- GPU synchronization (waiting for all operations)

## Constraint

**Do not remove `mx.eval`.** The no-restore-eval candidate proved that lazy arrays from the
I/O thread's stream cannot be consumed on the generation thread. Any solution must either:
- Keep materialization but reduce its cost, OR
- Transfer stream ownership before passing arrays to the generation thread

## Candidates

### A: Defer mx.contiguous (skip the memory copy)

Skip `mx.contiguous` in `_concat_kv_delta_caches` and `_concat_rotating_delta_caches`.
The KV cache arrays would be non-contiguous views into concatenated tensors.

- **Upside**: Eliminates the GPU memory copy cost (likely the dominant eval component)
- **Downside**: MLX attention kernels may require contiguous inputs; decode could be slower or fail
- **Risk**: Medium — needs decode correctness + performance validation
- **Validation gate**: Must pass quality compare AND not regress decode TPS

### B: Per-record eval during load (overlap with I/O)

Materialize each record's arrays immediately after `mx.load`, spreading GPU work across
the I/O timeline.

- **Upside**: May overlap eval with disk I/O for subsequent records
- **Downside**: Loaded arrays are already materialized; this only helps if `mx.load` returns
  lazy arrays (unlikely for safetensors)
- **Risk**: Low — still uses mx.eval, just moves it
- **Validation gate**: Benchmark restore detail time

### C: Stream-aware transfer

After assembly, use MLX stream mechanisms to transfer arrays to the default stream
without full materialization. Something like evaluating on a specific stream.

- **Upside**: Could move cost to generation thread where it overlaps with other work
- **Downside**: MLX stream API is limited; may not support partial transfer
- **Risk**: High — similar failure mode to no-restore-eval
- **Validation gate**: Must not produce cross-thread stream errors

### D: Pre-concatenated save format

During save, concatenate adjacent chunks into larger records. During restore, load
fewer, larger records → less concat work.

- **Upside**: Reduces number of concat operations at restore time
- **Downside**: Changes save format; less flexible chunk eviction
- **Risk**: Medium — format change, migration complexity
- **Validation gate**: Benchmark + quality compare

## Investigation Plan

### Phase 1: Profile the eval breakdown (spike)

Write a standalone script that:
1. Creates mock KV cache arrays of realistic size
2. Times concat, contiguous, and eval separately
3. Reports the cost breakdown

This tells us which operation dominates and guides candidate selection.

### Phase 2: Test the best candidate

1. Implement behind a feature flag or local experiment
2. Run the benchmark harness with `MLX_ENGINE_BATCHED_TIMING=1`
3. Run quality compare
4. Promote or reject based on evidence

### Phase 3: If promoted, harden

1. Clean up experiment code
2. Run full test suite
3. Update docs
4. Commit

## Next Action

Start Phase 1: write and run a profiling spike to understand the eval cost breakdown.

## Phase 1 Results (2026-06-19)

### Key Finding

`mx.load(str(path), format="safetensors")` returns **lazy arrays**. The actual GPU
transfer from CPU to GPU happens during `mx.eval`, not during `mx.load`. This was
confirmed with a microbenchmark:

- `mx.load` alone: ~0.02 ms (just returns a lazy handle)
- `mx.load` + `mx.eval`: ~0.28 ms (GPU transfer happens here)
- Second `mx.eval` on same array: ~0.001 ms (already materialized)

This means the ~14ms `eval_ms` in benchmarks is dominated by GPU transfers of loaded
safetensor data, not by the concat/contiguous operations (which are ~2ms for 420MB).

### Spike Results

With realistic dimensions (16 layers, 8 KV heads, head_dim=64, 15 chunks of 512 tokens,
~420 MB total):
- Concat+contiguous+eval: ~2.3 ms median
- Eval only (pre-materialized): 0-11.6 ms (high variance, GPU sync)

The concat/contiguous operations are fast. The eval cost is GPU synchronization.

## Phase 2 Implementation (2026-06-19)

### Candidate A: Per-Record Eager Materialization

**Implemented** in commit `64b3d2a`.

Change: After each `mx.load` call in `_load_one_chunk`, immediately `mx.eval(cache.state)`
to trigger GPU transfer. The GPU can then transfer data while the CPU reads the next
record from disk.

The final `mx.eval` at line 276 is preserved as the cross-thread stream safety barrier.
It now only materializes the concat/contiguous operations (arrays are already on GPU).

### Benchmark Command

Must be run from a **host tmux pane** (Metal not visible to sandboxed Codex):

```bash
cd /Users/jeffreycruz/Development/LLM_INFERENCE/mlx-bench-harness

# With timing enabled to see per-record eval breakdown:
MLX_ENGINE_BATCHED_TIMING=1 python3 shared_bench.py \
  --engine mlx-engine \
  --model /Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/models/lmstudio-community/LFM2.5-VL-1.6B-MLX-8bit \
  --runs 2 \
  --max-tokens 32 \
  --prompt-suite vlm_image_long_quality.json \
  --vlm-prompt-cache-storage-root /tmp/mlx-engine-vlm-cache-eager-eval \
  --vlm-prompt-cache-min-save-tokens 0

# Quality compare after benchmark:
python3 quality_compare.py \
  --a reports/<new-benchmark>.json \
  --b reports/20260619T000646Z-shared-bench.json \
  --prompt-ids image_long_toucan
```

### Expected Results

If GPU transfers overlap with disk I/O:
- `eval_ms` in `vlm_cache_restore_detail` should decrease (less work for final eval)
- `vlm_cache_record_eager_eval` events show per-record transfer times
- Total `duration_ms` should decrease if overlap is effective
- Warm TTFT should improve

If no improvement:
- GPU transfers may be serialized by MLX's stream scheduler
- Disk I/O may be faster than GPU transfers (no overlap opportunity)
- May need a different approach (stream transfer, pre-concatenated save format)

### Validation Gates

- [ ] Benchmark shows reduced `eval_ms` or `duration_ms`
- [ ] Quality compare passes (output text matches baseline)
- [ ] No cross-thread stream errors
- [ ] No decode TPS regression
