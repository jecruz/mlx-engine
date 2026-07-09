# M89 Small Cache/Perf Candidate Triage

This report is evidence only. It captures commit-level inspection of the
smallest plausible upstream cache/perf candidates and does not authorize a
cherry-pick or runtime promotion.

## Inspected Commits

- `81fc5d8` - `Avoid rebuilding VLM detokenizer per request`
- `8026180` - `Fix thread-local prompt cache tokens`
- `b758736` - `Speed up batched vision repeat penalty`
- `99e2328` - `Log lifetime prompt cache evictions`

## Triage Result

No new candidate was selected.

## Why

- `81fc5d8` is already functionally present in the current batched vision model
  kit: the tree already has `_new_detokenizer()` and uses it in the generation
  path.
- `8026180` is already present in `mlx_engine/cache_wrapper.py`: the current
  implementation already tracks live tokens as a host-side `List[int]` and uses
  that list for restore and checkpoint bookkeeping.
- `b758736` is already reflected in the current batched vision batch generator:
  the current tree already uses the `process_last_token` path and row-length
  guards that the patch introduced for the faster repeat-penalty handling.
- `99e2328` is already present in the current VLM prompt-cache store: the tree
  already tracks lifetime-evicted bytes and logs `lifetime_evicted_mib`.

## Evidence

- `git show --patch --unified=50 8026180 -- mlx_engine/cache_wrapper.py tests/test_cache_wrapper.py`
- `git show --patch --unified=40 81fc5d8 -- mlx_engine/model_kit/batched_vision/model_kit.py`
- `git show --patch --unified=40 b758736 -- mlx_engine/model_kit/batched_vision/batch_generator.py mlx_engine/model_kit/batched_vision/processors/repetition_penalty_processor.py tests/test_repetition_penalty_processor.py tests/test_batched_vision_batch_generator.py`
- `git show --patch --unified=40 99e2328 -- mlx_engine/model_kit/batched_vision/prompt_cache/cache_store.py`
- Current-tree inspection of `mlx_engine/model_kit/batched_vision/model_kit.py`
- Current-tree inspection of `mlx_engine/cache_wrapper.py`
- Current-tree inspection of `mlx_engine/model_kit/batched_vision/batch_generator.py`
- Current-tree inspection of `mlx_engine/model_kit/batched_vision/prompt_cache/cache_store.py`

## Next Step

Keep scanning for truly absent, isolated performance or stability patches.
Avoid re-reviewing commits that are already functionally present in the current
tree.
