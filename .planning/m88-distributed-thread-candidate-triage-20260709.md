# M88 Distributed Thread Candidate Triage

This report is evidence only. It records a narrower review of distributed and
VLM model-thread candidates and does not authorize a cherry-pick or runtime
promotion.

## Reviewed Commits

- `3e41fdf` - `Run MLX VLM generation on model thread`
- `c86c23a` - `Run Qwen VLM prompts on model thread`
- `b7019fc` - `Handle early distributed cancel requests`
- `4b6b826` - `Stop distributed stream when caller exits`

## Triage Result

No additional candidate was selected for promotion or patching in this pass.

## Why

- `4b6b826` is already an ancestor of the current branch, so it is not a fresh
  cherry-pick target.
- The current `mlx_engine/model_kit/distributed_model_kit.py` already contains
  the early-cancel machinery that `b7019fc` introduced, including a
  `_cancelled_request_ids` set and the cancel-before-insert path.
- `3e41fdf` and `c86c23a` target older `VisionModelKit`/`ModelKit` threading
  layouts that do not match the current batched-VLM and distributed code shape
  closely enough to treat them as isolated, reversible performance patches.
- The active tree already routes sequential distributed generation through the
  model thread in `mlx_engine/generate.py`, so the high-level behavior those
  commits were aiming for is already in place.

## Evidence

- `git merge-base --is-ancestor b7019fc HEAD`
- `git merge-base --is-ancestor 4b6b826 HEAD`
- `git show --patch --unified=40 b7019fc -- mlx_engine/model_kit/distributed_model_kit.py`
- `git show --patch --unified=30 4b6b826 -- mlx_engine/model_kit/distributed_model_kit.py`
- `git show --patch --unified=40 3e41fdf -- mlx_engine/generate.py mlx_engine/vision_model_kit/vision_model_kit.py`
- `git show --patch --unified=40 c86c23a -- mlx_engine/model_kit/model_kit.py mlx_engine/vision_model_kit/vision_model_kit.py`
- Current-tree inspection of `mlx_engine/generate.py`
- Current-tree inspection of `mlx_engine/model_kit/distributed_model_kit.py`
- Current-tree inspection of `mlx_engine/model_kit/batched_vision/model_kit.py`

## Next Step

Keep scanning for smaller, clearly isolated performance/cache/stability
commits, but do not promote or patch any of the reviewed thread-routing
candidates from this pass.
