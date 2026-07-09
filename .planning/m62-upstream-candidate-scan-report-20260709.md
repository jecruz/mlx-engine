# M62 Upstream Candidate Scan

## Summary

- Repository: `/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine`
- Head: `a5f2f74`
- Upstream main: `upstream/main` at `8ae2610`
- Origin branch: `origin/mlx-vlm-restore-eval-followup` at `8f0fa26`
- HEAD vs upstream/main: `{'left': 236, 'right': 1}`
- HEAD vs origin branch: `{'left': 41, 'right': 0}`

This report is readable evidence only. It does not make a promotion or
cherry-pick decision. Runtime changes still require retained benchmarks,
quality gates, candidate-vs-baseline deltas, and live LM Studio validation.

## Candidate Branches

| Branch | Head | Surface | Changed files | Unmatched commits | Subject |
| --- | --- | --- | ---: | ---: | --- |
| `upstream/neil/gemma4-tool-context` | `9aa3db2` | `broad` | 7 | 18 | Rename Gemma4 grammar edge case test |
| `upstream/yagil/dist` | `366ebd4` | `broad` | 84 | 0 | Preserve backpressure in distributed stream bridge |
| `upstream/yagil/mlx-dist-non-batched` | `c86c23a` | `broad` | 84 | 6 | Run Qwen VLM prompts on model thread |
| `upstream/neil/vlm-parity-ci` | `ea1a6bb` | `broad` | 44 | 82 | Relax VLM concurrency logprob parity |
| `upstream/will/lfm-2.5-unified` | `461015c` | `broad` | 83 | 1 | Add test for LFM 2.5 caching |
| `upstream/neil/img-caching` | `7dfe3cd` | `broad` | 94 | 13 | cleanup |

## `upstream/neil/gemma4-tool-context`

- Head: `9aa3db2`
- Subject: Rename Gemma4 grammar edge case test
- Ahead/behind vs upstream main: `{'left': 0, 'right': 17}`
- Change surface: `broad`
- Changed files: `7`
- Unmatched patch-id commits: `18`

### Changed Files

- `M	mlx_engine/generate.py`
- `M	mlx_engine/model_kit/batched_vision/batch_generator.py`
- `A	mlx_engine/tool_protocols.py`
- `A	mlx_engine/tool_runtime.py`
- `M	tests/test_batched_vision_batch_generator.py`
- `A	tests/test_tool_runtime_detection.py`
- `A	tests/test_tool_runtime_reasoning_guard.py`

### Unmatched Patch IDs

- `+ 8ae261033bc5bc16fdfc19a842bfc1d96db51348 Handle Gemma4 bidirectional visual prefill (#340)`
- `+ 1766468e46665ed6a02b77236787c1d5fec1f0d7 Add Gemma4 tool prompt context extraction`
- `+ 328064d949dbcac11e33feeef3a98d10b5666d47 Add Gemma4 VLM reasoning guard`
- `+ 8bfa083ee65dceeeafe60fe6ad4e1436460c65a9 Optimize Gemma4 reasoning guard token tracking`
- `+ 2ceeaec93d503e5bfd237f1cf22038d0c02c0c88 Simplify Gemma4 reasoning guard state`
- `+ 636d6bb867f81aee8cf4626e1e307d70a836f94e Document Gemma4 reasoning mask tradeoff`
- `+ e0bdfa5a91293bdb55ee7932f27a3f4aabb43e5d Simplify Gemma4 token helper assumptions`
- `+ 2316deb4ab2fb097ea8dcec26588aa52ec8fd7fd Optimize Gemma4 guard no-mask path`
- `+ 57a90c712c30ff64445bc2b805906e36a091bb0f Avoid Gemma4 guard token sync`
- `+ e35f614111dc6251d31a816c3c219552de186096 Simplify Gemma4 reasoning guard markers`
- `+ 6dda82dfef45b16d8497cba075375d4b52a39042 Add Gemma4 tool-call structure guard`
- `+ 0a7d07e9971aa76256a6475a96001a35b784ccf0 Use llguidance for Gemma4 tool grammar`
- `+ 7f3fae929e4b6221f8c183685097344c45baff51 Simplify Gemma4 llguidance guard`
- `+ 3035148802e84d42faa7034a33b452196b9fdd37 Move llguidance imports to module scope`
- `+ 91fe8bd1210f147c5a046c984c537f59ad1b5cbe Simplify Gemma4 tool runtime assumptions`
- `+ 05d1d64c38b211826115166dc344bbd151681d34 Relax Gemma4 tool grammar edge cases`
- `+ 864eeaf03fc699c4683d325051c8c671c8159f94 Trim Gemma4 tool runtime comments`
- `+ 9aa3db24f45630130386cf152b1b22699dca1813 Rename Gemma4 grammar edge case test`

## `upstream/yagil/dist`

- Head: `366ebd4`
- Subject: Preserve backpressure in distributed stream bridge
- Ahead/behind vs upstream main: `{'left': 5, 'right': 29}`
- Change surface: `broad`
- Changed files: `84`
- Unmatched patch-id commits: `0`

### Changed Files

- `M	mlx_engine/cache_wrapper.py`
- `A	mlx_engine/distributed_coordinator.py`
- `A	mlx_engine/distributed_rank.py`
- `A	mlx_engine/distributed_server.py`
- `A	mlx_engine/distributed_validation_harness.py`
- `A	mlx_engine/distributed_validation_rank_entry.py`
- `A	mlx_engine/distributed_validation_runner.py`
- `A	mlx_engine/distributed_worker.py`
- `M	mlx_engine/generate.py`
- `M	mlx_engine/model_kit/__init__.py`
- `M	mlx_engine/model_kit/batched_model_kit.py`
- `D	mlx_engine/model_kit/batched_vision/__init__.py`
- `D	mlx_engine/model_kit/batched_vision/batch_generator.py`
- `D	mlx_engine/model_kit/batched_vision/cache_io_thread.py`
- `D	mlx_engine/model_kit/batched_vision/model_kit.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/__init__.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/blob_store.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/cache_store.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/chunks.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/coordinator.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/disk_budget.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/image_spans.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/records.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/restore_planner.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/types.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_inputs.py`
- `D	mlx_engine/model_kit/batched_vision/qwen_mrope.py`
- `D	mlx_engine/model_kit/batched_vision/request_lifecycle.py`
- `D	mlx_engine/model_kit/batched_vision/vision_feature_memoizer.py`
- `A	mlx_engine/model_kit/distributed_model_kit.py`
- `M	mlx_engine/model_kit/model_kit.py`
- `M	mlx_engine/model_kit/patches/gemma4.py`
- `D	mlx_engine/model_kit/patches/lfm2_vl.py`
- `M	mlx_engine/model_kit/patches/qwen3_5.py`
- `A	mlx_engine/model_kit/vision_add_ons/base.py`
- `A	mlx_engine/model_kit/vision_add_ons/gemma3.py`
- `A	mlx_engine/model_kit/vision_add_ons/gemma3n.py`
- `A	mlx_engine/model_kit/vision_add_ons/gemma4.py`
- `A	mlx_engine/model_kit/vision_add_ons/lfm2_vl.py`
- `A	mlx_engine/model_kit/vision_add_ons/load_utils.py`
- `A	mlx_engine/model_kit/vision_add_ons/mistral3.py`
- `A	mlx_engine/model_kit/vision_add_ons/pixtral.py`
- `A	mlx_engine/model_kit/vision_add_ons/process_prompt_with_images.py`
- `A	mlx_engine/model_kit/vision_add_ons/qwen2_vl.py`
- `A	mlx_engine/model_kit/vision_add_ons/qwen3_5.py`
- `A	mlx_engine/model_kit/vision_add_ons/qwen3_5_moe.py`
- `A	mlx_engine/model_kit/vision_add_ons/qwen3_vl.py`
- `A	mlx_engine/model_kit/vision_add_ons/qwen3_vl_moe.py`
- `A	mlx_engine/model_kit/vision_add_ons/qwen_vl_utils.py`
- `A	mlx_engine/model_kit/vision_add_ons/vision_feature_memoizer.py`
- `M	mlx_engine/processors/repetition_penalty_processor.py`
- `M	mlx_engine/utils/eot_tokens.py`
- `M	mlx_engine/utils/generation_helpers.py`
- `A	mlx_engine/utils/mlx_lm_stream.py`
- `D	mlx_engine/utils/mlx_threading.py`
- `M	mlx_engine/utils/prompt_progress_reporter.py`
- `D	mlx_engine/utils/sampling.py`
- `M	mlx_engine/utils/speculative_decoding.py`
- `R097	mlx_engine/model_kit/batched_vision/transformers_compatibility.py	mlx_engine/vision_model_kit/_transformers_compatibility.py`
- `A	mlx_engine/vision_model_kit/vision_model_kit.py`
- `A	mlx_engine/vision_model_kit/vision_model_wrapper.py`
- `M	requirements.txt`
- `D	tests/test_batched_vision_batch_generator.py`
- `D	tests/test_batched_vision_blob_store.py`
- `D	tests/test_batched_vision_cache_io_thread.py`
- `D	tests/test_batched_vision_cache_store.py`
- `D	tests/test_batched_vision_chunks.py`
- `D	tests/test_batched_vision_coordinator.py`
- `D	tests/test_batched_vision_disk_budget.py`
- `D	tests/test_batched_vision_image_spans.py`
- `D	tests/test_batched_vision_model_kit.py`
- `D	tests/test_batched_vision_parity.py`
- `D	tests/test_batched_vision_prompt_inputs.py`
- `D	tests/test_batched_vision_qwen_mrope.py`
- `D	tests/test_batched_vision_records.py`
- `D	tests/test_batched_vision_request_lifecycle.py`
- `D	tests/test_batched_vision_restore_planner.py`
- `M	tests/test_cache_wrapper.py`
- `M	tests/test_patched_gemma4.py`
- `M	tests/test_patched_qwen3_5.py`
- `M	tests/test_prefill_step_size.py`
- `D	tests/test_repetition_penalty_processor.py`
- `M	tests/test_vision_feature_cache.py`
- `M	tests/test_vision_models.py`

### Unmatched Patch IDs

- None

## `upstream/yagil/mlx-dist-non-batched`

- Head: `c86c23a`
- Subject: Run Qwen VLM prompts on model thread
- Ahead/behind vs upstream main: `{'left': 5, 'right': 35}`
- Change surface: `broad`
- Changed files: `84`
- Unmatched patch-id commits: `6`

### Changed Files

- `M	mlx_engine/cache_wrapper.py`
- `A	mlx_engine/distributed_coordinator.py`
- `A	mlx_engine/distributed_rank.py`
- `A	mlx_engine/distributed_server.py`
- `A	mlx_engine/distributed_validation_harness.py`
- `A	mlx_engine/distributed_validation_rank_entry.py`
- `A	mlx_engine/distributed_validation_runner.py`
- `A	mlx_engine/distributed_worker.py`
- `M	mlx_engine/generate.py`
- `M	mlx_engine/model_kit/__init__.py`
- `M	mlx_engine/model_kit/batched_model_kit.py`
- `D	mlx_engine/model_kit/batched_vision/__init__.py`
- `D	mlx_engine/model_kit/batched_vision/batch_generator.py`
- `D	mlx_engine/model_kit/batched_vision/cache_io_thread.py`
- `D	mlx_engine/model_kit/batched_vision/model_kit.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/__init__.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/blob_store.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/cache_store.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/chunks.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/coordinator.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/disk_budget.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/image_spans.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/records.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/restore_planner.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/types.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_inputs.py`
- `D	mlx_engine/model_kit/batched_vision/qwen_mrope.py`
- `D	mlx_engine/model_kit/batched_vision/request_lifecycle.py`
- `D	mlx_engine/model_kit/batched_vision/vision_feature_memoizer.py`
- `A	mlx_engine/model_kit/distributed_model_kit.py`
- `M	mlx_engine/model_kit/model_kit.py`
- `M	mlx_engine/model_kit/patches/gemma4.py`
- `D	mlx_engine/model_kit/patches/lfm2_vl.py`
- `M	mlx_engine/model_kit/patches/qwen3_5.py`
- `A	mlx_engine/model_kit/vision_add_ons/base.py`
- `A	mlx_engine/model_kit/vision_add_ons/gemma3.py`
- `A	mlx_engine/model_kit/vision_add_ons/gemma3n.py`
- `A	mlx_engine/model_kit/vision_add_ons/gemma4.py`
- `A	mlx_engine/model_kit/vision_add_ons/lfm2_vl.py`
- `A	mlx_engine/model_kit/vision_add_ons/load_utils.py`
- `A	mlx_engine/model_kit/vision_add_ons/mistral3.py`
- `A	mlx_engine/model_kit/vision_add_ons/pixtral.py`
- `A	mlx_engine/model_kit/vision_add_ons/process_prompt_with_images.py`
- `A	mlx_engine/model_kit/vision_add_ons/qwen2_vl.py`
- `A	mlx_engine/model_kit/vision_add_ons/qwen3_5.py`
- `A	mlx_engine/model_kit/vision_add_ons/qwen3_5_moe.py`
- `A	mlx_engine/model_kit/vision_add_ons/qwen3_vl.py`
- `A	mlx_engine/model_kit/vision_add_ons/qwen3_vl_moe.py`
- `A	mlx_engine/model_kit/vision_add_ons/qwen_vl_utils.py`
- `A	mlx_engine/model_kit/vision_add_ons/vision_feature_memoizer.py`
- `M	mlx_engine/processors/repetition_penalty_processor.py`
- `M	mlx_engine/utils/eot_tokens.py`
- `M	mlx_engine/utils/generation_helpers.py`
- `A	mlx_engine/utils/mlx_lm_stream.py`
- `D	mlx_engine/utils/mlx_threading.py`
- `M	mlx_engine/utils/prompt_progress_reporter.py`
- `D	mlx_engine/utils/sampling.py`
- `M	mlx_engine/utils/speculative_decoding.py`
- `R097	mlx_engine/model_kit/batched_vision/transformers_compatibility.py	mlx_engine/vision_model_kit/_transformers_compatibility.py`
- `A	mlx_engine/vision_model_kit/vision_model_kit.py`
- `A	mlx_engine/vision_model_kit/vision_model_wrapper.py`
- `M	requirements.txt`
- `D	tests/test_batched_vision_batch_generator.py`
- `D	tests/test_batched_vision_blob_store.py`
- `D	tests/test_batched_vision_cache_io_thread.py`
- `D	tests/test_batched_vision_cache_store.py`
- `D	tests/test_batched_vision_chunks.py`
- `D	tests/test_batched_vision_coordinator.py`
- `D	tests/test_batched_vision_disk_budget.py`
- `D	tests/test_batched_vision_image_spans.py`
- `D	tests/test_batched_vision_model_kit.py`
- `D	tests/test_batched_vision_parity.py`
- `D	tests/test_batched_vision_prompt_inputs.py`
- `D	tests/test_batched_vision_qwen_mrope.py`
- `D	tests/test_batched_vision_records.py`
- `D	tests/test_batched_vision_request_lifecycle.py`
- `D	tests/test_batched_vision_restore_planner.py`
- `M	tests/test_cache_wrapper.py`
- `M	tests/test_patched_gemma4.py`
- `M	tests/test_patched_qwen3_5.py`
- `M	tests/test_prefill_step_size.py`
- `D	tests/test_repetition_penalty_processor.py`
- `M	tests/test_vision_feature_cache.py`
- `M	tests/test_vision_models.py`

### Unmatched Patch IDs

- `+ c8275b612e067be9091c5aac90788df843fd21bf Align distributed MLX prompt caching`
- `+ e5a6faf59bcc201d6d454232062930cdf3010eb0 Run distributed MLX scheduler on model thread`
- `+ 958ffb8ebe89cd54e5fa8977a6bbb40c3150f575 Run Gemma 4 MLX generation on model thread`
- `+ b7019fc3f59bfa4aeba89c66b1c3970057f4ca22 Handle early distributed cancel requests`
- `+ 3e41fdfcd3ee35127bf2b828ff7307eb21f4350c Run MLX VLM generation on model thread`
- `+ c86c23ae9ff957c1c4f34e9e21e6563b627ab2cc Run Qwen VLM prompts on model thread`

## `upstream/neil/vlm-parity-ci`

- Head: `ea1a6bb`
- Subject: Relax VLM concurrency logprob parity
- Ahead/behind vs upstream main: `{'left': 5, 'right': 82}`
- Change surface: `broad`
- Changed files: `44`
- Unmatched patch-id commits: `82`

### Changed Files

- `M	mlx_engine/generate.py`
- `M	mlx_engine/model_kit/__init__.py`
- `M	mlx_engine/model_kit/batched_model_kit.py`
- `M	mlx_engine/model_kit/batched_vision/batch_generator.py`
- `M	mlx_engine/model_kit/batched_vision/cache_io_thread.py`
- `M	mlx_engine/model_kit/batched_vision/model_kit.py`
- `M	mlx_engine/model_kit/batched_vision/prompt_cache/cache_store.py`
- `M	mlx_engine/model_kit/batched_vision/prompt_inputs.py`
- `D	mlx_engine/model_kit/batched_vision/vision_feature_memoizer.py`
- `M	mlx_engine/model_kit/model_kit.py`
- `M	mlx_engine/model_kit/patches/gemma4.py`
- `D	mlx_engine/model_kit/patches/lfm2_vl.py`
- `M	mlx_engine/model_kit/patches/qwen3_5.py`
- `A	mlx_engine/model_kit/vision_add_ons/base.py`
- `A	mlx_engine/model_kit/vision_add_ons/gemma3.py`
- `A	mlx_engine/model_kit/vision_add_ons/gemma3n.py`
- `A	mlx_engine/model_kit/vision_add_ons/gemma4.py`
- `A	mlx_engine/model_kit/vision_add_ons/lfm2_vl.py`
- `A	mlx_engine/model_kit/vision_add_ons/load_utils.py`
- `A	mlx_engine/model_kit/vision_add_ons/mistral3.py`
- `A	mlx_engine/model_kit/vision_add_ons/pixtral.py`
- `A	mlx_engine/model_kit/vision_add_ons/process_prompt_with_images.py`
- `A	mlx_engine/model_kit/vision_add_ons/qwen2_vl.py`
- `A	mlx_engine/model_kit/vision_add_ons/qwen3_5.py`
- `A	mlx_engine/model_kit/vision_add_ons/qwen3_5_moe.py`
- `A	mlx_engine/model_kit/vision_add_ons/qwen3_vl.py`
- `A	mlx_engine/model_kit/vision_add_ons/qwen3_vl_moe.py`
- `A	mlx_engine/model_kit/vision_add_ons/qwen_vl_utils.py`
- `A	mlx_engine/model_kit/vision_add_ons/vision_feature_memoizer.py`
- `D	mlx_engine/utils/mlx_threading.py`
- `R097	mlx_engine/model_kit/batched_vision/transformers_compatibility.py	mlx_engine/vision_model_kit/_transformers_compatibility.py`
- `A	mlx_engine/vision_model_kit/vision_model_kit.py`
- `A	mlx_engine/vision_model_kit/vision_model_wrapper.py`
- `M	requirements.txt`
- `M	tests/test_batched_vision_batch_generator.py`
- `M	tests/test_batched_vision_cache_store.py`
- `M	tests/test_batched_vision_coordinator.py`
- `D	tests/test_batched_vision_model_kit.py`
- `M	tests/test_batched_vision_parity.py`
- `M	tests/test_batched_vision_prompt_inputs.py`
- `D	tests/test_patched_gemma4.py`
- `M	tests/test_patched_qwen3_5.py`
- `M	tests/test_vision_feature_cache.py`
- `M	tests/test_vision_models.py`

### Unmatched Patch IDs

- `+ 0c3aede4b91faa1b8593d8d7205bbffba33466a6 Add batched vision backend`
- `+ 50f75c08cac1143b978bf8074b48ca41c8ddeb13 Add VLM prompt spill cache`
- `+ 5251729e4d5741ce7fb92c9856069daaead6482e Split VLM safetensor spool`
- `+ c9b679a86c6e62e2c3e7c31d676039bcede75169 Split VLM prompt cache payload`
- `+ 99b4730be6f30d75383b997c9a6a6e30c41d7c6c Simplify batched vision scheduler`
- `+ 7abc8cf60d47f5b67310dce5065a989152cd47c9 Split VLM prompt cache planning`
- `+ 2cc7f6cbbe65868fa6b5833f04fc7d6393c3c7ab Add span-aware VLM prompt spill cache`
- `+ 40d7761e86fa109acbcc3466ed1c297ca641d808 Clarify VLM prefix cache chunk naming`
- `+ 1322bf2465c1e3ac445bbb6458b3a27b5db13090 Add VLM hot prompt cache saves`
- `+ ec879a3304785bbbe40eee94931435a5f808724b Refine VLM spill cache disk budget`
- `+ 5631674d140c7d686be9975081337697f4f62ed2 Refine VLM spill cache eviction`
- `+ d6c1b9d5b2a63b12a6f8547fbf536966b7a617c6 Align VLM spill cache with physical records`
- `+ eff978fa801c2eb3780b64edb47090114eaad24f Refine VLM cache save points`
- `+ c1013c3ec88726bdc041151d39a2c118beb07f09 Clarify VLM cache save points`
- `+ 2064bb3da429491c004f4b31a2d4e62050cadaa6 Refactor batched vision cache modules`
- `+ 2fa9ea4645218c2d5d9da06fe3483fbd31302ba2 Simplify batched vision cache store`
- `+ 7fa5e3c9a14a336e9a78ec71ab2c38b1c023c78b Simplify batched vision generator`
- `+ 270f0a64085b5ff1e4795b25091029af57a6213d Add batched VLM logits processors`
- `+ 92691c9f030e1de442ba22d770e14ee2b70a78c3 Improve batched vision generation`
- `+ c0b9686e5c79d7d0044f3c1d8ec8ef3f03bfd4b2 Update batched vision tests`
- `+ 31418e4faf7754d1eaf545d2e7cbc4b3d895d8b2 Refine batched vision prompt cache restore`
- `+ 430d10999dcfc883e8075dce2d91edfad9bb5837 Add batched vision prompt cache tests`
- `+ 57d35bf3cb0e394250d28fe887e97934e793ad60 Add batched vision cache tests`
- `+ 7800d3e54130b2b5511e9ee4566a52aba562b7fb Add batched vision prompt input tests`
- `+ 085b85dfd5f6220b953b53eb9535cd56733fae72 Simplify generate model kit typing`
- `+ ddff92d704e9ab353410e6727ac9c7335b5adfbb Stop tracking batched vision smoke script`
- `+ cc80b490778d3966d81fd5e82302f3a105ac4214 Tighten batched vision cache lifecycle`
- `+ 0d7f069d9b18ec9cd27f699b0ddd56d71ab26948 Simplify batched vision cache chunking`
- `+ 55e30061809d70276ea05a36cfb965e8ae18ff1d Improve Qwen prompt cache checkpointing`
- `+ b400e74bc42cb449fdd6f0d5b7f99b2113fd05ad Guard rotating prompt cache saves`
- `+ 965ddd9c24dc721283dffa636ab606f02373b8df Preserve Gemma batched vision masks`
- `+ 2aaa61f0df11e5759535c2c866be1955eeb6162a Avoid default VLM top logprobs work`
- `+ 94af3ba574b52b52e8be21bb94a9ae57d918860c Fix cached VLM prompt kwargs restore`
- `+ cef9c62610b789807a27e726f83338263212cac1 Validate KV prompt cache record coverage`
- `+ 329eed8d986d0e799544f20b6f40676c3358628b Group VLM generation row state`
- `+ 58f9bb21f51e734874cad39a18777cfe50efb3be Tighten batched vision response metadata`
- `+ 4e63fa7816b6373570befe2c03a4ab1f89a9982a Tighten batched VLM request handling`
- `+ a8a0478f9bf1ddb56297633c5674ad72a255c402 Optimize single-row VLM batching`
- `+ d4c47361a0174fc8bf7d1ac16a8870077d8c2c04 Sync scalar Qwen RoPE state`
- `+ 884bccb3ec3b3f3127e0697ca6fe6645bd1758d8 Fix wrapped rotating cache records`
- `+ 12442833d00d89c2a8be22f9444054edc8278546 Preserve budget-fitting prompt cache restores`
- `+ c8bf1908c9b61a5a878d0b0254ff194f90ddf39e Evict failed prompt cache records`
- `+ 32de1b3ef440fbb303798d3573a8e40f6c482faf Avoid scalar cache response aliasing`
- `+ 6632e61b91851605ad8ce67b7f8a2e91f9c8ecf0 Tighten batched vision failure boundaries`
- `+ 2bfe647805f6f34600af15313476d2c3b302fdaf Make VLM cache snapshots best effort`
- `+ 8cce9b3fd8846cf95c4ea130f9d13843080f78d6 Update VLM prefill alignment test`
- `+ 95b06d2c1420852d408544667cd324abf868de56 Restore semantic cache test assertions`
- `+ 19b04bfa3e93597c5be747b90a5c4bad5e89a7fb Document Qwen MRoPE workaround`
- `+ 4180b437089528a51f0a8aa0c9c3c049f7873cbf Clarify VLM prompt cache restore names`
- `+ 67ed0dc3319dad9a4d529642b3c734672e7f8989 Preserve restore chains after partial cache saves`
- `+ f2fd3cfab2751df800d8a0d816011d626679d563 Keep hot cache fallback after disk restore miss`
- `+ c675d2712ee1fa29a7eb6eded47178f851b98483 Tighten VLM prompt cache restore handoffs`
- `+ caf7dd509a23bd0a5ecea977ae9552477041ce84 Fix VLM test compatibility`
- `+ 77a2acd36f58486ffc2ec2f38e8cb32369d73006 Fix VLM cache budget failure and trim tests`
- `+ d8614f81a19c821bb5558ce7077f88d88d25a8eb Disable disk cache on budget estimate failure`
- `+ 64801d2ae22b0d7e350570001ed1610186f63b55 Stabilize Gemma3n cache prompt test`
- `+ 93b1ade3adec7722837653f341309e8e447a4583 Stabilize Gemma3n cache reuse test`
- `+ 694109129017f1196290ef6eb718e8b433737f81 Restore Gemma3n cache semantic assertions`
- `+ 3742cf627c2b8441c03ac4cedde33843c3de176d Stabilize Gemma3n long prompt cache test`
- `+ 25f0f9acfc1549cad961c64a13e99f5f3df79b5a Stabilize Gemma3n long prompt author test`
- `+ 63b8ff6589e3dd74486f169f5b76680823ec6a29 Stabilize Gemma long prompt cache tests`
- `+ 7172a15bc0d9e6077cbb1004b6acb8f1ac79770d Fix Gemma3 batched vision mask handling`
- `+ 04cf7b72a7404e666650c2a83ef0493390864c62 Gate Gemma3 attention mask drop`
- `+ e4fcfb9a72d4d7286bbcef0dbfe5338a2c3b2ceb Route VLM attention masks in batched prefill`
- `+ bb20ef7958179bcd734e981f6e5e38f281316049 Adjust VLM prompt cache disk budget`
- `+ 20b94ded2c9f97d79fe7a76421e1e39962c9275e Fix Granite4 batched vision DeepStack slicing`
- `+ b7587360df0ff3d59fbb23d5847bd0f5a9393ea0 Speed up batched vision repeat penalty`
- `+ c9389c8dcd3db41f3b6f5b636961e6587ac2b6d0 Fix worker-safe sampling`
- `+ 80261802fec56533a04a5d6487039d80909830b5 Fix thread-local prompt cache tokens`
- `+ a814a9520da2db68fe29a9d008dcbccf1625794e Fix Qwen3.5 text fast path in batched vision`
- `+ 02b1d0ecabc40f9807287428c36c1cecb41fec2e Fix batched vision structured processors`
- `+ 81fc5d806e723504b2cecf0e0894abf4de0009bc Avoid rebuilding VLM detokenizer per request`
- `+ 6455eb7a1620983fc64bfa57ebab76590e8e2369 Run sequential model kit on owner thread`
- `+ 99e2328b198d85e564acf2b5fe5944d228ab15a2 Log lifetime prompt cache evictions`
- `+ b1456ee905f3253a1a96d245f1ec3e9795fd66c3 Fix batched VLM prompt progress reporting`
- `+ cf058d70196f1feb1b987f3d9f5871989924dec0 Fix VLM model kit setup edge cases`
- `+ b229b959aa30420778561b28abb335563369576e Normalize VLM concurrency limit`
- `+ 722f35145593e7fc7a2f7cf12c1d42a5eac2a07f Add Qwen3.5 VLM restore parity test`
- `+ c390565e401bd45886acf61c5440cf45eca3a2a5 Add Qwen3.5 VLM generation trace parity test`
- `+ 406af8a7f7d7ee891a41c518688d22c8149435c1 Add Qwen3.5 continuous batching parity test`
- `+ 9d4a4eeef701d70d880ebe9773f0e1add0b69110 Add batched VLM parity tests`
- `+ ea1a6bb164a83c7b0e4c4f6e75f431e5569502aa Relax VLM concurrency logprob parity`

## `upstream/will/lfm-2.5-unified`

- Head: `461015c`
- Subject: Add test for LFM 2.5 caching
- Ahead/behind vs upstream main: `{'left': 25, 'right': 1}`
- Change surface: `broad`
- Changed files: `83`
- Unmatched patch-id commits: `1`

### Changed Files

- `D	batched_demo.py`
- `M	demo.py`
- `M	mlx_engine/__init__.py`
- `M	mlx_engine/cache_wrapper.py`
- `M	mlx_engine/generate.py`
- `M	mlx_engine/model_kit/__init__.py`
- `D	mlx_engine/model_kit/batched_model_kit.py`
- `D	mlx_engine/model_kit/batched_model_kit_types.py`
- `D	mlx_engine/model_kit/batched_vision/__init__.py`
- `D	mlx_engine/model_kit/batched_vision/batch_generator.py`
- `D	mlx_engine/model_kit/batched_vision/cache_io_thread.py`
- `D	mlx_engine/model_kit/batched_vision/model_kit.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/__init__.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/blob_store.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/cache_store.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/chunks.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/coordinator.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/disk_budget.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/image_spans.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/records.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/restore_planner.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/types.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_inputs.py`
- `D	mlx_engine/model_kit/batched_vision/qwen_mrope.py`
- `D	mlx_engine/model_kit/batched_vision/request_lifecycle.py`
- `D	mlx_engine/model_kit/batched_vision/vision_feature_memoizer.py`
- `M	mlx_engine/model_kit/model_kit.py`
- `D	mlx_engine/model_kit/patches/gemma4.py`
- `D	mlx_engine/model_kit/patches/lfm2_vl.py`
- `D	mlx_engine/model_kit/patches/qwen3_5.py`
- `A	mlx_engine/model_kit/vision_add_ons/base.py`
- `A	mlx_engine/model_kit/vision_add_ons/gemma3.py`
- `A	mlx_engine/model_kit/vision_add_ons/gemma3n.py`
- `A	mlx_engine/model_kit/vision_add_ons/lfm2_vl.py`
- `A	mlx_engine/model_kit/vision_add_ons/load_utils.py`
- `A	mlx_engine/model_kit/vision_add_ons/mistral3.py`
- `A	mlx_engine/model_kit/vision_add_ons/pixtral.py`
- `A	mlx_engine/model_kit/vision_add_ons/process_prompt_with_images.py`
- `A	mlx_engine/model_kit/vision_add_ons/qwen2_vl.py`
- `A	mlx_engine/model_kit/vision_add_ons/qwen3_vl.py`
- `A	mlx_engine/model_kit/vision_add_ons/qwen3_vl_moe.py`
- `A	mlx_engine/model_kit/vision_add_ons/qwen_vl_utils.py`
- `M	mlx_engine/processors/repetition_penalty_processor.py`
- `M	mlx_engine/utils/eot_tokens.py`
- `D	mlx_engine/utils/generation_helpers.py`
- `D	mlx_engine/utils/generation_result.py`
- `D	mlx_engine/utils/mlx_threading.py`
- `M	mlx_engine/utils/prompt_progress_reporter.py`
- `D	mlx_engine/utils/sampling.py`
- `M	mlx_engine/utils/speculative_decoding.py`
- `R097	mlx_engine/model_kit/batched_vision/transformers_compatibility.py	mlx_engine/vision_model_kit/_transformers_compatibility.py`
- `A	mlx_engine/vision_model_kit/vision_model_kit.py`
- `A	mlx_engine/vision_model_kit/vision_model_wrapper.py`
- `M	requirements.txt`
- `M	tests/data/ben_franklin_autobiography_start.txt`
- `D	tests/data/equations.jpg`
- `D	tests/patched_model_test_utils.py`
- `M	tests/shared.py`
- `D	tests/test_batched_generation.py`
- `D	tests/test_batched_vision_batch_generator.py`
- `D	tests/test_batched_vision_blob_store.py`
- `D	tests/test_batched_vision_cache_io_thread.py`
- `D	tests/test_batched_vision_cache_store.py`
- `D	tests/test_batched_vision_chunks.py`
- `D	tests/test_batched_vision_coordinator.py`
- `D	tests/test_batched_vision_disk_budget.py`
- `D	tests/test_batched_vision_image_spans.py`
- `D	tests/test_batched_vision_model_kit.py`
- `D	tests/test_batched_vision_parity.py`
- `D	tests/test_batched_vision_prompt_inputs.py`
- `D	tests/test_batched_vision_qwen_mrope.py`
- `D	tests/test_batched_vision_records.py`
- `D	tests/test_batched_vision_request_lifecycle.py`
- `D	tests/test_batched_vision_restore_planner.py`
- `M	tests/test_cache_wrapper.py`
- `D	tests/test_patched_gemma4.py`
- `D	tests/test_patched_qwen3_5.py`
- `D	tests/test_prefill_step_size.py`
- `D	tests/test_repetition_penalty_processor.py`
- `M	tests/test_text_models.py`
- `D	tests/test_vision_feature_cache.py`
- `M	tests/test_vision_models.py`
- `M	tests/utils/test_prompt_progress_reporter.py`

### Unmatched Patch IDs

- `+ 461015c789cb88ebb8a5dea339ef38bd0c887ae7 Add test for LFM 2.5 caching`

## `upstream/neil/img-caching`

- Head: `7dfe3cd`
- Subject: cleanup
- Ahead/behind vs upstream main: `{'left': 39, 'right': 13}`
- Change surface: `broad`
- Changed files: `94`
- Unmatched patch-id commits: `13`

### Changed Files

- `D	batched_demo.py`
- `M	demo.py`
- `M	mlx_engine/__init__.py`
- `M	mlx_engine/cache_wrapper.py`
- `M	mlx_engine/external/models/ernie4_5/tokenization_ernie4_5.py`
- `D	mlx_engine/external/models/lfm2_vl/router_lfm2_vl_processor.py`
- `M	mlx_engine/generate.py`
- `M	mlx_engine/model_kit/__init__.py`
- `D	mlx_engine/model_kit/batched_model_kit.py`
- `D	mlx_engine/model_kit/batched_model_kit_types.py`
- `D	mlx_engine/model_kit/batched_vision/__init__.py`
- `D	mlx_engine/model_kit/batched_vision/batch_generator.py`
- `D	mlx_engine/model_kit/batched_vision/cache_io_thread.py`
- `D	mlx_engine/model_kit/batched_vision/model_kit.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/__init__.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/blob_store.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/cache_store.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/chunks.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/coordinator.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/disk_budget.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/image_spans.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/records.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/restore_planner.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_cache/types.py`
- `D	mlx_engine/model_kit/batched_vision/prompt_inputs.py`
- `D	mlx_engine/model_kit/batched_vision/qwen_mrope.py`
- `D	mlx_engine/model_kit/batched_vision/request_lifecycle.py`
- `D	mlx_engine/model_kit/batched_vision/vision_feature_memoizer.py`
- `M	mlx_engine/model_kit/model_kit.py`
- `D	mlx_engine/model_kit/patches/gemma4.py`
- `D	mlx_engine/model_kit/patches/lfm2_vl.py`
- `D	mlx_engine/model_kit/patches/qwen3_5.py`
- `A	mlx_engine/model_kit/vision_add_ons/base.py`
- `A	mlx_engine/model_kit/vision_add_ons/gemma3.py`
- `A	mlx_engine/model_kit/vision_add_ons/gemma3n.py`
- `A	mlx_engine/model_kit/vision_add_ons/lfm2_vl.py`
- `A	mlx_engine/model_kit/vision_add_ons/load_utils.py`
- `A	mlx_engine/model_kit/vision_add_ons/mistral3.py`
- `A	mlx_engine/model_kit/vision_add_ons/pixtral.py`
- `A	mlx_engine/model_kit/vision_add_ons/process_prompt_with_images.py`
- `A	mlx_engine/model_kit/vision_add_ons/qwen2_vl.py`
- `A	mlx_engine/model_kit/vision_add_ons/qwen3_vl.py`
- `A	mlx_engine/model_kit/vision_add_ons/qwen3_vl_moe.py`
- `A	mlx_engine/model_kit/vision_add_ons/qwen_vl_utils.py`
- `M	mlx_engine/processors/repetition_penalty_processor.py`
- `M	mlx_engine/utils/eot_tokens.py`
- `D	mlx_engine/utils/fix_mistral_pre_tokenizer.py`
- `D	mlx_engine/utils/generation_helpers.py`
- `D	mlx_engine/utils/generation_result.py`
- `D	mlx_engine/utils/mlx_threading.py`
- `A	mlx_engine/utils/progress_decorators.py`
- `M	mlx_engine/utils/prompt_processing.py`
- `D	mlx_engine/utils/prompt_progress_events.py`
- `D	mlx_engine/utils/prompt_progress_reporter.py`
- `M	mlx_engine/utils/register_models.py`
- `D	mlx_engine/utils/sampling.py`
- `M	mlx_engine/utils/speculative_decoding.py`
- `R097	mlx_engine/model_kit/batched_vision/transformers_compatibility.py	mlx_engine/vision_model_kit/_transformers_compatibility.py`
- `A	mlx_engine/vision_model_kit/vision_model_kit.py`
- `A	mlx_engine/vision_model_kit/vision_model_wrapper.py`
- `M	requirements.txt`
- `M	tests/data/ben_franklin_autobiography_start.txt`
- `D	tests/data/equations.jpg`
- `D	tests/patched_model_test_utils.py`
- `M	tests/shared.py`
- `D	tests/test_batched_generation.py`
- `D	tests/test_batched_vision_batch_generator.py`
- `D	tests/test_batched_vision_blob_store.py`
- `D	tests/test_batched_vision_cache_io_thread.py`
- `D	tests/test_batched_vision_cache_store.py`
- `D	tests/test_batched_vision_chunks.py`
- `D	tests/test_batched_vision_coordinator.py`
- `D	tests/test_batched_vision_disk_budget.py`
- `D	tests/test_batched_vision_image_spans.py`
- `D	tests/test_batched_vision_model_kit.py`
- `D	tests/test_batched_vision_parity.py`
- `D	tests/test_batched_vision_prompt_inputs.py`
- `D	tests/test_batched_vision_qwen_mrope.py`
- `D	tests/test_batched_vision_records.py`
- `D	tests/test_batched_vision_request_lifecycle.py`
- `D	tests/test_batched_vision_restore_planner.py`
- `M	tests/test_cache_wrapper.py`
- `D	tests/test_patched_gemma4.py`
- `D	tests/test_patched_qwen3_5.py`
- `D	tests/test_prefill_step_size.py`
- `D	tests/test_repetition_penalty_processor.py`
- `M	tests/test_stop_string_processor.py`
- `M	tests/test_text_models.py`
- `A	tests/test_vision_cache.py`
- `D	tests/test_vision_feature_cache.py`
- `M	tests/test_vision_models.py`
- `A	tests/utils/test_progress_decorators.py`
- `D	tests/utils/test_prompt_progress_events.py`
- `D	tests/utils/test_prompt_progress_reporter.py`

### Unmatched Patch IDs

- `+ db145daf763f55ac57ff05e883c6373beae5db7e refactor qwen2_vl vision add on`
- `+ 2ce6b9d77f861ce893e2618df1776e35989ee54a working p1`
- `+ 6a04c0e7e54e8d550e8da995a48ba8d1f6cc6707 use text-only hook`
- `+ c9777fe146a641b261d8b0a9e4b7025853b6338d working test`
- `+ cb2dba7b4227c6c3a71871168422035a7bbd8c94 simplify`
- `+ 1ae47321059103ad9db9263cccf70b7cf98580e3 test non-swa model`
- `+ 47e1d8a52d181f310a1799acc7082eb598ed0d11 checkpoint`
- `+ c05c2416bba386ff1261e1627e492b8a4efc6fc9 checkpoint`
- `+ 973907bdf0ca1cd8f6ab54524e0ce11860d92e9d checkpoint`
- `+ 4e94de98231f42ad01c3942d25e90c173d214b29 remove list`
- `+ 36c360a5ee9352439f954b43c182f570197d58f6 cleanup`
- `+ fd38585257b6c24ddd60bc4d7d43d89487cb9fca cleanup`
- `+ 7dfe3cdeeedf289b9b8bed72f67fcabe201765c2 cleanup`
