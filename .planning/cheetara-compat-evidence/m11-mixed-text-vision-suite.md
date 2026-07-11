# M11 Mixed Text + Vision Dogfood Suite

This document defines the repeatable cheetara dogfood suite for the M11 staged cutover.
It is the black-box app-facing task set that runs against the validated host-local
surfaces only. It does not use packaged GUI automation, does not touch
`vmlx.app.asar`, and does not involve LM Studio.

## Suite goal

Prove that cheetara can complete a small real-task sequence through `mlx-engine`
on both validated paths:

- M7 external adapter on `http://127.0.0.1:3180`
- M9 source-level local compatibility on `http://127.0.0.1:3181`

## Shared runner

Use `scripts/cheetara_m11_dogfood_suite.py` from the `mlx-engine` repo root.
The runner writes a single JSON report with:

- `suite_id`
- `suite_version`
- `path_label`
- `preflight`
- `tasks`
- `summary`
- `cleanup_rules`
- `evidence_paths`

### Invocation shapes

#### M7 external adapter

```bash
cd /Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine
.venv-py312/bin/python scripts/cheetara_m11_dogfood_suite.py \
  --base-url http://127.0.0.1:3180 \
  --model cheetara-m7 \
  --path-label m7-external \
  --image-path demo-data/toucan.jpeg \
  --output .planning/cheetara-compat-evidence/m11/m7-dogfood-report.json
```

#### M9 local compatibility

```bash
cd /Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine
.venv-py312/bin/python scripts/cheetara_m11_dogfood_suite.py \
  --base-url http://127.0.0.1:3181 \
  --model cheetara-m9 \
  --path-label m9-local \
  --image-path demo-data/toucan.jpeg \
  --output .planning/cheetara-compat-evidence/m11/m9-dogfood-report.json
```

## Task set

All tasks are run sequentially. The mixed follow-up reuses the earlier text and image
answers so the final request combines prior text reasoning with image evidence.

| Task id | Kind | Prompt | Expected outcome | Request shape | Evidence path |
|---|---|---|---|---|---|
| `text_status_update` | text | Write one concise status update for this session. Mention that the cheetara path is ready and that the adapter is responding. | Must contain `ready` and `responding`. | `POST /v1/chat/completions`, `stream=true`, text-only user message. | Report JSON for the chosen path. |
| `image_description` | image | Describe the bird in the attached photo in one short sentence. | Must contain `toucan`. | `POST /v1/chat/completions`, `stream=true`, multimodal `messages[].content` with `image_url` + text. | Report JSON for the chosen path. |
| `image_qna` | image | What bird is shown, and what visible feature makes it easy to recognize? Answer in one short sentence. | Must contain `toucan` and one of `beak` or `bill`. | `POST /v1/chat/completions`, `stream=true`, multimodal `messages[].content` with `image_url` + text. | Report JSON for the chosen path. |
| `mixed_followup` | image | Status note from earlier: {text_status_update}. Image evidence from earlier: {image_qna}. Using both, write one sentence that says the session is ready and names the bird. | Must contain `ready` and `toucan`. | `POST /v1/chat/completions`, `stream=true`, multimodal `messages[].content` with `image_url` + text. | Report JSON for the chosen path. |

## Preflight checks

Each run must begin with:

1. `GET /v1/models`
2. `GET /health`

The runner fails the run if the requested model is absent, if `/health` does not
return `status=ok`, or if `/health` does not report `supports_vision=true`.

## Evidence output paths

The suite records the following path-specific evidence files:

- `.planning/cheetara-compat-evidence/m11/m7-dogfood-report.json`
- `.planning/cheetara-compat-evidence/m11/m9-dogfood-report.json`

Those reports are the authoritative evidence for the task set. They record the
request shape, streaming completion state, task outputs, and the cleanup rules
that apply between runs.

## Cleanup rules

- Run only one path at a time, never M7 and M9 concurrently.
- Stop the path under test with the matching `services.yaml` stop command before
  starting the other path.
- Do not run Qwen LLMDYNAMIX or any other MLX-heavy service during M11 capture.
- Do not modify or repack `vmlx.app.asar`.
- If any temporary persistent-cache experiment is used, remove
  `/private/tmp/mlx-engine-vlm-cache-*` before the next run.

## Path comparison

After both reports are produced, compare the M7 and M9 JSON outputs task by task.
Record:

- pass or fail per task
- streaming behavior
- warnings
- latency observations
- resource notes
- cleanup confirmation

The comparison is the basis for the daily-use readiness note and the decision
to keep LM Studio integration deferred until M11 evidence passes.
