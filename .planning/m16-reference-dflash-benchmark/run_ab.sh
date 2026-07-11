#!/usr/bin/env bash
# M16 reference DFlash A/B benchmark: baseline vs reference-DFlash candidate.
# Each invocation uses one prompt, identical config, sequential.
# Outputs a JSON Lines evidence file per side.

set -u

TARGET="/Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/lmstudio-community/Qwen3.6-27B-MLX-8bit"
DRAFTER="/Volumes/StudioStackSSD4TB/Development/LLM/huggingface/hub/models--z-lab--Qwen3.5-27B-DFlash/snapshots/25ee0025ff950496a634e100b75c2db4515e9824"
MAX_TOKENS=16
TEMPERATURE=0.0
SEED=0
DRAFT_BLOCK_SIZE=8

# 3 short deterministic prompts. Identical for both baseline and candidate.
PROMPTS=(
  "Reply with the single word ok."
  "What is 2 + 2? Reply with a single digit."
  "List the first 5 prime numbers separated by commas."
)

OUT_DIR="/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.planning/m16-reference-dflash-benchmark"
mkdir -p "$OUT_DIR"
TS="$(date -u +%Y%m%dT%H%M%SZ)"

run_one() {
  local mode="$1"          # "baseline" or "dflash"
  local prompt="$2"
  local prompt_id="$3"

  local log_path="${OUT_DIR}/${mode}-${TS}-prompt${prompt_id}.log"
  local jsonl_path="${OUT_DIR}/${mode}-${TS}-prompt${prompt_id}.json"

  local cmd=( .venv-py312/bin/python -m mlx_vlm.generate
              --model "$TARGET"
              --prompt "$prompt"
              --max-tokens "$MAX_TOKENS"
              --temperature "$TEMPERATURE"
              --seed "$SEED"
              --verbose )

  if [[ "$mode" == "dflash" ]]; then
    cmd+=( --draft-model "$DRAFTER"
           --draft-kind dflash
           --draft-block-size "$DRAFT_BLOCK_SIZE" )
  fi

  # Run with wall-time measurement.
  local start_ns end_ns exit_code
  start_ns="$(date +%s%N)"

  "${cmd[@]}" >"$log_path" 2>&1
  exit_code=$?

  end_ns="$(date +%s%N)"
  local wall_s
  wall_s=$(awk -v s="$start_ns" -v e="$end_ns" 'BEGIN { printf "%.6f", (e - s) / 1e9 }')

  # Parse key fields from the log.
  local prompt_tokens gen_tokens gen_tps peak_mem out_text draft_stats
  prompt_tokens=$(grep -oE "Prompt: [0-9]+ tokens" "$log_path" | head -1 | grep -oE "[0-9]+" || echo "0")
  gen_tokens=$(grep -oE "Generation: [0-9]+ tokens" "$log_path" | head -1 | grep -oE "[0-9]+" || echo "0")
  gen_tps=$(grep -oE "Generation: [0-9]+ tokens, [0-9.]+ tokens-per-sec" "$log_path" | head -1 | grep -oE "[0-9.]+ tokens-per-sec" | awk '{print $1}' || echo "0")
  peak_mem=$(grep -oE "Peak memory: [0-9.]+ GB" "$log_path" | head -1 | awk '{print $3}' || echo "0")
  draft_stats=$(grep -oE "Speculative decoding: [^$]*" "$log_path" | head -1 || echo "")

  # Output text is between "==========" markers (between "Generation" block).
  # The verbose CLI prints: ======, then "Prompt: ..." paragraph, then "==========" before "Prompt: 19 tokens...".
  # We extract the assistant output by grabbing lines after the second "==========" but before "Prompt: 19 tokens..."
  out_text=$(awk '
    BEGIN { state=0 }
    /==========$/ { state++; if (state == 2) { next } }
    state == 2 && /==========$/ { state = 3; next }
    state == 2 { print }
    state == 3 { exit }
  ' "$log_path" | sed '/^Prompt: /d' | head -c 600)

  cat >"$jsonl_path" <<EOF
{"mode":"${mode}","prompt_id":${prompt_id},"prompt":$(printf '%s' "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'),"max_tokens":${MAX_TOKENS},"temperature":${TEMPERATURE},"seed":${SEED},"wall_time_s":${wall_s},"prompt_tokens":${prompt_tokens},"generation_tokens":${gen_tokens},"generation_tps":${gen_tps},"peak_memory_gb":${peak_mem},"exit_code":${exit_code},"output_text":$(printf '%s' "$out_text" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'),"draft_stats":${draft_stats:+$(printf '%s' "$draft_stats" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}}
EOF
  echo "[${mode} prompt${prompt_id}] exit=${exit_code} wall=${wall_s}s prompt_tok=${prompt_tokens} gen_tok=${gen_tokens} gen_tps=${gen_tps} peak_gb=${peak_mem}"
}

run_side() {
  local mode="$1"
  for i in "${!PROMPTS[@]}"; do
    run_one "$mode" "${PROMPTS[$i]}" $((i + 1))
  done
}

echo "=== M16 reference DFlash A/B ==="
echo "Date (UTC): $(date -u)"
echo "Target: $TARGET"
echo "Drafter: $DRAFTER"
echo "Max tokens: $MAX_TOKENS, Temperature: $TEMPERATURE, Seed: $SEED, Draft block size: $DRAFT_BLOCK_SIZE"
echo "Prompts: ${#PROMPTS[@]}"

echo "--- baseline (no DFlash) ---"
run_side baseline

echo "--- candidate (reference DFlash) ---"
run_side dflash

echo "--- done ---"
echo "Evidence directory: $OUT_DIR"
ls -la "$OUT_DIR" | head -40
