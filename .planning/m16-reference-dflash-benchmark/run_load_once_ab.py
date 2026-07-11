#!/usr/bin/env python3
"""M16 load-once A/B benchmark.

Loads the Qwen3.6-27B target model + z-lab DFlash drafter exactly once,
then runs the same 3 prompts sequentially in two modes:
  - baseline: pure autoregressive generation (no draft_model)
  - candidate: reference DFlash speculative generation

This eliminates per-CLI-invocation model+drafter load overhead (~16 s
each), so the per-prompt gen_tps reflects true generation cost.
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

from mlx_vlm.generate import generate as vl_generate
from mlx_vlm.utils import load as vl_load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.speculative.drafters import load_drafter

TARGET = "/Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/lmstudio-community/Qwen3.6-27B-MLX-8bit"
DRAFTER = "/Volumes/StudioStackSSD4TB/Development/LLM/huggingface/hub/models--z-lab--Qwen3.5-27B-DFlash/snapshots/25ee0025ff950496a634e100b75c2db4515e9824"

PROMPTS = [
    "Reply with the single word ok.",
    "What is 2 + 2? Reply with a single digit.",
    "List the first 5 prime numbers separated by commas.",
]

MAX_TOKENS = 16
TEMPERATURE = 0.0
SEED = 0

OUT_DIR = Path(
    "/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.planning/m16-reference-dflash-benchmark"
)


def format_speculative(draft_model):
    a = getattr(draft_model, "accept_lens", None) or []
    d = getattr(draft_model, "draft_lens", None) or []
    if not a or not d:
        return ""
    total_a = sum(int(x) for x in a)
    total_d = sum(int(x) for x in d)
    rounds = len(a)
    avg_a = total_a / rounds
    avg_d = total_d / rounds if rounds else 0
    rate = total_a / total_d if total_d > 0 else 0
    return (
        f"{avg_a:.2f} accepted tokens/round "
        f"({(avg_a - 1):.2f} accepted drafts/round after bonus, "
        f"{rate * 100:.1f}% acceptance, avg draft {avg_d:.2f}) over {rounds} rounds"
    )


def time_one(
    model,
    processor,
    prompt,
    *,
    draft_model=None,
    draft_kind=None,
    draft_block_size=None,
):
    """Run one prompt via mlx_vlm.generate and return a dict of metrics."""
    config = model.config
    chat_kwargs = {"enable_thinking": False}
    rendered = apply_chat_template(
        processor,
        config,
        prompt,
        num_images=0,
        num_audios=0,
        **chat_kwargs,
    )
    if draft_model is not None:
        draft_model.accept_lens = []
        draft_model.draft_lens = []
    gen_kwargs = {
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "seed": SEED,
        "verbose": False,
    }
    if draft_model is not None:
        gen_kwargs["draft_model"] = draft_model
        gen_kwargs["draft_kind"] = draft_kind
        if draft_block_size is not None:
            gen_kwargs["draft_block_size"] = draft_block_size

    # Warmup mx stream to keep timing clean.
    mx.synchronize()
    t0 = time.monotonic()
    result = vl_generate(model, processor, rendered, **gen_kwargs)
    mx.synchronize()
    wall = time.monotonic() - t0

    return {
        "wall_time_s": wall,
        "prompt_tokens": getattr(result, "prompt_tokens", 0),
        "generation_tokens": getattr(result, "generation_tokens", 0),
        "generation_tps": getattr(result, "generation_tps", 0.0),
        "peak_memory_gb": getattr(result, "peak_memory", 0.0),
        "output_text": (result.text or "").strip(),
        "draft_stats": format_speculative(draft_model)
        if draft_model is not None
        else None,
        "finish_reason": getattr(result, "finish_reason", None),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-prefix", default="load_once")
    p.add_argument("--draft-block-size", type=int, default=8)
    p.add_argument("--num-prompts", type=int, default=len(PROMPTS))
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== M16 load-once A/B ===", flush=True)
    print(f"Target: {TARGET}", flush=True)
    print(f"Drafter: {DRAFTER}", flush=True)
    print(
        f"Max tokens: {MAX_TOKENS}, Temperature: {TEMPERATURE}, Seed: {SEED}",
        flush=True,
    )
    print(f"Draft block size: {args.draft_block_size}", flush=True)
    print(f"Prompts: {args.num_prompts}", flush=True)

    # Load target once
    load_t0 = time.monotonic()
    print("\n[load] Loading target model ...", flush=True)
    model, processor = vl_load(TARGET)
    print(f"[load] Target loaded in {time.monotonic() - load_t0:.1f}s", flush=True)

    # Load drafter once (DFlash only — baseline mode below skips it)
    load_t1 = time.monotonic()
    print("[load] Loading drafter (dflash) ...", flush=True)
    draft_model, draft_kind = load_drafter(DRAFTER, kind="dflash")
    print(
        f"[load] Drafter loaded in {time.monotonic() - load_t1:.1f}s, kind={draft_kind}",
        flush=True,
    )

    prompts = PROMPTS[: args.num_prompts]
    rows = []
    # Run each prompt in two modes: baseline (no draft) then dflash (with draft)
    for i, prompt in enumerate(prompts, start=1):
        # Baseline first (no draft)
        print(f"\n[run] baseline prompt{i}: {prompt!r}", flush=True)
        try:
            r_base = time_one(model, processor, prompt)
        except Exception as e:
            print(f"  ! ERROR: {type(e).__name__}: {e}", flush=True)
            r_base = {
                "error": f"{type(e).__name__}: {e}",
                "wall_time_s": 0,
                "prompt_tokens": 0,
                "generation_tokens": 0,
                "generation_tps": 0,
                "peak_memory_gb": 0,
                "output_text": "",
                "draft_stats": None,
            }
        r_base["mode"] = "baseline"
        r_base["prompt_id"] = i
        r_base["prompt"] = prompt
        rows.append(r_base)
        print(
            f"  -> wall={r_base['wall_time_s']:.3f}s, gen_tok={r_base['generation_tokens']}, "
            f"gen_tps={r_base['generation_tps']:.2f}, peak={r_base['peak_memory_gb']:.1f} GB",
            flush=True,
        )
        print(f"     out={r_base['output_text']!r}", flush=True)

        # DFlash candidate
        print(f"\n[run] dflash   prompt{i}: {prompt!r}", flush=True)
        try:
            r_d = time_one(
                model,
                processor,
                prompt,
                draft_model=draft_model,
                draft_kind=draft_kind,
                draft_block_size=args.draft_block_size,
            )
        except Exception as e:
            print(f"  ! ERROR: {type(e).__name__}: {e}", flush=True)
            r_d = {
                "error": f"{type(e).__name__}: {e}",
                "wall_time_s": 0,
                "prompt_tokens": 0,
                "generation_tokens": 0,
                "generation_tps": 0,
                "peak_memory_gb": 0,
                "output_text": "",
                "draft_stats": None,
            }
        r_d["mode"] = "dflash"
        r_d["prompt_id"] = i
        r_d["prompt"] = prompt
        rows.append(r_d)
        print(
            f"  -> wall={r_d['wall_time_s']:.3f}s, gen_tok={r_d['generation_tokens']}, "
            f"gen_tps={r_d['generation_tps']:.2f}, peak={r_d['peak_memory_gb']:.1f} GB",
            flush=True,
        )
        print(f"     out={r_d['output_text']!r}", flush=True)
        if r_d.get("draft_stats"):
            print(f"     draft_stats={r_d['draft_stats']}", flush=True)

    out_path = OUT_DIR / f"{args.out_prefix}-results.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
