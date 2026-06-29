#!/usr/bin/env python3
"""M16 evidence aggregator — reads the per-prompt run logs directly and
produces aggregate-summary.json plus a human-readable print summary.

Robust to the broken baseline-N-...json files (which had an awk extraction
bug). We re-parse the .log files since they contain the authoritative
assistant output and the CLI metrics.
"""
import json
import re
import sys
from pathlib import Path

LOG_DIR = Path("/Users/jeffreycruz/Development/LLM_INFERENCE/mlx-engine/.planning/m16-reference-dflash-benchmark")


def parse_log(path):
    content = path.read_text()
    assistant_m = re.search(
        r"<\|im_start\|>assistant\n(.*?)={10,}", content, re.DOTALL
    )
    output_text = ""
    if assistant_m:
        body = assistant_m.group(1)
        output_text = body.rsplit("<|im_start\|>assistant\n", 1)[-1].strip()
    ptm = re.search(r"Prompt:\s+(\d+)\s+tokens,\s+([\d.]+)\s+tokens-per-sec", content)
    gtm = re.search(
        r"Generation:\s+(\d+)\s+tokens,\s+([\d.]+)\s+tokens-per-sec", content
    )
    mem = re.search(r"Peak memory:\s+([\d.]+)\s+GB", content)
    draft = re.search(r"Speculative decoding:\s+([^\n]+)", content)
    return {
        "path": str(path),
        "prompt_tokens": int(ptm.group(1)) if ptm else 0,
        "prompt_tps": float(ptm.group(2)) if ptm else 0.0,
        "generation_tokens": int(gtm.group(1)) if gtm else 0,
        "generation_tps": float(gtm.group(2)) if gtm else 0.0,
        "peak_memory_gb": float(mem.group(1)) if mem else 0.0,
        "draft_stats": draft.group(1) if draft else None,
        "output_text": output_text,
    }


def run_dir(mode, ts):
    return LOG_DIR


def main():
    ts = sys.argv[1] if len(sys.argv) > 1 else "20260629T221606Z"

    # Find baseline + dflash log files for the timestamp
    base_logs = sorted(LOG_DIR.glob(f"baseline-{ts}-prompt*.log"))
    dflash_logs = sorted(LOG_DIR.glob(f"dflash-{ts}-prompt*.log"))

    if not base_logs or not dflash_logs:
        print("No logs found for ts:", ts)
        sys.exit(1)

    # Re-read prompt from the JSON even if parsing is partial.
    def read_prompt(mode, pid):
        jpath = LOG_DIR / f"{mode}-{ts}-prompt{pid}.json"
        if not jpath.exists():
            return None
        try:
            data = json.loads(jpath.read_text())
            return data.get("prompt"), data.get("wall_time_s"), data.get("exit_code")
        except Exception:
            # Try to read just the prompt with regex
            text = jpath.read_text()
            pm = re.search(r'"prompt":"([^"]*)"', text)
            wm = re.search(r'"wall_time_s":([\d.]+)', text)
            return (pm.group(1) if pm else None,
                    float(wm.group(1)) if wm else None,
                    None)

    results = {"baseline": {}, "dflash": {}}
    for path in base_logs:
        pid_m = re.search(r"prompt(\d+)\.log", str(path))
        if not pid_m:
            continue
        pid = int(pid_m.group(1))
        info = parse_log(path)
        meta = read_prompt("baseline", pid)
        info["wall_time_s"] = meta[1] if meta else None
        info["exit_code"] = meta[2] if meta else None
        info["prompt"] = meta[0] if meta else None
        results["baseline"][pid] = info

    for path in dflash_logs:
        pid_m = re.search(r"prompt(\d+)\.log", str(path))
        if not pid_m:
            continue
        pid = int(pid_m.group(1))
        info = parse_log(path)
        meta = read_prompt("dflash", pid)
        info["wall_time_s"] = meta[1] if meta else None
        info["exit_code"] = meta[2] if meta else None
        info["prompt"] = meta[0] if meta else None
        results["dflash"][pid] = info

    # Compute per-prompt deltas
    per_prompt = []
    for pid in sorted(results["baseline"]):
        b = results["baseline"][pid]
        c = results["dflash"].get(pid)
        if not c:
            continue
        delta_tps = c["generation_tps"] - b["generation_tps"]
        delta_pct = (
            (delta_tps / b["generation_tps"]) * 100 if b["generation_tps"] > 0 else 0
        )
        per_prompt.append(
            {
                "prompt_id": pid,
                "prompt": b["prompt"],
                "baseline": {
                    "wall_time_s": b["wall_time_s"],
                    "generation_tokens": b["generation_tokens"],
                    "generation_tps": b["generation_tps"],
                    "peak_memory_gb": b["peak_memory_gb"],
                    "output_text": b["output_text"],
                },
                "dflash": {
                    "wall_time_s": c["wall_time_s"],
                    "generation_tokens": c["generation_tokens"],
                    "generation_tps": c["generation_tps"],
                    "peak_memory_gb": c["peak_memory_gb"],
                    "output_text": c["output_text"],
                    "draft_stats": c["draft_stats"],
                },
                "delta_tps": delta_tps,
                "delta_pct": delta_pct,
                "outputs_match": b["output_text"] == c["output_text"],
            }
        )

    # Aggregate stats
    agg = {}
    for mode in ("baseline", "dflash"):
        rows = list(results[mode].values())
        if not rows:
            continue
        agg[mode] = {
            "n_prompts": len(rows),
            "total_generation_tokens": sum(r["generation_tokens"] for r in rows),
            "total_wall_time_s": sum(
                r["wall_time_s"] for r in rows if r["wall_time_s"] is not None
            ),
            "avg_gen_tps": sum(r["generation_tps"] for r in rows) / len(rows),
            "max_peak_memory_gb": max(r["peak_memory_gb"] for r in rows),
            "min_peak_memory_gb": min(r["peak_memory_gb"] for r in rows),
        }

    # Decision factors
    decision = {
        "outputs_match_all_prompts": all(p["outputs_match"] for p in per_prompt),
        "non_empty_all_outputs": all(
            bool(p["baseline"]["output_text"]) and bool(p["dflash"]["output_text"])
            for p in per_prompt
        ),
        "keyword_expected_present_all_prompts": all(
            {
                1: "ok",
                2: "4",
                3: "2, 3, 5, 7, 11",
            }[p["prompt_id"]].lower() in (p["dflash"]["output_text"] or "").lower()
            for p in per_prompt
        ),
        "no_visible_thinking_leak": all(
            "think" not in (p["dflash"]["output_text"] or "").lower()
            for p in per_prompt
        ),
        "no_repetition_loops": all(
            not re.search(r"(.)\1{15,}", (p["dflash"]["output_text"] or ""))
            for p in per_prompt
        ),
        "dflash_wins_prompts": sum(
            1 for p in per_prompt if p["delta_pct"] > 5
        ),
        "dflash_loses_prompts": sum(
            1 for p in per_prompt if p["delta_pct"] < -5
        ),
        "dflash_mixed_prompts": sum(
            1 for p in per_prompt if abs(p["delta_pct"]) <= 5
        ),
    }

    # Final decision
    if (
        decision["keyword_expected_present_all_prompts"]
        and decision["no_visible_thinking_leak"]
        and decision["no_repetition_loops"]
    ):
        if decision["dflash_wins_prompts"] >= 2 and decision["dflash_loses_prompts"] == 0:
            decision_summary = "reference DFlash wins locally (clean throughput win, no quality loss)"
        elif decision["dflash_wins_prompts"] >= 1 and decision["dflash_loses_prompts"] >= 1:
            decision_summary = (
                "reference DFlash mixed locally (quality preserved; throughput win on "
                "longer outputs but overhead-dominated loss on short outputs)"
            )
        elif decision["dflash_loses_prompts"] >= 2:
            decision_summary = "reference DFlash fails locally (consistent throughput regression)"
        else:
            decision_summary = "reference DFlash neutral locally"
    else:
        decision_summary = "reference DFlash fails locally (quality regression detected)"

    summary = {
        "timestamp_utc": ts,
        "per_prompt": per_prompt,
        "aggregate": agg,
        "decision_factors": decision,
        "decision_summary": decision_summary,
    }

    out_path = LOG_DIR / "aggregate-summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote {out_path}")

    # Pretty print
    print("\n=== PER-PROMPT ===")
    for p in per_prompt:
        print(
            f"\nprompt {p['prompt_id']}: {p['prompt']!r}\n"
            f"  baseline: wall={p['baseline']['wall_time_s']:.1f}s, "
            f"gen_tok={p['baseline']['generation_tokens']}, "
            f"gen_tps={p['baseline']['generation_tps']:.2f}, "
            f"peak={p['baseline']['peak_memory_gb']:.1f} GB\n"
            f"          out={p['baseline']['output_text']!r}\n"
            f"  dflash:   wall={p['dflash']['wall_time_s']:.1f}s, "
            f"gen_tok={p['dflash']['generation_tokens']}, "
            f"gen_tps={p['dflash']['generation_tps']:.2f}, "
            f"peak={p['dflash']['peak_memory_gb']:.1f} GB\n"
            f"          out={p['dflash']['output_text']!r}\n"
            f"          draft_stats={p['dflash'].get('draft_stats', 'n/a')}\n"
            f"  Δ tps: {p['delta_tps']:+.2f} ({p['delta_pct']:+.1f}%), "
            f"outputs_match={p['outputs_match']}"
        )

    print("\n=== AGGREGATE ===")
    for mode, a in agg.items():
        print(
            f"  {mode}: avg_tps={a['avg_gen_tps']:.2f}, "
            f"total_tok={a['total_generation_tokens']}, "
            f"total_wall={a['total_wall_time_s']:.2f}s, "
            f"max_peak={a['max_peak_memory_gb']:.1f} GB, "
            f"min_peak={a['min_peak_memory_gb']:.1f} GB"
        )

    print("\n=== DECISION FACTORS ===")
    for k, v in decision.items():
        print(f"  {k}: {v}")
    print(f"\nDECISION: {decision_summary}")


if __name__ == "__main__":
    main()
