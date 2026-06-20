"""Model VLM prompt-cache KV record-layout write/restore costs.

The model is token-normalized: for a fixed model, KV bytes are proportional to
the number of token positions stored across full-attention KV layers. This lets
us compare write amplification and restore record count before implementing a
new persistent-cache format.
"""

from __future__ import annotations

import argparse
import json


def current_one_step_layout(chunk_count: int) -> dict[str, float | int | str]:
    """Return costs for the retained one-step KV span layout."""
    validate_chunk_count(chunk_count)
    write_units = (2 * chunk_count) - 1
    return {
        "layout": "current_one_step",
        "chunk_count": chunk_count,
        "write_kv_chunk_units": write_units,
        "restore_kv_chunk_units": chunk_count,
        "restore_kv_records": (chunk_count + 1) // 2,
        "write_amp_vs_current": 1.0,
    }


def rejected_full_prefix_layout(chunk_count: int) -> dict[str, float | int | str]:
    """Return costs for the rejected full-prefix-at-every-boundary layout."""
    validate_chunk_count(chunk_count)
    current_units = int(current_one_step_layout(chunk_count)["write_kv_chunk_units"])
    write_units = chunk_count * (chunk_count + 1) // 2
    return {
        "layout": "rejected_full_prefix_every_boundary",
        "chunk_count": chunk_count,
        "write_kv_chunk_units": write_units,
        "restore_kv_chunk_units": chunk_count,
        "restore_kv_records": 1,
        "write_amp_vs_current": round(write_units / current_units, 3),
    }


def terminal_packed_additive_layout(chunk_count: int) -> dict[str, float | int | str]:
    """Return costs for adding one full-prefix KV record at the terminal boundary."""
    validate_chunk_count(chunk_count)
    current_units = int(current_one_step_layout(chunk_count)["write_kv_chunk_units"])
    write_units = current_units + chunk_count
    return {
        "layout": "terminal_packed_additive",
        "chunk_count": chunk_count,
        "write_kv_chunk_units": write_units,
        "restore_kv_chunk_units": chunk_count,
        "restore_kv_records": 1,
        "write_amp_vs_current": round(write_units / current_units, 3),
    }


def terminal_packed_replace_final_layout(
    chunk_count: int,
) -> dict[str, float | int | str]:
    """Return costs for replacing only the final one-step span with full-prefix KV."""
    validate_chunk_count(chunk_count)
    current_units = int(current_one_step_layout(chunk_count)["write_kv_chunk_units"])
    removed_final_record_units = 1 if chunk_count == 1 else 2
    write_units = current_units - removed_final_record_units + chunk_count
    return {
        "layout": "terminal_packed_replace_final",
        "chunk_count": chunk_count,
        "write_kv_chunk_units": write_units,
        "restore_kv_chunk_units": chunk_count,
        "restore_kv_records": 1,
        "write_amp_vs_current": round(write_units / current_units, 3),
    }


def validate_chunk_count(chunk_count: int) -> None:
    """Reject invalid chunk counts for cost-model inputs."""
    if chunk_count < 1:
        raise ValueError("chunk_count must be positive")


def compare_layouts(chunk_count: int) -> list[dict[str, float | int | str]]:
    """Return all modeled record-layout costs for a restore boundary."""
    return [
        current_one_step_layout(chunk_count),
        terminal_packed_replace_final_layout(chunk_count),
        terminal_packed_additive_layout(chunk_count),
        rejected_full_prefix_layout(chunk_count),
    ]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the record-layout model."""
    parser = argparse.ArgumentParser(
        description="Model VLM prompt-cache KV record-layout write/restore costs."
    )
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    return parser.parse_args()


def main() -> None:
    """Run the model and print a concise comparison."""
    args = parse_args()
    result = compare_layouts(args.chunks)
    if args.json:
        print(json.dumps(result, sort_keys=True))
        return

    print("VLM record-layout cost model")
    for row in result:
        print(
            "{layout}: write_units={write_kv_chunk_units} "
            "restore_units={restore_kv_chunk_units} "
            "restore_records={restore_kv_records} "
            "write_amp_vs_current={write_amp_vs_current}".format(**row)
        )


if __name__ == "__main__":
    main()
