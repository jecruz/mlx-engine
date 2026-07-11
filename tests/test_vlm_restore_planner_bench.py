from benchmarks.vlm_restore_planner_bench import run_benchmark


def test_vlm_restore_planner_benchmark_reports_equivalent_planner_metrics():
    """The planner benchmark should compare equivalent restore chains."""
    result = run_benchmark(index_chunks=64, restore_chunks=16, iterations=3)

    assert result["index_chunks"] == 64
    assert result["restore_chunks"] == 16
    assert result["records_in_index"] == 64
    assert result["selected_records"] == 8
    assert result["indexed_median_ms"] >= 0
    assert result["legacy_scan_median_ms"] >= 0
    assert result["speedup"] > 0
