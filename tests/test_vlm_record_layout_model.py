from benchmarks.vlm_record_layout_model import compare_layouts


def test_vlm_record_layout_model_compares_eight_chunk_restore_boundary():
    """The 8-chunk model should match the latest timed VLM restore chain."""
    rows = {row["layout"]: row for row in compare_layouts(8)}

    assert rows["current_one_step"]["write_kv_chunk_units"] == 15
    assert rows["current_one_step"]["restore_kv_records"] == 4
    assert rows["terminal_packed_replace_final"]["write_kv_chunk_units"] == 21
    assert rows["terminal_packed_replace_final"]["restore_kv_records"] == 1
    assert rows["terminal_packed_replace_final"]["write_amp_vs_current"] == 1.4
    assert rows["terminal_packed_additive"]["write_kv_chunk_units"] == 23
    assert rows["terminal_packed_additive"]["write_amp_vs_current"] == 1.533
    assert rows["rejected_full_prefix_every_boundary"]["write_kv_chunk_units"] == 36
    assert rows["rejected_full_prefix_every_boundary"]["write_amp_vs_current"] == 2.4


def test_vlm_record_layout_model_keeps_restore_kv_units_constant():
    """Candidate layouts should reduce records, not the KV bytes needed by decode."""
    for row in compare_layouts(8):
        assert row["restore_kv_chunk_units"] == 8
