import pytest

from mlx_engine.utils.suffix_decoding import propose_suffix_decoding_tokens


def test_propose_suffix_decoding_tokens_returns_none_when_no_suffix_repeats():
    proposal = propose_suffix_decoding_tokens([1, 2, 3, 4, 5], max_draft_tokens=4)

    assert proposal is None


def test_propose_suffix_decoding_tokens_chooses_longest_non_overlapping_suffix():
    proposal = propose_suffix_decoding_tokens(
        [4, 5, 6, 7, 4, 5, 6, 7],
        max_draft_tokens=4,
    )

    assert proposal is not None
    assert proposal.source_start_index == 0
    assert proposal.matched_suffix_length == 4
    assert proposal.draft_tokens == (4, 5, 6, 7)


def test_propose_suffix_decoding_tokens_caps_draft_length():
    proposal = propose_suffix_decoding_tokens(
        [1, 2, 3, 4, 5, 6, 1, 2, 3, 4],
        max_draft_tokens=2,
    )

    assert proposal is not None
    assert proposal.matched_suffix_length == 4
    assert proposal.draft_tokens == (5, 6)


def test_propose_suffix_decoding_tokens_skips_overlapping_source_occurrences():
    proposal = propose_suffix_decoding_tokens(
        [7, 8, 9, 10, 11, 12, 7, 8, 9, 10, 11, 12],
        max_draft_tokens=4,
    )

    assert proposal is not None
    assert proposal.source_start_index == 0
    assert proposal.matched_suffix_length == 6
    assert proposal.draft_tokens == (7, 8, 9, 10)


@pytest.mark.parametrize(
    ("history", "kwargs", "expected_tokens"),
    [
        (
            [1, 2, 3, 4, 5, 6, 1, 2, 3],
            {"stop_token_ids": [5]},
            (4,),
        ),
        (
            [7, 8, 9, 10, 11, 12, 7, 8, 9],
            {"eos_token_ids": [11]},
            (10,),
        ),
    ],
)
def test_propose_suffix_decoding_tokens_respects_boundary_tokens(
    history,
    kwargs,
    expected_tokens,
):
    proposal = propose_suffix_decoding_tokens(
        history,
        max_draft_tokens=4,
        **kwargs,
    )

    assert proposal is not None
    assert proposal.draft_tokens == expected_tokens
