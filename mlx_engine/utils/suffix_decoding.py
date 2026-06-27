from dataclasses import dataclass
from collections.abc import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class SuffixDecodingProposal:
    """A bounded draft-token proposal produced from token history only."""

    source_start_index: int
    matched_suffix_length: int
    draft_tokens: tuple[int, ...]


def propose_suffix_decoding_tokens(
    token_history: Sequence[int],
    *,
    max_draft_tokens: int,
    stop_token_ids: Iterable[int] = (),
    eos_token_ids: Iterable[int] = (),
    min_match_length: int = 1,
) -> SuffixDecodingProposal | None:
    """Return a suffix/ngram draft proposal from token history, or ``None``.

    The helper is intentionally pure. It never touches model state, caches, or
    generation paths. The longest non-overlapping suffix match wins, and the
    proposed continuation is truncated before any stop/eos boundary token.
    """

    if max_draft_tokens <= 0:
        return None
    if min_match_length <= 0:
        raise ValueError("min_match_length must be positive")

    history = tuple(int(token) for token in token_history)
    if len(history) <= min_match_length:
        return None

    boundary_token_ids = {int(token) for token in stop_token_ids}
    boundary_token_ids.update(int(token) for token in eos_token_ids)

    longest_match_length = min(len(history) - 1, len(history) // 2)
    for match_length in range(longest_match_length, min_match_length - 1, -1):
        suffix_start = len(history) - match_length
        if suffix_start <= 0:
            continue

        suffix = history[suffix_start:]
        source_start_limit = suffix_start - match_length
        if source_start_limit < 0:
            continue

        for source_start in range(source_start_limit, -1, -1):
            if history[source_start : source_start + match_length] != suffix:
                continue

            draft_tokens: list[int] = []
            for token in history[source_start + match_length :]:
                if token in boundary_token_ids:
                    break
                draft_tokens.append(token)
                if len(draft_tokens) >= max_draft_tokens:
                    break

            if draft_tokens:
                return SuffixDecodingProposal(
                    source_start_index=source_start,
                    matched_suffix_length=match_length,
                    draft_tokens=tuple(draft_tokens),
                )

    return None
