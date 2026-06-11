"""Pure-Python matching logic for the DRY (Don't Repeat Yourself) sampler.

Kept free of torch/sglang imports so the matching logic is unit-testable
outside a GPU environment.
"""

from typing import Dict, List, Optional, Set

# Cap on how much history DRY scans per step.
DRY_CONTEXT_WINDOW = 2048
# Cap on the penalty exponent so base ** exponent cannot overflow.
DRY_MAX_EXPONENT = 32


def _z_array(seq: List[int]) -> List[int]:
    """Z-array: z[i] = length of the longest common prefix of seq and seq[i:]."""
    n = len(seq)
    z = [0] * n
    z[0] = n
    l = r = 0
    for i in range(1, n):
        if i < r:
            z[i] = min(r - i, z[i - l])
        while i + z[i] < n and seq[z[i]] == seq[i + z[i]]:
            z[i] += 1
        if i + z[i] > r:
            l, r = i, i + z[i]
    return z


def dry_candidate_penalties(
    tokens: List[int],
    multiplier: float,
    base: float,
    allowed_length: int,
    breaker_ids: Optional[Set[int]] = None,
) -> Dict[int, float]:
    """Return {token_id: penalty} for tokens that would extend a repeat.

    For each position p in the context, let k be the length of the match
    between the tokens just before p and the current suffix of the context.
    If k > allowed_length, emitting tokens[p] now would extend that repeat
    to length k + 1, so it is penalized with multiplier * base ** (k -
    allowed_length). Each candidate keeps its largest penalty across all
    such positions. Matches never include breaker tokens, so a repeat
    cannot extend across a breaker.
    """
    tokens = tokens[-DRY_CONTEXT_WINDOW:]
    n = len(tokens)
    if n < 2 or multiplier <= 0:
        return {}
    # Negative ids (e.g. unresolved future-token placeholders from the
    # overlap scheduler) and breakers become unique sentinels, so no match
    # can include or cross them and positions stay aligned.
    if breaker_ids or any(t < 0 for t in tokens):
        breaker_ids = breaker_ids or set()
        tokens = [
            -(idx + 1) if (t < 0 or t in breaker_ids) else t
            for idx, t in enumerate(tokens)
        ]
    z = _z_array(tokens[::-1])
    penalties: Dict[int, float] = {}
    for i in range(1, n):
        k = z[i]
        if k > allowed_length:
            candidate = tokens[n - i]
            if candidate < 0:
                # Breaker sentinel: breakers themselves are never penalized.
                continue
            exponent = min(k - allowed_length, DRY_MAX_EXPONENT)
            penalty = multiplier * (base**exponent)
            if penalty > penalties.get(candidate, 0.0):
                penalties[candidate] = penalty
    return penalties
