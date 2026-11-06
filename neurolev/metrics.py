"""Standard BCI performance metrics."""

import math


def information_transfer_rate(n_targets: int, accuracy: float, seconds_per_selection: float) -> float:
    """Wolpaw information transfer rate in bits per minute.

    seconds_per_selection should include the decision window AND the time a
    user needs to shift their gaze, or the number will be inflated.
    """
    if n_targets < 2:
        raise ValueError("Need at least 2 targets")
    if seconds_per_selection <= 0:
        raise ValueError("seconds_per_selection must be positive")
    p = min(max(accuracy, 0.0), 1.0)
    n = n_targets
    if p <= 1.0 / n:
        return 0.0
    bits = math.log2(n)
    if p < 1.0:
        bits += p * math.log2(p) + (1 - p) * math.log2((1 - p) / (n - 1))
    return bits * 60.0 / seconds_per_selection
