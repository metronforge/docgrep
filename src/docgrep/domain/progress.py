"""Pure arithmetic for build progress."""


def estimate_eta_seconds(
    *, done: int, total: int, elapsed_seconds: float
) -> float | None:
    """Seconds left at the pace so far, or None until it is computable."""
    if done <= 0 or elapsed_seconds <= 0:
        return None
    return max(total - done, 0) * elapsed_seconds / done
