"""Pure arithmetic for build progress."""


def estimate_eta_seconds(
    *, done: int, total: int, elapsed_seconds: float, carried: int = 0
) -> float | None:
    """Seconds left at the pace so far, or None until it is computable.

    ``carried`` counts sources that were already done when this process started
    (a resumed run); they say nothing about the pace.
    """
    processed = done - carried
    if processed <= 0 or elapsed_seconds <= 0:
        return None
    return max(total - done, 0) * elapsed_seconds / processed
