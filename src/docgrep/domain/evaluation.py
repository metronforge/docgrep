"""Known-item retrieval evaluation (ADR-0024): what the per-channel probe
reads from the index."""

from dataclasses import dataclass

# The most candidates one channel list of the probe may hold.
MAX_PROBE_DEPTH = 10_000


@dataclass(frozen=True, slots=True)
class ProbeHit:
    """One passage in a channel list; ``text`` only when it was asked for."""

    passage_id: str
    source_id: str
    text: str | None = None


@dataclass(frozen=True, slots=True)
class ChannelLists:
    """The dense, sparse and fused first-stage lists of one query, best first.

    ``sparse`` is None when the index has no sparse channel; it is empty
    when the query has no sparse terms. ``fused`` is the reciprocal rank
    fusion of the channels that took part.
    """

    dense: tuple[ProbeHit, ...]
    sparse: tuple[ProbeHit, ...] | None
    fused: tuple[ProbeHit, ...]
