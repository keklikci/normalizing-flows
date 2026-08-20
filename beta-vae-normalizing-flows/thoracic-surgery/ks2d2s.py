"""Compatibility wrapper for the packaged statistical utilities."""

from normalizing_flows.metrics import (
    avgmaxdist,
    energy,
    estat,
    estat2d,
    ks2d2s,
    maxdist,
    quadct,
)

__all__ = ["ks2d2s", "estat", "estat2d", "energy", "avgmaxdist", "maxdist", "quadct"]
