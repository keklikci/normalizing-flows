"""Two sample statistics used by the research experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.random import default_rng
from scipy.spatial.distance import cdist, pdist
from scipy.stats import genextreme, kstwobign, pearsonr


def _paired_samples(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(x, dtype=float)
    right = np.asarray(y, dtype=float)
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("samples must be one dimensional")
    if len(left) != len(right):
        raise ValueError("sample coordinates must have equal lengths")
    if not len(left):
        raise ValueError("samples must not be empty")
    return left, right


def ks2d2s(
    x1: Any,
    y1: Any,
    x2: Any,
    y2: Any,
    nboot: int | None = None,
    extra: bool = False,
) -> float | tuple[float, float]:
    """Run the two dimensional two sample Kolmogorov Smirnov test."""
    x1, y1 = _paired_samples(x1, y1)
    x2, y2 = _paired_samples(x2, y2)
    if nboot is not None and nboot < 1:
        raise ValueError("nboot must be positive")

    statistic = avgmaxdist(x1, y1, x2, y2)
    n1, n2 = len(x1), len(x2)
    if nboot is None:
        effective_n = np.sqrt(n1 * n2 / (n1 + n2))
        r1 = pearsonr(x1, y1)[0]
        r2 = pearsonr(x2, y2)[0]
        correlation = np.sqrt(1 - 0.5 * (r1**2 + r2**2))
        scaled = (
            statistic * effective_n / (1 + correlation * (0.25 - 0.75 / effective_n))
        )
        pvalue = float(kstwobign.sf(scaled))
    else:
        rng = default_rng(0)
        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])
        bootstrapped = np.empty(nboot)
        for index in range(nboot):
            sample = rng.choice(len(x), len(x), replace=True)
            first, second = sample[:n1], sample[n1:]
            bootstrapped[index] = avgmaxdist(x[first], y[first], x[second], y[second])
        pvalue = float(np.mean(bootstrapped > statistic))
    return (pvalue, statistic) if extra else pvalue


def avgmaxdist(x1: Any, y1: Any, x2: Any, y2: Any) -> float:
    """Return the symmetric average maximum quadrant distance."""
    return (maxdist(x1, y1, x2, y2) + maxdist(x2, y2, x1, y1)) / 2


def maxdist(x1: Any, y1: Any, x2: Any, y2: Any) -> float:
    """Return the maximum quadrant distance from one sample to another."""
    x1, y1 = _paired_samples(x1, y1)
    x2, y2 = _paired_samples(x2, y2)
    distances = []
    for x, y in zip(x1, y1):
        first = np.asarray(quadct(x, y, x1, y1))
        second = np.asarray(quadct(x, y, x2, y2))
        distances.extend(first - second)
    distances = np.asarray(distances)
    distances[::4] -= 1 / len(x1)
    return float(max(-distances.min(), distances.max() + 1 / len(x1)))


def quadct(x: float, y: float, xx: Any, yy: Any) -> tuple[float, float, float, float]:
    """Return the four empirical quadrant counts around a point."""
    xx, yy = _paired_samples(xx, yy)
    horizontal = xx <= x
    vertical = yy <= y
    first = np.mean(horizontal & vertical)
    second = np.mean(horizontal & ~vertical)
    third = np.mean(~horizontal & vertical)
    return float(first), float(second), float(third), float(1 - first - second - third)


def estat2d(x1: Any, y1: Any, x2: Any, y2: Any, **kwargs: Any) -> tuple[Any, ...]:
    """Run the energy statistic on two dimensional coordinate samples."""
    return estat(np.c_[x1, y1], np.c_[x2, y2], **kwargs)


def estat(
    x: Any,
    y: Any,
    nboot: int = 1000,
    replace: bool = False,
    method: str = "log",
    fitting: bool = False,
) -> tuple[Any, ...]:
    """Run a bootstrap energy distance test on two samples."""
    if nboot < 1:
        raise ValueError("nboot must be positive")
    first, second = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("samples must be two dimensional with matching features")
    if len(first) < 2 or len(second) < 2:
        raise ValueError("samples must contain at least two rows")
    stacked = np.vstack([first, second])
    stacked = (stacked - stacked.mean(0)) / stacked.std(0)
    rng = default_rng(0)
    total = len(stacked)
    statistic = energy(stacked[: len(first)], stacked[len(first) :], method)
    bootstrapped = np.empty(nboot)
    for index in range(nboot):
        indices = rng.integers(total, size=total) if replace else rng.permutation(total)
        bootstrapped[index] = energy(
            stacked[indices[: len(first)]], stacked[indices[len(first) :]], method
        )
    if fitting:
        parameters = genextreme.fit(bootstrapped)
        return float(genextreme.sf(statistic, *parameters)), statistic, parameters
    return float(np.mean(bootstrapped >= statistic)), statistic, bootstrapped


def energy(x: Any, y: Any, method: str = "log") -> float:
    """Return the energy distance statistic for two feature matrices."""
    distances_x, distances_y, distances_xy = pdist(x), pdist(y), cdist(x, y)
    if method == "log":
        distances_x, distances_y, distances_xy = (
            np.log(distances_x),
            np.log(distances_y),
            np.log(distances_xy),
        )
    elif method != "linear":
        raise ValueError(f"unsupported method: {method}")
    return float(
        distances_xy.sum() / (len(x) * len(y))
        - distances_x.sum() / len(x) ** 2
        - distances_y.sum() / len(y) ** 2
    )
