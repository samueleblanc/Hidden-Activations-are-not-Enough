"""Bootstrap CIs + permutation null for the reduce step."""
from typing import Callable, Tuple
import numpy as np


def bootstrap_ci(values, n_resamples: int = 10_000, alpha: float = 0.05) -> Tuple[float, float]:
    """Compute (1-alpha) percentile bootstrap CI for the mean of `values`."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    rng = np.random.default_rng(0)
    means = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means.append(arr[idx].mean())
    means = np.sort(np.asarray(means))
    lo = means[int((alpha / 2) * n_resamples)]
    hi = means[int((1 - alpha / 2) * n_resamples)]
    return float(lo), float(hi)


def permutation_null_pvalue(x, y, stat: Callable, n_shuffles: int = 1000) -> float:
    """Two-sided p-value: fraction of shuffled-y stats with |stat| >= |stat_observed|."""
    rng = np.random.default_rng(0)
    observed = abs(stat(x, y))
    n = len(y)
    count = 0
    for _ in range(n_shuffles):
        y_perm = y[rng.permutation(n)]
        if abs(stat(x, y_perm)) >= observed:
            count += 1
    return count / n_shuffles
