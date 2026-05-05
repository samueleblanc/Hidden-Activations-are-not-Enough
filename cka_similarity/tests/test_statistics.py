"""Tests for the bootstrap CI + permutation null helpers (Phase-1 Step C, Task 4.2)."""
import numpy as np
import pytest


def test_bootstrap_ci_covers_known_mean():
    from cka_similarity.reduce.statistics import bootstrap_ci
    np.random.seed(0)
    data = np.random.normal(loc=5.0, scale=1.0, size=200)
    lo, hi = bootstrap_ci(data, n_resamples=2000, alpha=0.05)
    assert lo < 5.0 < hi
    # CI should be tight for n=200
    assert (hi - lo) < 0.5


def test_permutation_null_low_for_correlated_data():
    from cka_similarity.reduce.statistics import permutation_null_pvalue
    np.random.seed(0)
    x = np.random.normal(size=100)
    y = x + 0.1 * np.random.normal(size=100)   # strongly correlated

    def stat(x, y):
        return float(np.corrcoef(x, y)[0, 1])

    p = permutation_null_pvalue(x, y, stat, n_shuffles=200)
    assert p < 0.05
