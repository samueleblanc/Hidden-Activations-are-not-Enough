"""Gram-identity cost in soft_matching must match the naive broadcast formula.

The broadcast form (p1, p2, n) is the reference semantics but OOMs at Phase-1
scale (~419 GB at p=2048, n=25000 — job 13963270); _pairwise_sq_dists must
reproduce it exactly on sizes where both fit.
"""
import numpy as np
import torch

from cka_similarity.measures.soft_matching import _pairwise_sq_dists


def _broadcast_sq_dists(X: torch.Tensor, Y: torch.Tensor) -> np.ndarray:
    Xn = X.to(torch.float64).numpy()
    Yn = Y.to(torch.float64).numpy()
    return np.linalg.norm(Xn[:, None, :] - Yn[None, :, :], axis=2) ** 2


def test_matches_broadcast_square():
    gen = torch.Generator().manual_seed(0)
    X = torch.randn(64, 200, generator=gen)
    Y = torch.randn(64, 200, generator=gen)
    np.testing.assert_allclose(
        _pairwise_sq_dists(X, Y), _broadcast_sq_dists(X, Y), rtol=1e-10, atol=1e-8
    )


def test_matches_broadcast_cross_dim():
    # p1 != p2, mimicking cross-arch D1/D2 panels (rows = neuron tuning vectors)
    gen = torch.Generator().manual_seed(1)
    X = torch.randn(48, 300, generator=gen)
    Y = torch.randn(96, 300, generator=gen)
    np.testing.assert_allclose(
        _pairwise_sq_dists(X, Y), _broadcast_sq_dists(X, Y), rtol=1e-10, atol=1e-8
    )


def test_diagonal_zero_on_self():
    gen = torch.Generator().manual_seed(2)
    X = torch.randn(32, 50, generator=gen)
    cost = _pairwise_sq_dists(X, X)
    assert (cost >= 0).all()
    np.testing.assert_allclose(np.diag(cost), np.zeros(32), atol=1e-8)
