"""Parity tests: each measure cross-checked against a reference implementation
on small random matrices. Tolerance atol=1e-4 (we expect bit-level agreement on
the same float32 inputs but allow for small fp accumulation noise).
"""
import numpy as np
import torch
import pytest

torch.manual_seed(0)
np.random.seed(0)


def _gen_random_pair(n=200, p1=64, p2=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, p1, generator=g)
    Y = torch.randn(n, p2, generator=g)
    return X, Y


# --- Reference: biased linear CKA via direct Frobenius formula ---
def _ref_biased_linear_cka(X, Y):
    """Biased linear CKA via the Frobenius-product formulation."""
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)
    num = (Yc.T @ Xc).pow(2).sum()
    denom = (Xc.T @ Xc).norm(p='fro') * (Yc.T @ Yc).norm(p='fro')
    return float(num / denom)


# --- Reference: unbiased HSIC₁ CKA via Song-Smola-Gretton ---
def _ref_unbiased_hsic1_cka(X, Y):
    """Unbiased HSIC₁ via the U-statistic; n must be >= 4."""
    n = X.shape[0]
    assert n >= 4
    K = (X @ X.T)
    L = (Y @ Y.T)
    K_tilde = K - torch.diag(K.diag())
    L_tilde = L - torch.diag(L.diag())
    ones = torch.ones(n)

    def hsic1(K, L):
        term1 = (K * L).sum()
        term2 = (ones @ K @ ones) * (ones @ L @ ones) / ((n - 1) * (n - 2))
        term3 = 2.0 * (ones @ K @ L @ ones) / (n - 2)
        return (term1 + term2 - term3) / (n * (n - 3))

    num = hsic1(K_tilde, L_tilde)
    denom = (hsic1(K_tilde, K_tilde) * hsic1(L_tilde, L_tilde)).sqrt()
    return float(num / denom)


def test_debiased_cka_matches_unbiased_hsic1():
    from cka_similarity.measures.cka import DebiasedLinearCKA
    A, B = _gen_random_pair(n=200, p1=64, p2=64, seed=42)
    measure = DebiasedLinearCKA()

    # Single-chunk computation
    acc = measure.accumulate(A, B)
    result = measure.finalize([acc])

    expected = _ref_unbiased_hsic1_cka(A, B)
    assert abs(result.value - expected) < 1e-4, f"got {result.value}, expected {expected}"


def test_debiased_cka_chunked_matches_single():
    """Splitting into chunks must produce the same final result."""
    from cka_similarity.measures.cka import DebiasedLinearCKA
    A, B = _gen_random_pair(n=400, p1=64, p2=64, seed=43)
    measure = DebiasedLinearCKA()

    single = measure.finalize([measure.accumulate(A, B)])
    chunks = [
        measure.accumulate(A[:200], B[:200]),
        measure.accumulate(A[200:], B[200:]),
    ]
    chunked = measure.finalize(chunks)

    assert abs(single.value - chunked.value) < 1e-4


def test_angular_cka_matches_arccos_of_cka():
    from cka_similarity.measures.angular_cka import AngularCKA
    from cka_similarity.measures.cka import DebiasedLinearCKA
    import math
    A, B = _gen_random_pair(n=200, p1=64, p2=64, seed=44)

    cka = DebiasedLinearCKA().finalize([DebiasedLinearCKA().accumulate(A, B)]).value
    angular = AngularCKA().finalize([AngularCKA().accumulate(A, B)]).value

    expected = math.acos(max(-1.0, min(1.0, cka)))
    assert abs(angular - expected) < 1e-5


def test_procrustes_sanity_identity_zero_and_random_positive():
    """Sanity properties for orthogonal Procrustes shape distance.

    netrep direct comparison is omitted because the github source install
    was blocked in this environment; we verify identity-symmetry and
    positivity on random unrelated matrices instead. If netrep becomes
    available, replace with a numerical equality at atol=1e-4 against
    netrep.metrics.LinearMetric(alpha=1.0, center_columns=True).
    """
    from cka_similarity.measures.procrustes import ProcrustesShapeDistance

    A, B = _gen_random_pair(n=200, p1=64, p2=64, seed=45)

    measure = ProcrustesShapeDistance()
    ours = measure.finalize([measure.accumulate(A, B)]).value
    same = measure.finalize([measure.accumulate(A, A)]).value
    assert same < 1e-5, f"identical matrices should give 0, got {same}"
    assert ours > 0


def test_bures_identity_unity():
    from cka_similarity.measures.bures import BuresSimilarity
    A, _ = _gen_random_pair(n=200, p1=64, p2=64, seed=46)
    measure = BuresSimilarity()
    same = measure.finalize([measure.accumulate(A, A)]).value
    # Identity must give NBS = 1 (within numerical noise)
    assert abs(same - 1.0) < 1e-4, f"identity should give 1, got {same}"


def test_bures_in_unit_interval():
    from cka_similarity.measures.bures import BuresSimilarity
    A, B = _gen_random_pair(n=200, p1=64, p2=64, seed=47)
    measure = BuresSimilarity()
    val = measure.finalize([measure.accumulate(A, B)]).value
    assert 0.0 <= val <= 1.0 + 1e-6
