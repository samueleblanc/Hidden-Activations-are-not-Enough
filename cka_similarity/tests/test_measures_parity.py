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
