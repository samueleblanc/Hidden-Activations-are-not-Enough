"""Normalized Bures Similarity (Harvey-Larsen-Williams UniReps 2023).

NBS(K_X, K_Y) = tr((K_X^{1/2} K_Y K_X^{1/2})^{1/2}) / sqrt(||K_X||_* * ||K_Y||_*)

= cosine of the Riemannian shape distance on the manifold of centered
Gram matrices. Lives in [0, 1].
"""
from typing import Dict, List
import torch
from .base import MeasureBase, MeasureResult


def _matrix_sqrt_psd(M: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """PSD matrix square root via eigendecomp; clamps negative eigenvalues to 0.

    Input must be PSD up to numerical noise; small negative eigenvalues
    (>= -eps) are silently clamped. A negative eigenvalue more negative
    than -eps indicates the input was not PSD (a bug in the caller); we
    warn and clamp anyway, but the output should be treated with caution.
    """
    M = (M + M.T) / 2  # symmetrize
    eigvals, eigvecs = torch.linalg.eigh(M.double())
    if eigvals.min() < -eps:
        import warnings
        warnings.warn(
            f"_matrix_sqrt_psd: input has eigenvalue {float(eigvals.min()):.2e} < -eps; "
            f"input may not be PSD. Clamping to {eps}.",
            RuntimeWarning,
        )
    eigvals = torch.clamp(eigvals, min=eps)
    return (eigvecs * eigvals.sqrt()) @ eigvecs.T


class BuresSimilarity(MeasureBase):
    name = "bures"
    cross_dim_native = True   # operates on N×N centered Grams

    def accumulate(self, A, B):
        return {"A_block": A.detach().cpu(), "B_block": B.detach().cpu(),
                "A_sum": A.sum(0), "B_sum": B.sum(0), "n": torch.tensor(A.shape[0])}

    def finalize(self, accumulators):
        n_total = sum(int(acc["n"]) for acc in accumulators)
        X_mean = sum(acc["A_sum"] for acc in accumulators) / n_total
        Y_mean = sum(acc["B_sum"] for acc in accumulators) / n_total

        X = torch.cat([acc["A_block"] for acc in accumulators], dim=0) - X_mean
        Y = torch.cat([acc["B_block"] for acc in accumulators], dim=0) - Y_mean

        # Centered Grams (n × n)
        K_X = X @ X.T
        K_Y = Y @ Y.T

        K_X_sqrt = _matrix_sqrt_psd(K_X)
        inner = K_X_sqrt @ K_Y.double() @ K_X_sqrt
        # tr((·)^{1/2}) = sum of square roots of eigenvalues of inner
        eigvals = torch.linalg.eigvalsh((inner + inner.T) / 2)
        eigvals = torch.clamp(eigvals, min=0.0)
        numerator = float(eigvals.sqrt().sum())

        norm_X = float(K_X.diagonal().sum())   # ||K_X||_* for PSD = trace
        norm_Y = float(K_Y.diagonal().sum())

        if norm_X <= 0 or norm_Y <= 0:
            return MeasureResult(value=float('nan'), extras={"norm_X": norm_X, "norm_Y": norm_Y})

        nbs = numerator / (norm_X * norm_Y) ** 0.5
        return MeasureResult(value=float(nbs), extras={"trace_sqrt": numerator, "norm_X": norm_X, "norm_Y": norm_Y})
