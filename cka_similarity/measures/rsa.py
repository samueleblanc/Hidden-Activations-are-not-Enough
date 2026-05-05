"""Representational Similarity Analysis with Spearman rank correlation
on representational dissimilarity matrices (Kriegeskorte 2008).

RDM_X[i,j] = 1 - cos_sim(X[i], X[j])
RSA(X, Y) = spearman(vec_uppertri(RDM_X), vec_uppertri(RDM_Y))

The RDM is N×N regardless of feature dim, so RSA is natively cross-dim.
"""
from typing import Dict, List
import torch
from .base import MeasureBase, MeasureResult


def _build_rdm(X: torch.Tensor) -> torch.Tensor:
    """Build representational dissimilarity matrix: 1 - cosine similarity."""
    X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-12)
    cos = X_norm @ X_norm.T
    return 1.0 - cos


class RSASpearman(MeasureBase):
    name = "rsa"
    cross_dim_native = True

    def accumulate(self, A, B):
        return {"A_block": A.detach().cpu(), "B_block": B.detach().cpu()}

    def finalize(self, accumulators):
        from scipy.stats import spearmanr
        A = torch.cat([acc["A_block"] for acc in accumulators], dim=0)
        B = torch.cat([acc["B_block"] for acc in accumulators], dim=0)
        rdm_A = _build_rdm(A)
        rdm_B = _build_rdm(B)
        n = A.shape[0]
        # Upper-triangle indices (excluding diagonal)
        iu = torch.triu_indices(n, n, offset=1)
        v_A = rdm_A[iu[0], iu[1]].numpy()
        v_B = rdm_B[iu[0], iu[1]].numpy()
        rho, _ = spearmanr(v_A, v_B)
        return MeasureResult(value=float(rho), extras={"n": int(n)})
