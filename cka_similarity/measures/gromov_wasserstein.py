"""Entropic Gromov-Wasserstein distance (Mémoli 2011, Peyré 2016).

Compares two metric-measure spaces. Natively handles unequal feature dims.
We compute on a 5K subsample for tractability — finalize subsamples if N > 5000.
"""
from typing import Dict, List
import torch
import numpy as np
from .base import MeasureBase, MeasureResult


class GromovWasserstein(MeasureBase):
    name = "gw"
    cross_dim_native = True

    def __init__(self, n_subsample: int = 5000, sinkhorn_reg: float = 1e-2, max_iter: int = 100):
        self.n_subsample = n_subsample
        self.reg = sinkhorn_reg
        self.max_iter = max_iter

    def accumulate(self, A, B):
        return {"A_block": A.detach().cpu(), "B_block": B.detach().cpu()}

    def finalize(self, accumulators):
        import ot
        A = torch.cat([acc["A_block"] for acc in accumulators], dim=0)
        B = torch.cat([acc["B_block"] for acc in accumulators], dim=0)
        n = A.shape[0]
        if n > self.n_subsample:
            torch.manual_seed(0)
            idx = torch.randperm(n)[:self.n_subsample]
            A = A[idx]
            B = B[idx]
            n = self.n_subsample
        # Pairwise distance matrices
        D_A = torch.cdist(A, A).numpy()
        D_B = torch.cdist(B, B).numpy()
        a_unif = np.ones(n) / n
        b_unif = np.ones(n) / n
        gw_val = ot.gromov.entropic_gromov_wasserstein2(
            D_A, D_B, a_unif, b_unif,
            loss_fun='square_loss', epsilon=self.reg, max_iter=self.max_iter,
        )
        return MeasureResult(value=float(gw_val), extras={"n_used": int(n)})
