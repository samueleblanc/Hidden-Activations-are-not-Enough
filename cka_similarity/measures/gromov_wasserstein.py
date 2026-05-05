"""Entropic Gromov-Wasserstein OBJECTIVE (Mémoli 2011, Peyré 2016).

Returns the GW *loss* (objective value of the optimization problem), not
the GW *distance*. Per Mémoli 2011, the GW distance is (1/2) * sqrt(loss);
since downstream tables compare measures by relative ranking across
(arch, arch') pairs, we report the loss directly. If you need the
literature GW distance, transform: dist = 0.5 * sqrt(value).

Compares two metric-measure spaces. Natively handles unequal feature dims.
We compute on a 5K subsample for tractability — finalize subsamples if N > 5000.
"""
from typing import Dict, List
import torch
import numpy as np
from .base import MeasureBase, MeasureResult


class GromovWasserstein(MeasureBase):
    """Entropic Gromov-Wasserstein objective on row-as-sample feature matrices.

    The reported value is the GW *loss* (objective of the entropic-regularized
    optimization), NOT the GW distance. Per Mémoli 2011, the GW distance is
    ``(1/2) * sqrt(loss)``; for ranking-style aggregation across (arch, arch')
    pairs, loss vs. distance only differs by a monotone transform, so we report
    the loss directly. ``MeasureResult.extras["unit"] == "loss"`` flags this for
    any downstream consumer that needs absolute scale.
    """
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
            gen = torch.Generator().manual_seed(0)
            idx = torch.randperm(n, generator=gen)[:self.n_subsample]
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
        return MeasureResult(
            value=float(gw_val),
            extras={
                "n_used": int(n),
                # Flag the semantics for downstream consumers: this is the GW
                # *loss* (objective value), not the GW distance. Mémoli 2011's
                # distance is (1/2) * sqrt(loss).
                "unit": "loss",
            },
        )
