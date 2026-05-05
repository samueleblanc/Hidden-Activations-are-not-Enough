"""Soft-matching distance (Khosla-Williams UniReps 2024).

Wasserstein-2 distance between empirical distributions of neuron tuning
vectors (treating each network's columns as samples in n-D space).

d_SM = min_{T ∈ Π(μ_X, μ_Y)} Σ_{i,j} T_{ij} ||x_:,i − y_:,j||²

where μ_X, μ_Y are uniform on the columns of X, Y. Natively cross-dimensional: accepts p1 ≠ p2.
"""
from typing import Dict, List
import torch
from .base import MeasureBase, MeasureResult


class SoftMatching(MeasureBase):
    name = "soft_matching"
    cross_dim_native = True

    def __init__(self, sinkhorn_reg: float = 1e-2, max_iter: int = 1000):
        self.reg = sinkhorn_reg
        self.max_iter = max_iter

    def accumulate(self, A, B):
        return {"A_block": A.detach().cpu(), "B_block": B.detach().cpu()}

    def finalize(self, accumulators):
        import ot
        import numpy as np
        A = torch.cat([acc["A_block"] for acc in accumulators], dim=0)
        B = torch.cat([acc["B_block"] for acc in accumulators], dim=0)
        # Center features
        A = A - A.mean(0, keepdim=True)
        B = B - B.mean(0, keepdim=True)

        # Neuron tuning vectors are columns; transpose so samples become features
        A_neurons = A.T.numpy()   # (p1, n)
        B_neurons = B.T.numpy()
        p1, p2 = A_neurons.shape[0], B_neurons.shape[0]

        # Pairwise squared distances between neuron tuning vectors
        cost = np.linalg.norm(A_neurons[:, None, :] - B_neurons[None, :, :], axis=2) ** 2

        a_unif = np.ones(p1) / p1
        b_unif = np.ones(p2) / p2

        # Use log-domain Sinkhorn for numerical stability when cost magnitude
        # is large relative to ``reg`` (typical for high-dim feature columns).
        T = ot.sinkhorn(
            a_unif, b_unif, cost, reg=self.reg,
            numItermax=self.max_iter, method="sinkhorn_log",
        )
        d_sq = float((T * cost).sum())
        return MeasureResult(value=d_sq ** 0.5, extras={"p1": p1, "p2": p2, "n": int(A.shape[0])})
