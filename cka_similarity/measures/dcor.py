"""Distance correlation (Székely-Rizzo-Bakirov 2007), wraps `dcor` library.

Uses the U-statistic (bias-corrected) form of squared distance correlation
and returns sqrt(max(0, ·)). The biased ``distance_correlation`` has a
strong positive bias in high dim (e.g. ~0.75 for two independent N(0,I_64)
samples at n=200), so the bias-corrected estimator is the right default
for cross-arch / model-comparison work.
"""
from typing import Dict, List
import torch
from .base import MeasureBase, MeasureResult


class DistanceCorrelation(MeasureBase):
    name = "dcor"
    cross_dim_native = True

    def accumulate(self, A, B):
        return {"A_block": A.detach().cpu(), "B_block": B.detach().cpu()}

    def finalize(self, accumulators):
        import dcor
        A = torch.cat([acc["A_block"] for acc in accumulators], dim=0).numpy()
        B = torch.cat([acc["B_block"] for acc in accumulators], dim=0).numpy()
        u_sqr = float(dcor.u_distance_correlation_sqr(A, B))
        val = float(max(0.0, u_sqr) ** 0.5)
        return MeasureResult(
            value=val,
            extras={"n": int(A.shape[0]), "u_dcor_sqr": u_sqr},
        )
