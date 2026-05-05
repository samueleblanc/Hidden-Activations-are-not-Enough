"""Orthogonal Procrustes shape distance (Williams-Kunz-Kornblith-Linderman 2021).

d_OP² = ||X̃||²_F + ||Ỹ||²_F − 2||X̃ᵀỸ||_*

where X̃, Ỹ are column-centered features and ||·||_* is the nuclear norm.
"""
from typing import Dict, List
import torch
from .base import MeasureBase, MeasureResult


class ProcrustesShapeDistance(MeasureBase):
    name = "procrustes"
    cross_dim_native = False  # needs same feature dim; cross-arch use requires PCA padding

    def accumulate(self, A: torch.Tensor, B: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Per-chunk cross-products (XtX, YtY, XtY) are intentionally NOT stored:
        # they are anchored to per-chunk means, so they cannot be combined
        # into the global re-centered cross-products. finalize re-derives
        # them from the concatenated, re-centered blocks.
        return {
            "A_sum":   A.sum(dim=0),                  # for centering
            "B_sum":   B.sum(dim=0),
            "n":       torch.tensor(A.shape[0]),
            "A_block": A.detach().cpu(),  # for re-centering after total mean known
            "B_block": B.detach().cpu(),
        }

    def finalize(self, accumulators: List[Dict[str, torch.Tensor]]) -> MeasureResult:
        # Total sample count
        n_total = sum(int(acc["n"]) for acc in accumulators)
        # True column means (across all chunks)
        X_mean = sum(acc["A_sum"] for acc in accumulators) / n_total
        Y_mean = sum(acc["B_sum"] for acc in accumulators) / n_total

        # Recenter and recompute the chunk blocks; promote to float64 for stable
        # nuclear-norm computation on near-identity inputs (otherwise
        # ||X̃||² + ||X̃||² - 2 ||X̃ᵀX̃||_* picks up O(1e-3) error from float32
        # singular-value summation even on n=200 random matrices).
        X_full = (torch.cat([acc["A_block"] for acc in accumulators], dim=0) - X_mean).double()
        Y_full = (torch.cat([acc["B_block"] for acc in accumulators], dim=0) - Y_mean).double()

        XtY = X_full.T @ Y_full

        norm_X_sq = float((X_full ** 2).sum())
        norm_Y_sq = float((Y_full ** 2).sum())
        # Nuclear norm of XtY = sum of singular values
        S = torch.linalg.svdvals(XtY)
        nuclear = float(S.sum())

        d_sq = norm_X_sq + norm_Y_sq - 2 * nuclear
        # Numerical clamp (small negative values from fp accumulation)
        d_sq = max(d_sq, 0.0)

        return MeasureResult(value=float(d_sq ** 0.5), extras={
            "norm_X_sq": norm_X_sq, "norm_Y_sq": norm_Y_sq, "nuclear": nuclear, "n": n_total,
        })
