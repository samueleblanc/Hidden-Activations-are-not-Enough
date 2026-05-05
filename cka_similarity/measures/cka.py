"""Debiased linear CKA via the unbiased HSIC₁ U-statistic
(Song-Smola-Gretton 2012; Nguyen-Raghu-Kornblith 2021).

Required for n < p regime per Murphy-Adolfi-Bowers ICLR 2024.

The chunked formulation accumulates per-chunk Gram-block sums and
combines them in finalize, since HSIC₁ depends on the full N×N Gram
matrices. We materialize the Gram matrix N×N in finalize (2.5 GB at
N=25K, fine on a node with 256 GB RAM) rather than streaming it.
"""
from typing import Dict, List
import torch
from .base import MeasureBase, MeasureResult


class DebiasedLinearCKA(MeasureBase):
    name = "debiased_cka"
    cross_dim_native = True  # CKA via Gram-form is dim-agnostic for the *accumulator*

    def accumulate(self, A: torch.Tensor, B: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Per-chunk accumulator: store the chunk's A and B row blocks.

        For a centralized HSIC₁ we need the full N×N Gram matrix, so per-chunk
        we just stash the row blocks. Gram construction happens in finalize.
        """
        return {"A_block": A.detach().cpu(), "B_block": B.detach().cpu()}

    def finalize(self, accumulators: List[Dict[str, torch.Tensor]]) -> MeasureResult:
        X = torch.cat([acc["A_block"] for acc in accumulators], dim=0)
        Y = torch.cat([acc["B_block"] for acc in accumulators], dim=0)
        n = X.shape[0]
        if n < 4:
            raise ValueError(f"HSIC₁ requires n >= 4, got n={n}")

        K = X @ X.T   # (n, n)
        L = Y @ Y.T
        K_tilde = K - torch.diag(K.diag())
        L_tilde = L - torch.diag(L.diag())
        ones = torch.ones(n, dtype=X.dtype)

        def hsic1(K, L):
            term1 = (K * L).sum()
            term2 = (ones @ K @ ones) * (ones @ L @ ones) / ((n - 1) * (n - 2))
            term3 = 2.0 * (ones @ K @ L @ ones) / (n - 2)
            return (term1 + term2 - term3) / (n * (n - 3))

        num = hsic1(K_tilde, L_tilde)
        d_x = hsic1(K_tilde, K_tilde)
        d_y = hsic1(L_tilde, L_tilde)

        if d_x <= 0 or d_y <= 0:
            return MeasureResult(value=float('nan'), extras={
                "hsic_xy": float(num), "hsic_xx": float(d_x), "hsic_yy": float(d_y),
                "warning": "non-positive HSIC self-term; debiased estimator can be negative",
            })

        cka = num / (d_x * d_y).sqrt()
        return MeasureResult(value=float(cka), extras={
            "hsic_xy": float(num), "hsic_xx": float(d_x), "hsic_yy": float(d_y),
            "n": int(n),
        })
