"""Murphy 2024 shuffled-pair control.

Permute the sample alignment between X and Y, then recompute each measure.
Expected: ≈ 0 for the debiased estimator. If non-zero, the unbiased
estimator is mis-implemented or the n is too small.
"""
import torch
from typing import Dict

from cka_similarity.measures.panel import PANEL


def compute_murphy_control(X: torch.Tensor, Y: torch.Tensor, seed: int = 0) -> Dict[str, float]:
    """Compute the panel on (X, Y_permuted)."""
    torch.manual_seed(seed)
    perm = torch.randperm(Y.shape[0])
    Y_shuffled = Y[perm]

    out = {}
    for cls in PANEL:
        m = cls()
        try:
            r = m.finalize([m.accumulate(X, Y_shuffled)])
            out[m.name] = r.value
        except Exception:
            out[m.name] = float("nan")
    return out
