"""sqrt-Jensen-Shannon-Divergence between output distributions.

Per-sample sqrt(JSD(softmax(f_A(x)), softmax(f_B(x)))), averaged across N samples.
sqrt-JSD is a metric (Endres-Schindelin 2003) on probability distributions.

Inputs A, B are LOGITS — softmax is applied internally.
"""
from typing import Dict, List
import math
import torch
import torch.nn.functional as F
from .base import MeasureBase, MeasureResult


def _per_sample_sqrt_jsd(p_logits, q_logits):
    """Compute sqrt(JSD) per sample. Inputs: (n, k) logits."""
    p = F.softmax(p_logits, dim=1)
    q = F.softmax(q_logits, dim=1)
    m = 0.5 * (p + q)
    eps = 1e-12
    kl_pm = (p * (torch.log(p + eps) - torch.log(m + eps))).sum(dim=1)
    kl_qm = (q * (torch.log(q + eps) - torch.log(m + eps))).sum(dim=1)
    jsd = 0.5 * (kl_pm + kl_qm)
    jsd = torch.clamp(jsd, min=0.0)
    return jsd.sqrt()


class OutputJSD(MeasureBase):
    name = "output_jsd"
    cross_dim_native = True  # JSD on output distributions handles different inner dims via softmax

    def accumulate(self, A, B):
        # A, B are logits — compute per-sample sqrt-JSD now and just sum + count
        per = _per_sample_sqrt_jsd(A, B)
        return {"sum": per.sum().item(), "count": A.shape[0]}

    def finalize(self, accumulators):
        total = sum(acc["sum"] for acc in accumulators)
        n = sum(acc["count"] for acc in accumulators)
        return MeasureResult(value=total / max(1, n), extras={"n": n})
