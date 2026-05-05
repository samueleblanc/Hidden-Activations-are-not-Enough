"""Angular CKA = arccos(debiased CKA), proper metric on the projective
sphere of centered Gram matrices (Williams et al. NeurIPS 2021)."""
import math
from typing import Dict, List
import torch
from .base import MeasureBase, MeasureResult
from .cka import DebiasedLinearCKA


class AngularCKA(MeasureBase):
    name = "angular_cka"
    cross_dim_native = True

    def __init__(self):
        self._inner = DebiasedLinearCKA()

    def accumulate(self, A, B):
        return self._inner.accumulate(A, B)

    def finalize(self, accumulators):
        cka_result = self._inner.finalize(accumulators)
        cka = max(-1.0, min(1.0, cka_result.value))   # numerical clamp for arccos
        if math.isnan(cka):
            return MeasureResult(value=float('nan'), extras={**cka_result.extras, "underlying_cka_nan": True})
        return MeasureResult(value=math.acos(cka), extras={**cka_result.extras, "cka": cka})
