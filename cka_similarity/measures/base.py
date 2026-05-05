"""MeasureBase ABC + per-chunk accumulator API.

A "measure" computes a similarity (or distance) between two stacked
feature matrices A (n x p_A) and B (n x p_B) where rows are aligned samples.
The naming is intentionally generic; concrete subclasses document what
they expect (penultimate activations, logits, post-softmax probabilities,
KM rows, etc.).

For chunked computation across many SLURM tasks, each measure exposes:
  - accumulate(A_chunk, B_chunk) -> dict of small tensors
  - finalize([acc_chunk_0, acc_chunk_1, ...]) -> scalar similarity

The accumulator dict is the data the chunk worker writes to disk; finalize
runs in the centralized reduce step.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

import torch


@dataclass(frozen=True)
class MeasureResult:
    value: float                     # the scalar measure
    extras: dict = field(default_factory=dict)  # auxiliary diagnostics


class MeasureBase(ABC):
    """Base class for chunk-parallel similarity measures."""
    name: str = "base"
    # Whether this measure accepts unequal feature dimensions natively.
    cross_dim_native: bool = False

    @abstractmethod
    def accumulate(self, A: torch.Tensor, B: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute the per-chunk accumulator from one chunk of paired features.

        A, B: (n_chunk, p_A), (n_chunk, p_B) -- measure-specific feature
        matrices. For penultimate-feature measures (CKA, Procrustes, etc.),
        A and B are typically penultimate activations h_W and h_~W. For
        output-distribution measures (output_jsd), A and B are logits or
        post-softmax probabilities. The naming is intentionally generic;
        concrete subclasses document what they expect.

        Returns: dict of small tensors that can be summed across chunks.
        """
        raise NotImplementedError

    @abstractmethod
    def finalize(self, accumulators: List[Dict[str, torch.Tensor]]) -> MeasureResult:
        """Aggregate per-chunk accumulators into the final scalar measure."""
        raise NotImplementedError
