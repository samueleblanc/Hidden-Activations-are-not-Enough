"""MeasureBase ABC + per-chunk accumulator API.

A "measure" computes a similarity (or distance) between two stacked
feature matrices X (n x p1) and Y (n x p2) where rows are aligned samples.

For chunked computation across many SLURM tasks, each measure exposes:
  - accumulate(X_chunk, Y_chunk) -> dict of small tensors
  - finalize([acc_chunk_0, acc_chunk_1, ...]) -> scalar similarity

The accumulator dict is the data the chunk worker writes to disk; finalize
runs in the centralized reduce step.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class MeasureResult:
    value: float           # the scalar measure
    extras: Dict[str, Any] # any auxiliary diagnostics (e.g., HSIC numerator)


class MeasureBase(ABC):
    """Base class for chunk-parallel similarity measures."""
    name: str = "base"
    # Whether this measure accepts unequal feature dimensions natively.
    cross_dim_native: bool = False

    @abstractmethod
    def accumulate(self, X: torch.Tensor, Y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute the per-chunk accumulator from one chunk of (X, Y) features.

        X: (n_chunk, p1) penultimate features (or KM rows) for network A
        Y: (n_chunk, p2) penultimate features for network B
        Returns: dict of small tensors that can be summed across chunks.
        """
        raise NotImplementedError

    @abstractmethod
    def finalize(self, accumulators: List[Dict[str, torch.Tensor]]) -> MeasureResult:
        """Aggregate per-chunk accumulators into the final scalar measure."""
        raise NotImplementedError
