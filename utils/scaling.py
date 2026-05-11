"""Canonical distance scaling for cross-space, cross-architecture comparisons.

The paper compares distances across three spaces with very different ambient
dimension:

- logits:      f(x) ∈ R^C                            (C = #classes)
- penultimate: h(x) ∈ R^D                            (D depends on arch)
- KM:          M(x) ∈ R^{C × (d+1)}                  (d = prod(input_shape))

Raw L2 / Frobenius distances are NOT comparable across these spaces because
they scale with sqrt(numel). The canonical fair-comparison metric used in
this codebase is **RMS-per-coordinate**:

    rms_dist(a, b) = ||a - b||_2 / sqrt(numel(a))

Under this convention:
- gamma_RMS = gamma_raw / sqrt(d+1)             (KM-vs-logit amplification)
- amp_h_RMS = amp_h_raw * sqrt(C / D_penult)    (penult-vs-logit amplification)
- amp_M_RMS = amp_M_raw / sqrt(d+1)             (KM-vs-logit amplification)
"""
from __future__ import annotations

from math import sqrt
from typing import Iterable

import numpy as np


# ----------------------------------------------------------------------------
# Ambient dimensions
# ----------------------------------------------------------------------------

PENULTIMATE_DIM = {
    "resnet152":   2048,
    "densenet121": 1024,
    "googlenet":   1024,
    "alexnet":     4096,
    "resnet18":     512,
    "resnet50":    2048,
    "vgg11":       4096,
    "vgg16":       4096,
}


IMAGENET_INPUT_NUMEL = 3 * 224 * 224   # = 150528
IMAGENET_NUM_CLASSES = 1000
CIFAR_INPUT_NUMEL = 3 * 32 * 32        # = 3072


def penultimate_dim(arch: str) -> int:
    """Penultimate-feature dimension for a known architecture.

    Raises KeyError for unknown architectures so callers must register them
    explicitly rather than silently using a wrong scale factor.
    """
    return PENULTIMATE_DIM[arch.lower()]


def km_numel(num_classes: int, input_numel: int) -> int:
    """Number of entries in a knowledge matrix: C * (d + 1)."""
    return num_classes * (input_numel + 1)


def logit_numel(num_classes: int) -> int:
    return num_classes


# ----------------------------------------------------------------------------
# RMS distance
# ----------------------------------------------------------------------------

def rms_distance(raw_norm: float, ambient_numel: int) -> float:
    """Convert a raw L2/Frobenius norm to RMS-per-coordinate.

    Args:
        raw_norm: ||a - b||_2 (vectors) or ||a - b||_F (matrices).
        ambient_numel: numel(a) — total number of entries in the underlying
            tensor (NOT a single dimension; for an M×N matrix this is M*N).
    """
    if ambient_numel <= 0:
        raise ValueError(f"ambient_numel must be positive, got {ambient_numel}")
    return float(raw_norm) / sqrt(ambient_numel)


def rms_distances(raw_norms: Iterable[float], ambient_numel: int) -> np.ndarray:
    """Vectorized rms_distance over an iterable of raw norms."""
    arr = np.asarray(list(raw_norms), dtype=float)
    return arr / sqrt(ambient_numel)


# ----------------------------------------------------------------------------
# Pillar-2 / Pillar-3 amplification ratios under RMS
# ----------------------------------------------------------------------------

def rescale_amp_M_to_rms(amp_raw: float, input_numel: int) -> float:
    """Rescale d_M/d_f from raw to RMS units.

    Under RMS-per-coordinate normalization:
        amp_M_RMS = (||M-M'||_F / sqrt(C(d+1))) / (||f-f'||_2 / sqrt(C))
                  = amp_M_raw * sqrt(C) / sqrt(C(d+1))
                  = amp_M_raw / sqrt(d+1)
    """
    return float(amp_raw) / sqrt(input_numel + 1)


def rescale_amp_h_to_rms(amp_raw: float, num_classes: int, penult_dim: int) -> float:
    """Rescale d_h/d_f from raw to RMS units.

    Under RMS-per-coordinate normalization:
        amp_h_RMS = (||h-h'||_2 / sqrt(D)) / (||f-f'||_2 / sqrt(C))
                  = amp_h_raw * sqrt(C) / sqrt(D)
    """
    return float(amp_raw) * sqrt(num_classes) / sqrt(penult_dim)


def rescale_gamma_to_rms(gamma_raw: float, input_numel: int) -> float:
    """gamma_RMS = gamma_raw / sqrt(d+1). gamma is min(d_M/d_f) which is just
    a particular order-statistic of the per-pair amp_M ratio, so it rescales
    by the same constant.
    """
    return float(gamma_raw) / sqrt(input_numel + 1)
