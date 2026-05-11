"""Unit tests for utils/scaling.py — canonical RMS-per-coordinate distance.

The canonical fair-comparison metric across logit / penultimate / KM spaces is
RMS-per-coordinate. This module pins the rescaling identities so any future
change to the convention is caught by CI rather than producing a silently
inconsistent set of paper numbers.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from utils.scaling import (
    IMAGENET_INPUT_NUMEL,
    IMAGENET_NUM_CLASSES,
    PENULTIMATE_DIM,
    km_numel,
    logit_numel,
    penultimate_dim,
    rescale_amp_M_to_rms,
    rescale_amp_h_to_rms,
    rescale_gamma_to_rms,
    rms_distance,
    rms_distances,
)


# ----------------------------------------------------------------------------
# Numerical constants
# ----------------------------------------------------------------------------

def test_imagenet_constants_match_paper():
    """Dataset shape constants must match the paper's Pillar 3 framing."""
    assert IMAGENET_NUM_CLASSES == 1000
    assert IMAGENET_INPUT_NUMEL == 3 * 224 * 224
    # KM shape: C × (d+1) = 1000 × 150529 — pinned in the paper.
    assert km_numel(IMAGENET_NUM_CLASSES, IMAGENET_INPUT_NUMEL) == 1000 * 150529


def test_penultimate_dim_phase1_archs():
    """Phase-1 architectures must have correctly-pinned penultimate dims.

    The 3 archs the paper compares cross-architecturally have distinct
    penultimate dims; mis-registering one would silently bias every
    d_h-based comparison in Pillars 2 and 3.
    """
    assert penultimate_dim("resnet152") == 2048
    assert penultimate_dim("densenet121") == 1024
    assert penultimate_dim("googlenet") == 1024


def test_penultimate_dim_unknown_arch_raises():
    """Unknown arch must raise KeyError so callers register dims explicitly
    rather than silently picking a wrong scale factor."""
    with pytest.raises(KeyError):
        penultimate_dim("totally_unknown_net")


# ----------------------------------------------------------------------------
# RMS distance
# ----------------------------------------------------------------------------

def test_rms_distance_definition():
    """rms_distance(||a-b||, numel(a)) == ||a-b|| / sqrt(numel(a))."""
    # Numel 100 vector
    raw = 7.0
    expected = 7.0 / math.sqrt(100)
    assert math.isclose(rms_distance(raw, 100), expected)


def test_rms_distance_vector_norm_consistency():
    """Computing rms_distance from a synthetic L2 norm reproduces the
    coordinate-wise RMS of the same vector."""
    rng = np.random.default_rng(0)
    n = 1000
    diff = rng.standard_normal(n)
    raw_l2 = float(np.linalg.norm(diff))
    rms_via_helper = rms_distance(raw_l2, n)
    rms_direct = float(np.sqrt(np.mean(diff ** 2)))
    assert math.isclose(rms_via_helper, rms_direct, rel_tol=1e-12)


def test_rms_distance_invalid_numel():
    with pytest.raises(ValueError):
        rms_distance(1.0, 0)
    with pytest.raises(ValueError):
        rms_distance(1.0, -5)


def test_rms_distances_vectorized():
    """Batched rms_distances matches a Python loop."""
    raw_norms = [1.0, 2.0, 3.0, 4.0]
    n = 16
    out = rms_distances(raw_norms, n)
    expected = np.array([v / math.sqrt(n) for v in raw_norms])
    np.testing.assert_allclose(out, expected, rtol=1e-12)


# ----------------------------------------------------------------------------
# Cross-space rescaling identities
# ----------------------------------------------------------------------------

def test_amp_M_rescale_identity():
    """amp_M_RMS = amp_M_raw / sqrt(d+1).

    Derivation:
        amp_M_raw  = ||M-M'||_F / ||f-f'||_2
        amp_M_RMS  = (||M-M'||_F / sqrt(C(d+1))) / (||f-f'||_2 / sqrt(C))
                   = amp_M_raw * sqrt(C) / sqrt(C(d+1))
                   = amp_M_raw / sqrt(d+1)
    """
    raw = 3.166
    d_plus_1 = IMAGENET_INPUT_NUMEL + 1
    expected = raw / math.sqrt(d_plus_1)
    assert math.isclose(rescale_amp_M_to_rms(raw, IMAGENET_INPUT_NUMEL),
                        expected, rel_tol=1e-12)


def test_amp_h_rescale_identity():
    """amp_h_RMS = amp_h_raw * sqrt(C / D_penult)."""
    raw = 0.751
    D = 2048  # ResNet152
    C = 1000
    expected = raw * math.sqrt(C / D)
    assert math.isclose(rescale_amp_h_to_rms(raw, C, D), expected, rel_tol=1e-12)


def test_gamma_rescale_identity():
    """gamma is just a particular order-statistic of amp_M ratios, so it
    rescales by the same constant: gamma_RMS = gamma_raw / sqrt(d+1)."""
    raw = 1.229
    expected = raw / math.sqrt(IMAGENET_INPUT_NUMEL + 1)
    assert math.isclose(rescale_gamma_to_rms(raw, IMAGENET_INPUT_NUMEL),
                        expected, rel_tol=1e-12)


def test_rms_pillar2_imagenet_factor():
    """The headline 'γ_RMS ≈ γ_raw / 388' relationship that drives every
    Pillar-2 number in the paper. Pinned for paper-narrative integrity."""
    factor = math.sqrt(IMAGENET_INPUT_NUMEL + 1)
    assert 387.5 < factor < 388.5  # = sqrt(150529) ≈ 388.0


# ----------------------------------------------------------------------------
# Order-statistic equivariance under uniform multiplicative rescaling
# ----------------------------------------------------------------------------

def test_min_is_scale_equivariant():
    """gamma = min(ratio). After uniform rescaling by c, min(c*X) = c*min(X)."""
    rng = np.random.default_rng(1)
    X = rng.uniform(0.1, 10.0, size=500)
    c = 1.0 / math.sqrt(IMAGENET_INPUT_NUMEL + 1)
    assert math.isclose(np.min(c * X), c * np.min(X), rel_tol=1e-12)


def test_percentile_is_scale_equivariant():
    """All percentile-based bootstrap CI bounds rescale linearly under c."""
    rng = np.random.default_rng(2)
    X = rng.uniform(0.1, 10.0, size=500)
    c = 0.7
    for q in (2.5, 25, 50, 75, 97.5):
        assert math.isclose(np.percentile(c * X, q),
                            c * np.percentile(X, q),
                            rel_tol=1e-12)


# ----------------------------------------------------------------------------
# End-to-end: an example pair through the whole pipeline
# ----------------------------------------------------------------------------

def test_end_to_end_pillar2_resnet152():
    """A ResNet152 PGD example pinned at the real saved value, fully
    rescaled — guards against silent regression in the rescale identities."""
    raw_gamma = 1.2293271957025365
    raw_amp_M = 3.166171765664605
    raw_amp_h = 0.7512820468508022
    C = IMAGENET_NUM_CLASSES
    d_plus_1 = IMAGENET_INPUT_NUMEL + 1
    D = penultimate_dim("resnet152")

    gamma_rms = rescale_gamma_to_rms(raw_gamma, IMAGENET_INPUT_NUMEL)
    amp_M_rms = rescale_amp_M_to_rms(raw_amp_M, IMAGENET_INPUT_NUMEL)
    amp_h_rms = rescale_amp_h_to_rms(raw_amp_h, C, D)

    # gamma collapses by factor ~388 from dim alone.
    assert 0.0028 < gamma_rms < 0.0036
    # KM amplification under RMS should be much less than penultimate
    # amplification — opposite of the raw narrative.
    assert amp_M_rms < amp_h_rms
    # Order-of-magnitude pin
    assert 0.005 < amp_M_rms < 0.012
    assert 0.45 < amp_h_rms < 0.60


def test_logit_numel_trivial():
    """logit_numel returns num_classes; pinned for callers that don't need
    a separate scaling helper but want a consistent name."""
    assert logit_numel(1000) == 1000
    assert logit_numel(10) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
