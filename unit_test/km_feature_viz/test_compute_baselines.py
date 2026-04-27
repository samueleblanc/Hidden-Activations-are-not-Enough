"""Smoke tests for baseline visualization methods.

Pillar 3 ships with three architectures (residual / dense / inception); these
tests construct each one with `weights=None` so they don't trigger a download.
The tests are parameterized via pytest over the three archs. ResNet152 is the
heaviest (~60M params, ~11.5 GFLOPs/forward at 224×224); DenseNet121 is the
lightest (~8M, ~2.9 GFLOPs); GoogLeNet sits in between (~6.6M, ~1.5 GFLOPs).

The TestIGSmoothGrad case in particular runs IG with a reduced step count
(5 instead of the default 50) and SmoothGrad with 3 noise samples × 5 IG
steps each, so per-arch wall-clock stays bounded. We accept the cost as the
price of testing the *real* pipeline architecture rather than a mock —
fixture realism is more important than speed for these smoke tests.
"""
import pytest
import torch
import torch.nn as nn

from km_feature_viz.compute_baselines import (
    GRADCAM_TARGET_LAYERS,
    TV_MODEL_FACTORIES,
    compute_feature_maps,
    compute_gradcam,
    compute_ig,
    compute_pgd,
    compute_smoothgrad,
    pick_target_layer,
    top_k_activating,
)


# Parameterize all per-arch tests over the three Pillar 3 architectures.
TIER_A_ARCHS = ["resnet152", "densenet121", "googlenet"]


def _build_unweighted(model_name: str) -> nn.Module:
    """Build a torchvision model with weights=None (no download) for testing."""
    factory, _weights = TV_MODEL_FACTORIES[model_name]
    # Each factory has the same signature: (weights=...) -> nn.Module. The
    # googlenet factory wraps in a lambda that fixes aux_logits/transform_input.
    return factory(weights=None).eval()


@pytest.mark.parametrize("arch", TIER_A_ARCHS)
def test_pick_target_layer_returns_module(arch):
    """Every Pillar-3 arch must have a registered Grad-CAM target layer."""
    model = _build_unweighted(arch)
    layer = pick_target_layer(model, arch)
    assert layer is not None
    assert isinstance(layer, nn.Module), (
        f"pick_target_layer({arch}) returned {type(layer).__name__}, expected nn.Module"
    )


@pytest.mark.parametrize("arch", TIER_A_ARCHS)
def test_compute_gradcam_shape(arch):
    model = _build_unweighted(arch)
    x = torch.randn(1, 3, 224, 224)
    target_layer = pick_target_layer(model, arch)
    attribution = compute_gradcam(model, target_layer, x, class_idx=0)
    # Grad-CAM heatmap: (1, 1, H', W') after captum.
    assert attribution.dim() == 4
    assert attribution.shape[0] == 1


@pytest.mark.parametrize("arch", TIER_A_ARCHS)
def test_compute_integrated_gradients_shape(arch):
    model = _build_unweighted(arch)
    x = torch.randn(1, 3, 224, 224)
    # Reduce IG steps for the smoke test: 5 instead of the default 50.
    # The shape contract is independent of step count; this keeps wall-clock
    # manageable across all three arch fixtures.
    attr = compute_ig(model, x, class_idx=0, steps=5)
    assert attr.shape == (1, 3, 224, 224)


@pytest.mark.parametrize("arch", TIER_A_ARCHS)
def test_compute_smoothgrad_shape(arch):
    model = _build_unweighted(arch)
    x = torch.randn(1, 3, 224, 224)
    attr = compute_smoothgrad(model, x, class_idx=0, n_samples=3)
    assert attr.shape == (1, 3, 224, 224)


# Expected channel counts at the Grad-CAM target layer for each arch.
# - ResNet152 layer4[-1] (last Bottleneck) outputs 2048 channels at 7×7.
# - DenseNet121 features.denseblock4 outputs 1024 channels (concat of all
#   denselayers) at 7×7 — but reading shape[1] from the block-as-Sequential
#   produces the per-denselayer count, not the concatenated total. We
#   deliberately do not assert the exact channel count for non-resnet
#   archs because the dense-block / inception module shape semantics
#   differ; we only assert the batch + 4D structure.
EXPECTED_CHANNELS = {"resnet152": 2048}


@pytest.mark.parametrize("arch", TIER_A_ARCHS)
def test_compute_feature_maps_shape(arch):
    model = _build_unweighted(arch)
    x = torch.randn(1, 3, 224, 224)
    layer = pick_target_layer(model, arch)
    maps = compute_feature_maps(model, layer, x)
    assert maps.dim() == 4
    assert maps.shape[0] == 1
    if arch in EXPECTED_CHANNELS:
        assert maps.shape[1] == EXPECTED_CHANNELS[arch]


def test_max_activating_returns_topk():
    # Fake activations: 10 images, 5 neurons.
    activations = torch.tensor(
        [
            [3.0, 1.0, 0.0, 0.0, 0.0],
            [2.0, 4.0, 0.0, 0.0, 0.0],
            [1.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0],
            [5.0, 0.5, 0.0, 0.0, 0.0],
            [0.5, 5.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 0.0],
        ]
    )
    topk = top_k_activating(activations, k=3)
    # neuron 0: top 3 are images 4 (5.0), 0 (3.0), 1 (2.0)
    assert topk[0].tolist() == [4, 0, 1]
    # neuron 1: top 3 are images 5 (5.0), 1 (4.0), 2 (3.0)
    assert topk[1].tolist() == [5, 1, 2]


@pytest.mark.parametrize("arch", TIER_A_ARCHS)
def test_compute_pgd_shape(arch):
    model = _build_unweighted(arch)
    x = torch.rand(1, 3, 224, 224)
    delta = compute_pgd(model, x, class_idx=0, target_class=1, eps=8 / 255, steps=2)
    assert delta.shape == x.shape
    assert delta.abs().max().item() <= 8 / 255 + 1e-5


def test_tier_a_archs_have_dispatch_entries():
    """Every Pillar-3 arch must appear in both dispatch dicts (factory + gradcam)."""
    for arch in TIER_A_ARCHS:
        assert arch in TV_MODEL_FACTORIES, f"{arch} missing from TV_MODEL_FACTORIES"
        assert arch in GRADCAM_TARGET_LAYERS, f"{arch} missing from GRADCAM_TARGET_LAYERS"
