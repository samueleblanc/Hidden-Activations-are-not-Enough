"""Tests for teleportation_experiment.py."""

import pytest
import torch
import torch.nn as nn
import numpy as np


class TestNeuralTeleportationSmoke:
    """Verify neuralteleportation works with COB-compatible models.

    The neuralteleportation library requires models built from its own COB
    (Change of Basis) layer types (Conv2dCOB, LinearCOB, ReLUCOB, etc.)
    rather than standard torchvision models, which use functional operations
    in forward() that break the JIT graph analysis.

    The correct usage pattern is:
    1. Create the COB model
    2. Wrap it with NeuralTeleportationModel (this modifies internal state)
    3. Set eval mode AFTER wrapping
    4. Capture baseline output
    5. Teleport
    6. Compare outputs
    """

    def test_resnet18_teleport_preserves_output(self):
        """Teleported resnet18COB should produce identical outputs."""
        from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
        from neuralteleportation.models.model_zoo.resnetcob import resnet18COB

        model = resnet18COB(pretrained=False, num_classes=10)
        tp = NeuralTeleportationModel(model, input_shape=(1, 3, 224, 224))
        tp.eval()

        x = torch.randn(4, 3, 224, 224)

        with torch.no_grad():
            out_before = tp(x).clone()

        tp.random_teleport(cob_range=1)

        with torch.no_grad():
            out_after = tp(x)

        assert out_before.shape == out_after.shape
        assert torch.allclose(out_before, out_after, atol=1e-4), (
            f"Max diff: {(out_before - out_after).abs().max().item():.2e}"
        )

    def test_vgg11_bn_teleport_preserves_output(self):
        """Teleported vgg11_bnCOB should produce identical outputs."""
        from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
        from neuralteleportation.models.model_zoo.vggcob import vgg11_bnCOB

        model = vgg11_bnCOB(pretrained=False, num_classes=10)
        tp = NeuralTeleportationModel(model, input_shape=(1, 3, 224, 224))
        tp.eval()

        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            out_before = tp(x).clone()

        tp.random_teleport(cob_range=1)

        with torch.no_grad():
            out_after = tp(x)

        assert torch.allclose(out_before, out_after, atol=1e-4), (
            f"Max diff: {(out_before - out_after).abs().max().item():.2e}"
        )
