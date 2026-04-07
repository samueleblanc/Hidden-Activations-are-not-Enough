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


class TestCreateModel:
    """Test model construction with adapted final FC."""

    @pytest.mark.parametrize("arch_name,num_classes,expected_dim", [
        ('resnet18', 10, 512),
        ('resnet50', 100, 2048),
        ('vgg11_bn', 10, 4096),
        ('vgg19_bn', 200, 4096),
    ])
    def test_create_model_output_shape(self, arch_name, num_classes, expected_dim):
        from teleportation_experiment import create_model, ARCHITECTURES
        model = create_model(arch_name, num_classes)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, num_classes)

    def test_create_model_invalid_arch(self):
        from teleportation_experiment import create_model
        with pytest.raises(KeyError):
            create_model('nonexistent', 10)


class TestPenultimateExtractor:
    """Test penultimate-layer feature extraction via hooks."""

    def test_resnet18_penultimate_shape(self):
        from teleportation_experiment import create_model, PenultimateExtractor
        model = create_model('resnet18', 10)
        model.eval()
        extractor = PenultimateExtractor(model, 'resnet18')
        data = torch.randn(8, 3, 224, 224)
        feats = extractor.extract(model, data, batch_size=4)
        extractor.remove()
        assert feats.shape == (8, 512)

    def test_resnet50_penultimate_shape(self):
        from teleportation_experiment import create_model, PenultimateExtractor
        model = create_model('resnet50', 100)
        model.eval()
        extractor = PenultimateExtractor(model, 'resnet50')
        data = torch.randn(4, 3, 224, 224)
        feats = extractor.extract(model, data, batch_size=2)
        extractor.remove()
        assert feats.shape == (4, 2048)

    def test_vgg11_bn_penultimate_shape(self):
        from teleportation_experiment import create_model, PenultimateExtractor
        model = create_model('vgg11_bn', 10)
        model.eval()
        extractor = PenultimateExtractor(model, 'vgg11_bn')
        data = torch.randn(4, 3, 224, 224)
        feats = extractor.extract(model, data, batch_size=2)
        extractor.remove()
        assert feats.shape == (4, 4096)

    def test_extractor_cleanup(self):
        """Hook is removed after calling remove()."""
        from teleportation_experiment import create_model, PenultimateExtractor
        model = create_model('resnet18', 10)
        extractor = PenultimateExtractor(model, 'resnet18')
        assert len(model.avgpool._forward_hooks) == 1
        extractor.remove()
        assert len(model.avgpool._forward_hooks) == 0
