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


class TestTeleportModel:
    def test_teleported_model_preserves_output(self):
        from teleportation_experiment import create_model, teleport_model
        model = create_model('resnet18', 10)
        model.eval()
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out_orig = model(x)
        model_tp = teleport_model(model, input_shape=(1, 3, 224, 224), seed=42)
        model_tp.eval()
        with torch.no_grad():
            out_tp = model_tp(x)
        assert torch.allclose(out_orig, out_tp, atol=1e-3), (
            f"Max diff: {(out_orig - out_tp).abs().max().item():.2e}"
        )

    def test_teleported_model_has_different_weights(self):
        from teleportation_experiment import create_model, teleport_model
        model = create_model('resnet18', 10)
        model.eval()
        orig_params = {k: v.clone() for k, v in model.named_parameters()}
        model_tp = teleport_model(model, input_shape=(1, 3, 224, 224), seed=42)
        changed = False
        for k, v in model_tp.named_parameters():
            if k in orig_params and not torch.equal(v.data, orig_params[k]):
                changed = True
                break
        assert changed, "Teleported model should have different weights"

    def test_different_seeds_produce_different_teleportations(self):
        from teleportation_experiment import create_model, teleport_model
        model = create_model('resnet18', 10)
        model.eval()
        tp1 = teleport_model(model, (1, 3, 224, 224), seed=42)
        tp2 = teleport_model(model, (1, 3, 224, 224), seed=99)
        # At least one parameter should differ between the two teleportations
        differ = False
        for (k1, v1), (k2, v2) in zip(tp1.named_parameters(), tp2.named_parameters()):
            if not torch.equal(v1.data, v2.data):
                differ = True
                break
        assert differ


class TestComputeNormalizedDistances:
    def test_identical_features_zero_distance(self):
        from teleportation_experiment import compute_normalized_distances
        feats = torch.randn(10, 512)
        dists = compute_normalized_distances(feats, feats)
        np.testing.assert_allclose(dists, 0.0, atol=1e-7)

    def test_known_distance(self):
        from teleportation_experiment import compute_normalized_distances
        a = torch.zeros(1, 4)
        b = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
        # ||a-b|| = 2.0, sqrt(dim) = 2.0, result = 1.0
        dists = compute_normalized_distances(a, b)
        np.testing.assert_allclose(dists, [1.0], atol=1e-7)

    def test_output_shape(self):
        from teleportation_experiment import compute_normalized_distances
        a = torch.randn(50, 2048)
        b = torch.randn(50, 2048)
        dists = compute_normalized_distances(a, b)
        assert dists.shape == (50,)


class TestDataLoading:
    def test_generate_random_inputs_shape(self):
        from teleportation_experiment import generate_random_inputs
        data = generate_random_inputs(50, shape=(3, 224, 224), seed=42)
        assert data.shape == (50, 3, 224, 224)

    def test_generate_random_inputs_is_normal(self):
        from teleportation_experiment import generate_random_inputs
        data = generate_random_inputs(10000, shape=(3, 32, 32), seed=42)
        flat = data.flatten()
        assert abs(flat.mean().item()) < 0.05
        assert abs(flat.std().item() - 1.0) < 0.05

    def test_generate_random_inputs_reproducible(self):
        from teleportation_experiment import generate_random_inputs
        d1 = generate_random_inputs(10, shape=(3, 32, 32), seed=42)
        d2 = generate_random_inputs(10, shape=(3, 32, 32), seed=42)
        assert torch.equal(d1, d2)

    def test_load_dataset_cifar10(self):
        from teleportation_experiment import load_dataset
        try:
            data = load_dataset('cifar10', 'test', 16, data_dir='data')
            assert data.shape == (16, 3, 224, 224)
        except (FileNotFoundError, RuntimeError):
            pytest.skip("CIFAR-10 data not available locally")


class TestVerifyEquivalence:
    def test_identical_models_pass(self):
        from teleportation_experiment import create_model, verify_equivalence
        model = create_model('resnet18', 10)
        model.eval()
        data = torch.randn(8, 3, 224, 224)
        result = verify_equivalence(model, model, data, batch_size=4)
        assert result['all_predictions_match'] is True
        assert result['max_logit_diff'] < 1e-10

    def test_different_models_differ(self):
        from teleportation_experiment import create_model, verify_equivalence
        m1 = create_model('resnet18', 10)
        m2 = create_model('resnet18', 10)
        m1.eval()
        m2.eval()
        data = torch.randn(8, 3, 224, 224)
        result = verify_equivalence(m1, m2, data)
        assert result['max_logit_diff'] > 0.01


class TestIntegration:
    """End-to-end integration test with a small model."""

    def test_full_pipeline_resnet18(self, tmp_path):
        """Run 2 teleportations on resnet18 with random data only."""
        from teleportation_experiment import (
            create_model, PenultimateExtractor, generate_random_inputs,
            run_single_teleportation,
        )

        arch_name = 'resnet18'
        input_shape = (1, 3, 224, 224)
        model = create_model(arch_name, 10)
        model.eval()

        # Save and reload weights (test the loading path)
        weights_path = tmp_path / 'weights.pth'
        torch.save(model.state_dict(), weights_path)
        model2 = create_model(arch_name, 10)
        model2.load_state_dict(torch.load(weights_path, weights_only=True))
        model2.eval()

        data = generate_random_inputs(16, shape=(3, 224, 224), seed=42)
        splits = {'random': data}

        extractor = PenultimateExtractor(model2, arch_name)
        orig_feats = {'random': extractor.extract(model2, data, batch_size=8)}
        extractor.remove()
        assert orig_feats['random'].shape == (16, 512)

        results = []
        for seed in [42, 43]:
            result = run_single_teleportation(
                model2, arch_name, input_shape, splits, orig_feats,
                tp_seed=seed, device=torch.device('cpu'),
            )
            results.append(result)

            assert result['output_equivalence']['all_predictions_match']
            assert result['output_equivalence']['max_logit_diff'] < 1e-3
            assert result['random']['mean'] > 0

        # Two seeds should give different distances
        assert results[0]['random']['mean'] != results[1]['random']['mean']

    def test_full_pipeline_vgg11_bn(self, tmp_path):
        """Run 1 teleportation on vgg11_bn with random data only."""
        from teleportation_experiment import (
            create_model, PenultimateExtractor, generate_random_inputs,
            run_single_teleportation,
        )

        model = create_model('vgg11_bn', 10)
        model.eval()

        data = generate_random_inputs(8, shape=(3, 224, 224), seed=42)
        splits = {'random': data}

        extractor = PenultimateExtractor(model, 'vgg11_bn')
        orig_feats = {'random': extractor.extract(model, data, batch_size=4)}
        extractor.remove()
        assert orig_feats['random'].shape == (8, 4096)

        result = run_single_teleportation(
            model, 'vgg11_bn', (1, 3, 224, 224), splits, orig_feats,
            tp_seed=42, device=torch.device('cpu'),
        )

        assert result['output_equivalence']['all_predictions_match']
        assert result['random']['mean'] > 0
