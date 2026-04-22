"""Smoke tests for baseline visualization methods."""
import tempfile
import unittest
from pathlib import Path

import torch
import torchvision.models as tvm

from km_feature_viz.compute_baselines import (
    compute_gradcam,
    pick_target_layer,
)


class TestGradCam(unittest.TestCase):

    def test_compute_gradcam_shape(self):
        model = tvm.alexnet(weights=None)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        target_layer = pick_target_layer(model, "alexnet")
        attribution = compute_gradcam(model, target_layer, x, class_idx=0)
        # Grad-CAM heatmap: (1, 1, H', W') after captum
        self.assertEqual(attribution.dim(), 4)
        self.assertEqual(attribution.shape[0], 1)


class TestIGSmoothGrad(unittest.TestCase):

    def setUp(self):
        self.model = tvm.alexnet(weights=None).eval()
        self.x = torch.randn(1, 3, 224, 224)

    def test_compute_integrated_gradients_shape(self):
        from km_feature_viz.compute_baselines import compute_ig
        attr = compute_ig(self.model, self.x, class_idx=0)
        self.assertEqual(attr.shape, (1, 3, 224, 224))

    def test_compute_smoothgrad_shape(self):
        from km_feature_viz.compute_baselines import compute_smoothgrad
        attr = compute_smoothgrad(self.model, self.x, class_idx=0, n_samples=3)
        self.assertEqual(attr.shape, (1, 3, 224, 224))


class TestFeatureMaps(unittest.TestCase):

    def test_compute_feature_maps_shape(self):
        from km_feature_viz.compute_baselines import compute_feature_maps, pick_target_layer

        model = tvm.alexnet(weights=None).eval()
        x = torch.randn(1, 3, 224, 224)
        layer = pick_target_layer(model, "alexnet")
        maps = compute_feature_maps(model, layer, x)
        # AlexNet's last conv: 256 channels
        self.assertEqual(maps.shape[0], 1)
        self.assertEqual(maps.shape[1], 256)


class TestMaxActivating(unittest.TestCase):

    def test_max_activating_returns_topk(self):
        from km_feature_viz.compute_baselines import top_k_activating

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
        self.assertEqual(topk[0].tolist(), [4, 0, 1])
        # neuron 1: top 3 are images 5 (5.0), 1 (4.0), 2 (3.0)
        self.assertEqual(topk[1].tolist(), [5, 1, 2])


class TestPGD(unittest.TestCase):

    def test_compute_pgd_shape(self):
        from km_feature_viz.compute_baselines import compute_pgd

        model = tvm.alexnet(weights=None).eval()
        x = torch.rand(1, 3, 224, 224)
        delta = compute_pgd(model, x, class_idx=0, target_class=1, eps=8 / 255, steps=2)
        self.assertEqual(delta.shape, x.shape)
        self.assertLessEqual(delta.abs().max().item(), 8 / 255 + 1e-5)
