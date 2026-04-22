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
