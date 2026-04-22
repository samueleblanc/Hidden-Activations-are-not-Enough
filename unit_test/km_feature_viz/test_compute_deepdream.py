"""Smoke test for DeepDream."""
import unittest

import torch
import torchvision.models as tvm

from km_feature_viz.compute_deepdream import deepdream_neuron


class TestDeepDream(unittest.TestCase):

    def test_deepdream_returns_image(self):
        model = tvm.alexnet(weights=None).eval()
        target_layer = model.features[10]
        img = deepdream_neuron(
            model, target_layer, neuron_idx=0, steps=2, lr=0.1, image_size=64
        )
        self.assertEqual(img.shape, (3, 64, 64))
