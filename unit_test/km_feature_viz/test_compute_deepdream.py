"""Smoke test for DeepDream.

Pillar 3 launched with ResNet152 only; we use an unweighted ResNet152
fixture so no download is triggered. The optimization itself only runs
2 steps at 64×64, so the cost difference vs. the prior AlexNet fixture
is small even though the architecture is much larger — most of the
wall-clock is the construction, not the 2-step optimization.
"""
import unittest

import torch
import torchvision.models as tvm

from km_feature_viz.compute_deepdream import deepdream_neuron


class TestDeepDream(unittest.TestCase):

    def test_deepdream_returns_image(self):
        model = tvm.resnet152(weights=None).eval()
        # `model.layer4[-1]` is the last Bottleneck of stage 4; its output
        # is the deepest feature map fed into the final avg-pool.
        target_layer = model.layer4[-1]
        img = deepdream_neuron(
            model, target_layer, neuron_idx=0, steps=2, lr=0.1, image_size=64
        )
        self.assertEqual(img.shape, (3, 64, 64))
