"""Test Jacobian sensitivity computation."""
import unittest

import torch
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.neural_net import NN

from km_feature_viz.jacobian_sensitivity import (
    sensitivity_heatmap,
)


class TinyCNN(NN):
    def __init__(self, input_shape, num_classes):
        super().__init__(input_shape)
        self.conv(input_shape[0], 4, kernel_size=3, padding=1)
        self.relu()
        self.adaptiveavgpool((1, 1))
        self.flatten()
        self.linear(4, num_classes)


class TestJacobianSensitivity(unittest.TestCase):

    def test_heatmap_shape_and_nonneg(self):
        # NOTE: NN.eval() returns None; do NOT chain.
        model = TinyCNN((3, 8, 8), 5)
        model.eval()
        x = torch.rand(3, 8, 8)
        s = sensitivity_heatmap(model, x, predicted_class=0)
        self.assertEqual(s.shape, (3, 8, 8))
        self.assertTrue((s >= 0).all())
