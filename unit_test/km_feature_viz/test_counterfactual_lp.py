"""Test the LP counterfactual on a tiny CNN where W_eff is computable."""
import unittest

import torch
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.neural_net import NN

from km_feature_viz.counterfactual_lp import (
    extract_weff_and_beff,
    solve_l1_lp,
    in_region,
)

torch.set_default_dtype(torch.float64)


class TinyCNN(NN):
    def __init__(self, input_shape, num_classes):
        super().__init__(input_shape)
        self.conv(input_shape[0], 4, kernel_size=3, padding=1)
        self.relu()
        self.adaptiveavgpool((1, 1))
        self.flatten()
        self.linear(4, num_classes)


class TestLPCounterfactual(unittest.TestCase):

    def setUp(self):
        # NOTE: NN.eval() returns None — call separately, do not chain.
        self.model = TinyCNN((3, 8, 8), 5)
        self.model.eval()
        self.x = torch.rand(3, 8, 8)

    def test_extract_weff_and_beff_consistency(self):
        W, b = extract_weff_and_beff(self.model, self.x)
        out_pred = W @ self.x.flatten() + b
        out_true = self.model(self.x).flatten()
        self.assertTrue(torch.allclose(out_pred, out_true, atol=1e-6))

    def test_solve_l1_lp_returns_valid_delta(self):
        W, b = extract_weff_and_beff(self.model, self.x)
        out_true = self.model(self.x).flatten()
        y, t = 0, 1
        delta = solve_l1_lp(
            W, b, self.x.flatten(), out_true, source=y, target=t, margin=0.01
        )
        self.assertEqual(delta.shape, self.x.flatten().shape)
        # Verify that within the linear region, the predicted swing satisfies the constraint
        new_logit_t = (W[t] @ (self.x.flatten() + delta)) + b[t]
        new_logit_y = (W[y] @ (self.x.flatten() + delta)) + b[y]
        self.assertGreaterEqual((new_logit_t - new_logit_y).item(), 0.01 - 1e-3)

    def test_in_region_self(self):
        self.assertTrue(in_region(self.model, self.x, self.x))
