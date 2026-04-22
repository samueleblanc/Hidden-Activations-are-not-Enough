"""KM correctness invariant: model.forward(x) == A.sum(1) at machine epsilon.

This test runs against a tiny CNN — it does NOT load the cached KM cache.
For a cache-wide invariant check, use the dedicated script that the notebook
calls before rendering figures.
"""
import unittest

import torch

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.neural_net import NN

torch.set_default_dtype(torch.float64)


class TestKMInvariant(unittest.TestCase):

    def test_invariant_on_small_cnn(self):
        # Create a small CNN using the NN factory pattern
        model = NN(input_shape=(3, 8, 8), save=True)
        model.conv(3, 8, kernel_size=3, padding=1)
        model.relu()
        model.adaptiveavgpool((1, 1))
        model.flatten()
        model.linear(8, 5)
        model.eval()

        x = torch.rand(3, 8, 8)
        A = KnowledgeMatrixComputer(model, batch_size=8).forward(x)
        out_true = model.forward(x).flatten()
        out_pred = A.sum(1)
        self.assertTrue(torch.allclose(out_true, out_pred, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
