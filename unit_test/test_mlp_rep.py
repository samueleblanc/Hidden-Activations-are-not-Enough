#!/usr/bin/env python
"""
    Test if building the matrix keeps the network function unchanged
"""
import sys
import os
import unittest
import torch
import random
from time import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matrix_construction.matrix_computation import MlpRepresentation
from model_zoo.mlp import MLP

DEVICE = "cpu"
torch.set_default_dtype(torch.float64)


class TestMLPRepresentation(unittest.TestCase):
    def generate_random_params(self):
        w = random.randint(28, 32)
        l = random.randint(1, 50)
        c = random.randint(1, 800)
        num_classes = random.randint(2, 100)
        return w, l, c, num_classes

    def create_random_model(self):
        w, l, c, num_classes = self.generate_random_params()
        input_shape = (3, w, w)
        x = torch.rand(input_shape)

        model = MLP(input_shape=input_shape,
                    num_classes=num_classes,
                    hidden_sizes=tuple(c for _ in range(l)),
                    bias=True,
                    residual=False
                    ).to(DEVICE)
        model.init()
        model.eval()
        model.save = True
        forward_pass = model(x)

        return model, x, forward_pass, l, c, num_classes

    def test_MLPRepBuild(self):
        for _ in range(50):
            start = time()
            model, x, forward_pass, l, c, num_classes = self.create_random_model()

            # Build representation and compute output
            rep = MlpRepresentation(model, build_rep=True, device=DEVICE)
            matrix = rep.forward(x)
            one = torch.flatten(torch.ones(model.matrix_input_dim))
            rep_forward = torch.matmul(matrix, one)
            diff = torch.norm(rep_forward - forward_pass).detach().numpy()

            self.assertAlmostEqual(diff, 0, places=None, msg=f"rep and forward_pass differ by {diff}.", delta=0.1)
            end = time()
            print(f"Test passed for input_shape={x.shape[0]}x{x.shape[1]}x{x.shape[2]}, number_of_layers={l}, neurons_per_layer={c}, num_classes={num_classes}")
            print(f"Difference: {diff}. Time: {round(end-start,7)}sec.")


if __name__ == "__main__":
    unittest.main()
