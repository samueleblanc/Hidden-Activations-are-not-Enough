#!/usr/bin/env python
"""
    Test if building the representation keeps the network function unchanged
"""
import sys
import os
import unittest
import torch
import random
from time import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matrix_construction.representation import ConvRepresentation_2D
from model_zoo.vgg import VGG

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float64)


class TestVGGRepresentation(unittest.TestCase):
    def generate_random_params(self):
        w = random.randint(15,25)
        num_classes = random.randint(10,20)
        return w, num_classes

    def create_random_model(self):
        w, num_classes = self.generate_random_params()
        input_shape = (random.randint(1,3),w,w)
        x = torch.rand(input_shape)

        model = VGG(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    pretrained=False,
                    max_pool=True,
                    batch_norm=True
                ).to(DEVICE)
        model.eval()
        model.save = True
        forward_pass = model(x)

        return model, x, forward_pass, num_classes

    def test_VGGRepBuild(self):
        for _ in range(5):
            start = time()
            model, x, forward_pass, num_classes = self.create_random_model()
            batch_size = random.randint(1,8)
            # Build representation and compute output
            rep = ConvRepresentation_2D(model, batch_size=batch_size)
            rep = rep.forward(x)
            one = torch.flatten(torch.ones(model.matrix_input_dim))
            rep_forward = torch.matmul(rep, one)
            diff = torch.norm(rep_forward - forward_pass).item()

            self.assertAlmostEqual(diff, 0, places=None, msg=f"rep and forward_pass differ by {diff}.", delta=0.1)
            end = time()
            print(f"Test passed for input_shape={x.shape[0]}x{x.shape[1]}x{x.shape[2]}, num_classes={num_classes}, and batch_size={batch_size}")
            print(f"Difference: {diff}. Time: {round(end-start,7)}sec.")


if __name__ == "__main__":
    unittest.main()
