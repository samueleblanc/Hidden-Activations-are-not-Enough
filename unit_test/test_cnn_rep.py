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
from model_zoo.cnn import CNN_2D

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float64)


class TestConvRepresentation_2D(unittest.TestCase):
    def generate_random_params(self):
        w = random.randint(20,40)
        channels = tuple(random.randint(3,50) for _ in range(random.randint(3,8)))
        fc = random.randint(200,400)
        num_classes = random.randint(5,15)
        return w, channels, fc, num_classes

    def create_random_model(self):
        w, channels, fc, num_classes = self.generate_random_params()
        input_shape = (random.randint(1,4), w, w)
        x = torch.rand(input_shape)

        model = CNN_2D(input_shape=input_shape,
                    num_classes=num_classes,
                    channels=channels,
                    fc=fc,
                    bias=False,
                    residual=False,
                    ).to(DEVICE)
        model.init()
        model.eval()
        model.save = True
        forward_pass = model(x)

        return model, x, forward_pass, w, channels, fc, num_classes

    def test_ConvRepBuild(self):
        for _ in range(10):
            start = time()
            model, x, forward_pass, w, channels, fc, num_classes = self.create_random_model()
            # Build representation and compute output
            rep = ConvRepresentation_2D(model)
            rep = rep.forward(x)
            one = torch.flatten(torch.ones(model.matrix_input_dim))
            rep_forward = torch.matmul(rep, one)
            diff = torch.norm(rep_forward - forward_pass).item()

            self.assertAlmostEqual(diff, 0, places=None, msg=f"rep and forward_pass differ by {diff}.", delta=0.1)
            end = time()
            print(f"Test passed for input_shape=3x{w}x{w}, channels={channels}, fc={fc}, num_classes={num_classes}")
            print(f"Difference: {diff}. Time: {round(end-start,7)}sec.")


if __name__ == "__main__":
    unittest.main()
