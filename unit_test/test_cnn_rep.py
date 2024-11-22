#!/usr/bin/env python
"""
    Test if building the representation keeps the network function unchanged
"""
import sys
import os
import unittest
import torch
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matrix_construction.representation import ConvRepresentation_2D
from model_zoo.cnn import CNN_2D

DEVICE = "cpu"
torch.set_default_dtype(torch.float64)


class TestConvRepresentation_2D(unittest.TestCase):
    def generate_random_params(self):
        w = 32
        channels = tuple(10 for _ in range(4))
        fc = 300
        num_classes = 10
        return w, channels, fc, num_classes

    def create_random_model(self):
        w, channels, fc, num_classes = self.generate_random_params()
        input_shape = (3, w, w)
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
            model, x, forward_pass, w, channels, fc, num_classes = self.create_random_model()
            # Build representation and compute output
            rep = ConvRepresentation_2D(model, DEVICE)
            rep = rep.forward(x)
            one = torch.flatten(torch.ones(model.matrix_input_dim))
            rep_forward = torch.matmul(rep, one)
            diff = torch.norm(rep_forward - forward_pass).detach().numpy()

            self.assertAlmostEqual(diff, 0, places=None, msg=f"rep and forward_pass differ by {diff}.", delta=0.1)
            print(f"Test passed for input_shape=3x{w}x{w}, channels={channels}, fc={fc}, num_classes={num_classes}")


if __name__ == "__main__":
    unittest.main()
