"""
Pillar 1: Penultimate Activation Instability under Neural Teleportation.

Demonstrates that penultimate-layer activations change substantially under
neural teleportation (quiver isomorphism), even though the network computes
the same function. This validates that penultimate features are NOT invariant
under weight-space symmetries.

Usage:
    python teleportation_experiment.py \
        --architecture resnet18 \
        --dataset cifar10 \
        --weights_path path/to/weights.pth \
        --num_teleportations 100 \
        --num_samples 500
"""

import argparse
import copy
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pathlib import Path

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

ARCHITECTURES = {
    'resnet18':  {'factory': models.resnet18,  'penultimate_dim': 512,  'family': 'resnet'},
    'resnet34':  {'factory': models.resnet34,  'penultimate_dim': 512,  'family': 'resnet'},
    'resnet50':  {'factory': models.resnet50,  'penultimate_dim': 2048, 'family': 'resnet'},
    'resnet101': {'factory': models.resnet101, 'penultimate_dim': 2048, 'family': 'resnet'},
    'resnet152': {'factory': models.resnet152, 'penultimate_dim': 2048, 'family': 'resnet'},
    'vgg11_bn':  {'factory': models.vgg11_bn,  'penultimate_dim': 4096, 'family': 'vgg'},
    'vgg13_bn':  {'factory': models.vgg13_bn,  'penultimate_dim': 4096, 'family': 'vgg'},
    'vgg16_bn':  {'factory': models.vgg16_bn,  'penultimate_dim': 4096, 'family': 'vgg'},
    'vgg19_bn':  {'factory': models.vgg19_bn,  'penultimate_dim': 4096, 'family': 'vgg'},
}

NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'tiny_imagenet': 200}


if __name__ == '__main__':
    print("teleportation_experiment.py loaded successfully")
