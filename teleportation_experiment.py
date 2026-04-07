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
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pathlib import Path

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.models.model_zoo.resnetcob import (
    resnet18COB, resnet34COB, resnet50COB, resnet101COB, resnet152COB,
)
from neuralteleportation.models.model_zoo.vggcob import (
    vgg11_bnCOB, vgg13_bnCOB, vgg16_bnCOB, vgg19_bnCOB,
)


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

ARCHITECTURES = {
    'resnet18':  {'factory': resnet18COB,  'penultimate_dim': 512,  'family': 'resnet'},
    'resnet34':  {'factory': resnet34COB,  'penultimate_dim': 512,  'family': 'resnet'},
    'resnet50':  {'factory': resnet50COB,  'penultimate_dim': 2048, 'family': 'resnet'},
    'resnet101': {'factory': resnet101COB, 'penultimate_dim': 2048, 'family': 'resnet'},
    'resnet152': {'factory': resnet152COB, 'penultimate_dim': 2048, 'family': 'resnet'},
    'vgg11_bn':  {'factory': vgg11_bnCOB,  'penultimate_dim': 4096, 'family': 'vgg'},
    'vgg13_bn':  {'factory': vgg13_bnCOB,  'penultimate_dim': 4096, 'family': 'vgg'},
    'vgg16_bn':  {'factory': vgg16_bnCOB,  'penultimate_dim': 4096, 'family': 'vgg'},
    'vgg19_bn':  {'factory': vgg19_bnCOB,  'penultimate_dim': 4096, 'family': 'vgg'},
}

NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'tiny_imagenet': 200}


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def create_model(arch_name, num_classes):
    """Create a COB model compatible with neuralteleportation.

    COB models have the same architecture as torchvision models but use
    COB layer types. Torchvision-trained state_dicts load directly.

    Args:
        arch_name: Key into ARCHITECTURES (e.g. 'resnet18', 'vgg11_bn').
        num_classes: Number of output classes for the final FC layer.

    Returns:
        nn.Module: A COB model instance.

    Raises:
        KeyError: If arch_name is not in ARCHITECTURES.
    """
    config = ARCHITECTURES[arch_name]
    model = config['factory'](pretrained=False, num_classes=num_classes)
    return model


# ---------------------------------------------------------------------------
# Penultimate-layer feature extraction
# ---------------------------------------------------------------------------

class PenultimateExtractor:
    """Extract penultimate-layer activations via a forward hook.

    ResNets: hooks on avgpool -> flatten -> (batch, channels)
    VGGs: hooks on classifier[4] (ReLU after 2nd-to-last Linear) -> (batch, 4096)
    """

    def __init__(self, model, arch_name):
        self._features = None
        config = ARCHITECTURES[arch_name]
        if config['family'] == 'resnet':
            target = model.avgpool
        elif config['family'] == 'vgg':
            target = model.classifier[4]
        else:
            raise ValueError(f"Unknown architecture family: {config['family']}")
        self._hook = target.register_forward_hook(self._capture)

    def _capture(self, module, input, output):
        self._features = output.detach()

    def extract(self, model, data, batch_size=64):
        """Extract penultimate features for all samples in data.

        Args:
            model: The model to run forward passes on.
            data: Tensor of shape (N, C, H, W).
            batch_size: Number of samples per forward pass.

        Returns:
            Tensor of shape (N, penultimate_dim).
        """
        model.eval()
        all_feats = []
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                model(batch)
                feat = self._features
                if feat.dim() > 2:
                    feat = feat.flatten(1)
                all_feats.append(feat.cpu())
        return torch.cat(all_feats, dim=0)

    def remove(self):
        """Remove the forward hook."""
        self._hook.remove()


if __name__ == '__main__':
    print("teleportation_experiment.py loaded successfully")
