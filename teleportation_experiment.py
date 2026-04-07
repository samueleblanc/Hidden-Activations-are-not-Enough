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


# ---------------------------------------------------------------------------
# Teleportation wrapper
# ---------------------------------------------------------------------------

def teleport_model(model, input_shape, seed):
    """Create a random teleportation of the model.

    Returns a new model with different weights but identical function.
    The original model is not modified.

    Args:
        model: nn.Module (COB model) to teleport.
        input_shape: tuple, e.g. (1, 3, 224, 224) for JIT tracing.
        seed: random seed for reproducibility.

    Returns:
        nn.Module: teleported model (deep copy with modified weights).
    """
    model_copy = copy.deepcopy(model)
    torch.manual_seed(seed)
    tp = NeuralTeleportationModel(model_copy, input_shape=input_shape)
    tp.random_teleport(cob_range=1)
    return model_copy


# ---------------------------------------------------------------------------
# Distance computation
# ---------------------------------------------------------------------------

def compute_normalized_distances(feats_orig, feats_teleported):
    """Compute ||y - y'|| / sqrt(dim) per sample.

    Args:
        feats_orig: Tensor of shape (N, D) -- original penultimate features.
        feats_teleported: Tensor of shape (N, D) -- teleported features.

    Returns:
        numpy array of shape (N,) with normalized L2 distances.
    """
    diff = feats_orig - feats_teleported
    l2_per_sample = torch.norm(diff, dim=1)
    dim = feats_orig.shape[1]
    return (l2_per_sample / (dim ** 0.5)).numpy()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_dataset(dataset_name, split, num_samples, data_dir='data', seed=42):
    """Load a random subset of a dataset as a (N, 3, 224, 224) tensor.

    Args:
        dataset_name: One of 'cifar10', 'cifar100', 'tiny_imagenet'.
        split: 'train' or 'test'.
        num_samples: Number of samples to load.
        data_dir: Root directory for datasets.
        seed: Random seed for reproducible subset selection.

    Returns:
        Tensor of shape (num_samples, 3, 224, 224).
    """
    is_train = (split == 'train')
    if dataset_name == 'cifar10':
        ds = datasets.CIFAR10(data_dir, train=is_train, download=False,
                              transform=IMAGENET_TRANSFORM)
    elif dataset_name == 'cifar100':
        ds = datasets.CIFAR100(data_dir, train=is_train, download=False,
                               transform=IMAGENET_TRANSFORM)
    elif dataset_name == 'tiny_imagenet':
        subdir = 'train' if is_train else 'val'
        path = Path(data_dir) / 'tiny-imagenet-200' / subdir
        ds = datasets.ImageFolder(str(path), transform=IMAGENET_TRANSFORM)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(ds), generator=g)[:num_samples]
    images = torch.stack([ds[int(i)][0] for i in indices])
    return images


def generate_random_inputs(num_samples, shape=(3, 224, 224), seed=42):
    """Generate i.i.d. N(0,1) random inputs.

    Args:
        num_samples: Number of random samples to generate.
        shape: Shape of each sample (default: (3, 224, 224)).
        seed: Random seed for reproducibility.

    Returns:
        Tensor of shape (num_samples, *shape).
    """
    g = torch.Generator().manual_seed(seed)
    return torch.randn(num_samples, *shape, generator=g)


# ---------------------------------------------------------------------------
# Output equivalence verification
# ---------------------------------------------------------------------------

def verify_equivalence(model_orig, model_teleported, data, batch_size=64):
    """Check that two models produce the same outputs.

    Args:
        model_orig: Original model.
        model_teleported: Teleported model.
        data: Tensor of shape (N, C, H, W).
        batch_size: Number of samples per forward pass.

    Returns:
        dict with 'max_logit_diff' (float) and 'all_predictions_match' (bool).
    """
    model_orig.eval()
    model_teleported.eval()
    max_diff = 0.0
    all_match = True
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            out_orig = model_orig(batch)
            out_tp = model_teleported(batch)
            diff = (out_orig - out_tp).abs().max().item()
            max_diff = max(max_diff, diff)
            if not torch.equal(out_orig.argmax(dim=1), out_tp.argmax(dim=1)):
                all_match = False
    return {'max_logit_diff': float(max_diff), 'all_predictions_match': all_match}


if __name__ == '__main__':
    print("teleportation_experiment.py loaded successfully")
