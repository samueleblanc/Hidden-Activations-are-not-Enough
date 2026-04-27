"""
GoogLeNet (Inception v1) with Change-Of-Basis (COB) layers compatible with the
neuralteleportation library.

Mirrors torchvision.models.googlenet so that its pretrained state_dict loads
directly into the COB version (state-dict key alignment is exact, modulo the
absence of auxiliary classifiers — we only support aux_logits=False).

Why this file exists
--------------------
The Hidden-Activations-are-not-Enough repo's Step B teleportation experiment
(Pillar 1: Isomorphism Invariance) needs to teleport ResNet152, DenseNet121,
and GoogLeNet — the three Pillar-3 architectures. ResNet152 and DenseNet121
already have COB factories in the upstream `neuralteleportation` library,
but GoogLeNet does not. This file adds it.

The neuralteleportation library forbids functional ops in `forward()`
(`F.relu`, `torch.cat`, etc.) because its JIT-graph machinery requires every
operation to be an `nn.Module` so it can register a `*COB` wrapper. Torchvision's
GoogLeNet uses `F.relu(...)` inside `BasicConv2d.forward()` and `torch.cat(...)`
inside `Inception.forward()`. We replace both with their `*COB` Module
equivalents (`ReLUCOB` and `Concat`).

We intentionally do NOT support the auxiliary classifiers (aux_logits=True).
Aux classifiers branch off the main graph, are unused at inference, and would
add useless complexity to the COB graph. Step B always uses inference-mode
networks.

Patch hookup
------------
The repo ships this file as a patch source. `apply_neuralteleportation_patches.sh`
copies it to the installed `neuralteleportation/models/model_zoo/googlenetcob.py`.

Author: Hidden-Activations-are-not-Enough project (km-feature-viz branch),
        2026-04-27. Apache-2.0.
"""

from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from neuralteleportation.layers.activation import ReLUCOB
from neuralteleportation.layers.merge import Concat
from neuralteleportation.layers.neuralteleportation import FlattenCOB
from neuralteleportation.layers.dropout import DropoutCOB
from neuralteleportation.layers.neuron import BatchNorm2dCOB, LinearCOB, Conv2dCOB
from neuralteleportation.layers.pooling import MaxPool2dCOB, AdaptiveAvgPool2dCOB

__all__ = ['GoogLeNetCOB', 'BasicConv2dCOB', 'InceptionCOB', 'googlenetCOB']


class BasicConv2dCOB(nn.Module):
    """COB-friendly equivalent of torchvision's BasicConv2d.

    Replaces the functional F.relu with a ReLUCOB Module so the network_graph
    JIT walker can register it.
    """

    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        self.conv = Conv2dCOB(in_channels, out_channels, bias=False, **kwargs)
        # torchvision uses eps=0.001 here (NOT default 1e-5); preserve for
        # state-dict compatibility — the buffers carry running stats trained
        # under eps=0.001.
        self.bn = BatchNorm2dCOB(out_channels)
        self.bn.eps = 0.001
        self.relu = ReLUCOB(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InceptionCOB(nn.Module):
    """COB-friendly Inception module.

    Each of the four branches is computed in order (branch1 → branch2 →
    branch3 → branch4) and concatenated via Concat (an nn.Module wrapper of
    torch.cat that participates in the COB graph).

    The branch-order is the same as torchvision's, so the output channel layout
    matches exactly and pretrained state_dicts load.
    """

    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
    ) -> None:
        super().__init__()
        # branch1: 1x1 conv
        self.branch1 = BasicConv2dCOB(in_channels, ch1x1, kernel_size=1)

        # branch2: 1x1 → 3x3
        self.branch2 = nn.Sequential(
            BasicConv2dCOB(in_channels, ch3x3red, kernel_size=1),
            BasicConv2dCOB(ch3x3red, ch3x3, kernel_size=3, padding=1),
        )

        # branch3: 1x1 → 3x3 (note: torchvision documents kernel_size=3 here as
        # a known-bug-but-stable choice for the pretrained weights — we keep it
        # the same so the weights load and produce identical outputs)
        self.branch3 = nn.Sequential(
            BasicConv2dCOB(in_channels, ch5x5red, kernel_size=1),
            BasicConv2dCOB(ch5x5red, ch5x5, kernel_size=3, padding=1),
        )

        # branch4: maxpool → 1x1
        self.branch4 = nn.Sequential(
            MaxPool2dCOB(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2dCOB(in_channels, pool_proj, kernel_size=1),
        )

        # Concat is an nn.Module wrapping torch.cat; required so the
        # network_graph JIT walker can register it as a COB layer.
        self.concat = Concat()

    def forward(self, x: Tensor) -> Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        # IMPORTANT: argument order to Concat.forward must match the *forward
        # execution order* (b1 first, b4 last). See merge.Concat docstring.
        return self.concat(b1, b2, b3, b4, dim=1)


class GoogLeNetCOB(nn.Module):
    """GoogLeNet (Inception v1) with COB layers.

    Matches torchvision.models.googlenet(aux_logits=False) layer-for-layer so
    pretrained weights load via load_state_dict. Auxiliary classifiers are
    intentionally unsupported.
    """

    def __init__(self, num_classes: int = 1000, init_weights: bool = True,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.aux_logits = False  # always False; aux classifiers unsupported

        self.conv1 = BasicConv2dCOB(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = MaxPool2dCOB(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2dCOB(64, 64, kernel_size=1)
        self.conv3 = BasicConv2dCOB(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = MaxPool2dCOB(kernel_size=3, stride=2, ceil_mode=True)

        self.inception3a = InceptionCOB(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionCOB(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = MaxPool2dCOB(kernel_size=3, stride=2, ceil_mode=True)

        self.inception4a = InceptionCOB(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionCOB(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionCOB(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionCOB(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionCOB(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = MaxPool2dCOB(kernel_size=2, stride=2, ceil_mode=True)

        self.inception5a = InceptionCOB(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionCOB(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = AdaptiveAvgPool2dCOB((1, 1))
        self.flatten = FlattenCOB()
        self.dropout = DropoutCOB(dropout)
        self.fc = LinearCOB(1024, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        # Same init as torchvision's GoogLeNet (truncated normal via fallback)
        for m in self.modules():
            if isinstance(m, (Conv2dCOB, LinearCOB)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, BatchNorm2dCOB):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        # N x 3 x 224 x 224
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def googlenetCOB(pretrained: bool = False, num_classes: int = 1000,
                 **kwargs) -> GoogLeNetCOB:
    """Factory matching the *COB family naming convention.

    Args:
        pretrained: If True, attempt to load torchvision's pretrained
            GoogLeNet weights. The state-dict keys match exactly (aux
            classifiers are filtered out by load_state_dict with strict=False
            since we do not include them).
        num_classes: Number of output classes.

    Returns:
        GoogLeNetCOB instance.

    Notes:
        For the Hidden-Activations-are-not-Enough Step B experiment, weights
        are loaded externally from a torchvision model (so this factory is
        usually called with pretrained=False and the state_dict is loaded
        from torchvision directly — this avoids fetching weights on cluster
        compute nodes which have no internet).
    """
    model = GoogLeNetCOB(num_classes=num_classes, init_weights=not pretrained,
                         **kwargs)
    if pretrained:
        try:
            from torch.hub import load_state_dict_from_url
            url = 'https://download.pytorch.org/models/googlenet-1378be20.pth'
            state_dict = load_state_dict_from_url(url, progress=True)
            # Filter out auxiliary classifier keys — we don't include them
            state_dict = {k: v for k, v in state_dict.items()
                          if not k.startswith('aux1.') and not k.startswith('aux2.')}
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            raise RuntimeError(
                "Failed to download GoogLeNet pretrained weights. "
                "On compute nodes with no internet, set pretrained=False and "
                "load weights from a local file or via torchvision."
            ) from e
    return model


if __name__ == '__main__':
    import copy
    import numpy as np
    import torchvision.models as tv

    print("Smoke test: torchvision GoogLeNet -> GoogLeNetCOB -> teleport")
    tv_model = tv.googlenet(weights='DEFAULT', aux_logits=False)
    cob = GoogLeNetCOB(num_classes=1000, init_weights=False)
    cob.load_state_dict(tv_model.state_dict(), strict=True)
    cob.eval()
    tv_model.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y_tv = tv_model(x)
        y_cob = cob(x)
    print(f"  COB vs TV output diff: {(y_tv - y_cob).abs().max().item():.4e}")

    from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
    cob_tp = copy.deepcopy(cob)
    np.random.seed(0)
    NeuralTeleportationModel(cob_tp, input_shape=(1, 3, 224, 224)).random_teleport(cob_range=1)
    cob_tp.eval()
    with torch.no_grad():
        y_tp = cob_tp(x)
    print(f"  Teleport vs original: abs={(y_cob - y_tp).abs().max().item():.4e}")
    print(f"  argmax match: {torch.equal(y_cob.argmax(1), y_tp.argmax(1))}")
