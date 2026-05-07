"""Pretrained ImageNet model factories for the knowledgematrix library.

Wraps `torchvision`'s ResNet-152 / DenseNet-121 / GoogLeNet pretrained
weights via the `knowledgematrix` library's NN classes so callers get a
forward-compatible model that the `KnowledgeMatrixComputer` can consume
on `(3, 224, 224)` ImageNet inputs.
"""
from __future__ import annotations

import torch


def _build_resnet152(device: str) -> torch.nn.Module:
    from knowledgematrix.models.resnet152 import ResNet152

    return ResNet152(
        input_shape=(3, 224, 224),
        num_classes=1000,
        pretrained=True,
        device=device,
    )


def _build_densenet121(device: str) -> torch.nn.Module:
    # Lazy import: the wrapper is cluster-pinned in requirements-slurm.txt
    # and may not be installed locally on every dev machine.
    from knowledgematrix.models.densenet import DenseNet

    return DenseNet(
        input_shape=(3, 224, 224),
        num_classes=1000,
        pretrained=True,
        device=device,
    )


def _build_googlenet(device: str) -> torch.nn.Module:
    # Lazy import: the GoogLeNet (InceptionV1) wrapper lives on a parallel
    # knowledgematrix branch that may not resolve until the requirements
    # pin is bumped — keep the import inside the factory.
    from knowledgematrix.models.googlenet import GoogLeNet

    return GoogLeNet(
        input_shape=(3, 224, 224),
        num_classes=1000,
        pretrained=True,
        device=device,
    )


KM_MODEL_FACTORIES = {
    "resnet152":   _build_resnet152,
    "densenet121": _build_densenet121,
    "googlenet":   _build_googlenet,
}


def build_model(model_name: str, device: str) -> torch.nn.Module:
    """Construct a pretrained knowledgematrix-wrapped ImageNet model.

    Phase-1 architectures only: residual / dense / inception family
    coverage. Extend by adding a factory + dict entry above.
    """
    if model_name not in KM_MODEL_FACTORIES:
        raise ValueError(
            f"Unknown model {model_name!r}; "
            f"available: {sorted(KM_MODEL_FACTORIES)}"
        )
    model = KM_MODEL_FACTORIES[model_name](device)
    model.to(device)
    # knowledgematrix stores residual projection modules inside plain Python
    # lists under nn.ModuleDict values, so model.to(device) misses them.
    # Move them explicitly so subsequent forward passes don't crash with a
    # device-mismatch error on the residual path.
    for entries in getattr(model, "residuals", {}).values():
        for _start, projection in entries:
            for sub in projection:
                sub.to(device)
    model.eval()
    return model
