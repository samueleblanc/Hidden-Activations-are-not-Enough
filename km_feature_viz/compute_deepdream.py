"""Step 03: DeepDream-style activation maximization per neuron.

For each (model, layer, neuron) triple, optimize an input image to maximize
the mean activation of that neuron. Saves the optimized image and the
per-step activation trajectory.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms.functional as TF

from km_feature_viz import paths, state
from km_feature_viz.compute_baselines import (
    build_torchvision_model,
    pick_target_layer,
)
from km_feature_viz.compute_kms import IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)


def deepdream_neuron(
    model: nn.Module,
    target_layer: nn.Module,
    neuron_idx: int,
    steps: int = 200,
    lr: float = 0.05,
    jitter: int = 8,
    image_size: int = 224,
) -> torch.Tensor:
    """Optimize an image to maximize neuron_idx's mean activation in target_layer.
    Returns the de-normalized image tensor of shape (3, H, W)."""
    captured = {}

    def hook(module, inp, out):
        captured["out"] = out

    handle = target_layer.register_forward_hook(hook)

    device = next(model.parameters()).device
    img = torch.randn(1, 3, image_size, image_size, device=device, requires_grad=True) * 0.1
    img = img.detach().requires_grad_(True)

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    optimizer = torch.optim.Adam([img], lr=lr)

    try:
        for step in range(steps):
            ox, oy = torch.randint(-jitter, jitter + 1, (2,)).tolist()
            jittered = torch.roll(img, shifts=(ox, oy), dims=(2, 3))
            normed = (jittered - mean) / std
            model(normed)
            act = captured["out"]
            # Mean over (batch, H, W) for the chosen neuron channel
            score = act[:, neuron_idx].mean()
            loss = -score
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                img.clamp_(0, 1)
    finally:
        handle.remove()

    return img.detach().squeeze(0).cpu()


def neurons_per_layer(model_name: str) -> List[tuple]:
    """For Tier A: 5 neurons in each of 3 chosen layers per model."""
    if model_name == "alexnet":
        return [
            ("features.3", n) for n in range(5)
        ] + [
            ("features.6", n) for n in range(5)
        ] + [
            ("features.10", n) for n in range(5)
        ]
    if model_name == "resnet18":
        return [
            ("layer2.1", n) for n in range(5)
        ] + [
            ("layer3.1", n) for n in range(5)
        ] + [
            ("layer4.1", n) for n in range(5)
        ]
    if model_name == "vgg11":
        return [
            ("features.6", n) for n in range(5)
        ] + [
            ("features.13", n) for n in range(5)
        ] + [
            ("features.18", n) for n in range(5)
        ]
    raise ValueError(model_name)


def get_layer_by_name(model: nn.Module, name: str) -> nn.Module:
    layer = model
    for part in name.split("."):
        layer = layer[int(part)] if part.isdigit() else getattr(layer, part)
    return layer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--models", nargs="+", default=["alexnet", "resnet18", "vgg11"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    completed = state.load_completed(paths.state_path("03_deepdream"))

    for model_name in args.models:
        model = build_torchvision_model(model_name, args.device)
        for layer_name, neuron in neurons_per_layer(model_name):
            key = f"{model_name}/{layer_name}/{neuron}"
            if key in completed:
                continue
            try:
                target_layer = get_layer_by_name(model, layer_name)
                img = deepdream_neuron(
                    model, target_layer, neuron_idx=neuron, steps=args.steps
                )
                out_path = paths.deepdream_path(model_name, layer_name, neuron)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(img.to(torch.float16), out_path)
                state.mark_completed(paths.state_path("03_deepdream"), key)
                logger.info("done %s", key)
            except Exception as e:
                state.log_error(
                    paths.errors_path(), step="03_deepdream", sample_id=key,
                    error_type=type(e).__name__, message=str(e),
                    tb=state.capture_traceback(),
                )
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    sys.exit(main())
