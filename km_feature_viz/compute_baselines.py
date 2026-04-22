"""Step 02: compute hidden-activation baseline visualizations.

Grad-CAM, Integrated Gradients, SmoothGrad, raw feature maps, max-activating
images, and PGD adversarials. Each baseline saves to a separate subtree
under results/.../baselines/<method>/.
"""
import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tvm

from km_feature_viz import paths, state
from km_feature_viz.compute_kms import IMAGENET_TRANSFORM, load_image
from km_feature_viz.manifest import read_manifest, sample_key

logger = logging.getLogger(__name__)


def pick_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Choose the standard last-conv layer for Grad-CAM per architecture."""
    if model_name == "alexnet":
        return model.features[10]  # last conv before AvgPool
    if model_name == "resnet18":
        return model.layer4[-1]    # last bottleneck
    if model_name == "vgg11":
        return model.features[18]  # last conv
    raise ValueError(f"Unknown model: {model_name}")


def build_torchvision_model(model_name: str, device: str) -> nn.Module:
    """Build a *plain* torchvision model (NOT the knowledgematrix wrapper).
    Used by Grad-CAM/IG/SmoothGrad which need standard nn.Module forward."""
    factory = {
        "alexnet": tvm.alexnet,
        "resnet18": tvm.resnet18,
        "vgg11": tvm.vgg11,
    }[model_name]
    weights_enum = {
        "alexnet": tvm.AlexNet_Weights.IMAGENET1K_V1,
        "resnet18": tvm.ResNet18_Weights.IMAGENET1K_V1,
        "vgg11": tvm.VGG11_Weights.IMAGENET1K_V1,
    }[model_name]
    model = factory(weights=weights_enum).to(device).eval()
    return model


def compute_gradcam(model: nn.Module, target_layer: nn.Module, x: torch.Tensor, class_idx: int) -> torch.Tensor:
    from captum.attr import LayerGradCam

    gc = LayerGradCam(model, target_layer)
    return gc.attribute(x, target=class_idx)


def compute_ig(model: nn.Module, x: torch.Tensor, class_idx: int, steps: int = 50) -> torch.Tensor:
    from captum.attr import IntegratedGradients

    ig = IntegratedGradients(model)
    return ig.attribute(x, target=class_idx, n_steps=steps)


def compute_smoothgrad(
    model: nn.Module, x: torch.Tensor, class_idx: int, n_samples: int = 25, stdev: float = 0.15
) -> torch.Tensor:
    from captum.attr import IntegratedGradients, NoiseTunnel

    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    return nt.attribute(
        x, target=class_idx, nt_type="smoothgrad", nt_samples=n_samples, stdevs=stdev
    )


def compute_feature_maps(model: nn.Module, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Forward x through model, return the activation tensor at `layer`."""
    captured = {}

    def hook(module, inp, out):
        captured["out"] = out.detach()

    handle = layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(x)
    finally:
        handle.remove()
    return captured["out"]


def top_k_activating(activations: torch.Tensor, k: int) -> torch.Tensor:
    """Given (N_images, N_neurons), return (N_neurons, k) of image indices with
    highest activation per neuron.

    Note: this is a notebook-side post-processing helper. It is intentionally
    not dispatched from main() — call it after collecting feature_maps tensors
    across the full dataset in a notebook or analysis script.
    """
    _, topk = activations.topk(k, dim=0)  # (k, N_neurons)
    return topk.T  # (N_neurons, k)


def compute_pgd(
    model: nn.Module,
    x: torch.Tensor,
    class_idx: int,
    target_class: int,
    eps: float = 8 / 255,
    steps: int = 20,
) -> torch.Tensor:
    """Run targeted PGD; return only the perturbation delta = x_adv - x."""
    import torchattacks

    atk = torchattacks.PGD(model, eps=eps, alpha=eps / steps * 2.5, steps=steps)
    atk.set_mode_targeted_by_label(quiet=True)
    target = torch.full((x.size(0),), target_class, dtype=torch.long, device=x.device)
    x_adv = atk(x, target)
    return x_adv - x


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--methods", nargs="+", default=["gradcam"],
                        help="Which baselines to compute")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    entries = read_manifest(args.manifest)

    by_model = {}
    for e in entries:
        by_model.setdefault(e.model, []).append(e)

    for model_name, model_entries in by_model.items():
        logger.info("Loading torchvision %s", model_name)
        model = build_torchvision_model(model_name, args.device)

        if "gradcam" in args.methods:
            target_layer = pick_target_layer(model, model_name)
            method = "gradcam"
            completed = state.load_completed(paths.state_path(f"02_{method}"))
            for entry in model_entries:
                key = sample_key(entry)
                if key in completed:
                    continue
                try:
                    x = load_image(entry.image_path).unsqueeze(0).to(args.device)
                    attribution = compute_gradcam(model, target_layer, x, class_idx=entry.class_id)
                    out_path = paths.baseline_path(method, entry.model, entry.class_id, entry.image_id)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(attribution.detach().cpu().to(torch.float16), out_path)
                    state.mark_completed(paths.state_path(f"02_{method}"), key)
                except Exception as e:
                    state.log_error(
                        paths.errors_path(), step=f"02_{method}", sample_id=key,
                        error_type=type(e).__name__, message=str(e),
                        tb=state.capture_traceback(),
                    )

        if "ig" in args.methods:
            method = "ig"
            completed = state.load_completed(paths.state_path(f"02_{method}"))
            for entry in model_entries:
                key = sample_key(entry)
                if key in completed:
                    continue
                try:
                    x = load_image(entry.image_path).unsqueeze(0).to(args.device)
                    attribution = compute_ig(model, x, class_idx=entry.class_id)
                    out_path = paths.baseline_path(method, entry.model, entry.class_id, entry.image_id)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(attribution.detach().cpu().to(torch.float16), out_path)
                    state.mark_completed(paths.state_path(f"02_{method}"), key)
                except Exception as e:
                    state.log_error(
                        paths.errors_path(), step=f"02_{method}", sample_id=key,
                        error_type=type(e).__name__, message=str(e),
                        tb=state.capture_traceback(),
                    )

        if "smoothgrad" in args.methods:
            method = "smoothgrad"
            completed = state.load_completed(paths.state_path(f"02_{method}"))
            for entry in model_entries:
                key = sample_key(entry)
                if key in completed:
                    continue
                try:
                    x = load_image(entry.image_path).unsqueeze(0).to(args.device)
                    attribution = compute_smoothgrad(model, x, class_idx=entry.class_id)
                    out_path = paths.baseline_path(method, entry.model, entry.class_id, entry.image_id)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(attribution.detach().cpu().to(torch.float16), out_path)
                    state.mark_completed(paths.state_path(f"02_{method}"), key)
                except Exception as e:
                    state.log_error(
                        paths.errors_path(), step=f"02_{method}", sample_id=key,
                        error_type=type(e).__name__, message=str(e),
                        tb=state.capture_traceback(),
                    )

        if "feature_maps" in args.methods:
            method = "feature_maps"
            target_layer = pick_target_layer(model, model_name)
            completed = state.load_completed(paths.state_path(f"02_{method}"))
            for entry in model_entries:
                key = sample_key(entry)
                if key in completed:
                    continue
                try:
                    x = load_image(entry.image_path).unsqueeze(0).to(args.device)
                    maps = compute_feature_maps(model, target_layer, x)
                    out_path = paths.baseline_path(method, entry.model, entry.class_id, entry.image_id)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(maps.cpu().to(torch.float16), out_path)
                    state.mark_completed(paths.state_path(f"02_{method}"), key)
                except Exception as e:
                    state.log_error(
                        paths.errors_path(), step=f"02_{method}", sample_id=key,
                        error_type=type(e).__name__, message=str(e),
                        tb=state.capture_traceback(),
                    )

        if "pgd" in args.methods:
            method = "pgd"
            completed = state.load_completed(paths.state_path(f"02_{method}"))
            for entry in model_entries:
                key = sample_key(entry)
                if key in completed:
                    continue
                try:
                    x = load_image(entry.image_path).unsqueeze(0).to(args.device)
                    # Pick target_class as next class up (cyclic) for a deterministic pair.
                    target_class = (entry.class_id + 1) % 1000
                    delta = compute_pgd(
                        model, x, class_idx=entry.class_id, target_class=target_class
                    )
                    out_path = paths.baseline_path(method, entry.model, entry.class_id, entry.image_id)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {"delta": delta.cpu().to(torch.float16), "target": target_class},
                        out_path,
                    )
                    state.mark_completed(paths.state_path(f"02_{method}"), key)
                except Exception as e:
                    state.log_error(
                        paths.errors_path(), step=f"02_{method}", sample_id=key,
                        error_type=type(e).__name__, message=str(e),
                        tb=state.capture_traceback(),
                    )

        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    sys.exit(main())
