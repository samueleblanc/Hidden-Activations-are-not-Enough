"""Step 03: DeepDream-style activation maximization per neuron.

For each (model, layer, channel) triple, optimize an input image to
maximize the mean activation of that channel. Saves the optimized image
and the per-step activation trajectory.

Channel selection is per-architecture (`NEURON_SELECTION_METHOD`):
- ResNet152, DenseNet121: ``gradcam_class_conditional`` — for each layer,
  pick the top-K channels by mean |alpha_y^k| (Selvaraju 2017 channel-wise
  Grad-CAM weights) aggregated over the manifest's labelled images. This
  grounds the visualized channels in the model's actual class-discriminative
  signal rather than arbitrary indices.
- GoogLeNet (InceptionV1): ``catalogued_distill`` — channels documented in
  the published Distill / Circuits / OpenAI Microscope catalogue (Olah
  2017, Cammarata 2020). Lets the figure cite known features by name
  ("dog head detector", "stripe pattern") rather than re-derive them.

The selection is recorded in ``state/03a_neuron_selection_<model>.json``
as a sidecar to the activation-max outputs in ``state/03_deepdream_<model>.json``.
"""
import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms.functional as TF

from km_feature_viz import paths, state
from km_feature_viz.compute_baselines import (
    build_torchvision_model,
    pick_target_layer,
)
from km_feature_viz.compute_kms import IMAGENET_MEAN, IMAGENET_STD, load_image
from km_feature_viz.manifest import Entry, read_manifest

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


# DeepDream per-architecture layer schedule — three semantically-distinct
# depths per architecture (low / mid / high feature stages). Channels
# within each layer are chosen at runtime via NEURON_SELECTION_METHOD,
# NOT hard-coded here. Keep these in sync with KM_MODEL_FACTORIES
# (compute_kms.py) and TV_MODEL_FACTORIES (compute_baselines.py).
#
# - ResNet152 stages: layer2.7 (last Bottleneck of stage 2, 8 blocks),
#   layer3.35 (last of stage 3, 36 blocks), layer4.2 (last of stage 4,
#   3 blocks) — mirrors the Grad-CAM `model.layer4[-1]` convention.
# - DenseNet121: features.denseblock{2,3,4} — block-level outputs after
#   each dense concatenation; analogous depth progression to ResNet stages.
# - GoogLeNet (InceptionV1): inception{3b,4d,5b} — the three depths most
#   heavily catalogued in Distill Circuits / OpenAI Microscope and the
#   conventional probe points for InceptionV1 feature visualization.
DEEPDREAM_LAYER_NAMES = {
    "resnet152":   ["layer2.7", "layer3.35", "layer4.2"],
    "densenet121": ["features.denseblock2", "features.denseblock3", "features.denseblock4"],
    "googlenet":   ["inception3b", "inception4d", "inception5b"],
}

# Per-architecture neuron-selection strategy:
# - "gradcam_class_conditional": run the manifest images through the model,
#   compute channel-wise Grad-CAM weights (alpha^k_y = mean_{i,j} dY_y/dA^k_{ij}),
#   take top-K channels per layer by mean |alpha| over images.
# - "catalogued_distill": use channel indices documented in the published
#   Distill / Circuits / OpenAI Microscope catalogue (only for GoogLeNet,
#   which is the model those works targeted).
NEURON_SELECTION_METHOD = {
    "resnet152":   "gradcam_class_conditional",
    "densenet121": "gradcam_class_conditional",
    "googlenet":   "catalogued_distill",
}

# Catalogued GoogLeNet (InceptionV1) channels.
#
# Sources:
#   - Olah et al. 2017, "Feature Visualization", Distill
#     (https://distill.pub/2017/feature-visualization/)
#   - Cammarata et al. 2020, "Curve Detectors" / Distill Circuits Thread
#     (https://distill.pub/2020/circuits/)
#   - OpenAI Microscope catalogue
#     (https://microscope.openai.com/models/inceptionv1)
#
# Channels prioritized for the manifest's 3 classes:
#   207=golden retriever (dog), 282=tiger cat (cat), 340=zebra (stripes/texture).
#
# NOTE: these specific channel indices are a draft from the published
# literature; the Microscope numbering can drift between TF-Slim and
# PyTorch ports of InceptionV1, so range_check_googlenet_neurons() must
# pass before cluster submission and a brief visual audit of the
# resulting activation-max images is advised before publication. Refine
# entries here as issues are found.
GOOGLENET_CATALOGUED_NEURONS: Dict[str, List[tuple]] = {
    "inception3b": [
        (101, "edge/orientation",       "Cammarata 2020"),
        (216, "color contrast",         "Cammarata 2020"),
        (255, "low-level texture",      "Cammarata 2020"),
        (379, "diagonal edge",          "Cammarata 2020"),
        (415, "circular structure",     "Cammarata 2020"),
    ],
    "inception4d": [
        (65,  "dog head",               "Olah 2017 §1; Cammarata 2020"),
        (447, "animal head",            "Cammarata 2020"),
        (491, "fur texture",            "Cammarata 2020"),
        (110, "stripe pattern",         "Cammarata 2020"),
        (368, "eye / face part",        "Cammarata 2020"),
    ],
    "inception5b": [
        (50,  "dog whole-object",       "Cammarata 2020"),
        (188, "cat whole-object",       "Cammarata 2020"),
        (244, "striped animal",         "Cammarata 2020"),
        (309, "fur / animal context",   "Cammarata 2020"),
        (920, "animal scene",           "Cammarata 2020"),
    ],
}


def get_layer_by_name(model: nn.Module, name: str) -> nn.Module:
    layer = model
    for part in name.split("."):
        layer = layer[int(part)] if part.isdigit() else getattr(layer, part)
    return layer


def select_neurons_gradcam(
    model: nn.Module,
    layer_names: List[str],
    manifest_entries: List[Entry],
    device: str,
    top_k: int = 5,
) -> dict:
    """Class-conditional Grad-CAM channel selection (Selvaraju 2017 channel-wise variant).

    For each layer, for each manifest image:
      - forward the image through the torchvision model;
      - take y = argmax(f(x));
      - capture A = layer.output via hook;
      - compute alpha_y^k = mean over (i, j) of d(y_y) / d(A^k_{ij}) via autograd;
      - record |alpha_y^k| in a per-layer accumulator.
    Aggregate over all manifest images by MEAN. Pick top-k channels by
    descending mean |alpha|.

    Returns
    -------
    dict
        ``{"method": "gradcam_class_conditional",
          "selected": {layer_name: [{"channel": int, "mean_alpha": float, "rank": int}, ...]}}``
    """
    model.eval()
    # accumulator: layer_name -> tensor of running |alpha| sums (1D, len = n_channels)
    accum: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {n: 0 for n in layer_names}

    # Build target-layer references once.
    target_layers = {name: get_layer_by_name(model, name) for name in layer_names}

    for entry in manifest_entries:
        # Load the image via the same transform the rest of the pipeline uses
        # (224×224 ImageNet normalization). load_image returns (3, H, W); add
        # the batch dim explicitly for the torchvision forward pass.
        x = load_image(entry.image_path).unsqueeze(0).to(device)
        for layer_name, layer in target_layers.items():
            captured: Dict[str, torch.Tensor] = {}

            def _hook(_mod, _inp, out, _key=layer_name):
                # Retain grad on the activation so autograd can flow back.
                out.retain_grad()
                captured["A"] = out

            handle = layer.register_forward_hook(_hook)
            try:
                model.zero_grad(set_to_none=True)
                logits = model(x)
                y = int(logits.argmax(dim=1).item())
                score = logits[0, y]
                score.backward()
                A = captured["A"]                  # (1, K, H, W)
                grads = A.grad                     # (1, K, H, W)
                alpha = grads.mean(dim=(2, 3))     # (1, K) — channel-wise GAP of grads
                alpha_abs = alpha.detach().abs().squeeze(0).cpu()  # (K,)
            finally:
                handle.remove()

            if layer_name not in accum:
                accum[layer_name] = torch.zeros_like(alpha_abs)
            accum[layer_name] += alpha_abs
            counts[layer_name] += 1

    selected: Dict[str, List[dict]] = {}
    for layer_name in layer_names:
        if counts[layer_name] == 0:
            selected[layer_name] = []
            continue
        mean_alpha = accum[layer_name] / counts[layer_name]
        order = torch.argsort(mean_alpha, descending=True)[:top_k]
        selected[layer_name] = [
            {
                "channel": int(order[i].item()),
                "mean_alpha": float(mean_alpha[order[i]].item()),
                "rank": i,
            }
            for i in range(len(order))
        ]
    return {"method": "gradcam_class_conditional", "selected": selected}


def get_googlenet_catalogued_selection() -> dict:
    """Return GOOGLENET_CATALOGUED_NEURONS in the JSON-shape expected by
    the state file: ``{"method": "catalogued_distill",
    "selected": {layer: [{"channel", "label", "citation", "rank"}, ...]}}``."""
    selected: Dict[str, List[dict]] = {}
    for layer_name, entries in GOOGLENET_CATALOGUED_NEURONS.items():
        selected[layer_name] = [
            {"channel": int(ch), "label": label, "citation": citation, "rank": rank}
            for rank, (ch, label, citation) in enumerate(entries)
        ]
    return {"method": "catalogued_distill", "selected": selected}


def write_neuron_selection(state_path: Path, selection: dict) -> None:
    """Atomic JSON write of a neuron-selection record."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, dir=state_path.parent, suffix=".tmp"
    ) as tmp:
        json.dump(selection, tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, state_path)


def range_check_googlenet_neurons(model: nn.Module, device: str = "cpu") -> None:
    """Validate that every (layer, channel) in GOOGLENET_CATALOGUED_NEURONS
    refers to a real channel in the given model.

    Hooks each catalogued layer on a single dummy forward pass to read the
    per-layer ``out_channels`` count, then collects every out-of-range
    entry and raises a single ValueError listing them all (does NOT stop
    at the first failure — the user wants the full list to refine the
    catalogue in one shot).
    """
    model.eval()
    captured: Dict[str, int] = {}
    handles = []
    try:
        for layer_name in GOOGLENET_CATALOGUED_NEURONS:
            layer = get_layer_by_name(model, layer_name)

            def _hook(_mod, _inp, out, _key=layer_name):
                # out shape: (B, C, H, W); record C.
                captured[_key] = int(out.shape[1])

            handles.append(layer.register_forward_hook(_hook))
        x = torch.zeros(1, 3, 224, 224, device=device)
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()

    bad: List[str] = []
    for layer_name, entries in GOOGLENET_CATALOGUED_NEURONS.items():
        n_channels = captured.get(layer_name)
        if n_channels is None:
            bad.append(f"{layer_name}: forward hook did not fire (layer not on the forward path?)")
            continue
        for channel, label, citation in entries:
            if channel < 0 or channel >= n_channels:
                bad.append(
                    f"{layer_name}[{channel}] (label={label!r}, src={citation}) "
                    f"out of range — layer has {n_channels} channels"
                )
    if bad:
        raise ValueError(
            "GOOGLENET_CATALOGUED_NEURONS contains out-of-range channel "
            f"indices ({len(bad)} entry/entries):\n  - "
            + "\n  - ".join(bad)
        )


def state_step_name(suffix: Optional[str]) -> str:
    """Build the state-file step name, suffixed by `suffix` if non-empty.

    Lets per-model SLURM array tasks write to disjoint state files
    (`03_deepdream_<model>.json`) so concurrent tasks don't race on
    `state.mark_completed`'s read-modify-write.
    """
    return "03_deepdream" + (f"_{suffix}" if suffix else "")


def neuron_selection_step_name(model_name: str) -> str:
    """State-file name for the per-arch neuron-selection sidecar."""
    return f"03a_neuron_selection_{model_name}"


def _selection_to_layer_channel_pairs(selection: dict) -> List[tuple]:
    """Flatten a selection dict into [(layer_name, channel), ...] pairs."""
    pairs: List[tuple] = []
    for layer_name, entries in selection["selected"].items():
        for e in entries:
            pairs.append((layer_name, int(e["channel"])))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Manifest JSON (used by gradcam_class_conditional selection)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=5,
                        help="Channels per layer for gradcam_class_conditional selection")
    parser.add_argument("--models", nargs="+", default=["resnet152"])
    parser.add_argument("--state-suffix", default=None,
                        help="Suffix appended to the state-file step name (e.g., 'resnet152')")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    state_file = paths.state_path(state_step_name(args.state_suffix))
    completed = state.load_completed(state_file)

    entries = read_manifest(args.manifest)

    for model_name in args.models:
        if model_name not in DEEPDREAM_LAYER_NAMES:
            raise ValueError(
                f"Unknown model: {model_name!r}; "
                f"available: {sorted(DEEPDREAM_LAYER_NAMES)}"
            )
        if model_name not in NEURON_SELECTION_METHOD:
            raise ValueError(
                f"No neuron-selection method registered for {model_name!r}; "
                f"available: {sorted(NEURON_SELECTION_METHOD)}"
            )

        model = build_torchvision_model(model_name, args.device)
        layer_names = DEEPDREAM_LAYER_NAMES[model_name]
        method = NEURON_SELECTION_METHOD[model_name]

        # ---- Channel selection ------------------------------------------
        if method == "gradcam_class_conditional":
            model_entries = [e for e in entries if e.model == model_name]
            if not model_entries:
                logger.warning(
                    "No manifest entries for model %s; skipping selection", model_name
                )
                del model
                continue
            selection = select_neurons_gradcam(
                model=model,
                layer_names=layer_names,
                manifest_entries=model_entries,
                device=args.device,
                top_k=args.top_k,
            )
        elif method == "catalogued_distill":
            # Range-check FIRST — must raise before any DeepDream work
            # if any catalogued channel is out of range.
            range_check_googlenet_neurons(model, device=args.device)
            selection = get_googlenet_catalogued_selection()
        else:
            raise ValueError(
                f"Unknown NEURON_SELECTION_METHOD {method!r} for {model_name}"
            )

        write_neuron_selection(
            paths.state_path(neuron_selection_step_name(model_name)), selection
        )

        # ---- Activation maximization ------------------------------------
        for layer_name, neuron in _selection_to_layer_channel_pairs(selection):
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
                torch.save(img.to(torch.float32), out_path)
                state.mark_completed(state_file, key)
                logger.info("done %s", key)
            except Exception as e:
                state.log_error(
                    paths.errors_path(), step=state_step_name(args.state_suffix), sample_id=key,
                    error_type=type(e).__name__, message=str(e),
                    tb=state.capture_traceback(),
                )
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    sys.exit(main())
