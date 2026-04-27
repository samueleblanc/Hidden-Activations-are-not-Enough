"""Step 01: compute knowledge matrices for every entry in the manifest.

Slices the KM down to the in-scope class rows before saving (storage
optimization; see spec §6).
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image

from km_feature_viz import paths, state
from km_feature_viz.manifest import (
    Entry,
    TIER_A_CLASSES,
    read_manifest,
    sample_key,
)

logger = logging.getLogger(__name__)


# torchvision pretrained ImageNet preprocessing (standard)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_TRANSFORM = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)
# Same crop, no Normalize — what we save for side-by-side paper figures.
IMAGENET_TRANSFORM_RAW = T.Compose(
    [T.Resize(256), T.CenterCrop(224), T.ToTensor()]
)


def slice_class_rows(km: torch.Tensor, in_scope_classes: List[int]) -> torch.Tensor:
    """Return only the rows of the KM corresponding to in-scope classes."""
    return km[in_scope_classes]


def save_km(path: Path, tensor: torch.Tensor, in_scope_classes: List[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"km": tensor, "classes": in_scope_classes}, path)


def load_km_slice(path: Path) -> Tuple[torch.Tensor, List[int]]:
    payload = torch.load(path, weights_only=False)
    return payload["km"], payload["classes"]


def load_image(image_path: Path) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    return IMAGENET_TRANSFORM(img)  # (3, 224, 224)


def save_raw_image(src: Path, dst: Path) -> None:
    """Save the 224×224 center-cropped uint8 RGB tensor at `dst` (idempotent)."""
    if dst.exists():
        return
    raw = IMAGENET_TRANSFORM_RAW(Image.open(src).convert("RGB"))
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save((raw * 255).round().clamp(0, 255).to(torch.uint8), dst)


def _build_resnet152(device: str) -> torch.nn.Module:
    from knowledgematrix.models.resnet152 import ResNet152

    return ResNet152(
        input_shape=(3, 224, 224),
        num_classes=1000,
        pretrained=True,
        device=device,
    )


def _build_densenet121(device: str) -> torch.nn.Module:
    # Lazy import: the knowledgematrix wrapper isn't always present locally
    # (cluster-only pin in requirements-slurm.txt). Importing inside the
    # factory means module-load doesn't fail on dev machines that don't
    # have the wrapper installed.
    from knowledgematrix.models.densenet import DenseNet

    return DenseNet(
        input_shape=(3, 224, 224),
        num_classes=1000,
        pretrained=True,
        device=device,
    )


def _build_googlenet(device: str) -> torch.nn.Module:
    # Lazy import: the GoogLeNet (InceptionV1) wrapper is being added in a
    # parallel knowledgematrix branch and may not resolve until the
    # requirements pin is bumped. Keep the import inside the factory so
    # the module still loads on machines without the new wrapper.
    from knowledgematrix.models.googlenet import GoogLeNet

    return GoogLeNet(
        input_shape=(3, 224, 224),
        num_classes=1000,
        pretrained=True,
        device=device,
    )


# Architecture dispatcher: name -> factory(device) -> knowledgematrix NN.
# Each factory must return a knowledgematrix NN already constructed with
# pretrained ImageNet weights and (3, 224, 224) inputs. Pillar 3 launches
# with these three architectures (residual / dense / inception family
# coverage); extend by adding a new factory and dict entry here, plus
# matching entries in compute_baselines.TV_MODEL_FACTORIES,
# compute_baselines.GRADCAM_TARGET_LAYERS, and
# compute_deepdream.DEEPDREAM_LAYER_NAMES.
KM_MODEL_FACTORIES = {
    "resnet152":   _build_resnet152,
    "densenet121": _build_densenet121,
    "googlenet":   _build_googlenet,
}


def build_model(model_name: str, device: str) -> torch.nn.Module:
    """Wrap a pretrained torchvision model in the knowledgematrix NN."""
    if model_name not in KM_MODEL_FACTORIES:
        raise ValueError(
            f"Unknown model {model_name!r}; "
            f"available: {sorted(KM_MODEL_FACTORIES)}"
        )
    model = KM_MODEL_FACTORIES[model_name](device)
    model.to(device)
    # knowledgematrix stores residual projection modules inside plain Python lists
    # under nn.ModuleDict values, so model.to(device) misses them. Move explicitly.
    for entries in getattr(model, "residuals", {}).values():
        for _start, projection in entries:
            for sub in projection:
                sub.to(device)
    model.eval()
    return model


def compute_one(
    entry: Entry,
    model: torch.nn.Module,
    in_scope_classes: List[int],
    batch_size: int,
    device: str,
) -> torch.Tensor:
    from knowledgematrix.matrix_computer import KnowledgeMatrixComputer

    x = load_image(entry.image_path).to(device)
    computer = KnowledgeMatrixComputer(model, batch_size=batch_size, device=device)
    full = computer.forward(x)  # (1000, 150529)
    # fp32 is non-negotiable here: fp16 storage caused inf overflow on
    # ResNet18 KMs and broke the M(x).sum(1) == f(x) completeness invariant.
    return slice_class_rows(full, in_scope_classes).to(torch.float32)


def state_step_name(suffix: Optional[str]) -> str:
    """Build the state-file step name, suffixed by `suffix` if non-empty.

    Lets per-model SLURM array tasks write to disjoint state files
    (`01_compute_kms_<model>.json`) so concurrent tasks don't race on
    `state.mark_completed`'s read-modify-write.
    """
    return "01_compute_kms" + (f"_{suffix}" if suffix else "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-models", nargs="*", default=None,
                        help="Only run these model names (default: all in manifest)")
    parser.add_argument("--state-suffix", default=None,
                        help="Suffix appended to the state-file step name (e.g., 'resnet152')")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    entries = read_manifest(args.manifest)
    if args.limit_models is not None:
        entries = [e for e in entries if e.model in args.limit_models]
    state_file = paths.state_path(state_step_name(args.state_suffix))
    completed = state.load_completed(state_file)
    todo = [e for e in entries if sample_key(e) not in completed]
    logger.info("Completed: %d  Todo: %d", len(completed), len(todo))

    # Group by model so we only build each model once.
    by_model = {}
    for e in todo:
        by_model.setdefault(e.model, []).append(e)

    for model_name, model_entries in by_model.items():
        logger.info("Loading model %s", model_name)
        model = build_model(model_name, args.device)
        for entry in model_entries:
            try:
                save_raw_image(entry.image_path, paths.image_path(entry.class_id, entry.image_id))
                km = compute_one(entry, model, TIER_A_CLASSES, args.batch_size, args.device)
                save_km(
                    paths.km_path(entry.model, entry.class_id, entry.image_id),
                    km,
                    in_scope_classes=TIER_A_CLASSES,
                )
                state.mark_completed(state_file, sample_key(entry))
                logger.info("done %s", sample_key(entry))
            except torch.cuda.OutOfMemoryError as e:
                state.log_error(
                    paths.errors_path(),
                    step=state_step_name(args.state_suffix),
                    sample_id=sample_key(entry),
                    error_type="OOM",
                    message=str(e),
                    tb=state.capture_traceback(),
                )
                torch.cuda.empty_cache()
            except Exception as e:
                state.log_error(
                    paths.errors_path(),
                    step=state_step_name(args.state_suffix),
                    sample_id=sample_key(entry),
                    error_type=type(e).__name__,
                    message=str(e),
                    tb=state.capture_traceback(),
                )
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    sys.exit(main())
