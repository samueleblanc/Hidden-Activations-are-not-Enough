"""Canonical (model, class, image) enumeration for the km-feature-viz experiment.

A single source of truth: every downstream script consumes this manifest
and verifies its hash. Hash mismatch on rsync → notebook refuses to render.

ImageNet layout: uses the same convention as other experiments in this repo
(theorem45, isomorphism, teleportation) — `get_imagenet_val_dataset` from
`utils/utils.py` auto-detects ImageFolder (synset subdirs) vs flat
ILSVRC2012_val_*.JPEG + ground-truth file. Default base path on nibi:
`/datashare/imagenet/ILSVRC2012`.
"""
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

TIER_A_CLASSES = [0, 1, 2, 282, 207, 340, 386, 546, 717, 963]
TIER_A_IMAGES_PER_CLASS = 50
TIER_A_SEED = 20260421
TIER_A_MODELS = ["alexnet", "resnet18", "vgg11"]
NIBI_IMAGENET_ROOT = "/datashare/imagenet/ILSVRC2012"


@dataclass(frozen=True)
class Entry:
    model: str
    class_id: int
    image_id: str
    image_path: Path


def sample_key(entry: Entry) -> str:
    """Stable identifier used as both state key and filename stem."""
    return f"{entry.model}/{entry.class_id}/{entry.image_id}"


def _load_paths_and_labels(imagenet_root: Path) -> Tuple[List[str], List[int]]:
    """Use the repo's existing imagenet val loader; return (image_paths, labels).

    Auto-detects ImageFolder (synset subdirs) vs flat (ILSVRC2012_val_*.JPEG +
    ground-truth file) — same as `utils.utils.get_imagenet_val_dataset`."""
    from utils.utils import get_imagenet_val_dataset
    _, val_set = get_imagenet_val_dataset(str(imagenet_root))
    if hasattr(val_set, "image_paths"):
        # ImageNetVal (flat format)
        paths = list(val_set.image_paths)
        labels = list(val_set.labels)
    else:
        # torchvision.datasets.ImageFolder (synset subdirs)
        paths = [s[0] for s in val_set.samples]
        labels = [s[1] for s in val_set.samples]
    return paths, labels


def enumerate_entries(
    imagenet_val_dir: Path,
    classes: List[int],
    images_per_class: int,
    seed: int,
    models: List[str],
) -> List[Entry]:
    """Build the manifest by selecting `images_per_class` images per class
    deterministically given the seed, and crossing with the model list.

    `imagenet_val_dir` is the BASE imagenet path (e.g.,
    `/datashare/imagenet/ILSVRC2012`), not the val subdir directly — matches
    the convention used by other experiments in this repo.
    """
    entries: List[Entry] = []
    paths, labels = _load_paths_and_labels(imagenet_val_dir)

    # Group image paths by class label (0–999, matching ImageNet class indices).
    by_class: dict = {}
    for path, label in zip(paths, labels):
        by_class.setdefault(int(label), []).append(path)
    for c in by_class:
        by_class[c].sort()  # deterministic ordering before sampling

    rng = random.Random(seed)
    for class_id in classes:
        class_paths = by_class.get(class_id, [])
        if len(class_paths) < images_per_class:
            raise ValueError(
                f"Class {class_id}: requested {images_per_class} images, "
                f"only {len(class_paths)} present (base={imagenet_val_dir})"
            )
        chosen = rng.sample(class_paths, images_per_class)
        for path in chosen:
            image_path = Path(path)
            image_id = image_path.stem
            for model in models:
                entries.append(
                    Entry(
                        model=model,
                        class_id=class_id,
                        image_id=image_id,
                        image_path=image_path,
                    )
                )
    return entries


def write_manifest(path: Path, entries: List[Entry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "model": e.model,
            "class_id": e.class_id,
            "image_id": e.image_id,
            "image_path": str(e.image_path),
        }
        for e in entries
    ]
    with path.open("w") as f:
        json.dump({"hash": manifest_hash(entries), "entries": payload}, f, indent=2)


def read_manifest(path: Path) -> List[Entry]:
    with path.open() as f:
        data = json.load(f)
    return [
        Entry(
            model=d["model"],
            class_id=d["class_id"],
            image_id=d["image_id"],
            image_path=Path(d["image_path"]),
        )
        for d in data["entries"]
    ]


def manifest_hash(entries: List[Entry]) -> str:
    """SHA256 over the sorted (model, class_id, image_id) triples — order-invariant."""
    triples = sorted((e.model, e.class_id, e.image_id) for e in entries)
    blob = json.dumps(triples).encode()
    return hashlib.sha256(blob).hexdigest()
