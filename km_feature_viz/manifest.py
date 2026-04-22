"""Canonical (model, class, image) enumeration for the km-feature-viz experiment.

A single source of truth: every downstream script consumes this manifest
and verifies its hash. Hash mismatch on rsync → notebook refuses to render.
"""
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

TIER_A_CLASSES = [0, 1, 2, 282, 207, 340, 386, 546, 717, 963]
TIER_A_IMAGES_PER_CLASS = 50
TIER_A_SEED = 20260421
TIER_A_MODELS = ["alexnet", "resnet18", "vgg11"]


@dataclass(frozen=True)
class Entry:
    model: str
    class_id: int
    image_id: str
    image_path: Path


def sample_key(entry: Entry) -> str:
    """Stable identifier used as both state key and filename stem."""
    return f"{entry.model}/{entry.class_id}/{entry.image_id}"


def _list_class_images(class_dir: Path) -> List[str]:
    """Return sorted image filenames (without extension) in a class subdir."""
    if not class_dir.exists():
        raise FileNotFoundError(f"Class directory missing: {class_dir}")
    images = sorted(p.stem for p in class_dir.iterdir() if p.suffix.lower() in (".jpeg", ".jpg", ".png"))
    if not images:
        raise FileNotFoundError(f"No images found in {class_dir}")
    return images


def enumerate_entries(
    imagenet_val_dir: Path,
    classes: List[int],
    images_per_class: int,
    seed: int,
    models: List[str],
) -> List[Entry]:
    """Build the manifest by selecting `images_per_class` images per class
    deterministically given the seed, and crossing with the model list."""
    entries: List[Entry] = []
    rng = random.Random(seed)
    for class_id in classes:
        class_dir = imagenet_val_dir / str(class_id)
        all_imgs = _list_class_images(class_dir)
        if len(all_imgs) < images_per_class:
            raise ValueError(
                f"Class {class_id}: requested {images_per_class} images, "
                f"only {len(all_imgs)} present in {class_dir}"
            )
        chosen = rng.sample(all_imgs, images_per_class)
        for image_id in chosen:
            ext = next(
                p.suffix
                for p in class_dir.iterdir()
                if p.stem == image_id and p.suffix.lower() in (".jpeg", ".jpg", ".png")
            )
            image_path = class_dir / f"{image_id}{ext}"
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
