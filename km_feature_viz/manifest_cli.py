"""CLI wrapper to build and write the manifest.

Defaults `--imagenet-root` to nibi's ImageNet base (`/datashare/imagenet/ILSVRC2012`),
matching the convention used by validate_theorem45.py / teleportation_experiment.py.
The path is auto-detected as ImageFolder (synset subdirs) or flat (ILSVRC2012_val_*.JPEG
+ ground-truth file) by `utils.utils.get_imagenet_val_dataset`.
"""
import argparse
from pathlib import Path

from km_feature_viz.manifest import (
    NIBI_IMAGENET_ROOT,
    TIER_A_CLASSES,
    TIER_A_IMAGES_PER_CLASS,
    TIER_A_MODELS,
    TIER_A_SEED,
    enumerate_entries,
    write_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imagenet-root",
        type=Path,
        default=Path(NIBI_IMAGENET_ROOT),
        help="Base ImageNet path. Default = /datashare/imagenet/ILSVRC2012 (nibi).",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    entries = enumerate_entries(
        imagenet_val_dir=args.imagenet_root,
        classes=TIER_A_CLASSES,
        images_per_class=TIER_A_IMAGES_PER_CLASS,
        seed=TIER_A_SEED,
        models=TIER_A_MODELS,
    )
    write_manifest(args.output, entries)
    print(f"Wrote {len(entries)} entries to {args.output}")
    return 0


if __name__ == "__main__":
    main()
