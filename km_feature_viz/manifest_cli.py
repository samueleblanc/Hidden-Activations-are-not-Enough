"""CLI wrapper to build and write the manifest."""
import argparse
from pathlib import Path

from km_feature_viz.manifest import (
    TIER_A_CLASSES,
    TIER_A_IMAGES_PER_CLASS,
    TIER_A_MODELS,
    TIER_A_SEED,
    enumerate_entries,
    write_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-val", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    entries = enumerate_entries(
        imagenet_val_dir=args.imagenet_val,
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
