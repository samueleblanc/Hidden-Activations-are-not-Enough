#!/usr/bin/env python3
"""Restructure Tiny ImageNet val/ directory for torchvision ImageFolder compatibility.

Tiny ImageNet ships with a flat val/images/ directory and a val_annotations.txt
mapping file. torchvision.datasets.ImageFolder expects class subdirectories.
This script moves each validation image into val/{class_id}/images/{image_name}.

Usage:
    python slurm/restructure_tiny_imagenet_val.py --dir data/tiny-imagenet-200
"""

import argparse
import os
import shutil


def restructure_val(tiny_imagenet_dir: str) -> None:
    """Reorganize val/images/ into val/{class_id}/images/ subdirectories.

    Args:
        tiny_imagenet_dir: Root directory of the Tiny ImageNet dataset.
    """
    val_dir = os.path.join(tiny_imagenet_dir, "val")
    images_dir = os.path.join(val_dir, "images")

    # Already restructured — nothing to do
    if not os.path.isdir(images_dir):
        print(f"Directory {images_dir} does not exist. "
              "Validation set appears already restructured (or not downloaded). Skipping.")
        return

    annotations_path = os.path.join(val_dir, "val_annotations.txt")
    if not os.path.isfile(annotations_path):
        raise FileNotFoundError(
            f"Expected annotation file at {annotations_path} but it does not exist."
        )

    # Parse val_annotations.txt: tab-separated, columns are
    # image_name, class_id, x1, y1, x2, y2
    moved = 0
    with open(annotations_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            image_name = parts[0]
            class_id = parts[1]

            # Create class subdirectory
            class_images_dir = os.path.join(val_dir, class_id, "images")
            os.makedirs(class_images_dir, exist_ok=True)

            # Move image
            src = os.path.join(images_dir, image_name)
            dst = os.path.join(class_images_dir, image_name)
            if os.path.isfile(src):
                shutil.move(src, dst)
                moved += 1

    # Remove the now-empty flat images/ directory
    try:
        os.rmdir(images_dir)
    except OSError:
        remaining = os.listdir(images_dir)
        print(f"Warning: val/images/ not empty after restructure "
              f"({len(remaining)} files remain). Not removing.")

    print(f"Restructured Tiny ImageNet val/: moved {moved} images into class subdirectories.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restructure Tiny ImageNet val/ for ImageFolder compatibility."
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Root directory of Tiny ImageNet (e.g., data/tiny-imagenet-200)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        raise FileNotFoundError(f"Tiny ImageNet directory not found: {args.dir}")

    restructure_val(args.dir)


if __name__ == "__main__":
    main()
