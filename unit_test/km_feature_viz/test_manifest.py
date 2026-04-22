"""Tests for manifest enumeration."""
import json
import tempfile
import unittest
from pathlib import Path

from km_feature_viz.manifest import (
    TIER_A_CLASSES,
    TIER_A_IMAGES_PER_CLASS,
    TIER_A_MODELS,
    TIER_A_SEED,
    Entry,
    enumerate_entries,
    write_manifest,
    read_manifest,
    manifest_hash,
    sample_key,
)


class FakeImageNetDir:
    """Stand-in for an ImageNet val directory: per-class subdirs of jpegs."""

    def __init__(self, root: Path, classes: list, images_per_class: int):
        self.root = root
        for c in classes:
            class_dir = root / str(c)
            class_dir.mkdir(parents=True)
            for i in range(images_per_class):
                (class_dir / f"img_{i:05d}.JPEG").write_bytes(b"fake")


class TestManifest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.imagenet_root = Path(self.tmp.name)
        FakeImageNetDir(self.imagenet_root, TIER_A_CLASSES, images_per_class=200)

    def tearDown(self):
        self.tmp.cleanup()

    def test_enumerate_entries_count(self):
        entries = enumerate_entries(
            imagenet_val_dir=self.imagenet_root,
            classes=TIER_A_CLASSES,
            images_per_class=TIER_A_IMAGES_PER_CLASS,
            seed=TIER_A_SEED,
            models=TIER_A_MODELS,
        )
        # 10 classes * 50 images * 3 models = 1500
        self.assertEqual(len(entries), 1500)

    def test_enumerate_entries_deterministic(self):
        entries1 = enumerate_entries(
            self.imagenet_root, TIER_A_CLASSES, 50, TIER_A_SEED, TIER_A_MODELS
        )
        entries2 = enumerate_entries(
            self.imagenet_root, TIER_A_CLASSES, 50, TIER_A_SEED, TIER_A_MODELS
        )
        self.assertEqual(entries1, entries2)

    def test_enumerate_entries_seed_changes_selection(self):
        entries1 = enumerate_entries(
            self.imagenet_root, TIER_A_CLASSES, 50, seed=1, models=["alexnet"]
        )
        entries2 = enumerate_entries(
            self.imagenet_root, TIER_A_CLASSES, 50, seed=2, models=["alexnet"]
        )
        ids1 = sorted(e.image_id for e in entries1)
        ids2 = sorted(e.image_id for e in entries2)
        self.assertNotEqual(ids1, ids2)

    def test_entry_image_path_resolves(self):
        entries = enumerate_entries(
            self.imagenet_root, TIER_A_CLASSES[:1], 1, TIER_A_SEED, ["alexnet"]
        )
        self.assertTrue(entries[0].image_path.exists())

    def test_write_and_read_manifest(self):
        entries = enumerate_entries(
            self.imagenet_root, TIER_A_CLASSES[:2], 5, TIER_A_SEED, ["alexnet"]
        )
        manifest_file = Path(self.tmp.name) / "manifest.json"
        write_manifest(manifest_file, entries)
        loaded = read_manifest(manifest_file)
        self.assertEqual(len(loaded), len(entries))
        self.assertEqual(loaded[0].model, entries[0].model)
        self.assertEqual(loaded[0].image_id, entries[0].image_id)

    def test_manifest_hash_stable_under_reorder(self):
        entries_a = enumerate_entries(
            self.imagenet_root, TIER_A_CLASSES[:2], 5, TIER_A_SEED, ["alexnet"]
        )
        entries_b = list(reversed(entries_a))
        self.assertEqual(manifest_hash(entries_a), manifest_hash(entries_b))

    def test_manifest_hash_changes_under_content_change(self):
        entries_a = enumerate_entries(
            self.imagenet_root, TIER_A_CLASSES[:2], 5, TIER_A_SEED, ["alexnet"]
        )
        entries_b = enumerate_entries(
            self.imagenet_root, TIER_A_CLASSES[:2], 5, TIER_A_SEED + 1, ["alexnet"]
        )
        self.assertNotEqual(manifest_hash(entries_a), manifest_hash(entries_b))

    def test_sample_key_format(self):
        e = Entry(
            model="resnet18",
            class_id=207,
            image_id="img_00001",
            image_path=Path("/tmp/img_00001.JPEG"),
        )
        self.assertEqual(sample_key(e), "resnet18/207/img_00001")


if __name__ == "__main__":
    unittest.main()
