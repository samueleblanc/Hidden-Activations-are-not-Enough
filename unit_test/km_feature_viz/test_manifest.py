"""Tests for manifest enumeration.

The real loader uses `utils.utils.get_imagenet_val_dataset` to auto-detect
ImageFolder vs flat layout. We patch that loader with a fake (image_paths,
labels) pair so the tests don't need an actual ImageNet dataset on disk.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


class _FakeFlatDataset:
    """Mimic ImageNetVal: has .image_paths and .labels."""
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels


def _make_fake_loader(root: Path, classes: list, images_per_class: int):
    """Build a fake loader that yields (paths, labels) covering `classes`."""
    paths, labels = [], []
    for c in classes:
        for i in range(images_per_class):
            p = root / f"ILSVRC2012_val_{c:04d}_{i:05d}.JPEG"
            p.write_bytes(b"fake")
            paths.append(str(p))
            labels.append(c)

    def fake_loader(data_path, *args, **kwargs):
        return None, _FakeFlatDataset(paths, labels)

    return fake_loader


class TestManifest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.imagenet_root = Path(self.tmp.name)
        # 200 images per class — enough headroom for sampling tests
        self.fake_loader = _make_fake_loader(
            self.imagenet_root, TIER_A_CLASSES, images_per_class=200
        )
        self._patcher = patch(
            "utils.utils.get_imagenet_val_dataset", self.fake_loader
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self.tmp.cleanup()

    def test_enumerate_entries_count(self):
        entries = enumerate_entries(
            imagenet_val_dir=self.imagenet_root,
            classes=TIER_A_CLASSES,
            images_per_class=TIER_A_IMAGES_PER_CLASS,
            seed=TIER_A_SEED,
            models=TIER_A_MODELS,
        )
        # Pillar 3 budget: sum(per-class counts) * len(models) =
        #   sum({207: 7, 282: 7, 340: 6}.values())
        #     * len(["resnet152", "densenet121", "googlenet"]) = 20 * 3 = 60
        expected = sum(TIER_A_IMAGES_PER_CLASS.values()) * len(TIER_A_MODELS)
        self.assertEqual(len(entries), expected)
        self.assertEqual(len(entries), 60)

    def test_enumerate_entries_per_class_counts_respected(self):
        """The dict form must produce exactly the requested per-class count.

        With multiple models, each (class_id, image_id) is repeated once per
        model — but the *unique* image_ids per class must still match the
        requested per-class count.
        """
        entries = enumerate_entries(
            imagenet_val_dir=self.imagenet_root,
            classes=TIER_A_CLASSES,
            images_per_class=TIER_A_IMAGES_PER_CLASS,
            seed=TIER_A_SEED,
            models=TIER_A_MODELS,
        )
        # Group by class — image_ids must be unique within each class because
        # rng.sample (no replacement) is used. Across models, the same image_id
        # appears len(models) times per class, so we deduplicate via set.
        per_class_image_ids: dict = {}
        for e in entries:
            per_class_image_ids.setdefault(e.class_id, set()).add(e.image_id)
        for class_id, expected_count in TIER_A_IMAGES_PER_CLASS.items():
            self.assertEqual(
                len(per_class_image_ids[class_id]),
                expected_count,
                f"class {class_id} should have {expected_count} unique images",
            )

    def test_enumerate_entries_int_form_backwards_compat(self):
        """Passing a plain int still works (legacy uniform form)."""
        entries = enumerate_entries(
            imagenet_val_dir=self.imagenet_root,
            classes=TIER_A_CLASSES,
            images_per_class=5,  # uniform across all 3 classes
            seed=TIER_A_SEED,
            models=TIER_A_MODELS,
        )
        # 3 classes × 5 images × 3 models = 45
        self.assertEqual(len(entries), 3 * 5 * len(TIER_A_MODELS))
        per_class_count: dict = {}
        for e in entries:
            per_class_count[e.class_id] = per_class_count.get(e.class_id, 0) + 1
        # Per-class count is 5 images × len(models) (each image once per model).
        for class_id in TIER_A_CLASSES:
            self.assertEqual(per_class_count[class_id], 5 * len(TIER_A_MODELS))

    def test_enumerate_entries_mapping_missing_class_raises(self):
        """A dict that omits a requested class should raise KeyError."""
        bad_mapping = {207: 5, 282: 5}  # missing 340
        with self.assertRaises(KeyError):
            enumerate_entries(
                imagenet_val_dir=self.imagenet_root,
                classes=TIER_A_CLASSES,
                images_per_class=bad_mapping,
                seed=TIER_A_SEED,
                models=TIER_A_MODELS,
            )

    def test_enumerate_entries_deterministic(self):
        entries1 = enumerate_entries(
            self.imagenet_root, TIER_A_CLASSES, TIER_A_IMAGES_PER_CLASS,
            TIER_A_SEED, TIER_A_MODELS,
        )
        entries2 = enumerate_entries(
            self.imagenet_root, TIER_A_CLASSES, TIER_A_IMAGES_PER_CLASS,
            TIER_A_SEED, TIER_A_MODELS,
        )
        self.assertEqual(entries1, entries2)

    def test_enumerate_entries_seed_changes_selection(self):
        entries1 = enumerate_entries(
            self.imagenet_root, TIER_A_CLASSES, 50, seed=1, models=["resnet152"]
        )
        entries2 = enumerate_entries(
            self.imagenet_root, TIER_A_CLASSES, 50, seed=2, models=["resnet152"]
        )
        ids1 = sorted(e.image_id for e in entries1)
        ids2 = sorted(e.image_id for e in entries2)
        self.assertNotEqual(ids1, ids2)

    def test_entry_image_path_resolves(self):
        entries = enumerate_entries(
            self.imagenet_root, TIER_A_CLASSES[:1], 1, TIER_A_SEED, ["resnet152"]
        )
        self.assertTrue(entries[0].image_path.exists())

    def test_write_and_read_manifest(self):
        entries = enumerate_entries(
            self.imagenet_root, TIER_A_CLASSES[:2], 5, TIER_A_SEED, ["resnet152"]
        )
        manifest_file = Path(self.tmp.name) / "manifest.json"
        write_manifest(manifest_file, entries)
        loaded = read_manifest(manifest_file)
        self.assertEqual(len(loaded), len(entries))
        self.assertEqual(loaded[0].model, entries[0].model)
        self.assertEqual(loaded[0].image_id, entries[0].image_id)

    def test_manifest_hash_stable_under_reorder(self):
        entries_a = enumerate_entries(
            self.imagenet_root, TIER_A_CLASSES[:2], 5, TIER_A_SEED, ["resnet152"]
        )
        entries_b = list(reversed(entries_a))
        self.assertEqual(manifest_hash(entries_a), manifest_hash(entries_b))

    def test_manifest_hash_changes_under_content_change(self):
        entries_a = enumerate_entries(
            self.imagenet_root, TIER_A_CLASSES[:2], 5, TIER_A_SEED, ["resnet152"]
        )
        entries_b = enumerate_entries(
            self.imagenet_root, TIER_A_CLASSES[:2], 5, TIER_A_SEED + 1, ["resnet152"]
        )
        self.assertNotEqual(manifest_hash(entries_a), manifest_hash(entries_b))

    def test_sample_key_format(self):
        e = Entry(
            model="resnet152",
            class_id=207,
            image_id="img_00001",
            image_path=Path("/tmp/img_00001.JPEG"),
        )
        self.assertEqual(sample_key(e), "resnet152/207/img_00001")

    def test_tier_a_constants_pillar3_shape(self):
        """Sanity check the Pillar-3 constants: 3 classes, 3 models, 60 entries total.

        Pillar 3 launches with three architectures spanning the dominant CNN
        family lines (residual / dense / inception). 20 images × 3 models = 60
        manifest entries.
        """
        self.assertEqual(TIER_A_CLASSES, [207, 282, 340])
        self.assertEqual(TIER_A_MODELS, ["resnet152", "densenet121", "googlenet"])
        self.assertEqual(len(TIER_A_MODELS), 3)
        self.assertEqual(TIER_A_IMAGES_PER_CLASS, {207: 7, 282: 7, 340: 6})
        self.assertEqual(sum(TIER_A_IMAGES_PER_CLASS.values()), 20)
        # Total manifest size: 20 images × 3 archs = 60.
        self.assertEqual(
            sum(TIER_A_IMAGES_PER_CLASS.values()) * len(TIER_A_MODELS), 60
        )


if __name__ == "__main__":
    unittest.main()
