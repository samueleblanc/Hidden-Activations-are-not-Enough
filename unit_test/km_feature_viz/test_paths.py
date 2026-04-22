"""Tests for storage layout helpers."""
import unittest
from pathlib import Path

from km_feature_viz.paths import (
    km_path,
    baseline_path,
    deepdream_path,
    dictionary_path,
    counterfactual_path,
    jacobian_path,
    state_path,
    errors_path,
    manifest_path,
    figure_path,
    RESULTS_ROOT,
)


class TestPaths(unittest.TestCase):

    def test_km_path(self):
        p = km_path("resnet18", class_id=207, image_id="ILSVRC2012_val_00000123")
        self.assertEqual(
            p,
            RESULTS_ROOT / "kms" / "resnet18" / "207" / "ILSVRC2012_val_00000123.pt",
        )

    def test_baseline_path(self):
        p = baseline_path("gradcam", "alexnet", class_id=282, image_id="img1")
        self.assertEqual(
            p,
            RESULTS_ROOT / "baselines" / "gradcam" / "alexnet" / "282" / "img1.pt",
        )

    def test_deepdream_path(self):
        p = deepdream_path("vgg11", layer_name="features.8", neuron=42)
        self.assertEqual(
            p,
            RESULTS_ROOT / "deepdream" / "vgg11" / "features.8" / "neuron_0042.pt",
        )

    def test_dictionary_path(self):
        p = dictionary_path("resnet18", class_id=340, kind="components")
        self.assertEqual(
            p,
            RESULTS_ROOT / "formulations" / "dictionary" / "resnet18" / "340" / "components.pt",
        )

    def test_counterfactual_path(self):
        p = counterfactual_path("alexnet", class_id=282, image_id="img1", target=207)
        self.assertEqual(
            p,
            RESULTS_ROOT / "formulations" / "counterfactual" / "alexnet" / "282" / "img1__to_207.json",
        )

    def test_jacobian_path(self):
        p = jacobian_path("vgg11", class_id=386, image_id="img2")
        self.assertEqual(
            p,
            RESULTS_ROOT / "formulations" / "jacobian" / "vgg11" / "386" / "img2.pt",
        )

    def test_state_path(self):
        p = state_path("01_compute_kms")
        self.assertEqual(p, RESULTS_ROOT / "state" / "01_compute_kms.json")

    def test_errors_path(self):
        self.assertEqual(errors_path(), RESULTS_ROOT / "errors.json")

    def test_manifest_path(self):
        self.assertEqual(manifest_path(), RESULTS_ROOT / "manifest.json")

    def test_figure_path(self):
        self.assertEqual(
            figure_path("cell_a_attribution_vs_feature_maps"),
            Path("docs/km-feature-viz/figures") / "cell_a_attribution_vs_feature_maps.pdf",
        )


if __name__ == "__main__":
    unittest.main()
