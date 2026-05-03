"""Tests for the DeepDream neuron-selection helpers.

Covers both methodologies plumbed through `compute_deepdream`:
  - `select_neurons_gradcam` (Selvaraju 2017 channel-wise variant) for
    resnet152 / densenet121.
  - `get_googlenet_catalogued_selection` (Olah 2017 / Cammarata 2020 Distill
    Circuits Thread + OpenAI Microscope) for googlenet.

Also exercises `write_neuron_selection` round-trip and the JSON shape contract
the renderer / bundle gate depends on.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from km_feature_viz.compute_deepdream import (
    GOOGLENET_CATALOGUED_NEURONS,
    get_googlenet_catalogued_selection,
    range_check_googlenet_neurons,
    select_neurons_gradcam,
    write_neuron_selection,
)
from km_feature_viz.manifest import Entry


class TinyCNN(nn.Module):
    """Two-layer CNN with named layers `conv1` (4 ch) and `conv2` (8 ch).

    Acts as the model under test for `select_neurons_gradcam` so the test
    runs in milliseconds without any pretrained-weights download.
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _fake_entry(image_path: Path, model: str = "test", class_id: int = 0,
                image_id: str = "img") -> Entry:
    return Entry(model=model, class_id=class_id, image_id=image_id,
                 image_path=image_path)


class TestSelectNeuronsGradcam(unittest.TestCase):

    def setUp(self):
        # `select_neurons_gradcam` calls `load_image(entry.image_path)` which
        # reads from disk via PIL. We patch it module-side to skip I/O and
        # return a deterministic 224×224 tensor.
        self._patch = unittest.mock.patch(
            "km_feature_viz.compute_deepdream.load_image",
            return_value=torch.randn(3, 224, 224),
        )
        self._patch.start()

    def tearDown(self):
        self._patch.stop()

    def test_returns_expected_schema(self):
        torch.manual_seed(0)
        model = TinyCNN()
        entries = [_fake_entry(Path(f"/dev/null/img_{i}")) for i in range(3)]
        result = select_neurons_gradcam(
            model=model,
            layer_names=["conv1", "conv2"],
            manifest_entries=entries,
            device="cpu",
            top_k=2,
        )
        self.assertEqual(result["method"], "gradcam_class_conditional")
        self.assertIn("conv1", result["selected"])
        self.assertIn("conv2", result["selected"])
        # top_k=2 → 2 channels per layer
        self.assertEqual(len(result["selected"]["conv1"]), 2)
        self.assertEqual(len(result["selected"]["conv2"]), 2)

    def test_per_channel_record_has_required_keys(self):
        torch.manual_seed(0)
        model = TinyCNN()
        entries = [_fake_entry(Path(f"/dev/null/img_{i}")) for i in range(2)]
        result = select_neurons_gradcam(
            model=model, layer_names=["conv2"],
            manifest_entries=entries, device="cpu", top_k=3,
        )
        for record in result["selected"]["conv2"]:
            self.assertIn("channel", record)
            self.assertIn("mean_alpha", record)
            self.assertIn("rank", record)
            self.assertIsInstance(record["channel"], int)
            self.assertIsInstance(record["mean_alpha"], float)
            self.assertIsInstance(record["rank"], int)

    def test_ranks_are_dense_ascending(self):
        torch.manual_seed(0)
        model = TinyCNN()
        entries = [_fake_entry(Path(f"/dev/null/img_{i}")) for i in range(2)]
        result = select_neurons_gradcam(
            model=model, layer_names=["conv1"],
            manifest_entries=entries, device="cpu", top_k=4,
        )
        ranks = [r["rank"] for r in result["selected"]["conv1"]]
        self.assertEqual(ranks, [0, 1, 2, 3])

    def test_channels_in_range(self):
        torch.manual_seed(0)
        model = TinyCNN()
        entries = [_fake_entry(Path(f"/dev/null/img_{i}")) for i in range(2)]
        result = select_neurons_gradcam(
            model=model, layer_names=["conv1", "conv2"],
            manifest_entries=entries, device="cpu", top_k=2,
        )
        for ch_record in result["selected"]["conv1"]:
            # conv1 has 4 output channels
            self.assertLess(ch_record["channel"], 4)
        for ch_record in result["selected"]["conv2"]:
            # conv2 has 8 output channels
            self.assertLess(ch_record["channel"], 8)

    def test_mean_alpha_descending(self):
        torch.manual_seed(0)
        model = TinyCNN()
        entries = [_fake_entry(Path(f"/dev/null/img_{i}")) for i in range(3)]
        result = select_neurons_gradcam(
            model=model, layer_names=["conv2"],
            manifest_entries=entries, device="cpu", top_k=4,
        )
        alphas = [r["mean_alpha"] for r in result["selected"]["conv2"]]
        # Selection is by descending |mean_alpha| — ranks 0..k-1 must be
        # sorted in non-increasing order.
        for i in range(len(alphas) - 1):
            self.assertGreaterEqual(alphas[i], alphas[i + 1])


class TestCataloguedSelection(unittest.TestCase):

    def test_method_field(self):
        result = get_googlenet_catalogued_selection()
        self.assertEqual(result["method"], "catalogued_distill")

    def test_three_layers(self):
        result = get_googlenet_catalogued_selection()
        self.assertEqual(set(result["selected"].keys()),
                         {"inception3b", "inception4d", "inception5b"})

    def test_five_channels_per_layer(self):
        result = get_googlenet_catalogued_selection()
        for layer_name, records in result["selected"].items():
            self.assertEqual(len(records), 5,
                             f"layer {layer_name}: expected 5 catalogued channels")

    def test_per_channel_record_has_required_keys(self):
        result = get_googlenet_catalogued_selection()
        for records in result["selected"].values():
            for record in records:
                self.assertIn("channel", record)
                self.assertIn("label", record)
                self.assertIn("citation", record)
                self.assertIn("rank", record)
                self.assertIsInstance(record["channel"], int)
                self.assertIsInstance(record["label"], str)
                self.assertIsInstance(record["citation"], str)
                self.assertIsInstance(record["rank"], int)

    def test_ranks_are_dense_ascending(self):
        result = get_googlenet_catalogued_selection()
        for records in result["selected"].values():
            self.assertEqual([r["rank"] for r in records], list(range(len(records))))

    def test_constant_keys_match_helper(self):
        # The helper must surface every layer present in the constant.
        result = get_googlenet_catalogued_selection()
        self.assertEqual(set(result["selected"].keys()),
                         set(GOOGLENET_CATALOGUED_NEURONS.keys()))


class TestWriteNeuronSelection(unittest.TestCase):

    def test_round_trip(self):
        payload = {
            "method": "catalogued_distill",
            "selected": {
                "inception3b": [
                    {"channel": 101, "label": "edge", "citation": "Cammarata 2020", "rank": 0},
                ],
            },
        }
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "03a_neuron_selection_test.json"
            write_neuron_selection(target, payload)
            self.assertTrue(target.exists())
            with target.open() as f:
                loaded = json.load(f)
            self.assertEqual(loaded, payload)

    def test_creates_parent_dir(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "nested" / "dir" / "selection.json"
            write_neuron_selection(target, {"method": "x", "selected": {}})
            self.assertTrue(target.exists())


class TestRangeCheckGooglenetNeurons(unittest.TestCase):

    def _stub_googlenet_with_channels(self, channel_counts: dict) -> nn.Module:
        """Build a stub nn.Module with attribute names matching
        GOOGLENET_CATALOGUED_NEURONS keys ('inception3b', 'inception4d',
        'inception5b'); each attribute is a small Conv2d whose out_channels
        controls the range check."""
        model = nn.Module()
        for name, ch in channel_counts.items():
            setattr(model, name, nn.Conv2d(3, ch, kernel_size=1))
        # `range_check_googlenet_neurons` runs a dummy forward by hooking each
        # layer and pushing a `(1, 3, 224, 224)` tensor through. The stub
        # model has no `forward`, so we override it to call each named layer
        # in turn — only the hook capture matters for the range check.
        layer_names = list(channel_counts)

        def _forward(x):
            for name in layer_names:
                getattr(model, name)(x)
            return x

        model.forward = _forward
        model.eval = lambda: model  # no-op
        return model

    def test_passes_when_all_channels_in_range(self):
        # Build a stub whose channel counts strictly exceed the largest
        # catalogued index per layer.
        max_per_layer = {
            name: max(ch for (ch, _label, _cite) in entries) + 1
            for name, entries in GOOGLENET_CATALOGUED_NEURONS.items()
        }
        model = self._stub_googlenet_with_channels(max_per_layer)
        # Should not raise
        range_check_googlenet_neurons(model, device="cpu")

    def test_raises_with_out_of_range_channel(self):
        # Force one layer to have FEWER channels than the largest catalogued
        # index → the check must raise and name the offending entry.
        offending_layer = next(iter(GOOGLENET_CATALOGUED_NEURONS))
        small = {
            name: 2 if name == offending_layer else
                  max(ch for (ch, _l, _c) in entries) + 1
            for name, entries in GOOGLENET_CATALOGUED_NEURONS.items()
        }
        model = self._stub_googlenet_with_channels(small)
        with self.assertRaises(ValueError) as ctx:
            range_check_googlenet_neurons(model, device="cpu")
        # Error message must mention the offending layer name so the user can
        # locate the bad catalogue entry.
        self.assertIn(offending_layer, str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
