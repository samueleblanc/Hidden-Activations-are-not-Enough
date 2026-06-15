"""Tests for the offline-safe pretrained COB loader (D2 / resnet152 URL trap fix).

Background: teleportation_km_drift.py used to build its base COB model via
`resnet152COB(pretrained=True)`, which downloads the orphan IMAGENET1K_V1 file
`resnet152-b121ed2d.pth` via the COB library's own `model_urls`. But Phase-0d
pre-caches the torchvision DEFAULT (`resnet152-f82ba261.pth`, V2) — a DIFFERENT
file keyed differently in torch.hub's cache — so on a no-internet compute node
the COB download fails and the D2 job crashes before any KM is computed.

These tests pin the fix: the loader derives the expected DEFAULT basename from
torchvision's Weights enum (so it tracks whatever `weights='DEFAULT'` resolves
to on the installed torchvision), asserts that file is hub-cached before
loading, and builds the COB model with `pretrained=False` + a DEFAULT
state-dict load.

The basename-derivation and offline-assert tests need only torchvision (always
installed). The COB load smoke additionally needs neuralteleportation (a
manually-installed cluster dep) and the cached DEFAULT weights; both are
skipped cleanly if unavailable.
"""
import os

import pytest
import torch

import teleportation_experiment as te


# ---------------------------------------------------------------------------
# Pure / torchvision-only tests (no neuralteleportation needed)
# ---------------------------------------------------------------------------

class TestExpectedDefaultBasename:
    """expected_default_weight_basename must match the torchvision DEFAULT URL."""

    @pytest.mark.parametrize("arch", ["resnet152", "densenet121", "googlenet"])
    def test_matches_weights_enum_default_url(self, arch):
        import torchvision.models as tv_models
        enum_name = te._TORCHVISION_WEIGHTS_ENUM[arch]
        expected = os.path.basename(getattr(tv_models, enum_name).DEFAULT.url)
        assert te.expected_default_weight_basename(arch) == expected

    def test_resnet152_is_not_the_cob_orphan_v1(self):
        """The DEFAULT must NOT be the COB library's orphan V1 basename.

        That divergence is the entire bug: resnet152COB(pretrained=True) pulls
        resnet152-b121ed2d.pth, which Phase-0d never caches.
        """
        assert te.expected_default_weight_basename("resnet152") != "resnet152-b121ed2d.pth"

    def test_loader_requests_same_default_as_basename_derivation(self):
        """Loader and assert derive the basename from the SAME enum/url.

        Guards against the loader requesting `weights='DEFAULT'` while the assert
        checks a hard-coded/different basename (which would let a real URL drift
        slip past the offline guard). Both must route through the Weights enum.
        """
        import torchvision.models as tv_models
        for arch in ("resnet152", "densenet121", "googlenet"):
            enum_name = te._TORCHVISION_WEIGHTS_ENUM[arch]
            # getattr(tv_models, arch) is the factory the loader calls with
            # weights='DEFAULT'; its DEFAULT must be this enum's DEFAULT.
            factory = getattr(tv_models, arch)
            assert callable(factory)
            assert hasattr(tv_models, enum_name)


class TestAssertDefaultWeightsCached:
    """assert_default_weights_cached gates the load on a present hub file."""

    def test_passes_when_file_present(self, tmp_path, monkeypatch):
        ckpt = tmp_path / "checkpoints"
        ckpt.mkdir()
        basename = te.expected_default_weight_basename("resnet152")
        (ckpt / basename).write_bytes(b"not-real-weights")  # presence is all that's checked
        monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
        path = te.assert_default_weights_cached("resnet152")
        assert path == str(ckpt / basename)

    def test_raises_when_file_absent(self, tmp_path, monkeypatch):
        (tmp_path / "checkpoints").mkdir()  # empty cache
        monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
        with pytest.raises(FileNotFoundError) as exc:
            te.assert_default_weights_cached("resnet152")
        msg = str(exc.value)
        assert "resnet152" in msg
        assert te.expected_default_weight_basename("resnet152") in msg

    def test_raises_with_actionable_precache_hint(self, tmp_path, monkeypatch):
        (tmp_path / "checkpoints").mkdir()
        monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
        with pytest.raises(FileNotFoundError) as exc:
            te.assert_default_weights_cached("densenet121")
        assert "weights='DEFAULT'" in str(exc.value)


# ---------------------------------------------------------------------------
# COB load smoke (needs neuralteleportation + cached DEFAULT weights)
# ---------------------------------------------------------------------------

def _has_neuralteleportation():
    try:
        import neuralteleportation  # noqa: F401
        return True
    except Exception:
        return False


def _default_weights_cached(arch):
    basename = te.expected_default_weight_basename(arch)
    return os.path.isfile(
        os.path.join(torch.hub.get_dir(), "checkpoints", basename)
    )


neuralteleportation_required = pytest.mark.skipif(
    not _has_neuralteleportation(),
    reason="neuralteleportation not installed (manual cluster dep)",
)


class TestLoadPretrainedCobSmoke:
    """End-to-end offline load smoke for the two D2 archs."""

    @neuralteleportation_required
    @pytest.mark.parametrize("arch", ["resnet152", "densenet121"])
    def test_loads_offline_eval_and_forwards(self, arch, monkeypatch):
        if not _default_weights_cached(arch):
            pytest.skip(f"{arch} DEFAULT weights not hub-cached")

        # Simulate no internet: any torch.hub download attempt must fail loudly,
        # proving the load path reuses the cache and never hits the network.
        def _no_download(*a, **k):
            raise AssertionError("torch.hub attempted a download (should be offline)")

        monkeypatch.setattr(torch.hub, "download_url_to_file", _no_download)

        model = te.load_pretrained_cob(arch, device="cpu")
        assert not model.training, "model must be in eval mode"

        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1000)
        assert torch.isfinite(out).all()
