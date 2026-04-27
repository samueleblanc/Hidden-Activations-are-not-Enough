"""Unit tests for the --state-suffix arg added to compute_kms and compute_deepdream.

We test the helper that resolves the step name; running the full main() requires
loading a model and is exercised by the SLURM scripts at runtime.
"""
import pytest

from km_feature_viz.compute_kms import state_step_name as kms_step_name

try:
    from km_feature_viz.compute_deepdream import state_step_name as deepdream_step_name
    HAS_DEEPDREAM_HELPER = True
except ImportError:
    HAS_DEEPDREAM_HELPER = False


def test_kms_step_name_no_suffix_keeps_legacy_name():
    assert kms_step_name(None) == "01_compute_kms"
    assert kms_step_name("") == "01_compute_kms"


def test_kms_step_name_with_suffix_appends_underscore():
    assert kms_step_name("resnet152") == "01_compute_kms_resnet152"
    # Forward-looking: when DenseNet/Inception arrive, the same suffix
    # contract must hold — these checks document the API for future
    # architectures without coupling to weight availability.
    assert kms_step_name("densenet201") == "01_compute_kms_densenet201"


@pytest.mark.skipif(not HAS_DEEPDREAM_HELPER, reason="Task 2 not yet implemented")
def test_deepdream_step_name_no_suffix_keeps_legacy_name():
    assert deepdream_step_name(None) == "03_deepdream"
    assert deepdream_step_name("") == "03_deepdream"


@pytest.mark.skipif(not HAS_DEEPDREAM_HELPER, reason="Task 2 not yet implemented")
def test_deepdream_step_name_with_suffix_appends_underscore():
    assert deepdream_step_name("resnet152") == "03_deepdream_resnet152"
    assert deepdream_step_name("inception_v3") == "03_deepdream_inception_v3"
