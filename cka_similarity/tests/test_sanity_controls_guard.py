"""Guard tests for the Phase-1 sanity gate's controls-ran check.

Regression cover for the silent-controls bug: if the Cui/Murphy controls
crash at scale (missing GPU/val/weights), `_try_compute_controls` returns
EMPTY dicts and the threshold checks iterate over nothing → trivially pass.
The gate must FAIL when controls were supposed to run but are absent, while
still passing when the operator deliberately passed --skip_controls.
"""
import json

from cka_similarity.reduce.sanity import (
    write_sanity_report, check_km_correctness, check_murphy_near_zero,
    check_cui_below_threshold,
)


# --- Minimal fake sanity inputs (no cluster data needed) -------------------
#
# Shapes mirror what aggregate_s{1,2,3} produce and what the controls return:
#   s1: {(arch, teleport_id): {measure_name: {"value": float}}}
#   s2: {pair_name: {"D1": {measure: {"value": float}}, "D2": {...}}}
#   s3: {(arch, attack): {"per_pair": [...], "panel": {measure: {"value": float}}}}
#   cui: {arch: {measure_name: float}}  (must include "debiased_cka")
#   murphy: {arch: {measure_name: float}}


def _good_s1():
    return {("resnet152", 0): {"debiased_cka": {"value": 0.99}}}


def _good_s2():
    return {"resnet152|densenet121": {"D1": {"debiased_cka": {"value": 0.4}},
                                      "D2": {"debiased_cka": {"value": 0.3}}}}


def _good_s3():
    # One (arch, attack) with a single pair whose completeness residuals are
    # well below atol, so check_km_correctness passes; finite panel value.
    return {
        ("resnet152", "fgsm"): {
            "per_pair": [
                {"completeness_residual_clean": 1e-6,
                 "completeness_residual_adv": 1e-6},
            ],
            "panel": {"debiased_cka": {"value": 0.5}},
        }
    }


def _good_cui():
    return {"resnet152": {"debiased_cka": 0.1}}


def _good_murphy():
    return {"resnet152": {"debiased_cka": 0.01}}


def _read(out_path):
    with open(out_path) as f:
        return json.load(f)


# --- (a) controls present + within thresholds + skip_controls=False --------
def test_controls_present_within_thresholds_passes(tmp_path):
    out = tmp_path / "sanity_report.json"
    report = write_sanity_report(
        str(out), _good_s1(), _good_s2(), _good_s3(),
        _good_cui(), _good_murphy(), skip_controls=False,
    )
    assert report["all_pass"] is True
    # The new guard ran and confirms controls are present.
    assert report["controls_ran"]["passed"] is True
    assert report["controls_ran"]["controls_present"] is True
    assert report["controls_ran"].get("skipped", False) is False
    # Original four checks intact.
    for name in ("km_correctness", "no_nan",
                 "cui_random_below_threshold", "murphy_shuffled_near_zero"):
        assert report[name]["passed"] is True


# --- (b) controls EMPTY + skip_controls=False → THE bug being fixed --------
def test_silent_control_failure_fails_gate(tmp_path):
    out = tmp_path / "sanity_report.json"
    report = write_sanity_report(
        str(out), _good_s1(), _good_s2(), _good_s3(),
        {}, {}, skip_controls=False,
    )
    # Vacuous pass must NOT happen: the gate fails.
    assert report["all_pass"] is False
    # The failure is attributed to the new guard, not a content check.
    assert report["controls_ran"]["passed"] is False
    assert report["controls_ran"]["controls_present"] is False
    assert "controls_error" in report["controls_ran"]
    # The threshold checks still "pass" vacuously — proving they cannot be the
    # signal, which is exactly why the guard is needed.
    assert report["cui_random_below_threshold"]["passed"] is True
    assert report["murphy_shuffled_near_zero"]["passed"] is True
    # Diagnostic is persisted to disk for an operator inspecting the file.
    on_disk = _read(out)
    assert on_disk["all_pass"] is False
    assert on_disk["controls_ran"]["controls_present"] is False


def test_partial_control_failure_fails_gate(tmp_path):
    """Only one of cui/murphy missing is still a silent failure."""
    out = tmp_path / "sanity_report.json"
    # cui present, murphy empty
    report = write_sanity_report(
        str(out), _good_s1(), _good_s2(), _good_s3(),
        _good_cui(), {}, skip_controls=False,
    )
    assert report["all_pass"] is False
    assert report["controls_ran"]["passed"] is False
    assert report["controls_ran"]["controls_present"] is False


# --- (c) skip_controls=True + controls empty → recorded as skipped ---------
def test_deliberate_skip_does_not_fail_gate(tmp_path):
    out = tmp_path / "sanity_report.json"
    report = write_sanity_report(
        str(out), _good_s1(), _good_s2(), _good_s3(),
        {}, {}, skip_controls=True,
    )
    # Operator opted out; absence is legitimate → other checks decide all_pass.
    assert report["all_pass"] is True
    assert report["controls_ran"]["passed"] is True
    assert report["controls_ran"]["skipped"] is True
    assert report["controls_ran"]["controls_present"] is False


# --- (d) genuine control-threshold failure still fails (regression) --------
def test_genuine_cui_threshold_failure_still_fails(tmp_path):
    out = tmp_path / "sanity_report.json"
    bad_cui = {"resnet152": {"debiased_cka": 0.9}}  # >= 0.5 threshold
    report = write_sanity_report(
        str(out), _good_s1(), _good_s2(), _good_s3(),
        bad_cui, _good_murphy(), skip_controls=False,
    )
    assert report["all_pass"] is False
    # Content check is what fails; the presence guard passes (controls ran).
    assert report["cui_random_below_threshold"]["passed"] is False
    assert report["controls_ran"]["passed"] is True
    assert report["controls_ran"]["controls_present"] is True


def test_genuine_murphy_threshold_failure_still_fails(tmp_path):
    out = tmp_path / "sanity_report.json"
    bad_murphy = {"resnet152": {"debiased_cka": 0.9}}  # > 0.05 threshold
    report = write_sanity_report(
        str(out), _good_s1(), _good_s2(), _good_s3(),
        _good_cui(), bad_murphy, skip_controls=False,
    )
    assert report["all_pass"] is False
    assert report["murphy_shuffled_near_zero"]["passed"] is False
    assert report["controls_ran"]["passed"] is True


# --- Backwards-compatibility: skip_controls defaults to False --------------
def test_skip_controls_defaults_false_empty_controls_fail(tmp_path):
    """Calling without the new kwarg must still catch silent failures."""
    out = tmp_path / "sanity_report.json"
    report = write_sanity_report(
        str(out), _good_s1(), _good_s2(), _good_s3(), {}, {},
    )
    assert report["all_pass"] is False
    assert report["controls_ran"]["passed"] is False


# --- km_correctness atol = float32 completeness ceiling --------------------
def _s3_with_residuals(values):
    """One (arch, attack) cell whose pairs carry the given (clean, adv) residuals."""
    return {("resnet152", "deepfool"): {
        "per_pair": [{"completeness_residual_clean": c,
                      "completeness_residual_adv": a} for c, a in values],
        "panel": {"debiased_cka": {"value": 0.5}},
    }}


def test_km_correctness_passes_at_float32_ceiling():
    """Residuals at the documented fp32 ceiling (~0.15-0.30 on ResNet-152) must
    PASS the gate (atol=0.35) — they are physically correct float32 KMs, not a
    wiring bug."""
    # 100 pairs all near the observed fp32 max (0.295) -> below 0.35 -> pass.
    r = check_km_correctness(_s3_with_residuals([(0.29, 0.295)] * 100))
    assert r["passed"] is True
    assert r["max_residual"] <= 0.35


def test_km_correctness_fails_on_logit_scale_residual():
    """A real defect yields O(10) (logit-scale) residuals — the gate must still
    FAIL on those, i.e. atol=0.35 keeps its teeth."""
    # 99 fine pairs + enough broken (residual ~8.0) to drop below fraction 0.99.
    vals = [(0.01, 0.01)] * 90 + [(8.0, 8.0)] * 10
    r = check_km_correctness(_s3_with_residuals(vals))
    assert r["passed"] is False
    assert r["max_residual"] >= 1.0


# --- Murphy shuffled-pair: only the zero-null estimators are checked ---------
def test_murphy_ignores_distance_measures():
    """Distance/dissimilarity measures have non-zero shuffled-pair nulls (angular
    CKA -> pi/2, Procrustes/soft-matching large, Bures floor, output-JSD nonzero)
    and must NOT trip the Murphy control; only debiased_cka/dcor/rsa are checked."""
    murphy = {"resnet152": {
        "debiased_cka": 2.4e-5, "dcor": 0.0, "rsa": -0.006,        # zero-null: pass
        "angular_cka": 1.5708, "procrustes": 484.0, "bures": 0.28,
        "output_jsd": 0.43, "soft_matching": 17.3, "gw": 0.0,      # distances: ignored
    }}
    r = check_murphy_near_zero(murphy)
    assert r["passed"] is True
    assert r["measures_checked"] == ["dcor", "debiased_cka", "rsa"]


def test_murphy_fails_on_high_debiased_cka():
    """A genuinely high debiased CKA under shuffling (estimator bug) must FAIL."""
    r = check_murphy_near_zero({"resnet152": {"debiased_cka": 0.9, "dcor": 0.0, "rsa": 0.0}})
    assert r["passed"] is False


# --- Cui control: non-finite values fail (no vacuous NaN pass) ---------------
def test_cui_nan_fails():
    """A non-finite Cui CKA (e.g. a deep untrained net exploding in eval-mode BN)
    must FAIL, not pass vacuously via NaN >= threshold == False."""
    assert check_cui_below_threshold({"resnet152": {"debiased_cka": float("nan")}})["passed"] is False
    assert check_cui_below_threshold({"resnet152": {"debiased_cka": float("inf")}})["passed"] is False
    # A finite value below threshold still passes (the normal case).
    assert check_cui_below_threshold({"resnet152": {"debiased_cka": 0.11}})["passed"] is True
