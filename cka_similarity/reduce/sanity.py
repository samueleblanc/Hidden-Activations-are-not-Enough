"""Sanity gating for Phase-1 Step C.

Boolean pass/fail per check; tarball production (Step D) gated on all-pass.
"""
import json
from typing import Dict, List


def check_km_correctness(s3_results: Dict, atol: float = 1e-2, fraction_required: float = 0.99) -> Dict:
    """Verify M(x).sum(1) == f(x) for ≥ fraction_required of computed KMs.

    The S3 worker recorded completeness_residual_clean and completeness_residual_adv
    per pair. We check that the residual is below atol on at least fraction_required
    of pairs across all (arch, attack).
    """
    n_total = 0
    n_pass = 0
    max_residual = 0.0
    for (arch, attack), d in s3_results.items():
        for pair in d["per_pair"]:
            n_total += 2  # one for clean, one for adv
            for key in ["completeness_residual_clean", "completeness_residual_adv"]:
                r = pair[key]
                max_residual = max(max_residual, r)
                if r < atol:
                    n_pass += 1
    fraction = n_pass / n_total if n_total > 0 else 0.0
    return {
        "passed": fraction >= fraction_required,
        "fraction_pass": fraction,
        "max_residual": max_residual,
        "atol": atol,
        "fraction_required": fraction_required,
        "n_total": n_total,
    }


def check_no_nan_in_results(s1_results, s2_results, s3_results) -> Dict:
    """All measure values must be finite (no NaN, no Inf)."""
    import math
    n_nan = 0
    n_total = 0
    for (arch, tid), m in s1_results.items():
        for mname, d in m.items():
            n_total += 1
            if math.isnan(d["value"]) or math.isinf(d["value"]):
                n_nan += 1
    for pname, d in s2_results.items():
        for cat in ["D1", "D2"]:
            for mname, mv in d.get(cat, {}).items():
                n_total += 1
                if math.isnan(mv["value"]) or math.isinf(mv["value"]):
                    n_nan += 1
    for (arch, attack), d in s3_results.items():
        for mname, mv in d.get("panel", {}).items():
            n_total += 1
            if math.isnan(mv["value"]) or math.isinf(mv["value"]):
                n_nan += 1
    return {"passed": n_nan == 0, "n_nan": n_nan, "n_total": n_total}


def check_cui_below_threshold(cui_results: Dict, threshold: float = 0.5) -> Dict:
    """Random-network CKA must be < threshold to claim the input-confound is not dominant."""
    failed = []
    for arch, mvals in cui_results.items():
        cka_val = mvals.get("debiased_cka", float("nan"))
        if cka_val >= threshold:
            failed.append({"arch": arch, "cka": cka_val})
    return {"passed": len(failed) == 0, "failures": failed, "threshold": threshold}


def check_murphy_near_zero(murphy_results: Dict, threshold: float = 0.05) -> Dict:
    """Shuffled-pair CKA must be near zero (debiased estimator sanity)."""
    failed = []
    for key, mvals in murphy_results.items():
        for mname, val in mvals.items():
            if abs(val) > threshold and not (mname == "soft_matching"):  # exclude OT-based metric
                failed.append({"key": key, "measure": mname, "value": val})
    return {"passed": len(failed) == 0, "failures": failed, "threshold": threshold}


def write_sanity_report(out_path: str, s1_results, s2_results, s3_results,
                       cui_results, murphy_results) -> Dict:
    from utils.atomic_io import atomic_json_dump

    checks = {
        "km_correctness": check_km_correctness(s3_results),
        "no_nan": check_no_nan_in_results(s1_results, s2_results, s3_results),
        "cui_random_below_threshold": check_cui_below_threshold(cui_results),
        "murphy_shuffled_near_zero": check_murphy_near_zero(murphy_results),
    }
    checks["all_pass"] = all(c["passed"] for c in checks.values())
    atomic_json_dump(out_path, checks)
    return checks
