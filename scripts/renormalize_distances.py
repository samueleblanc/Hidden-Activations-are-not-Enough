"""Post-hoc RMS-per-coordinate renormalization for distance comparisons.

The codebase computes raw L2/Frobenius distances on logits, penultimate
features, and knowledge matrices — three spaces of very different ambient
dimension. Cross-space ratios (gamma, amplification factors) inherit a
sqrt(numel-ratio) confound from dimension alone.

This script walks the on-disk result tree, computes the corresponding
RMS-per-coordinate quantities ALONGSIDE the raw values, and writes them
back under a top-level `rms` key. Raw fields are never modified.

Idempotent: skip files that already have a `rms` block (override with --force).

Coverage:
  1. experiments/{arch}_{dataset}/theorem45/per_attack/*.json
  2. experiments/{arch}_{dataset}/theorem45/theorem45_results.json
  3. experiments/{arch}_{dataset}/theorem45/theorem45_checkpoint.json
  4. results/cross_model/**/*.json     (per_sample arrays available)
  5. results/cka_similarity/s2/**/*.json  (per-pair KM Frobenius lists)
  6. results/cka_similarity/s3/**/*.json  (per-pair {d_f, d_h, d_M} dicts)

Pillar 1 (teleportation_experiment.py) is already in RMS units — skipped.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from math import sqrt
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.scaling import (  # noqa: E402
    IMAGENET_INPUT_NUMEL,
    IMAGENET_NUM_CLASSES,
    CIFAR_INPUT_NUMEL,
    penultimate_dim,
    km_numel,
    rescale_amp_M_to_rms,
    rescale_amp_h_to_rms,
    rescale_gamma_to_rms,
)


# ----------------------------------------------------------------------------
# Experiment metadata parsing
# ----------------------------------------------------------------------------

_EXPERIMENT_PATTERN = re.compile(r"^([a-z0-9_]+?)_(imagenet|cifar10|cifar100|mnist)$")


def parse_experiment_name(name: str) -> tuple[str, str]:
    """Split 'resnet152_imagenet' → ('resnet152', 'imagenet')."""
    m = _EXPERIMENT_PATTERN.match(name.lower())
    if not m:
        raise ValueError(f"Cannot parse experiment name {name!r}")
    return m.group(1), m.group(2)


def dataset_dims(dataset: str) -> tuple[int, int]:
    """(num_classes, input_numel) for a known dataset."""
    if dataset == "imagenet":
        return IMAGENET_NUM_CLASSES, IMAGENET_INPUT_NUMEL
    if dataset == "cifar10":
        return 10, CIFAR_INPUT_NUMEL
    if dataset == "cifar100":
        return 100, CIFAR_INPUT_NUMEL
    if dataset == "mnist":
        return 10, 28 * 28
    raise ValueError(f"Unknown dataset {dataset!r}")


# ----------------------------------------------------------------------------
# Pillar-2 (validate_theorem45) per-attack rescaling
# ----------------------------------------------------------------------------

def rescale_stats(stats: dict, scale: float) -> dict:
    """Multiply every entry in a {mean,std,min,max,median} dict by `scale`.

    Std is scale-equivariant under affine x → c*x because std uses centered
    moments; min/max/median are order-statistics; mean is linear — so all
    five are correctly rescaled by uniform multiplication.
    """
    return {k: float(v) * scale for k, v in stats.items()}


def rescale_iqr(iqr: list, scale: float) -> list:
    """Rescale a [q25, q75] pair. Percentiles are order-statistics → linear."""
    return [float(iqr[0]) * scale, float(iqr[1]) * scale]


def rescale_theorem45_result(
    result: dict, arch: str, num_classes: int, input_numel: int
) -> dict:
    """Build the `rms` sub-block for a per-attack theorem45 result dict.

    Scale factors:
      d_f:   1 / sqrt(C)
      d_h:   1 / sqrt(D_penult)
      d_M:   1 / sqrt(C * (d+1))
      amp_M: 1 / sqrt(d+1)          (i.e., (d_M/d_f)_RMS = (d_M/d_f)_raw / sqrt(d+1))
      amp_h: sqrt(C / D_penult)
      gamma: same as amp_M
    """
    D = penultimate_dim(arch)
    s_f = 1.0 / sqrt(num_classes)
    s_h = 1.0 / sqrt(D)
    s_M = 1.0 / sqrt(km_numel(num_classes, input_numel))
    rms = {}
    if "d_f_stats" in result:
        rms["d_f_stats"] = rescale_stats(result["d_f_stats"], s_f)
    if "d_h_stats" in result:
        rms["d_h_stats"] = rescale_stats(result["d_h_stats"], s_h)
    if "d_M_stats" in result:
        rms["d_M_stats"] = rescale_stats(result["d_M_stats"], s_M)
    if "gamma_empirical" in result:
        rms["gamma_empirical"] = rescale_gamma_to_rms(
            result["gamma_empirical"], input_numel
        )
    if "gamma_ci_95" in result:
        gci = result["gamma_ci_95"]
        rms["gamma_ci_95"] = [
            rescale_gamma_to_rms(gci[0], input_numel),
            rescale_gamma_to_rms(gci[1], input_numel),
        ]
    for k in (
        "amplification_M_median",
        "amplification_M_mean",
        "amplification_M_std",
    ):
        if k in result:
            rms[k] = rescale_amp_M_to_rms(result[k], input_numel)
    if "amplification_M_iqr" in result:
        rms["amplification_M_iqr"] = [
            rescale_amp_M_to_rms(result["amplification_M_iqr"][0], input_numel),
            rescale_amp_M_to_rms(result["amplification_M_iqr"][1], input_numel),
        ]
    for k in (
        "amplification_h_median",
        "amplification_h_mean",
        "amplification_h_std",
    ):
        if k in result:
            rms[k] = rescale_amp_h_to_rms(result[k], num_classes, D)
    if "amplification_h_iqr" in result:
        rms["amplification_h_iqr"] = [
            rescale_amp_h_to_rms(result["amplification_h_iqr"][0], num_classes, D),
            rescale_amp_h_to_rms(result["amplification_h_iqr"][1], num_classes, D),
        ]
    rms["_scale_factors"] = {
        "s_f": s_f, "s_h": s_h, "s_M": s_M,
        "amp_M": 1.0 / sqrt(input_numel + 1),
        "amp_h": sqrt(num_classes / D),
        "num_classes": num_classes,
        "penult_dim": D,
        "input_numel": input_numel,
    }
    return rms


def process_theorem45_per_attack(path: Path, force: bool) -> bool:
    data = json.loads(path.read_text())
    exp_name = data.get("experiment")
    if not exp_name:
        return False
    if "rms" in data.get("result", {}) and not force:
        return False
    arch, dataset = parse_experiment_name(exp_name)
    num_classes, input_numel = dataset_dims(dataset)
    data["result"]["rms"] = rescale_theorem45_result(
        data["result"], arch, num_classes, input_numel
    )
    path.write_text(json.dumps(data, indent=2))
    return True


def process_theorem45_results(path: Path, force: bool) -> bool:
    """theorem45_results.json contains {'experiment', 'num_samples', 'per_attack'}.

    per_attack is a dict {attack_name: result_dict}. We rescale each.
    """
    data = json.loads(path.read_text())
    exp_name = data.get("experiment")
    if not exp_name:
        return False
    per_attack = data.get("per_attack", {})
    if not per_attack:
        return False
    arch, dataset = parse_experiment_name(exp_name)
    num_classes, input_numel = dataset_dims(dataset)
    changed = False
    for attack, result in per_attack.items():
        if "rms" in result and not force:
            continue
        result["rms"] = rescale_theorem45_result(
            result, arch, num_classes, input_numel
        )
        changed = True
    # Aggregate block, if present
    agg = data.get("aggregate")
    if agg and ("rms" not in agg or force):
        rms_agg = {}
        if "gamma_global" in agg:
            rms_agg["gamma_global"] = rescale_gamma_to_rms(
                agg["gamma_global"], input_numel
            )
        if "mean_amplification_M" in agg:
            rms_agg["mean_amplification_M"] = rescale_amp_M_to_rms(
                agg["mean_amplification_M"], input_numel
            )
        if "mean_amplification_h" in agg:
            rms_agg["mean_amplification_h"] = rescale_amp_h_to_rms(
                agg["mean_amplification_h"], num_classes, penultimate_dim(arch)
            )
        if "ratio_M_over_h" in agg and agg["ratio_M_over_h"] is not None:
            # Under RMS: ratio_M_over_h scales by amp_M_factor / amp_h_factor
            amp_M_factor = 1.0 / sqrt(input_numel + 1)
            amp_h_factor = sqrt(num_classes / penultimate_dim(arch))
            rms_agg["ratio_M_over_h"] = float(agg["ratio_M_over_h"]) * (
                amp_M_factor / amp_h_factor
            )
        agg["rms"] = rms_agg
        changed = True
    if changed:
        path.write_text(json.dumps(data, indent=2))
    return changed


# ----------------------------------------------------------------------------
# Cross-model experiment (Pillar 3a) — per_sample arrays available
# ----------------------------------------------------------------------------

def process_cross_model(path: Path, force: bool) -> bool:
    data = json.loads(path.read_text())
    if "rms" in data and not force:
        return False
    arch = data.get("arch")
    if not arch:
        return False
    # cross_model is always ImageNet — assert via aggregate keys
    num_classes, input_numel = IMAGENET_NUM_CLASSES, IMAGENET_INPUT_NUMEL
    D = penultimate_dim(arch)
    per_sample = data.get("per_sample", {})
    d_KM = np.asarray(per_sample.get("d_KM", []), dtype=float)
    d_h = np.asarray(per_sample.get("d_h", []), dtype=float)
    d_logit = np.asarray(per_sample.get("d_logit", []), dtype=float)
    if d_KM.size == 0 or d_h.size == 0 or d_logit.size == 0:
        return False
    s_KM = 1.0 / sqrt(km_numel(num_classes, input_numel))
    s_h = 1.0 / sqrt(D)
    s_f = 1.0 / sqrt(num_classes)
    d_KM_rms = d_KM * s_KM
    d_h_rms = d_h * s_h
    d_f_rms = d_logit * s_f
    valid = d_f_rms > 1e-12
    if valid.sum() > 0:
        ratio = d_KM_rms[valid] / d_f_rms[valid]
        gamma_cross_rms = float(ratio.min())
        rng = np.random.default_rng(42)
        boot = [
            float(ratio[rng.integers(0, len(ratio), len(ratio))].min())
            for _ in range(1000)
        ]
        gamma_ci_rms = [
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)),
        ]
    else:
        gamma_cross_rms = float("nan")
        gamma_ci_rms = [float("nan"), float("nan")]
    data["rms"] = {
        "per_sample": {
            "d_KM":    d_KM_rms.tolist(),
            "d_h":     d_h_rms.tolist(),
            "d_logit": d_f_rms.tolist(),
        },
        "aggregate": {
            "d_KM_mean":    float(d_KM_rms.mean()),
            "d_KM_std":     float(d_KM_rms.std()),
            "d_h_mean":     float(d_h_rms.mean()),
            "d_h_std":      float(d_h_rms.std()),
            "d_logit_mean": float(d_f_rms.mean()),
            "d_logit_std":  float(d_f_rms.std()),
            "gamma_cross":  gamma_cross_rms,
            "gamma_cross_ci_95": gamma_ci_rms,
        },
        "_scale_factors": {
            "s_KM": s_KM, "s_h": s_h, "s_f": s_f,
            "arch": arch, "num_classes": num_classes,
            "penult_dim": D, "input_numel": input_numel,
        },
    }
    path.write_text(json.dumps(data, indent=2))
    return True


# ----------------------------------------------------------------------------
# CKA-similarity S2 / S3 — per-pair JSON lists / dicts
# ----------------------------------------------------------------------------

def process_cka_s2(path: Path, force: bool) -> bool:
    """S2 cross-arch KM Frobenius distance lists.

    Files are `results/cka_similarity/s2/.../<a>_<b>_km_chunk*.json` or
    similar; they hold a JSON list of floats (one per pair) — raw Frobenius.
    Write a sibling file `<...>_rms.json` with the rescaled list.
    """
    if "_rms" in path.stem:
        return False
    sibling = path.with_stem(path.stem + "_rms")
    if sibling.exists() and not force:
        return False
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        return False
    s_KM = 1.0 / sqrt(km_numel(IMAGENET_NUM_CLASSES, IMAGENET_INPUT_NUMEL))
    rms = [float(v) * s_KM for v in raw]
    sibling.write_text(json.dumps(rms))
    return True


def process_cka_s3(path: Path, force: bool) -> bool:
    """S3 distance amplification per-pair files.

    Each entry has {pair_idx, d_f, d_h, d_M, completeness_residual_*}.
    Augment each entry with d_f_rms, d_h_rms, d_M_rms.

    Penultimate dim must be inferred from the filename pattern
    `{arch}_{attack}_chunk*.json`.
    """
    m = re.match(r"^([a-z0-9]+)_[A-Za-z0-9]+_(?:panel_)?chunk\d+\.json$", path.name)
    if not m:
        return False
    arch = m.group(1)
    if arch not in {"resnet152", "densenet121", "googlenet"}:
        return False
    raw = json.loads(path.read_text())
    if not isinstance(raw, list) or not raw:
        return False
    if "d_f_rms" in raw[0] and not force:
        return False
    D = penultimate_dim(arch)
    s_f = 1.0 / sqrt(IMAGENET_NUM_CLASSES)
    s_h = 1.0 / sqrt(D)
    s_M = 1.0 / sqrt(km_numel(IMAGENET_NUM_CLASSES, IMAGENET_INPUT_NUMEL))
    for entry in raw:
        if "d_f" in entry:
            entry["d_f_rms"] = float(entry["d_f"]) * s_f
        if "d_h" in entry:
            entry["d_h_rms"] = float(entry["d_h"]) * s_h
        if "d_M" in entry:
            entry["d_M_rms"] = float(entry["d_M"]) * s_M
    path.write_text(json.dumps(raw))
    return True


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def walk_and_process(root: Path, force: bool, dry_run: bool) -> dict:
    summary = {"theorem45": 0, "cross_model": 0, "cka_s2": 0, "cka_s3": 0, "skipped": 0}

    # Pillar 2: per-attack
    for path in sorted(root.glob("experiments/*/theorem45/per_attack/*.json")):
        if dry_run:
            print(f"[dry-run] would process theorem45/per_attack: {path}")
            continue
        try:
            if process_theorem45_per_attack(path, force):
                summary["theorem45"] += 1
                print(f"  + theorem45/per_attack: {path.relative_to(root)}")
            else:
                summary["skipped"] += 1
        except KeyError as e:
            print(f"  ! skipped {path.relative_to(root)} — unknown arch {e}",
                  file=sys.stderr)
        except Exception as e:
            print(f"  ! failed: {path}: {e}", file=sys.stderr)

    # Pillar 2: combined / checkpoint
    for pattern in (
        "experiments/*/theorem45/theorem45_results.json",
        "experiments/*/theorem45/theorem45_checkpoint.json",
    ):
        for path in sorted(root.glob(pattern)):
            if dry_run:
                print(f"[dry-run] would process theorem45 combined: {path}")
                continue
            try:
                if process_theorem45_results(path, force):
                    summary["theorem45"] += 1
                    print(f"  + theorem45 combined: {path.relative_to(root)}")
                else:
                    summary["skipped"] += 1
            except KeyError as e:
                print(f"  ! skipped {path.relative_to(root)} — unknown arch {e}",
                      file=sys.stderr)
            except Exception as e:
                print(f"  ! failed: {path}: {e}", file=sys.stderr)

    # Pillar 3a: cross-model
    for path in sorted(root.glob("results/cross_model/**/*.json")):
        if dry_run:
            print(f"[dry-run] would process cross_model: {path}")
            continue
        try:
            if process_cross_model(path, force):
                summary["cross_model"] += 1
                print(f"  + cross_model: {path.relative_to(root)}")
            else:
                summary["skipped"] += 1
        except Exception as e:
            print(f"  ! failed: {path}: {e}", file=sys.stderr)

    # Pillar 3b: CKA S2 (cross-arch KM Frobenius lists)
    for path in sorted(root.glob("results/cka_similarity/**/s2/**/*.json")):
        if dry_run:
            print(f"[dry-run] would process cka s2: {path}")
            continue
        try:
            if process_cka_s2(path, force):
                summary["cka_s2"] += 1
                print(f"  + cka_s2: {path.relative_to(root)}")
            else:
                summary["skipped"] += 1
        except Exception as e:
            print(f"  ! failed: {path}: {e}", file=sys.stderr)

    # Pillar 3b: CKA S3 (per-pair d_f/d_h/d_M)
    for path in sorted(root.glob("results/cka_similarity/**/s3/**/*.json")):
        if dry_run:
            print(f"[dry-run] would process cka s3: {path}")
            continue
        try:
            if process_cka_s3(path, force):
                summary["cka_s3"] += 1
                print(f"  + cka_s3: {path.relative_to(root)}")
            else:
                summary["skipped"] += 1
        except Exception as e:
            print(f"  ! failed: {path}: {e}", file=sys.stderr)

    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(REPO_ROOT),
                    help="Repository root (default: repo containing this script).")
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if a `rms` block already exists.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List files that would be processed; don't modify anything.")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    print(f"Scanning {root} for distance result files...")
    summary = walk_and_process(root, force=args.force, dry_run=args.dry_run)
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k:>12}: {v}")


if __name__ == "__main__":
    main()
