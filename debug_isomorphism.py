"""
Diagnostic script for isomorphism invariance of knowledge matrices.

Tests whether the matrix_change observed in isomorphism_experiment.py
is caused by float32 precision accumulation or a logical bug.

Diagnostics performed:
  1. Compute knowledge matrices for original and permuted models
  2. Report absolute and RELATIVE error (||M_orig - M_perm|| / ||M_orig||)
  3. Repeat in float64 — if error drops to ~1e-10, it's precision
  4. Test multiple architectures to isolate ResNet-specific issues

Usage:
    python debug_isomorphism.py --experiment resnet_imagenet
    python debug_isomorphism.py --experiment resnet_imagenet --num_samples 3
    python debug_isomorphism.py --experiment alexnet_imagenet
"""

import torch
import torch.nn as nn
import numpy as np
import copy
import time
from argparse import ArgumentParser

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer

from utils.utils import (
    get_dataset, get_input_shape, get_num_classes,
    get_device, subset, get_architecture, _move_residuals_to_device,
)
from constants.constants import DEFAULT_EXPERIMENTS
from isomorphism_experiment import permute_network


def compute_matrix_with_norms(model, sample, device, batch_size_mc=1800):
    """Compute knowledge matrix and return it with its Frobenius norm."""
    mc = KnowledgeMatrixComputer(model, batch_size=batch_size_mc, device=device)
    mat = mc.forward(sample)
    return mat


def run_diagnostic(experiment_name, num_samples=5, seed=42,
                   matrix_batch_size=1800):
    """Run isomorphism diagnostic for one experiment."""
    exp_config = DEFAULT_EXPERIMENTS[experiment_name]
    dataset = exp_config['dataset']
    arch_idx = exp_config['architecture_index']
    input_shape = get_input_shape(dataset)
    num_classes = get_num_classes(dataset)
    device = get_device()

    # Load model
    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC: {experiment_name}")
    print(f"{'='*70}")

    if exp_config.get('pretrained', False) and exp_config['epochs'] == 0:
        print("Using pretrained torchvision weights", flush=True)
        model = get_architecture(
            architecture_index=arch_idx, input_shape=input_shape,
            num_classes=num_classes, pretrained=True,
            freeze_features=False,
        ).to(device)
        _move_residuals_to_device(model, device)
    else:
        raise NotImplementedError("Non-pretrained models not supported in diagnostic")

    model.eval()

    # Load data (fall back to random data if dataset unavailable)
    try:
        _, test_set = get_dataset(dataset, data_loader=False)
        test_data, _ = subset(test_set, num_samples, input_shape)
    except (FileNotFoundError, OSError) as e:
        print(f"  Dataset not available ({e}), using random data.", flush=True)
        test_data = torch.randn(num_samples, *input_shape)

    # Create permuted model
    permuted_model, perm = permute_network(model, seed=seed)
    permuted_model.to(device)
    _move_residuals_to_device(permuted_model, device)
    permuted_model.eval()

    # --- Test 1: float32 (default) ---
    print(f"\n--- Float32 test ({num_samples} samples) ---")
    _run_comparison(model, permuted_model, test_data, device,
                    matrix_batch_size, dtype_label="float32")

    # --- Test 2: float64 ---
    print(f"\n--- Float64 test ({num_samples} samples) ---")
    model_f64 = copy.deepcopy(model).double().to(device)
    _move_residuals_to_device(model_f64, device)
    model_f64.eval()

    perm_f64 = copy.deepcopy(permuted_model).double().to(device)
    _move_residuals_to_device(perm_f64, device)
    perm_f64.eval()

    test_data_f64 = test_data.double()

    _run_comparison(model_f64, perm_f64, test_data_f64, device,
                    matrix_batch_size, dtype_label="float64")

    # Cleanup
    del model_f64, perm_f64, test_data_f64
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Test 3: Verify outputs match ---
    print(f"\n--- Output equivalence check ---")
    with torch.no_grad():
        for i in range(min(num_samples, 3)):
            sample = test_data[i:i+1].to(device).float()
            out_orig = model(sample)
            out_perm = permuted_model(sample)
            max_diff = (out_orig - out_perm).abs().max().item()
            preds_match = out_orig.argmax() == out_perm.argmax()
            print(f"  Sample {i}: max_output_diff={max_diff:.2e}, "
                  f"preds_match={preds_match}")


def _run_comparison(model_orig, model_perm, data, device, batch_size_mc,
                    dtype_label="float32"):
    """Compare knowledge matrices for original vs permuted model."""
    n = len(data)
    diffs = []
    norms_orig = []
    norms_perm = []
    rel_errors = []

    for i in range(n):
        sample = data[i].to(device)

        mat_orig = compute_matrix_with_norms(model_orig, sample, device,
                                              batch_size_mc)
        mat_perm = compute_matrix_with_norms(model_perm, sample, device,
                                              batch_size_mc)

        norm_orig = torch.linalg.norm(mat_orig.double()).item()
        norm_perm = torch.linalg.norm(mat_perm.double()).item()
        diff = torch.linalg.norm((mat_orig.double() - mat_perm.double())).item()
        rel = diff / norm_orig if norm_orig > 0 else float('inf')

        diffs.append(diff)
        norms_orig.append(norm_orig)
        norms_perm.append(norm_perm)
        rel_errors.append(rel)

        print(f"  Sample {i}: ||M_orig||={norm_orig:.4e}  "
              f"||M_perm||={norm_perm:.4e}  "
              f"||diff||={diff:.4e}  "
              f"rel_err={rel:.4e}")

        del mat_orig, mat_perm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n  [{dtype_label}] Summary:")
    print(f"    Mean ||M_orig||:  {np.mean(norms_orig):.4e}")
    print(f"    Mean ||diff||:    {np.mean(diffs):.4e}")
    print(f"    Max  ||diff||:    {np.max(diffs):.4e}")
    print(f"    Mean rel_error:   {np.mean(rel_errors):.4e}")
    print(f"    Max  rel_error:   {np.max(rel_errors):.4e}")

    if np.mean(rel_errors) < 1e-5:
        print(f"    → PASS: relative error consistent with {dtype_label} precision")
    else:
        print(f"    → FAIL: relative error too large for {dtype_label} precision")


def parse_args():
    parser = ArgumentParser(
        description="Diagnostic for isomorphism invariance of knowledge matrices."
    )
    parser.add_argument(
        "--experiment", type=str, default="resnet_imagenet",
        help="Experiment name."
    )
    parser.add_argument(
        "--num_samples", type=int, default=5,
        help="Number of samples to test (default: 5, enough for diagnostic)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for permutation."
    )
    parser.add_argument(
        "--matrix_batch_size", type=int, default=1800,
        help="Batch size for KnowledgeMatrixComputer."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("Isomorphism Diagnostic")
    print(f"  Experiment: {args.experiment}")
    print(f"  Samples:    {args.num_samples}")
    print(f"  Seed:       {args.seed}")

    t0 = time.perf_counter()
    run_diagnostic(
        experiment_name=args.experiment,
        num_samples=args.num_samples,
        seed=args.seed,
        matrix_batch_size=args.matrix_batch_size,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nTotal diagnostic time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
