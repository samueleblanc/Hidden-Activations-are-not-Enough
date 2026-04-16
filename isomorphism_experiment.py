"""
Isomorphism invariance demonstration.

Validates the paper's core theoretical claim that knowledge matrices are
invariant under neuron permutations (isomorphisms) while penultimate-layer
activations are not.

The experiment:
  1. Loads a trained network (specified by --experiment).
  2. Applies random neuron permutations within the penultimate layer,
     creating an isomorphic network that computes the same function.
  3. Shows that:
     - The permuted network produces IDENTICAL outputs (same predictions).
     - Penultimate-layer activations CHANGE after permutation.
     - Knowledge matrices remain the SAME (up to numerical precision).

Usage:
    python isomorphism_experiment.py --experiment alexnet_cifar10
    python isomorphism_experiment.py --experiment alexnet_cifar10 --num_permutations 5 --num_samples 500
    python isomorphism_experiment.py --experiment alexnet_cifar10 --temp_dir $SLURM_TMPDIR
"""

import os
import copy
import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer

from utils.utils import (
    get_model, get_dataset, get_input_shape, get_num_classes,
    get_device, subset, get_architecture, _move_residuals_to_device,
)
from constants.constants import DEFAULT_EXPERIMENTS
from utils.features import extract_penultimate_features


# ---------------------------------------------------------------------------
# Network permutation utilities
# ---------------------------------------------------------------------------

def find_linear_layer_indices(model):
    """Find indices of all nn.Linear layers in model.layers.

    Returns:
        list of int: indices into model.layers that are nn.Linear.
    """
    indices = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, nn.Linear):
            indices.append(i)
    return indices


def find_penultimate_linear_pair(model):
    """Find the last two nn.Linear layers in model.layers.

    These form the pair (second-to-last Linear, last Linear) where
    permuting the output neurons of the second-to-last and the input
    neurons of the last preserves the network function but changes
    the penultimate representation.

    Returns:
        (int, int): indices of (penultimate Linear, final Linear) in
        model.layers, or None if fewer than 2 Linear layers exist.
    """
    linear_indices = find_linear_layer_indices(model)
    if len(linear_indices) < 2:
        return None
    return linear_indices[-2], linear_indices[-1]


def permute_penultimate_layer(model, seed=42):
    """Permute neurons of the penultimate layer only.

    Find the last two Linear layers. Permute:
      - Output neurons (rows of weight, bias) of second-to-last Linear
      - Input features (columns of weight) of last Linear
    This changes penultimate activations but preserves model output.

    Also permutes any BatchNorm1d/LayerNorm between the two layers if
    present (not typical for these architectures, but handled for safety).

    Args:
        model: a knowledgematrix NN model (with .layers attribute).
        seed: random seed for reproducibility.

    Returns:
        permuted_model: a deep copy with permuted weights.
        perm: the permutation tensor used (for diagnostics).
    """
    permuted = copy.deepcopy(model)
    pair = find_penultimate_linear_pair(permuted)
    if pair is None:
        raise ValueError(
            "Cannot find two Linear layers in model.layers. "
            "This architecture may not support penultimate permutation."
        )
    penult_idx, final_idx = pair

    penult_layer = permuted.layers[penult_idx]
    final_layer = permuted.layers[final_idx]

    n_neurons = penult_layer.out_features
    assert final_layer.in_features == n_neurons, (
        f"Dimension mismatch: penultimate out_features={n_neurons}, "
        f"final in_features={final_layer.in_features}"
    )

    # Generate random permutation
    rng = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(n_neurons, generator=rng)

    # Permute output neurons of the penultimate Linear layer
    with torch.no_grad():
        penult_layer.weight.data = penult_layer.weight.data[perm]
        if penult_layer.bias is not None:
            penult_layer.bias.data = penult_layer.bias.data[perm]

        # Permute input neurons of the final Linear layer
        final_layer.weight.data = final_layer.weight.data[:, perm]

    return permuted, perm


def permute_conv_penultimate(model, seed=42):
    """Permute the channel group before the final Linear layer for
    architectures with only one Linear layer (e.g., ResNet18).

    For ResNet18, the structure ends with:
        ... -> [block4: Conv2d/BN layers with 512 channels] ->
        AdaptiveAvgPool2d -> Flatten -> Linear(512, num_classes)

    We permute ALL layers in the final channel group (e.g., all 512-ch
    Conv2d and BN layers in ResNet18's block 4), including residual
    projection layers, and the corresponding input features of the
    final Linear.  This correctly handles skip connections: both the
    main pathway and the residual pathway are permuted, so their
    addition remains consistent.

    Args:
        model: a knowledgematrix NN model (with .layers and .residuals).
        seed: random seed.

    Returns:
        permuted_model: deep copy with permuted weights.
        perm: the permutation tensor used.
    """
    permuted = copy.deepcopy(model)

    linear_indices = find_linear_layer_indices(permuted)
    if not linear_indices:
        raise ValueError("No Linear layers found in model.")

    final_linear_idx = linear_indices[-1]
    final_linear = permuted.layers[final_linear_idx]

    # Determine channel count from the final Linear's input features.
    # After AdaptiveAvgPool2d → Flatten, in_features = n_channels * spatial_size.
    # For ResNet18: pool output is (512, 1, 1), so spatial_size = 1.
    # We detect n_channels from the last Conv2d before the Linear.
    last_conv_idx = None
    for i in range(final_linear_idx - 1, -1, -1):
        if isinstance(permuted.layers[i], nn.Conv2d):
            last_conv_idx = i
            break
    if last_conv_idx is None:
        raise ValueError("No Conv2d layer found before the final Linear.")

    n_channels = permuted.layers[last_conv_idx].out_channels

    rng = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(n_channels, generator=rng)

    with torch.no_grad():
        # --- 1. Walk backward through model.layers, permuting the channel group ---
        for i in range(final_linear_idx - 1, -1, -1):
            layer = permuted.layers[i]

            if isinstance(layer, nn.Conv2d):
                if (layer.out_channels == n_channels
                        and layer.in_channels == n_channels):
                    # Fully inside the channel group: permute both
                    layer.weight.data = layer.weight.data[perm][:, perm]
                    if layer.bias is not None:
                        layer.bias.data = layer.bias.data[perm]
                elif layer.out_channels == n_channels:
                    # Boundary (e.g., Conv2d(256→512)): permute output only
                    layer.weight.data = layer.weight.data[perm]
                    if layer.bias is not None:
                        layer.bias.data = layer.bias.data[perm]
                    break  # reached the boundary of the channel group
                else:
                    break  # exited the channel group

            elif isinstance(layer, nn.BatchNorm2d):
                if layer.num_features == n_channels:
                    layer.weight.data = layer.weight.data[perm]
                    layer.bias.data = layer.bias.data[perm]
                    layer.running_mean.data = layer.running_mean.data[perm]
                    layer.running_var.data = layer.running_var.data[perm]
                else:
                    break  # exited the channel group

            # Skip non-parametric layers (ReLU, Flatten, Pool, etc.)

        # --- 2. Permute residual projection layers in the channel group ---
        if hasattr(permuted, 'residuals'):
            for connections in permuted.residuals.values():
                for _, proj_layers in connections:
                    for proj_layer in proj_layers:
                        if isinstance(proj_layer, nn.Conv2d):
                            if proj_layer.out_channels == n_channels:
                                proj_layer.weight.data = (
                                    proj_layer.weight.data[perm])
                                if proj_layer.bias is not None:
                                    proj_layer.bias.data = (
                                        proj_layer.bias.data[perm])
                        elif isinstance(proj_layer, nn.BatchNorm2d):
                            if proj_layer.num_features == n_channels:
                                proj_layer.weight.data = (
                                    proj_layer.weight.data[perm])
                                proj_layer.bias.data = (
                                    proj_layer.bias.data[perm])
                                proj_layer.running_mean.data = (
                                    proj_layer.running_mean.data[perm])
                                proj_layer.running_var.data = (
                                    proj_layer.running_var.data[perm])

        # --- 3. Permute final Linear input features ---
        in_features = final_linear.in_features
        spatial_size = in_features // n_channels
        assert in_features == n_channels * spatial_size, (
            f"in_features={in_features} not divisible by n_channels={n_channels}"
        )

        if spatial_size == 1:
            final_linear.weight.data = final_linear.weight.data[:, perm]
        else:
            full_perm = torch.zeros(in_features, dtype=torch.long)
            for new_pos, old_pos in enumerate(perm):
                src_start = old_pos * spatial_size
                dst_start = new_pos * spatial_size
                full_perm[dst_start:dst_start + spatial_size] = torch.arange(
                    src_start, src_start + spatial_size
                )
            final_linear.weight.data = final_linear.weight.data[:, full_perm]

    return permuted, perm


def permute_network(model, seed=42):
    """Apply a neuron permutation to create an isomorphic network.

    Automatically selects the right permutation strategy based on the
    number of Linear layers:
      - >= 2 Linear layers: permute the penultimate Linear pair.
      - 1 Linear layer: permute the last Conv2d channels.

    Args:
        model: a knowledgematrix NN model.
        seed: random seed.

    Returns:
        permuted_model: deep copy with permuted weights.
        perm: the permutation tensor used.
    """
    linear_indices = find_linear_layer_indices(model)
    if len(linear_indices) >= 2:
        return permute_penultimate_layer(model, seed=seed)
    elif len(linear_indices) == 1:
        return permute_conv_penultimate(model, seed=seed)
    else:
        raise ValueError("No Linear layers found in model.")


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def verify_output_equivalence(model_orig, model_perm, data, device,
                              batch_size=64):
    """Verify that original and permuted models produce identical outputs.

    Returns:
        dict with 'max_diff', 'mean_diff', 'all_match' (predictions agree).
    """
    model_orig.eval()
    model_perm.eval()
    max_diff = 0.0
    total_diff = 0.0
    n_samples = 0
    all_match = True

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size].to(device).float()
            out_orig = model_orig(batch)
            out_perm = model_perm(batch)
            diff = (out_orig - out_perm).abs()
            max_diff = max(max_diff, diff.max().item())
            total_diff += diff.sum().item()
            n_samples += diff.numel()
            # Check prediction agreement
            if not torch.equal(out_orig.argmax(dim=-1),
                               out_perm.argmax(dim=-1)):
                all_match = False

    return {
        'max_diff': float(max_diff),
        'mean_diff': float(total_diff / max(n_samples, 1)),
        'all_predictions_match': all_match,
    }


def compute_activation_distances(model_orig, model_perm, data, device,
                                 batch_size=64):
    """Compute L2 distances between penultimate features of original and
    permuted models.

    Returns:
        dict with 'mean_l2', 'max_l2', 'min_l2', 'all_distances'.
    """
    feats_orig = extract_penultimate_features(model_orig, data,
                                              batch_size=batch_size)
    feats_perm = extract_penultimate_features(model_perm, data,
                                              batch_size=batch_size)

    # Per-sample L2 distance
    diffs = np.linalg.norm(feats_orig - feats_perm, axis=1)
    return {
        'mean_l2': float(np.mean(diffs)),
        'max_l2': float(np.max(diffs)),
        'min_l2': float(np.min(diffs)),
        'std_l2': float(np.std(diffs)),
        'features_shape': list(feats_orig.shape),
    }


def compute_matrix_distances(model_orig, model_perm, data, device,
                             num_samples=50, batch_size_mc=1800,
                             debug=False):
    """Compute L2 distances between knowledge matrices of original and
    permuted models for a subset of samples.

    Knowledge matrices should be IDENTICAL for isomorphic networks.

    Args:
        model_orig, model_perm: the two models.
        data: tensor of input samples (N, C, H, W).
        device: torch device.
        num_samples: how many samples to check (matrix computation is expensive).
        batch_size_mc: batch_size for KnowledgeMatrixComputer.
        debug: if True, print matrix norms and relative errors.

    Returns:
        dict with 'mean_l2', 'max_l2', 'min_l2', distances list.
    """
    model_orig.eval()
    model_perm.eval()

    mc_orig = KnowledgeMatrixComputer(model_orig, batch_size=batch_size_mc,
                                      device=device)
    mc_perm = KnowledgeMatrixComputer(model_perm, batch_size=batch_size_mc,
                                      device=device)

    n = min(num_samples, len(data))
    distances = []
    norms_orig = []
    rel_errors = []

    for i in range(n):
        # KnowledgeMatrixComputer expects 3D input (C, H, W)
        sample = data[i].to(device).float()
        mat_orig = mc_orig.forward(sample)
        mat_perm = mc_perm.forward(sample)

        norm_orig = torch.linalg.norm(mat_orig.double()).item()
        dist = torch.linalg.norm(
            (mat_orig.double() - mat_perm.double())).item()
        rel = dist / norm_orig if norm_orig > 0 else float('inf')
        distances.append(dist)
        norms_orig.append(norm_orig)
        rel_errors.append(rel)

        if (i + 1) % 10 == 0 or i == n - 1:
            if debug:
                print(f"    Matrix {i + 1}/{n}: ||M||={norm_orig:.4e}  "
                      f"||diff||={dist:.4e}  rel={rel:.4e}", flush=True)
            else:
                print(f"    Matrix comparison {i + 1}/{n}: "
                      f"L2 dist = {dist:.6e}", flush=True)

        del mat_orig, mat_perm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    distances = np.array(distances)
    result = {
        'mean_l2': float(np.mean(distances)),
        'max_l2': float(np.max(distances)),
        'min_l2': float(np.min(distances)),
        'std_l2': float(np.std(distances)),
        'n_samples': n,
        'distances': [float(d) for d in distances],
    }

    if debug:
        result['mean_norm_orig'] = float(np.mean(norms_orig))
        result['mean_relative_error'] = float(np.mean(rel_errors))
        result['max_relative_error'] = float(np.max(rel_errors))
        print(f"    DEBUG: mean ||M_orig||={np.mean(norms_orig):.4e}  "
              f"mean_rel_err={np.mean(rel_errors):.4e}", flush=True)

    return result


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_isomorphism_experiment(experiment_name, num_permutations=5,
                               num_samples=500, temp_dir=None,
                               num_matrix_samples=50,
                               matrix_batch_size=1800, debug=False):
    """Run the full isomorphism invariance experiment.

    Args:
        experiment_name: key into DEFAULT_EXPERIMENTS.
        num_permutations: number of random permutations to try.
        num_samples: number of data samples for feature/detection experiments.
        temp_dir: optional temp directory (cluster).
        num_matrix_samples: samples for matrix comparison (expensive).
        matrix_batch_size: batch_size for KnowledgeMatrixComputer.

    Returns:
        dict: full results structure.
    """
    exp_config = DEFAULT_EXPERIMENTS[experiment_name]
    dataset = exp_config['dataset']
    arch_idx = exp_config['architecture_index']
    epoch = exp_config['epochs']

    input_shape = get_input_shape(dataset)
    num_classes = get_num_classes(dataset)
    device = get_device()

    base = (f'{temp_dir}/experiments/{experiment_name}'
            if temp_dir else f'experiments/{experiment_name}')

    # Load model
    if exp_config.get('pretrained', False) and epoch == 0:
        print("Using pretrained torchvision weights (no local weight file)",
              flush=True)
        model = get_architecture(
            architecture_index=arch_idx, input_shape=input_shape,
            num_classes=num_classes, pretrained=True,
            freeze_features=False,
        ).to(device)
        _move_residuals_to_device(model, device)
    else:
        weights_dir = Path(base) / 'weights'
        weights_path = None
        for candidate_epoch in [epoch, epoch - 1]:
            candidate = weights_dir / f'epoch_{candidate_epoch}.pth'
            if candidate.exists():
                weights_path = candidate
                break
        if weights_path is None:
            epoch_files = sorted(weights_dir.glob('epoch_*.pth'),
                                 key=lambda p: int(p.stem.split('_')[1]))
            if epoch_files:
                weights_path = epoch_files[-1]
            else:
                raise FileNotFoundError(
                    f"No weights found in {weights_dir}"
                )
        print(f"Using weights: {weights_path}", flush=True)
        model = get_model(weights_path, arch_idx, input_shape, num_classes,
                          device)
    model.eval()

    # Load data
    print("Loading datasets...", flush=True)
    train_set, test_set = get_dataset(dataset, data_loader=False,
                                      data_path=temp_dir)
    train_data, train_labels = subset(train_set, num_samples, input_shape)
    test_data, test_labels = subset(test_set,
                                    min(num_samples, 2000), input_shape)

    # -----------------------------------------------------------------------
    # Per-permutation experiments
    # -----------------------------------------------------------------------
    all_permutation_results = []

    for perm_idx in range(num_permutations):
        seed = 42 + perm_idx
        print(f"\n{'=' * 70}", flush=True)
        print(f"  PERMUTATION {perm_idx + 1}/{num_permutations} (seed={seed})",
              flush=True)
        print(f"{'=' * 70}", flush=True)

        t0 = time.perf_counter()

        # Create permuted model
        permuted_model, perm = permute_network(model, seed=seed)
        permuted_model.to(device)
        _move_residuals_to_device(permuted_model, device)
        permuted_model.eval()

        # 1. Verify output equivalence
        print("  Verifying output equivalence...", flush=True)
        output_equiv = verify_output_equivalence(
            model, permuted_model, test_data, device
        )
        print(f"    Max output diff: {output_equiv['max_diff']:.2e}",
              flush=True)
        print(f"    Mean output diff: {output_equiv['mean_diff']:.2e}",
              flush=True)
        print(f"    All predictions match: {output_equiv['all_predictions_match']}",
              flush=True)

        # 2. Compute activation distances
        print("  Computing penultimate activation distances...", flush=True)
        act_dist = compute_activation_distances(
            model, permuted_model, test_data, device
        )
        print(f"    Mean L2 distance: {act_dist['mean_l2']:.4f}", flush=True)
        print(f"    Max L2 distance: {act_dist['max_l2']:.4f}", flush=True)

        # 3. Compute matrix distances
        print("  Computing knowledge matrix distances...", flush=True)
        mat_dist = compute_matrix_distances(
            model, permuted_model, test_data, device,
            num_samples=num_matrix_samples,
            batch_size_mc=matrix_batch_size,
            debug=debug,
        )
        print(f"    Mean L2 distance: {mat_dist['mean_l2']:.6e}", flush=True)
        print(f"    Max L2 distance: {mat_dist['max_l2']:.6e}", flush=True)

        elapsed = time.perf_counter() - t0

        perm_result = {
            'seed': seed,
            'output_equivalence': output_equiv,
            'activation_change': act_dist,
            'matrix_change': mat_dist,
            'elapsed_seconds': float(elapsed),
        }
        all_permutation_results.append(perm_result)

        # Cleanup
        del permuted_model, perm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Aggregate results
    # -----------------------------------------------------------------------
    print(f"\n\n{'#' * 70}", flush=True)
    print(f"  SUMMARY: {experiment_name}", flush=True)
    print(f"{'#' * 70}", flush=True)

    # Average activation and matrix distances across permutations
    avg_act_l2 = float(np.mean([
        r['activation_change']['mean_l2'] for r in all_permutation_results
    ]))
    avg_mat_l2 = float(np.mean([
        r['matrix_change']['mean_l2'] for r in all_permutation_results
    ]))
    max_mat_l2 = float(np.max([
        r['matrix_change']['max_l2'] for r in all_permutation_results
    ]))
    all_outputs_match = all(
        r['output_equivalence']['all_predictions_match']
        for r in all_permutation_results
    )

    print(f"  Avg penultimate activation L2 change: {avg_act_l2:.4f}",
          flush=True)
    print(f"  Avg knowledge matrix L2 change:       {avg_mat_l2:.6e}",
          flush=True)
    print(f"  Max knowledge matrix L2 change:       {max_mat_l2:.6e}",
          flush=True)
    print(f"  All predictions match:                {all_outputs_match}",
          flush=True)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    out_dir = Path(f'experiments/{experiment_name}/isomorphism/')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'isomorphism_results.json'

    save_data = {
        'experiment': experiment_name,
        'dataset': dataset,
        'architecture_index': arch_idx,
        'num_permutations': num_permutations,
        'num_samples': num_samples,
        'num_matrix_samples': num_matrix_samples,
        'summary': {
            'activation_change': avg_act_l2,
            'matrix_change': avg_mat_l2,
            'max_matrix_change': max_mat_l2,
            'all_predictions_match': all_outputs_match,
        },
        'per_permutation': all_permutation_results,
    }

    with open(out_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {out_file}", flush=True)

    return save_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = ArgumentParser(
        description="Isomorphism invariance experiment: demonstrates that "
                    "knowledge matrices are invariant under neuron permutations "
                    "while penultimate activations are not."
    )
    parser.add_argument(
        "--experiment", type=str, default="alexnet_cifar10",
        help="Experiment name (key in DEFAULT_EXPERIMENTS)."
    )
    parser.add_argument(
        "--num_permutations", type=int, default=5,
        help="Number of random permutations to test."
    )
    parser.add_argument(
        "--num_samples", type=int, default=500,
        help="Number of data samples for feature extraction and detection."
    )
    parser.add_argument(
        "--num_matrix_samples", type=int, default=50,
        help="Number of samples for matrix comparison (expensive)."
    )
    parser.add_argument(
        "--matrix_batch_size", type=int, default=1800,
        help="Batch size for KnowledgeMatrixComputer."
    )
    parser.add_argument(
        "--temp_dir", type=str, default=None,
        help="Temporary directory (cluster SLURM_TMPDIR)."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print matrix norms and relative errors for diagnostics."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Isomorphism Invariance Experiment", flush=True)
    print(f"  Experiment:       {args.experiment}", flush=True)
    print(f"  Permutations:     {args.num_permutations}", flush=True)
    print(f"  Samples:          {args.num_samples}", flush=True)
    print(f"  Matrix samples:   {args.num_matrix_samples}", flush=True)
    print(f"  Temp dir:         {args.temp_dir}", flush=True)

    t_start = time.perf_counter()

    run_isomorphism_experiment(
        experiment_name=args.experiment,
        num_permutations=args.num_permutations,
        num_samples=args.num_samples,
        temp_dir=args.temp_dir,
        num_matrix_samples=args.num_matrix_samples,
        matrix_batch_size=args.matrix_batch_size,
        debug=args.debug,
    )

    elapsed = time.perf_counter() - t_start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed / 3600:.2f}h)", flush=True)


if __name__ == "__main__":
    main()
