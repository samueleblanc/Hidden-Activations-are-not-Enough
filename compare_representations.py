"""
Fair comparison of adversarial detection across representations.

Applies the SAME detector (Mahalanobis distance) to three representations:
  1. Knowledge Matrices    — M(W,f)(x), the full forward-pass matrix
  2. Penultimate Features  — activations from the last hidden layer
  3. All-Layer Features    — concatenation of all hidden-layer activations

By using the same detector, any performance difference is due entirely to
the REPRESENTATION, not the detection algorithm.  This is the core experiment
for the claim "Hidden Activations Are Not Enough."

Usage:
    # On cluster (after pipeline steps A-F complete):
    python compare_representations.py --experiment lenet_cifar10
    python compare_representations.py --experiment alexnet_cifar10 --temp_dir $SLURM_TMPDIR

    # Run for multiple experiments:
    python compare_representations.py --experiment lenet_cifar10 alexnet_cifar10 resnet_cifar10 vgg_cifar10
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from typing import Union
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.decomposition import TruncatedSVD

from utils.utils import (
    get_model, get_dataset, get_input_shape, get_num_classes,
    get_device, subset,
)
from constants.constants import DEFAULT_EXPERIMENTS, ATTACKS, ATTACK_CATEGORIES


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

class AllLayerExtractor:
    """Extracts activations from every layer using forward hooks.
    Architecture-agnostic: works with any nn.Module."""

    def __init__(self, model):
        self.model = model
        self.features = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.ReLU, torch.nn.ELU, torch.nn.Tanh,
                                   torch.nn.LeakyReLU, torch.nn.PReLU,
                                   torch.nn.Sigmoid, torch.nn.GELU)):
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def _make_hook(self, name):
        def hook_fn(module, input, output):
            self.features[name] = output.detach().cpu()
        return hook_fn

    def extract(self, x):
        """Run forward pass and return concatenated features from all activation layers."""
        self.features = {}
        with torch.no_grad():
            _ = self.model(x)
        if not self.features:
            return None
        parts = []
        for name in sorted(self.features.keys()):
            feat = self.features[name]
            parts.append(feat.reshape(feat.shape[0], -1))
        return torch.cat(parts, dim=1).numpy()

    def cleanup(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def extract_penultimate_features(model, data, batch_size=128):
    """Extract penultimate-layer features in batches."""
    model.eval()
    device = next(model.parameters()).device
    feats = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size].to(device).float()
            f = model.forward(batch, return_penultimate=True)
            feats.append(f.detach().cpu().numpy().reshape(f.shape[0], -1))
    return np.vstack(feats) if feats else np.zeros((0,))


def extract_all_layer_features(model, data, batch_size=128):
    """Extract concatenated all-layer activation features."""
    device = next(model.parameters()).device
    extractor = AllLayerExtractor(model)
    feats = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size].to(device).float()
        f = extractor.extract(batch)
        if f is not None:
            feats.append(f)
    extractor.cleanup()
    return np.vstack(feats) if feats else np.zeros((0,))


def load_matrices_as_features(base_path, attack_name, max_samples=None):
    """Load pre-computed knowledge matrices and flatten them into feature vectors."""
    mat_dir = Path(base_path) / 'adversarial_matrices' / attack_name
    mats = []
    i = 0
    while True:
        mat_path = mat_dir / str(i) / 'matrix.pth'
        if not mat_path.exists():
            break
        m = torch.load(mat_path, map_location='cpu')
        mats.append(m.numpy().ravel())
        i += 1
        if max_samples and i >= max_samples:
            break
    return np.array(mats) if mats else None


def load_train_matrices(base_path, num_classes, per_class=1000):
    """Load training matrices and flatten."""
    mat_dir = Path(base_path) / 'matrices'
    mats = []
    labels = []
    for c in range(num_classes):
        class_dir = mat_dir / str(c)
        if not class_dir.exists():
            continue
        count = 0
        j = 0
        while count < per_class:
            mat_path = class_dir / str(j) / 'matrix.pt'
            if not mat_path.exists():
                break
            m = torch.load(mat_path, map_location='cpu')
            mats.append(m.numpy().ravel())
            labels.append(c)
            count += 1
            j += 1
    return np.array(mats), np.array(labels)


# ---------------------------------------------------------------------------
# Mahalanobis detector (same for all representations)
# ---------------------------------------------------------------------------

class MahalanobisDetector:
    """Per-class Mahalanobis distance detector with LedoitWolf covariance.
    Uses TruncatedSVD to handle high-dimensional features."""

    def __init__(self, max_components=256):
        self.max_components = max_components
        self.svd = None
        self.class_means = {}
        self.class_precisions = {}

    def fit(self, features, labels, num_classes):
        """Fit the detector on training data."""
        N, D = features.shape
        n_comp = min(self.max_components, D, max(1, N - 1))

        # Reduce dimensionality if needed
        if D > n_comp:
            self.svd = TruncatedSVD(n_components=n_comp, random_state=0)
            proj = self.svd.fit_transform(features)
        else:
            self.svd = None
            proj = features

        # Global fallback
        try:
            lw_global = LedoitWolf().fit(proj)
            global_mean = lw_global.location_
            global_prec = lw_global.precision_
        except Exception:
            global_mean = np.mean(proj, axis=0)
            global_prec = np.eye(proj.shape[1])

        # Per-class statistics
        for c in range(num_classes):
            mask = (labels == c)
            class_data = proj[mask]
            if class_data.shape[0] < 2:
                self.class_means[c] = global_mean
                self.class_precisions[c] = global_prec
            else:
                try:
                    lw = LedoitWolf().fit(class_data)
                    self.class_means[c] = lw.location_
                    self.class_precisions[c] = lw.precision_
                except Exception:
                    self.class_means[c] = global_mean
                    self.class_precisions[c] = global_prec

    def score(self, features):
        """Compute min Mahalanobis distance to any class (lower = more normal)."""
        if self.svd is not None:
            proj = self.svd.transform(features.reshape(features.shape[0], -1))
        else:
            proj = features.reshape(features.shape[0], -1)

        N = proj.shape[0]
        min_dists = np.full(N, np.inf)
        for c in self.class_means:
            diff = proj - self.class_means[c]
            prec = self.class_precisions[c]
            dists_sq = np.einsum('ij,jk,ik->i', diff, prec, diff)
            min_dists = np.minimum(min_dists, np.sqrt(np.maximum(dists_sq, 0.0)))
        return min_dists


# ---------------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------------

def run_comparison(experiment_name, temp_dir=None):
    """Run fair comparison for one experiment."""
    exp_config = DEFAULT_EXPERIMENTS[experiment_name]
    dataset = exp_config['dataset']
    arch_idx = exp_config['architecture_index']
    epoch = exp_config['epochs']

    input_shape = get_input_shape(dataset)
    num_classes = get_num_classes(dataset)
    device = get_device()

    base = f'{temp_dir}/experiments/{experiment_name}' if temp_dir else f'experiments/{experiment_name}'

    # Training saves epoch_{epochs-1}.pth as final checkpoint; try both conventions
    weights_dir = Path(base) / 'weights'
    weights_path = None
    for candidate_epoch in [epoch, epoch - 1]:
        candidate = weights_dir / f'epoch_{candidate_epoch}.pth'
        if candidate.exists():
            weights_path = candidate
            break
    if weights_path is None:
        # Fall back to highest-numbered epoch file
        epoch_files = sorted(weights_dir.glob('epoch_*.pth'),
                             key=lambda p: int(p.stem.split('_')[1]))
        if epoch_files:
            weights_path = epoch_files[-1]
        else:
            print(f"  Skipping {experiment_name}: no weights in {weights_dir}")
            return None
    print(f"  Using weights: {weights_path.name}")

    # Check if adversarial matrices exist
    adv_mat_dir = Path(base) / 'adversarial_matrices'
    if not adv_mat_dir.exists():
        print(f"  Skipping {experiment_name}: no adversarial matrices")
        return None

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {experiment_name}")
    print(f"  Architecture: {['MLP','CNN','CNN','CNN','CNN','CNN','CNN','CNN','CNN2D','CNN2D','CNN2D','AlexNet','ResNet18','VGG11'][arch_idx] if arch_idx >= -4 else 'custom'}")
    print(f"  Dataset: {dataset}, Classes: {num_classes}")
    print(f"{'='*70}")

    model = get_model(weights_path, arch_idx, input_shape, num_classes, device)
    model.eval()

    # -----------------------------------------------------------------------
    # 1. Load training data for fitting detectors
    # -----------------------------------------------------------------------
    print("  Loading training data...")
    train_set, _ = get_dataset(dataset, data_loader=False, data_path=temp_dir)
    train_data, train_labels = subset(train_set, 5000, input_shape)
    train_labels_np = train_labels.numpy().astype(int)

    # -----------------------------------------------------------------------
    # 2. Extract training representations
    # -----------------------------------------------------------------------
    print("  Extracting training PENULTIMATE features...")
    train_penult = extract_penultimate_features(model, train_data)
    print(f"    Shape: {train_penult.shape}")

    print("  Extracting training ALL-LAYER features...")
    train_alllayer = extract_all_layer_features(model, train_data)
    print(f"    Shape: {train_alllayer.shape}")

    print("  Loading training KNOWLEDGE MATRICES...")
    train_matrices, train_mat_labels = load_train_matrices(base, num_classes, per_class=500)
    if train_matrices is None or len(train_matrices) == 0:
        print("    WARNING: No training matrices found, skipping matrix comparison")
        train_matrices = None
    else:
        print(f"    Shape: {train_matrices.shape}")
        train_mat_labels = train_mat_labels.astype(int)

    # -----------------------------------------------------------------------
    # 3. Fit Mahalanobis detectors on each representation
    # -----------------------------------------------------------------------
    detectors = {}

    print("  Fitting Mahalanobis on PENULTIMATE features...")
    det_penult = MahalanobisDetector(max_components=256)
    det_penult.fit(train_penult, train_labels_np, num_classes)
    detectors['penultimate'] = det_penult

    print("  Fitting Mahalanobis on ALL-LAYER features...")
    det_alllayer = MahalanobisDetector(max_components=256)
    det_alllayer.fit(train_alllayer, train_labels_np, num_classes)
    detectors['all_layer'] = det_alllayer

    if train_matrices is not None:
        print("  Fitting Mahalanobis on KNOWLEDGE MATRICES...")
        det_matrix = MahalanobisDetector(max_components=256)
        det_matrix.fit(train_matrices, train_mat_labels, num_classes)
        detectors['knowledge_matrix'] = det_matrix

    # -----------------------------------------------------------------------
    # 4. Score clean test data (for FPR calibration)
    # -----------------------------------------------------------------------
    print("  Scoring clean test data...")
    _, test_set = get_dataset(dataset, data_loader=False, data_path=temp_dir)
    test_data, test_labels = subset(test_set, 2000, input_shape)

    clean_scores = {}
    clean_scores['penultimate'] = det_penult.score(
        extract_penultimate_features(model, test_data))
    clean_scores['all_layer'] = det_alllayer.score(
        extract_all_layer_features(model, test_data))
    if 'knowledge_matrix' in detectors:
        test_mats = load_matrices_as_features(base, 'test', max_samples=2000)
        if test_mats is not None and len(test_mats) > 0:
            clean_scores['knowledge_matrix'] = det_matrix.score(test_mats)
        else:
            # Cannot score clean test matrices, remove from comparison
            del detectors['knowledge_matrix']

    # -----------------------------------------------------------------------
    # 5. Score adversarial data per attack
    # -----------------------------------------------------------------------
    results = {}
    available_attacks = []
    for attack in ATTACKS:
        adv_path = Path(base) / 'adversarial_examples' / attack / 'adversarial_examples.pth'
        if adv_path.exists():
            available_attacks.append(attack)

    if not available_attacks:
        print("  No adversarial examples found!")
        return None

    print(f"  Found {len(available_attacks)} attacks: {available_attacks}")

    for attack in available_attacks:
        print(f"  Scoring attack: {attack}...")
        adv_data = torch.load(
            Path(base) / 'adversarial_examples' / attack / 'adversarial_examples.pth',
            map_location='cpu')

        # Limit to avoid OOM
        if len(adv_data) > 2000:
            adv_data = adv_data[:2000]

        attack_results = {}

        for rep_name, detector in detectors.items():
            if rep_name == 'penultimate':
                adv_feats = extract_penultimate_features(model, adv_data)
            elif rep_name == 'all_layer':
                adv_feats = extract_all_layer_features(model, adv_data)
            elif rep_name == 'knowledge_matrix':
                adv_feats = load_matrices_as_features(base, attack, max_samples=2000)
                if adv_feats is None or len(adv_feats) == 0:
                    continue

            adv_scores = detector.score(adv_feats)
            clean = clean_scores[rep_name]

            # Compute AUROC: higher Mahalanobis distance = more adversarial
            n_clean = len(clean)
            n_adv = len(adv_scores)
            labels = np.concatenate([np.zeros(n_clean), np.ones(n_adv)])
            scores = np.concatenate([clean, adv_scores])

            try:
                auroc = roc_auc_score(labels, scores)
            except ValueError:
                auroc = 0.5

            # TPR at fixed FPR thresholds
            fpr_arr, tpr_arr, _ = roc_curve(labels, scores)
            tpr_at_5 = tpr_arr[np.searchsorted(fpr_arr, 0.05, side='right') - 1] if len(fpr_arr) > 1 else 0
            tpr_at_10 = tpr_arr[np.searchsorted(fpr_arr, 0.10, side='right') - 1] if len(fpr_arr) > 1 else 0

            attack_results[rep_name] = {
                'auroc': float(auroc),
                'tpr_at_fpr5': float(tpr_at_5),
                'tpr_at_fpr10': float(tpr_at_10),
                'n_adv': n_adv,
                'n_clean': n_clean,
            }

        results[attack] = attack_results

    # -----------------------------------------------------------------------
    # 6. Aggregate and report
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  RESULTS: {experiment_name}")
    print(f"{'='*70}")

    # Per-attack table
    rep_names = list(detectors.keys())
    header = f"  {'Attack':<12s}"
    for rn in rep_names:
        short = {'penultimate': 'Penult.', 'all_layer': 'AllLayer', 'knowledge_matrix': 'KnowMat'}[rn]
        header += f" | {short:>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    aggregate = {rn: [] for rn in rep_names}
    for attack in available_attacks:
        if attack not in results:
            continue
        line = f"  {attack:<12s}"
        for rn in rep_names:
            if rn in results[attack]:
                auroc = results[attack][rn]['auroc']
                line += f" | {auroc:>8.4f}"
                aggregate[rn].append(auroc)
            else:
                line += f" | {'---':>8s}"
        print(line)

    # Average AUROC
    print("  " + "-" * (len(header) - 2))
    line = f"  {'AVERAGE':<12s}"
    for rn in rep_names:
        if aggregate[rn]:
            avg = np.mean(aggregate[rn])
            line += f" | {avg:>8.4f}"
        else:
            line += f" | {'---':>8s}"
    print(line)

    # Category breakdown
    print(f"\n  BY CATEGORY:")
    for cat_name, cat_attacks in ATTACK_CATEGORIES.items():
        cat_line = f"    {cat_name:<16s}"
        for rn in rep_names:
            cat_aurocs = [results[a][rn]['auroc'] for a in cat_attacks
                         if a in results and rn in results[a]]
            if cat_aurocs:
                cat_line += f" | {np.mean(cat_aurocs):>8.4f}"
            else:
                cat_line += f" | {'---':>8s}"
        print(cat_line)

    # TPR@FPR=5% summary
    print(f"\n  TPR @ FPR=5% (averaged across attacks):")
    for rn in rep_names:
        tprs = [results[a][rn]['tpr_at_fpr5'] for a in available_attacks
                if a in results and rn in results[a]]
        short = {'penultimate': 'Penultimate Features', 'all_layer': 'All-Layer Features',
                 'knowledge_matrix': 'Knowledge Matrices'}[rn]
        if tprs:
            print(f"    {short:<25s}: {np.mean(tprs):.4f}")

    # Save results
    out_dir = Path(f'experiments/{experiment_name}/comparison/')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'representation_comparison.json'
    save_data = {
        'experiment': experiment_name,
        'dataset': dataset,
        'architecture_index': arch_idx,
        'representations': rep_names,
        'per_attack': results,
        'average_auroc': {rn: float(np.mean(aggregate[rn])) if aggregate[rn] else None
                          for rn in rep_names},
    }
    with open(out_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {out_file}")

    return save_data


def print_cross_experiment_summary(all_results):
    """Print comparison across all experiments."""
    if not all_results:
        return

    print(f"\n\n{'#'*70}")
    print(f"  CROSS-EXPERIMENT SUMMARY")
    print(f"{'#'*70}")

    # Collect all representation names
    all_reps = set()
    for r in all_results:
        all_reps.update(r['representations'])
    all_reps = sorted(all_reps)

    header = f"  {'Experiment':<20s}"
    for rn in all_reps:
        short = {'penultimate': 'Penult.', 'all_layer': 'AllLayer', 'knowledge_matrix': 'KnowMat'}[rn]
        header += f" | {short:>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in all_results:
        line = f"  {r['experiment']:<20s}"
        for rn in all_reps:
            avg = r['average_auroc'].get(rn)
            if avg is not None:
                line += f" | {avg:>8.4f}"
            else:
                line += f" | {'---':>8s}"
        print(line)

    # Win counts
    print(f"\n  WINS (highest AUROC per attack):")
    wins = {rn: 0 for rn in all_reps}
    total = 0
    for r in all_results:
        for attack, attack_res in r['per_attack'].items():
            best_auroc = -1
            best_rep = None
            for rn in all_reps:
                if rn in attack_res and attack_res[rn]['auroc'] > best_auroc:
                    best_auroc = attack_res[rn]['auroc']
                    best_rep = rn
            if best_rep:
                wins[best_rep] += 1
                total += 1
    for rn in all_reps:
        short = {'penultimate': 'Penultimate Features', 'all_layer': 'All-Layer Features',
                 'knowledge_matrix': 'Knowledge Matrices'}[rn]
        print(f"    {short:<25s}: {wins[rn]}/{total} ({100*wins[rn]/total:.1f}%)")


def parse_args():
    parser = ArgumentParser(description="Fair comparison of adversarial detection representations")
    parser.add_argument("--experiment", nargs="+",
                        default=["lenet_cifar10"],
                        help="Experiment name(s)")
    parser.add_argument("--temp_dir", type=str, default=None,
                        help="Temporary directory for cluster")
    return parser.parse_args()


def main():
    args = parse_args()
    all_results = []
    for exp in args.experiment:
        result = run_comparison(exp, args.temp_dir)
        if result:
            all_results.append(result)

    if len(all_results) > 1:
        print_cross_experiment_summary(all_results)


if __name__ == "__main__":
    main()
