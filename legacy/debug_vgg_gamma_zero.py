"""
Diagnostic: confirm why VGG+ImageNet Theorem 4.5 runs report gamma = 0 for
certain attacks (APGD, DeepFool, Square).

Hypothesis: for a small number of (clean, adversarial) pairs the attack
produces a perturbation small enough that BOTH inputs fall inside the
SAME linear region of the piecewise-linear VGG network — i.e. every
ReLU has the same sign pattern and every MaxPool has the same argmax
everywhere. In that regime knowledgematrix linearizes ReLU/MaxPool via
the *identical* `vertices = post_act / pre_act` and identical argmax
indices (see env/lib/python3.11/site-packages/knowledgematrix/matrix_computer.py:95-122),
so the two knowledge matrices are byte-identical and ||M(x)-M(x')|| = 0
exactly while logits still differ slightly (numerical round-off in the
final Linear layer only).

This script reproduces that: for each sample i it
    (a) generates an adversarial example with the chosen attack,
    (b) computes clean & adversarial logits and KMs,
    (c) snapshots the ReLU sign pattern AND maxpool argmax after each
        forward pass through KnowledgeMatrixComputer,
    (d) reports per-sample d_f, d_M, ReLU Hamming distance and maxpool
        mismatch count.

Expected finding: the sample(s) with d_M < 1e-12 have relu_hamming == 0
AND maxpool_mismatches == 0 (i.e. same linear region), which confirms
the mechanism.

Usage:
    python debug_vgg_gamma_zero.py --attack APGD
    python debug_vgg_gamma_zero.py --attack DeepFool --num_samples 200
    python debug_vgg_gamma_zero.py --attack Square --temp_dir $SLURM_TMPDIR
"""

import json
import time
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch import nn

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer

from utils.utils import (
    get_architecture, _move_residuals_to_device, get_dataset, subset,
    get_device, get_input_shape, get_num_classes,
)
from validate_theorem45 import generate_adversarial_pairs


# ---------------------------------------------------------------------------
# Activation-pattern snapshotting
# ---------------------------------------------------------------------------

def get_relu_sign_pattern(model):
    """Flatten the sign pattern of all populated ReLU pre-activations.

    Must be called IMMEDIATELY after `KnowledgeMatrixComputer.forward(x)` —
    the pre_acts/acts lists are (re-)allocated on every forward with
    save=True (see knowledgematrix/neural_net.py:360-363), so a later
    forward would overwrite them.

    Returns a 1-D bool tensor on CPU.
    """
    parts = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, nn.ReLU):
            if i < len(model.pre_acts) and model.pre_acts[i] is not None:
                pre = model.pre_acts[i]
                parts.append((pre > 0).flatten().cpu())
    if not parts:
        return torch.zeros(0, dtype=torch.bool)
    return torch.cat(parts)


def get_maxpool_argmax_snapshot(model):
    """Flatten the argmax indices of every populated MaxPool2d layer.

    See knowledgematrix/neural_net.py:376-378 — maxpool_indices[i] is
    populated in the save=True forward. Returns a 1-D int tensor on CPU.
    """
    parts = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, (nn.MaxPool2d, nn.AdaptiveMaxPool2d)):
            if i < len(model.maxpool_indices) and model.maxpool_indices[i] is not None:
                parts.append(model.maxpool_indices[i].flatten().cpu())
    if not parts:
        return torch.zeros(0, dtype=torch.long)
    return torch.cat(parts)


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def debug_vgg_gamma_zero(attack_name, num_samples=200, temp_dir=None):
    experiment_name = 'vgg_imagenet'
    dataset = 'imagenet'
    arch_idx = -1  # VGG11
    input_shape = get_input_shape(dataset)
    num_classes = get_num_classes(dataset)
    device = get_device()

    # --- Load pretrained VGG11 (matches validate_theorem45.py:320-328) ---
    print("Loading pretrained VGG11...", flush=True)
    model = get_architecture(
        architecture_index=arch_idx, input_shape=input_shape,
        num_classes=num_classes, pretrained=True,
        freeze_features=False,
    ).to(device)
    _move_residuals_to_device(model, device)
    model.eval()

    # --- Load test data (matches validate_theorem45.py:350-354) ---
    print("Loading test data...", flush=True)
    _, test_set = get_dataset(dataset, data_loader=False, data_path=temp_dir)
    test_data, test_labels = subset(test_set, num_samples, input_shape)
    print(f"Test subset: {test_data.shape}", flush=True)

    # --- Generate paired adversarial examples ---
    print(f"\nGenerating adversarial examples with {attack_name}...",
          flush=True)
    t0 = time.perf_counter()
    clean, adv = generate_adversarial_pairs(
        model, test_data, test_labels, attack_name, device,
        dataset=dataset, experiment_name=experiment_name,
    )
    if clean is None:
        print(f"Attack {attack_name} failed to produce pairs. Aborting.",
              flush=True)
        return
    print(f"  {len(clean)} pairs in {time.perf_counter() - t0:.1f}s",
          flush=True)

    # --- KnowledgeMatrixComputer ---
    # Use a smaller batch size than validate_theorem45's 1800 default to be
    # safer on GPU memory — the diagnostic walks one sample at a time anyway.
    kmc = KnowledgeMatrixComputer(model, batch_size=512, device=device)

    n = len(clean)

    # Resume from checkpoint if available — per-sample work is slow on VGG
    # ImageNet (~12-15s/sample), so 200 samples can exceed the SLURM
    # walltime. Save after every sample and resume the next run.
    out_dir = Path('experiments/vgg_imagenet/theorem45')
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = out_dir / f'debug_gamma_zero_{attack_name}.ckpt.json'

    per_sample = []
    total_pattern_len = None
    total_maxpool_len = None
    start_idx = 0

    if ckpt_file.exists():
        try:
            with open(ckpt_file) as f:
                ckpt = json.load(f)
            if ckpt.get('attack') == attack_name and ckpt.get('num_samples') == n:
                per_sample = ckpt.get('per_sample', [])
                total_pattern_len = ckpt.get('total_pattern_len')
                total_maxpool_len = ckpt.get('total_maxpool_len')
                start_idx = len(per_sample)
                print(f"  Resuming from checkpoint: {start_idx}/{n} samples done",
                      flush=True)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARNING: corrupt checkpoint ({e}), starting fresh.",
                  flush=True)

    for i in range(start_idx, n):
        # 3D inputs — no unsqueeze (see CLAUDE.md "Critical Patterns")
        sample_c = clean[i].to(device).float()
        sample_a = adv[i].to(device).float()

        # Clean forward (also gives us logits via kmc.current_output)
        mat_c = kmc.forward(sample_c)
        out_c = kmc.current_output.detach().clone()
        pat_c = get_relu_sign_pattern(model)
        mp_c = get_maxpool_argmax_snapshot(model)

        # Adv forward
        mat_a = kmc.forward(sample_a)
        out_a = kmc.current_output.detach().clone()
        pat_a = get_relu_sign_pattern(model)
        mp_a = get_maxpool_argmax_snapshot(model)

        d_f = float(torch.linalg.norm(
            out_c.double() - out_a.double()).item())
        d_M = float(torch.linalg.norm(
            (mat_c.double() - mat_a.double())).item())

        if pat_c.numel() != pat_a.numel():
            # Shouldn't happen — same model, same input shape
            relu_hamming = -1
            pat_len = -1
        else:
            relu_hamming = int((pat_c != pat_a).sum().item())
            pat_len = int(pat_c.numel())

        if mp_c.numel() != mp_a.numel():
            mp_mismatch = -1
            mp_len = -1
        else:
            mp_mismatch = int((mp_c != mp_a).sum().item())
            mp_len = int(mp_c.numel())

        if total_pattern_len is None:
            total_pattern_len = pat_len
        if total_maxpool_len is None:
            total_maxpool_len = mp_len

        per_sample.append({
            'idx': i,
            'd_f': d_f,
            'd_M': d_M,
            'relu_hamming': relu_hamming,
            'relu_pattern_len': pat_len,
            'maxpool_mismatches': mp_mismatch,
            'maxpool_total_positions': mp_len,
        })

        # Memory hygiene — pre_acts tensors are large on VGG + ImageNet
        del mat_c, mat_a, out_c, out_a, pat_c, pat_a, mp_c, mp_a
        del sample_c, sample_a
        if torch.cuda.is_available() and (i + 1) % 25 == 0:
            torch.cuda.empty_cache()

        if (i + 1) % 10 == 0 or i == n - 1 or i == 0:
            last = per_sample[-1]
            print(f"  [{i + 1}/{n}] d_f={last['d_f']:.3e}  "
                  f"d_M={last['d_M']:.3e}  "
                  f"relu_hamming={last['relu_hamming']}/{last['relu_pattern_len']}  "
                  f"maxpool_mismatches={last['maxpool_mismatches']}/"
                  f"{last['maxpool_total_positions']}", flush=True)

        # Incremental checkpoint (atomic): tmp write + rename
        ckpt_data = {
            'attack': attack_name,
            'experiment': experiment_name,
            'num_samples': n,
            'total_pattern_len': total_pattern_len,
            'total_maxpool_len': total_maxpool_len,
            'per_sample': per_sample,
        }
        ckpt_tmp = ckpt_file.with_suffix('.tmp')
        with open(ckpt_tmp, 'w') as f:
            json.dump(ckpt_data, f)
        ckpt_tmp.rename(ckpt_file)

    # --- Summary ---
    exact_zero_idx = [s['idx'] for s in per_sample if s['d_M'] == 0.0]
    near_zero_idx = [s['idx'] for s in per_sample if s['d_M'] < 1e-12]
    zero_hamming = [s['relu_hamming'] for s in per_sample if s['d_M'] < 1e-12]
    zero_mp = [s['maxpool_mismatches'] for s in per_sample
               if s['d_M'] < 1e-12]

    # Smallest d_M among samples where sign patterns differ — sets a
    # "floor" on the numerical noise we should expect when regions differ.
    positive_hamming = [(s['d_M'], s['relu_hamming'])
                        for s in per_sample if s['relu_hamming'] > 0]
    min_d_M_pos_hamming = min((d for d, _ in positive_hamming),
                              default=None)

    all_zero_have_hamming_zero = (
        len(near_zero_idx) > 0
        and all(h == 0 for h in zero_hamming)
        and all(m == 0 for m in zero_mp)
    )

    print(f"\n{'#' * 60}")
    print(f"  SUMMARY  ({attack_name})")
    print(f"{'#' * 60}")
    print(f"  num_samples                          : {n}")
    print(f"  num with d_M == 0.0 exactly          : {len(exact_zero_idx)}")
    print(f"  num with d_M < 1e-12                 : {len(near_zero_idx)}")
    print(f"  relu_pattern_len                     : {total_pattern_len}")
    print(f"  maxpool_total_positions              : {total_maxpool_len}")
    if near_zero_idx:
        print(f"  d_M<1e-12 sample indices             : {near_zero_idx}")
        print(f"  their relu_hamming distances         : {zero_hamming}")
        print(f"  their maxpool_mismatches             : {zero_mp}")
        print(f"  all of them have hamming==0 AND mp==0: "
              f"{all_zero_have_hamming_zero}")
    if min_d_M_pos_hamming is not None:
        print(f"  min d_M across samples w/ ham>0      : "
              f"{min_d_M_pos_hamming:.3e}")
    else:
        print(f"  min d_M across samples w/ ham>0      : (none)")

    # --- Save final results (ckpt file above has the same data + resumption
    # metadata; final file is the canonical deliverable) ---
    out_file = out_dir / f'debug_gamma_zero_{attack_name}.json'
    save_data = {
        'attack': attack_name,
        'experiment': experiment_name,
        'num_samples': n,
        'num_samples_d_M_exact_zero': len(exact_zero_idx),
        'num_samples_d_M_lt_1e-12': len(near_zero_idx),
        'total_pattern_len': total_pattern_len,
        'total_maxpool_len': total_maxpool_len,
        'per_sample': per_sample,
        'summary': {
            'd_M_exact_zero_indices': exact_zero_idx,
            'd_M_lt_1e-12_indices': near_zero_idx,
            'd_M_lt_1e-12_relu_hamming_distances': zero_hamming,
            'd_M_lt_1e-12_maxpool_mismatches': zero_mp,
            'all_d_M_zero_have_hamming_zero_and_mp_zero':
                all_zero_have_hamming_zero,
            'min_d_M_across_samples_with_positive_hamming':
                min_d_M_pos_hamming,
        },
    }
    with open(out_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved: {out_file}")

    # Clean up checkpoint now that the final file is written
    if ckpt_file.exists():
        ckpt_file.unlink()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = ArgumentParser(
        description="Diagnostic: find VGG+ImageNet samples where d_M = 0 "
                    "and confirm that clean & adversarial share the same "
                    "ReLU/MaxPool linear region."
    )
    parser.add_argument(
        "--attack", type=str, default="APGD",
        help="Attack to run. One of FGSM, PGD, CW, DeepFool, APGD, Square.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=200,
        help="Number of test samples to use.",
    )
    parser.add_argument(
        "--temp_dir", type=str, default=None,
        help="Temporary directory (cluster SLURM_TMPDIR).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"VGG gamma=0 diagnostic", flush=True)
    print(f"  Attack:    {args.attack}", flush=True)
    print(f"  Samples:   {args.num_samples}", flush=True)
    print(f"  Temp dir:  {args.temp_dir}", flush=True)
    t_start = time.perf_counter()
    debug_vgg_gamma_zero(
        attack_name=args.attack,
        num_samples=args.num_samples,
        temp_dir=args.temp_dir,
    )
    print(f"\nTotal time: {time.perf_counter() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
