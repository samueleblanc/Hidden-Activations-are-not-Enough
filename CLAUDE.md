# Knowledge Matrices as Canonical Neural Network Representations

## Overview

Research implementation of "Hidden Activations Are Not Enough: A General Approach to Neural Network Predictions" (arXiv:2409.13163) by Samuel Leblanc, Aiky Rasolomanana, and Marco Armenta. Computes "knowledge matrices" via quiver representations of neural networks and demonstrates their superiority over penultimate-layer activations as canonical representations.

- **Language:** Python 3.11 + Bash (Slurm job scripts)
- **Cluster:** Compute Canada Alliance HPC (Rorqual, H100 GPUs)
- **License:** Apache 2.0
- **Key dependencies:** torch 2.2.2, torchvision 0.17.2, torchattacks 3.5.1, knowledgematrix (git+MarcoArmenta/knowledgematrix-cluster@fe64a13 — Phase-1 fork; pinned in `requirements-{slurm,local}.txt`), scikit-learn 1.3.2, scipy 1.10.1, neuralteleportation (for teleportation experiment)

## TMLR Resubmission Direction

Paper rejected by TMLR (Nov 2024). New direction: **"Knowledge Matrices as Canonical Neural Network Representations"** — dropping adversarial detection claims entirely. Three studies:

1. **Study 1 — Isomorphism Invariance.** KMs are provably invariant under neuron permutations and broader quiver isomorphisms; no other practically computable representation has this. Sub-studies: 1a (random neuron permutation, retired from orchestrator), 1b (neural teleportation), 1c (within-arch invariance under the 9-measure similarity panel).
2. **Study 2 — Theorem 4.5 Distance Lower Bound.** KM distances are guaranteed to exceed logit-space distances. Novel empirical finding: the amplification factor splits cleanly by attack type.
3. **Study 3 — Cross-architecture canonical comparison.** KMs are uniformly $1000 \times 150{,}529$ for any feedforward network on $224 \times 224$ ImageNet inputs, so KM Frobenius distance compares ResNet-152, DenseNet-121, and GoogLeNet directly without any alignment step. Includes a same-arch cross-recipe positioning experiment (Step E).

Phase 1 (added 2026-05-03) extends Studies 1, 2, and 3 with a 9-measure representation-similarity panel and Cui/Murphy controls (`docs/superpowers/specs/2026-05-03-cka-similarity-experiments-design.md`).

Additionally: testing how penultimate activation distances behave when increasing network size (using pretrained torchvision models directly, no training needed).

## Directory Structure

```
├── isomorphism_experiment.py      # Study 1a: KM invariance under neuron permutations (retired)
├── validate_theorem45.py          # Study 2: empirical Theorem 4.5 validation
├── teleportation_experiment.py    # Penultimate activation instability under teleportation
├── training.py                    # Model training with checkpointing
├── generate_matrices.py           # Clean knowledge matrix computation (chunked)
├── generate_theorem45_tables.py   # LaTeX table generation for Theorem 4.5 results
├── plot_graphs.py                 # Training curve visualization
├── constants/
│   └── constants.py               # Experiment configs, architectures, attack lists
├── matrix_construction/
│   ├── parallel.py                # ParallelMatrixConstruction — chunked computation
│   └── matrix_computation.py      # MlpRepresentation / ConvRepresentation_2D
├── model_zoo/                     # AlexNet, ResNet, VGG, CNN_2D, MLP definitions
├── utils/
│   ├── utils.py                   # get_architecture(), get_model(), get_dataset(), get_device()
│   ├── features.py                # extract_penultimate_features()
│   └── atomic_io.py               # atomic_torch_save(), atomic_json_dump()
├── patches/                       # neuralteleportation PyTorch 2.x compatibility patches
├── unit_test/                     # Test suite
├── docs/
│   ├── new_direction.tex          # New paper direction strategy document
│   └── rebuttal/                  # LaTeX paper sources
├── job_training.sh                # SLURM job: model training
├── job_matrices.sh                # SLURM job: matrix computation
├── experiments/                   # Per-experiment output (weights, matrices, results)
└── legacy/                        # Archived old adversarial detection pipeline
```

## Experiments

### Study 1a: Isomorphism Invariance (`isomorphism_experiment.py`)

Applies random neuron permutations to create isomorphic networks. Shows:
- Permuted network produces identical outputs
- Penultimate activations CHANGE after permutation
- Knowledge matrices remain the SAME (up to numerical precision)

```bash
python isomorphism_experiment.py --experiment alexnet_cifar10
```

### Study 1b: Teleportation (`teleportation_experiment.py`)

Demonstrates penultimate activation instability under neural teleportation (quiver isomorphism via the `neuralteleportation` library's COB models).

```bash
python teleportation_experiment.py --arch resnet18 --dataset cifar10 --num_teleportations 10
```

### Study 2: Theorem 4.5 Validation (`validate_theorem45.py`)

Empirically validates `||M(x) - M(x')|| >= gamma * ||f(x) - f(x')||`. Generates adversarial pairs on-the-fly, computes logit/penultimate/KM distances, estimates gamma with bootstrap 95% CI.

```bash
python validate_theorem45.py --experiment alexnet_cifar10 --temp_dir $SLURM_TMPDIR
```

### LaTeX Tables (`generate_theorem45_tables.py`)

```bash
python generate_theorem45_tables.py --experiments alexnet_cifar10 resnet_cifar10 --output tables/
```

## Experiment Configuration

Defined in `constants/constants.py`:

- `DEFAULT_EXPERIMENTS`: dict mapping experiment names to hyperparams
- `ATTACKS`: 16 adversarial attack names (used by validate_theorem45.py for generating perturbed pairs)
- `ARCHITECTURES`: indexed by negative integers — -4=LeNet, -3=AlexNet, -2=ResNet, -1=VGG

Main experiments: `alexnet_cifar10`, `resnet_cifar10`, `resnet_cifar100`, `vgg_cifar100`

## Critical Patterns

### KnowledgeMatrixComputer expects 3D input
The `knowledgematrix` library's `KnowledgeMatrixComputer.forward()` expects **3D tensors** `(C, H, W)`, NOT 4D `(1, C, H, W)`. Never use `.unsqueeze(0)` before calling `forward()`.

### No internet on compute nodes
Compute nodes cannot download anything. All model creation must use `pretrained=False`. Pretrained weights loaded from local files. Datasets must be pre-downloaded on login nodes.

### torchattacks version compatibility
Cluster has torchattacks 3.3.0, local uses 3.5.1. Use lazy `getattr(torchattacks, cls_name, None)` to handle missing attack classes.

### freeze_features compatibility
The `knowledgematrix` package's AlexNet may not accept `freeze_features` kwarg. `utils/utils.py:get_architecture()` uses `inspect.signature` to check before passing it.

### neuralteleportation patches
The `neuralteleportation` library requires patches for PyTorch 2.x compatibility. Run `bash patches/apply_neuralteleportation_patches.sh` after installation.

### Canonical distance metric: RMS-per-coordinate
Cross-space distance comparisons (logit vs penultimate vs KM) use **RMS-per-coordinate** (`||·|| / sqrt(numel)`) as the canonical fair-comparison metric, not raw L2/Frobenius. Raw norms inflate KM distances by a factor of `sqrt(d+1) ≈ 388` for ImageNet purely from dimensionality, confounding Theorem-4.5 γ values and amplification ratios.

- Helpers: `utils/scaling.py` (`rms_distance`, `rescale_gamma_to_rms`, `rescale_amp_M_to_rms`, `rescale_amp_h_to_rms`, `penultimate_dim`, `km_numel`).
- Source-of-truth save: `validate_theorem45.py`, `cross_model_experiment.py`, `cka_similarity/workers/s2_cross_architecture.py`, `cka_similarity/workers/s3_distance_amplification.py` all save both raw and RMS values in their result JSONs.
- Post-hoc salvage: `python scripts/renormalize_distances.py` walks existing result trees and adds `rms` blocks to legacy JSONs without modifying raw fields (idempotent).
- Consumers (`generate_theorem45_tables.py`, `cka_similarity/reduce/tables.py`, `scripts/paper_figures.py`, `wire_paper_results.py`) default to RMS; pass `--raw` / `--metric raw` for the legacy view.
- Background: `docs/Final-twist/km-notes.md` (2026-05-10 entry).

## Environment

- **Cluster modules:** `StdEnv/2023 python/3.11.5 scipy-stack/2025a`
- **Virtual env:** `env/` (created via `python -m venv env`)
- **Cluster deps:** `requirements-slurm.txt` (torch 2.2.2)
- **Local deps:** `requirements-local.txt` (torch 2.6.0)
- **Key packages:** knowledgematrix (`git+MarcoArmenta/knowledgematrix-cluster@fe64a13` — Phase-1 fork stacking `extract_weff` + `densenet121` + `googlenet` + `resnet152` PRs), neuralteleportation (with patches)

## Legacy Code

The `legacy/` directory contains the old adversarial detection pipeline that was part of the original paper submission. This includes:
- Adversarial example generation and matrix computation
- 6-detector x 3-representation comparison framework
- Lee et al. (2018) multi-layer Mahalanobis baseline
- Full SLURM pipeline orchestrator with auto-retry
- HP tuning pipeline
- Associated tests and documentation

This code is preserved for reference but is not part of the current paper direction.
