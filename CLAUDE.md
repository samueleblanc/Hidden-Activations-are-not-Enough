# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research implementation of "Hidden Activations Are Not Enough" — adversarial example detection via knowledge matrices (quiver representations) of neural networks. Runs on Compute Canada Alliance HPC clusters (Rorqual, H100 GPUs).

The core research question: **are knowledge matrices better than penultimate-layer activations for detecting adversarial examples?**

- **Knowledge matrices** (quiver representations capturing full forward-pass structure) — the proposed method.
- **Penultimate-layer activations** (last hidden layer features) — the baseline.
- Step E runs 6 detectors (KNN, KDE, GMM, OCSVM, Isolation Forest, Mahalanobis) on 3 representations (penultimate activations, matrices, SVD-reduced matrices) and compares them.
- Step F generates LaTeX tables from the comparison results.

## Pipeline Architecture

7-step Slurm pipeline orchestrated by `run_experiment.sh`:

```
A (Training) → B (Matrices ×8) ──────────────→ E (RepComparison) ─→ F (LaTeXTables)
             → C (AdvExamples) → D (AdvMats ×8) → E                ↗
             → G (Theorem4.5) ─────────────────────────────────────→ F
```

Steps B, D run as 8 parallel Slurm jobs (chunks). Steps A, B, C, D, E, G require GPU (H100). Step F is CPU-only.
Step G runs in parallel with B/C/D/E (depends only on A).

## Key Commands

```bash
# Full pipeline (with audit/checkpointing)
bash run_experiment.sh alexnet_cifar10

# Quick test (small samples, short times)
bash run_experiment.sh --test --skip-audit alexnet_cifar10

# Skip audit, direct submission
bash run_experiment.sh --skip-audit alexnet_cifar10

# Dry run (generate scripts only)
bash run_experiment.sh --dry-run --test alexnet_cifar10

# Post-run report
python pipeline_report.py --experiment alexnet_cifar10
python pipeline_report.py --experiment alexnet_cifar10 --test

# Run individual steps manually
python training.py --experiment_name alexnet_cifar10 --temp_dir $SLURM_TMPDIR                    # Step A
python generate_matrices.py --experiment alexnet_cifar10 --chunk_id 0 --total_chunks 8 --batch_size 1800  # Step B
python generate_adversarial_examples.py --experiment_name alexnet_cifar10 --temp_dir $SLURM_TMPDIR       # Step C
python generate_adversarial_matrices.py --experiment_name alexnet_cifar10 --chunk_id 0 --total_chunks 8  # Step D
python compare_representations.py --experiment alexnet_cifar10 --temp_dir $SLURM_TMPDIR --svd_ablation   # Step E
python validate_theorem45.py --experiment alexnet_cifar10 --temp_dir $SLURM_TMPDIR                       # Step G
python generate_latex_tables.py --output tables/                                                          # Step F

# Unit tests
python -m pytest unit_test/
```

## Critical Patterns

### KnowledgeMatrixComputer expects 3D input
The `knowledgematrix` library's `KnowledgeMatrixComputer.forward()` expects **3D tensors** `(C, H, W)`, NOT 4D `(1, C, H, W)`. Never use `.unsqueeze(0)` before calling `forward()`. The library's `_get_start_layer()` reads `C, H, W = x.shape[0], x.shape[1], x.shape[2]`.

### No internet on compute nodes
Compute nodes cannot download anything. All model creation must use `pretrained=False`. Pretrained weights are loaded from local files at `experiments/{arch}_{dataset}/weights/pretrained-weights.pth`. Datasets must be pre-downloaded on the login node.

### torchattacks version compatibility
Cluster has torchattacks 3.3.0 (via computecanada), local uses 3.5.1. Use lazy `getattr(torchattacks, cls_name, None)` to handle missing attack classes gracefully.

### freeze_features compatibility
The `knowledgematrix` package's AlexNet may not accept `freeze_features` kwarg. `utils/utils.py:get_architecture()` uses `inspect.signature` to check before passing it.

## Experiment Configuration

Defined in `constants/constants.py`:
- `DEFAULT_EXPERIMENTS`: dict mapping experiment names to hyperparameters (epochs, lr, batch_size, architecture_index, dataset, optimizer, scheduler)
- `ATTACKS`: list of 17 adversarial attack method names
- `ARCHITECTURES`: list of network architectures (index -3=AlexNet, -2=ResNet, -1=VGG, -4=LeNet)

Main experiments: `alexnet_cifar10`, `resnet_cifar10`, `resnet_cifar100`, `vgg_cifar100`

## Output Structure

```
experiments/{experiment}/
├── weights/epoch_{N}.pth           # Step A output
├── matrices_task_{0-7}.zip         # Step B output (chunked zips)
├── adversarial_examples/{attack}/  # Step C output
├── adv_matrices_task_{0-7}.zip     # Step D output
├── comparison/representation_comparison.json  # Step E output
├── theorem45/theorem45_results.json           # Step G output
├── isomorphism/isomorphism_results.json       # Isomorphism experiment
├── calibration.json                # GPU calibration results
├── checkpoints/                    # Per-step completion tracking
└── orchestrator_jobs/              # Generated Slurm scripts
tables/*.tex                           # Step F output (LaTeX tables)
```

## Key Files

| File | Role |
|------|------|
| `run_experiment.sh` | Main orchestrator — generates and submits all Slurm jobs |
| `constants/constants.py` | All experiment configs, attack lists, architectures |
| `utils/utils.py` | `get_architecture()`, `get_model()`, `get_dataset()`, `get_device()` |
| `utils/data_integrity.py` | `verify_experiment()`, zip verification |
| `matrix_construction/parallel.py` | `ParallelMatrixConstruction` — chunked matrix computation |
| `calibrate.py` | GPU batch_size binary search, SLURM time/memory estimation |
| `pipeline_report.py` | Post-run analysis with error classification |
| `compare_representations.py` | Step E: representation comparison (6 detectors × 3 representations) + Lee et al. baseline |
| `generate_latex_tables.py` | Step F: generates comparison LaTeX tables |
| `validate_theorem45.py` | Step G: empirical validation of Theorem 4.5 (distance lower bound) |
| `baselines/lee2018.py` | Lee et al. (2018) multi-layer Mahalanobis baseline detector |
| `isomorphism_experiment.py` | Isomorphism invariance demonstration (Stage 6) |

## Slurm Log Locations

- Normal mode: `slurm_out/`, `slurm_err/`
- Test mode: `slurm_out_test/`, `slurm_err_test/`
- GPU monitoring: `gpu-monitor/{experiment}.{step}.{chunk}.log`
- Reports: `reports/{experiment}_report_{timestamp}.txt`
- Log filename pattern: `PIPE_{STEP}_{experiment}[_c{chunk}]_{jobid}.{out|err}`

## Environment

- Cluster modules: `StdEnv/2023 python/3.11.5 scipy-stack/2025a`
- Virtual env: `env/` (created via `python -m venv env`)
- Key deps: PyTorch, torchattacks, knowledgematrix (git+samueleblanc@0d26c7a)
