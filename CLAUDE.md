# CLAUDE.md

Research implementation of "Hidden Activations Are Not Enough" — adversarial example detection via knowledge matrices (quiver representations) of neural networks. Runs on Compute Canada Alliance HPC clusters (Rorqual, H100 GPUs).

## TMLR Revision Context

Paper rejected by TMLR (Nov 2024). Three reviewers (Kv2C, BFn5, Vj72) + action editor (t5mo / Grigorios Chrysos). Core reframing for resubmission: **"representation comparison, not detector proposal"** — same standard detectors applied to knowledge matrices vs. penultimate features.

Reviewer concerns and how the codebase addresses each:

1. **Non-standard metrics** (Kv2C, Vj72) → AUROC / AUPR / FPR@95TPR for every cell (Step 4)
2. **Unfair HP comparison** (BFn5) → 6 fixed-default detectors on all 3 representations (no per-detector tuning)
3. **Dimensionality confound** (BFn5, Vj72) → SVD rank ablation {16, 32, 64, 128, 256, 512}; if KMs at rank 64 beat penultimate at rank 256, advantage is structural
4. **Weak baselines** (Kv2C, Vj72) → KNN, KDE, GMM, OCSVM, IsolationForest, Mahalanobis + Lee et al. (2018) multi-layer Mahalanobis
5. **Computational cost unreported** (Vj72) → wall-clock time + peak GPU memory instrumentation per representation
6. **Toy datasets** (BFn5) → CIFAR-10/100 with 4 CNN architectures (AlexNet, ResNet, VGG, LeNet)
7. **Theorem 4.5 never validated** (BFn5) → new Step 2c empirical validation of distance lower bound

See `debate/` for full reviewer feedback (`reviews.pdf`), multi-agent debate (`rebuttal.md`), and implementation plan (`README-rebuttal.md`).

## Experimental Design

**Core claim: the same standard detectors, applied to knowledge matrices instead of penultimate-layer features, detect adversarial examples more consistently.**

| Dimension | Values | Count |
|-----------|--------|-------|
| Representations | Penultimate activations, Knowledge matrices, SVD-reduced matrices | 3 |
| Detectors | KNN, KDE, GMM, OCSVM, IsolationForest, Mahalanobis | 6 |
| Attacks | See ATTACKS in `constants/constants.py` | 16 |
| Experiments | alexnet_cifar10, resnet_cifar10, resnet_cifar100, vgg_cifar100 | 4 |

**Metrics:** AUROC (primary), AUPR, FPR@95TPR — all three reported for every cell in the 6×3 grid.

**SVD ablation:** ranks {16, 32, 64, 128, 256, 512}. Default rank 256 when raw dimensionality exceeds 256. Flag: `--svd_ablation`.

**Lee et al. (2018):** Multi-layer Mahalanobis baseline (`baselines/lee2018.py`). Runs separately from the 6×3 grid — uses its own logistic regression combiner across layers.

**16 attacks** across 5 categories: gradient-based (7), AutoAttack ensemble (4), gradient-free (3), elastic-net (2), baseline noise (1). Note: Square appears in both AutoAttack and gradient-free categories (M4).

## Known Bugs / Code Review Status

**v2 review** in `docs/deep_code_review_v2.md`: 15-agent review found **8 critical, 21 high, 40+ medium, 20+ low** (total open); 21 verified correct. Previous v1 review in `docs/deep_code_review.md`.

Of the original 17 critical+high bugs: **13 fixed**, 1 open (C5), 3 partially addressed (H4 code exists but broken, H5 main path fixed, H6 acknowledged).

### NEW Critical (v2) — Scientific Validity Blockers

| ID | Location | Description |
|----|----------|-------------|
| NC1 | `baselines/lee2018.py:162-189` | **Lee et al. input preprocessing is a no-op** — `feat.detach().cpu().numpy()` severs gradient graph; `loss.backward()` produces zero gradients; epsilon perturbation never fires. Weakens strongest baseline. |
| NC2 | `compare_representations.py:599,607` | **Clean test data differs between representations** — penultimate/all-layer use random 2000-sample subset; KMs use different subset from Step 2b. AUROC baselines not comparable. |
| NC3 | `compare_representations.py:645-654` | **Adversarial sample count mismatch** — 2000 for penultimate/all-layer vs ~200 for KMs. 10x statistical power difference. |
| NC4 | `training.py:237-259` | **Epoch-60 unfreeze nukes LR for from-scratch training** — unconditionally recreates optimizer with lr=1e-5. Affects `resnet_cifar100` and `resnet_cifar10` (2 of 4 main experiments). |
| NC5 | `validate_theorem45.py:329-330` | **Theorem 4.5 bound satisfaction is tautological** — gamma=min(d_M/d_f) guarantees 100% by construction. |
| NC6 | `run_experiment.sh:28-30` | **No CLI argument parsing** — `--test`, `--skip-audit`, `--dry-run` documented but silently ignored. |
| NC7 | `run_experiment.sh:108-172` | **EXPERIMENT_LIST exported after dataset check** — pre-flight never verifies datasets. |

### Still Open from v1

| ID | Location | Description |
|----|----------|-------------|
| C5 | `constants/constants.py:43` | ATTACKS list has 16 entries; LaTeX spec claims 17. Square double-counted across categories. |
| H5† | `baselines/lee2018.py:147-195` | Gradient hooks in Lee preprocessing lack try/finally (main AllLayerExtractor is fixed). |
| H6† | `compare_representations.py:534-566` | KM cost = disk I/O. Acknowledged in code note but Jacobian cost not integrated. |
| H12 | `generate_latex_tables.py:705` | Per-experiment SVD labels use `tab:svd_ablation_{exp}` vs spec `tab:svd_ablation`. |

### NEW High (v2) — Top Priority

| ID | Location | Description |
|----|----------|-------------|
| NH1 | `compare_representations.py:738` | Lee eval_clean_scores uses wrong LR model across attack iterations. |
| NH2 | `compare_representations.py:507,537` | Training data subsets differ between representations. |
| NH3 | `compare_representations.py:274` | KDE bandwidth=1.0 hardcoded — undermines "fair comparison" claim. |
| NH4 | `compare_representations.py:317` | GMM full covariance severely underdetermined (330K params from 5K samples). |
| NH5 | `compare_representations.py:170-237` | Per-class Mahalanobis degenerate with 50 samples/class (CIFAR-100). |
| NH6 | `training.py:104` | Loss divided by batch count not sample count — inflated ~16x in history.json. |
| NH7 | `training.py:220-228` | Checkpoint resume doesn't restore optimizer/scheduler state. |
| NH8 | `generate_adversarial_matrices.py:86` | No OOM retry logic (unlike parallel.py which has 4-attempt retry). |
| NH9 | `generate_adversarial_matrices.py` | No done_file checkpoint — partial failures indistinguishable from success. |
| NH10 | `experiment_config.sh:149` | `double_mem()` has no cap — can exceed 480GB node physical memory. |
| NH11 | `run_experiment.sh:844` | Step 3 missing dependency on Step 1 (trained weights). |
| NH12 | `job_recovery.sh:80,91` | Hardcoded wrong GPU type (10GB A100 MIG) and wrong venv name. |
| NH13 | All Python entry points | No global random seed — results not reproducible across runs. |

**Suggested fix priority:** NC1 → NC4 → NC2+NC3 → NC5 → NH13 → C5 → NH3 → NH4 → NC6+NC7 → NH10+NH11.

See `docs/deep_code_review_v2.md` for the complete 15-agent report with code snippets, line numbers, and cross-agent agreement matrix.

**Suggested fix priority:** C4 → H1-H4 (Lee baseline) → C5 → H6 (cost) → C3 → H7 → H10 → C1 → C2 → H5 → H8 → H9 → H11-H12.

## Pipeline Architecture

7-step Slurm pipeline orchestrated by `run_experiment.sh`. Step IDs encode dependency depth:

```
1 (Training) → 2a (Matrices ×8) ──────────────→ 4 (RepComparison) ─→ 5 (LaTeXTables)
             → 2b (AdvExamples) → 3 (AdvMats ×8) → 4                ↗
             → 2c (Theorem4.5) ────────────────────────────────────→ 5
```

Steps 2a, 3 run as 8 parallel Slurm jobs (chunks). Steps 1, 2a, 2b, 3, 4, 2c require GPU (H100). Step 5 is CPU-only. Step 2c runs in parallel with 2a/2b/3/4 (depends only on 1).

**Infrastructure:** GPU calibration (`calibration.sh`), OOM retry with memory doubling, checkpoint exit codes, error classification (`utils/error_classification.py`), auto-resubmit (`auto_resubmit.py`), job recovery (`job_recovery.sh`).

## Key Commands

```bash
# Calibrate first, then run pipeline
bash calibration.sh
bash run_experiment.sh

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

# Error collection / recovery
python collect_errors.py --experiment alexnet_cifar10
bash job_recovery.sh alexnet_cifar10

# Run individual steps manually
python training.py --experiment_name alexnet_cifar10 --temp_dir $SLURM_TMPDIR                    # Step 1
python generate_matrices.py --experiment alexnet_cifar10 --chunk_id 0 --total_chunks 8 --batch_size 1800  # Step 2a
python generate_adversarial_examples.py --experiment_name alexnet_cifar10 --temp_dir $SLURM_TMPDIR       # Step 2b
python generate_adversarial_matrices.py --experiment_name alexnet_cifar10 --chunk_id 0 --total_chunks 8  # Step 3
python compare_representations.py --experiment alexnet_cifar10 --temp_dir $SLURM_TMPDIR --svd_ablation   # Step 4
python validate_theorem45.py --experiment alexnet_cifar10 --temp_dir $SLURM_TMPDIR                       # Step 2c
python generate_latex_tables.py --output tables/                                                          # Step 5

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
- `ATTACKS`: list of **16** adversarial attack method names (not 17 — see C5)
- `ARCHITECTURES`: list of network architectures (index -4=LeNet, -3=AlexNet, -2=ResNet, -1=VGG)

Main experiments: `alexnet_cifar10`, `resnet_cifar10`, `resnet_cifar100`, `vgg_cifar100`

## Output Structure

```
experiments/{experiment}/
├── weights/epoch_{N}.pth                       # Step 1
├── matrices_task_{0-7}.zip                     # Step 2a (chunked)
├── adversarial_examples/{attack}/              # Step 2b
├── adv_matrices_task_{0-7}.zip                 # Step 3 (chunked)
├── comparison/representation_comparison.json   # Step 4 (6×3 grid + cost)
├── comparison/svd_ablation.json                # Step 4 --svd_ablation
├── theorem45/theorem45_results.json            # Step 2c
├── isomorphism/isomorphism_results.json        # Isomorphism experiment
├── calibration.json                            # GPU calibration
├── checkpoints/                                # Per-step completion tracking
└── orchestrator_jobs/                          # Generated Slurm scripts
tables/*.tex                                    # Step 5 (LaTeX tables)
slurm_out[_test]/, slurm_err[_test]/            # Slurm logs
gpu-monitor/{experiment}.{step}.{chunk}.log     # GPU monitoring
reports/{experiment}_report_{timestamp}.txt     # Pipeline reports
```

Log filename pattern: `PIPE_{STEP_ID}_{experiment}[_c{chunk}]_{jobid}.{out|err}` (step IDs: 1, 2a, 2b, 2c, 3, 4, 5)

## Key Files

| File | Role |
|------|------|
| `experiment_config.sh` | Shared config — accounts, resource profiles, helper functions |
| `calibration.sh` | Standalone GPU calibration — produces `calibration.json` |
| `run_experiment.sh` | Main orchestrator — generates and submits all Slurm jobs |
| `constants/constants.py` | All experiment configs, attack lists, architectures |
| `utils/utils.py` | `get_architecture()`, `get_model()`, `get_dataset()`, `get_device()` |
| `utils/data_integrity.py` | `verify_experiment()`, zip verification |
| `utils/error_classification.py` | Slurm error parsing and classification |
| `matrix_construction/parallel.py` | `ParallelMatrixConstruction` — chunked matrix computation |
| `calibrate.py` | GPU batch_size binary search, SLURM time/memory estimation |
| `pipeline_report.py` | Post-run analysis with error classification |
| `collect_errors.py` | Aggregate and report pipeline errors |
| `auto_resubmit.py` | Automatic OOM resubmission with memory doubling |
| `job_recovery.sh` | Manual job recovery for failed steps |
| `compare_representations.py` | Step 4: 6 detectors × 3 representations + Lee et al. baseline |
| `generate_latex_tables.py` | Step 5: generates comparison LaTeX tables |
| `validate_theorem45.py` | Step 2c: empirical validation of Theorem 4.5 (distance lower bound) |
| `baselines/lee2018.py` | Lee et al. (2018) multi-layer Mahalanobis baseline detector |
| `isomorphism_experiment.py` | Isomorphism invariance demonstration |
| `docs/deep_code_review.md` | Full code review: 47 findings with severity and fix status |

## Environment

- Cluster modules: `StdEnv/2023 python/3.11.5 scipy-stack/2025a`
- Virtual env: `env/` (created via `python -m venv env`)
- Key deps: PyTorch, torchattacks, knowledgematrix (git+samueleblanc@0d26c7a)
