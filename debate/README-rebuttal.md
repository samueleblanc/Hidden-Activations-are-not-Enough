# Rebuttal Experiment Plan — Implementation Summary

**Paper:** "Hidden Activations Are Not Enough"
**Target:** TMLR resubmission
**Status:** Steps 1–8 implemented

---

## Core Change: Representation Comparison, Not Detector Proposal

The original submission positioned the ellipsoid detector as the main contribution. The resubmission leads with: *when the same standard detectors are applied to knowledge matrices instead of penultimate features, detection improves consistently.*

This is tested by a **6 detectors x 3 representations** factorial experiment.

---

## What Was Implemented

### Steps 1–5: `compare_representations.py` (919 lines)

| Step | What | Status |
|------|------|--------|
| 1 | 6 detector classes (Mahalanobis, KNN, KDE, GMM, OCSVM, IsolationForest) | Done |
| 2 | `run_comparison()` iterates all 6 detectors x 3 representations | Done |
| 3 | AUROC + AUPR + FPR@95TPR metrics via `compute_detection_metrics()` | Done |
| 4 | SVD rank ablation (`--svd_ablation` flag, ranks 16–512) | Done |
| 5 | Computational cost measurement (wall-clock time + GPU memory) | Done |

Output: `experiments/{experiment}/comparison/representation_comparison.json`

### Step 6: `generate_latex_tables.py` — 4 New Tables

| Table | File | Content |
|-------|------|---------|
| Representation comparison | `representation_comparison.tex` | 6 detectors x 3 reps, mean AUROC |
| Per-attack AUROC | `per_attack_auroc_{exp}.tex` | Best detector per rep, per attack |
| Computational cost | `cost_comparison.tex` | Time, memory, dimensionality |
| SVD ablation | `svd_ablation_{exp}.tex` | Rank vs. AUROC per representation |

### Step 7: Step Gc in `run_experiment.sh`

- GPU job (H100, 8 CPUs, 8h, 64G)
- Depends on A + all B + C + all F (same as Gb)
- Copies weights, training matrices, adversarial examples, adversarial matrices to `$SLURM_TMPDIR`
- Runs `compare_representations.py --experiment $EXPERIMENT --temp_dir $SLURM_TMPDIR --svd_ablation`
- Copies results back, writes checkpoint
- Step H now depends on Ga + Gb + **Gc**
- Recovery path included

### Step 8: Unit Tests — `unit_test/test_detectors.py`

20 tests covering:
- All 6 detector fit/score APIs
- Score polarity (higher = more anomalous)
- SVD path activation
- Edge cases: single-class data, zero-variance features, small datasets, empty inputs
- `compute_detection_metrics()` correctness

### LaTeX Rebuttal Document: `debate/rebuttal-experiments.tex`

Standalone LaTeX sections for the paper revision covering:
1. **Experimental methodology** — factorial design justification
2. **SVD rank ablation theory** — dimensionality confound argument
3. **Metric justification** — AUROC/AUPR/FPR@95TPR (Carlini et al. 2019, RobustBench)
4. **Reviewer concern mapping** — table linking each concern to resolution
5. **Adaptive attack discussion** — reframing argument + theoretical + empirical evidence
6. **Narrative reframing** — "representation comparison" framing

---

## How to Run

```bash
# Run the representation comparison (on cluster with GPU):
python compare_representations.py --experiment alexnet_cifar10 --temp_dir $SLURM_TMPDIR --svd_ablation

# Run for all experiments:
python compare_representations.py --experiment lenet_cifar10 alexnet_cifar10 resnet_cifar10 vgg_cifar10 --svd_ablation

# Generate LaTeX tables (after comparison runs):
python generate_latex_tables.py --output tables/

# Run unit tests:
python -m pytest unit_test/test_detectors.py -v

# Full pipeline (includes Step Gc):
bash run_experiment.sh alexnet_cifar10
```

---

## Key Arguments for Reviewers

### Why this experiment is definitive
The factorial design eliminates detector-design confounds. Any performance difference between representations is due entirely to the representation, since the same detector with the same hyperparameters is applied to each.

### Why SVD ablation matters
KMs are 8–75x larger than penultimate features before SVD. If KMs at rank 64 beat penultimate features at rank 256, the advantage is **structural** (algebraic structure of quiver representations), not **dimensional** (more raw features).

### Why adaptive attacks are less critical under reframing
We claim representation quality, not defense robustness. The 17-attack suite with 4 categories (gradient, AutoAttack, gradient-free, elastic-net) probes whether the representation advantage persists under diverse attack strategies.

---

## File Map

| File | Role |
|------|------|
| `compare_representations.py` | Steps 1–5: detectors, comparison loop, metrics, SVD ablation, cost |
| `generate_latex_tables.py` | Step 6: 4 new AUROC-based LaTeX tables |
| `run_experiment.sh` | Step 7: Step Gc integration + recovery |
| `unit_test/test_detectors.py` | Step 8: 20 unit tests |
| `debate/rebuttal-experiments.tex` | LaTeX text for paper revision |
| `debate/rebuttal.md` | Original 4-agent debate + 8-step plan |
| `debate/report.md` | Initial academic debate report |
