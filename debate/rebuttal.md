# Rebuttal Debate: Converging on an Implementation Plan

**Paper:** "Hidden Activations Are Not Enough"
**Context:** TMLR rejection. Four agents debate the path to resubmission, converging on a concrete 8-step implementation plan centered on reframing the paper as a representation comparison.
**Participants:** Kv2C (Adversarial ML Expert), BFn5 (Critical Experimentalist), Vj72 (Sympathetic Theorist), Defender (Authors' Advocate)

---

## 1. Opening Statements

> **Kv2C:** Let me be direct. The paper was rejected because the evaluation is not legible to the detection community. You report `good_defence - wrong_rejection` in `grid_search.py` (line 70, the `--rej_lev` sweep), and no reviewer in adversarial ML will take that seriously. AUROC, AUPR, and FPR@95TPR are non-negotiable for any detection paper published after 2019. Carlini et al. (2019) settled this, and RobustBench codified it. The 17-attack suite is strong --- I acknowledged that in the first debate --- but without standard metrics, none of that work is visible to the community.

> **BFn5:** I agree with Kv2C on metrics, but the deeper problem is the comparison is rigged. In `grid_search.py`, the ellipsoid detector sweeps 9 x 9 x 9 = 729 combinations of `t_epsilon`, `epsilon`, and `epsilon_p` (lines 28--47). Meanwhile, each baseline in `detect_adversarial_examples.py` gets 3 hyperparameter settings per detector. That is not a fair fight. More fundamentally, knowledge matrices for AlexNet on CIFAR-10 are 30,720-dimensional, while penultimate features are 4,096-dimensional. The code applies `TruncatedSVD(n_components=256)` in `MahalanobisDetector.fit()` (line 174 of `compare_representations.py`), so both end up at 256 dimensions --- but we never check whether the SVD rank matters. An SVD ablation sweeping rank from 16 to 512 is the minimum needed to control the dimensionality confound.

> **Vj72:** The right experiment already exists in embryonic form. `compare_representations.py` has a `MahalanobisDetector` with a clean `fit()`/`score()` API, AUROC computation via `roc_auc_score` (line 390), and extraction functions for all three representations. The file already imports `NearestNeighbors`, `KernelDensity`, `GaussianMixture`, `OneClassSVM`, and `IsolationForest` (lines 34--38) but only defines one detector class. What I want is simple: 6 detectors x 3 representations, all sharing the same `fit(features, labels, num_classes)` / `score(features)` interface. That produces an 18-cell grid. The central table in the paper should be that grid, averaged across attacks. The ellipsoid becomes a bonus row, not the headline.

> **Defender:** The reframing is what saves this paper. The original submission said "we propose a detector." The resubmission should say "we compare representations." Once you do that, the central claim becomes: *when the same standard detectors are applied to knowledge matrices instead of penultimate features, detection improves consistently.* That claim is tested by Vj72's 18-cell grid, not by the ellipsoid's 729-point grid search. The ellipsoid results move to a separate section titled "Exploiting KM Geometry" --- a bonus that leverages the convexity theorem (Theorem 4.4). This reframing neutralizes three of the four fatal objections: non-standard metrics (the grid uses AUROC), unfair HP comparison (same default HPs for all detectors on all representations), and the adaptive attack concern (we claim representation quality, not defense robustness).

---

## 2. Points of Agreement

All four agents agree on the following:

1. **Standard metrics are mandatory.** AUROC, AUPR, and FPR@95TPR must be reported for every cell in the comparison. The code already computes AUROC (`roc_auc_score` at line 390 of `compare_representations.py`) and `roc_curve` (line 395); AUPR via `average_precision_score` is imported (line 39) but not called. FPR@95TPR is a simple threshold lookup on the existing ROC curve arrays.

2. **Same 6 detectors x 3 representations is the right experiment.** KNN, KDE, GMM, OCSVM, Isolation Forest, and Mahalanobis applied to knowledge matrices, penultimate features, and all-layer features. This is a clean factorial design that isolates the representation's contribution.

3. **Computational cost table is essential.** Wall-clock time and peak GPU memory for each representation's extraction, per architecture. The pipeline already calibrates GPU batch sizes in `calibrate.py`; timing instrumentation is a small addition.

4. **The ellipsoid should be presented separately.** It is a KM-specific detector that exploits Theorem 4.4's convexity guarantee. It belongs in its own section ("Exploiting KM Geometry"), not as the main result competing against standard baselines.

---

## 3. Points of Contention and Resolution

### Dimensionality confound

> **BFn5:** Any performance advantage of knowledge matrices could simply be "more features, not better features." KMs are 8--75x larger than penultimate vectors before SVD. We need a controlled experiment.

> **Vj72:** I agree this is a valid concern, but concatenating random intermediate layers is not a clean control either --- those layers have different semantics. A better approach: sweep the SVD rank applied to KMs and show that detection quality does not collapse at low rank. If KMs at rank 64 still beat penultimate features at their natural dimensionality, the advantage is structural, not dimensional.

> **Defender:** The SVD rank ablation resolves this cleanly. Add a `--svd_ablation` flag to `compare_representations.py` that sweeps `max_components` in `{16, 32, 64, 128, 256, 512}` for all representations equally.

**Resolution:** SVD rank ablation is Step 4 in the plan. Sweep rank across all representations. If KMs win at matched rank, the confound is controlled.

### Confidence intervals

> **BFn5:** Results without error bars are anecdotal. I want 3--5 random seeds.

> **Kv2C:** On an HPC cluster with 8 parallel Slurm jobs per matrix computation step, repeating 4 experiments x 5 seeds would require 160 Slurm array jobs for matrices alone, plus adversarial generation. That is not feasible within a reasonable compute budget.

> **Defender:** The cross-attack standard deviation is a meaningful proxy. With 17 attacks per experiment, we have 17 AUROC values per cell. Reporting mean and standard deviation across attacks shows how robust the representation advantage is without re-running the full pipeline.

**Resolution:** Report cross-attack standard deviation as the variability measure. Acknowledge the absence of multi-seed runs as a limitation. BFn5 accepts this under protest.

### Softmax baseline (MaxProb / ODIN)

> **Kv2C:** Any detection paper should include a softmax-confidence baseline. It is trivial to implement and is the weakest reasonable baseline.

> **Vj72:** Softmax scores are not a "representation" in the same sense as feature vectors. They are scalar per class. Including them in the Representation x Detector grid would require a different pipeline --- you cannot fit a KDE on a 10-dimensional softmax vector and compare it to fitting a KDE on 4,096-dimensional penultimate features.

> **Defender:** Include MaxProb AUROC as a reference line in the table (one column, no detector decomposition), but in the appendix. The core experiment stays clean: 3 representations x 6 detectors.

**Resolution:** MaxProb appears as an appendix-only reference. Not part of the main grid.

---

## 4. The Incremental Plan (8 Steps)

The agents converge on 8 concrete implementation steps:

### Step 1: Add 5 detector classes to `compare_representations.py`

Define `KNNDetector`, `KDEDetector`, `GMMDetector`, `OCSVMDetector`, and `IsolationForestDetector`, each with `fit(features, labels, num_classes)` and `score(features)` methods matching `MahalanobisDetector` (line 157). All imports are already present (lines 34--38). Each detector wraps its sklearn counterpart with the same SVD projection used by `MahalanobisDetector`:

- **KNNDetector**: fits `NearestNeighbors(n_neighbors=5)` on training data; `score()` returns mean distance to k neighbors (higher = more anomalous).
- **KDEDetector**: fits `KernelDensity(bandwidth=1.0)` per class; `score()` returns negative log-likelihood.
- **GMMDetector**: fits `GaussianMixture(n_components=5)` per class; `score()` returns negative log-likelihood.
- **OCSVMDetector**: fits `OneClassSVM(kernel='rbf', nu=0.1)` per class; `score()` returns negative decision function.
- **IsolationForestDetector**: fits `IsolationForest(n_estimators=100)` on training data; `score()` returns negative anomaly score.

Default hyperparameters for all detectors, no per-detector tuning.

### Step 2: Modify `run_comparison()` to iterate over all 6 detectors

Currently `run_comparison()` (line 226) fits only `MahalanobisDetector`. Change it to iterate over a `DETECTORS` dict:

```python
DETECTORS = {
    'mahalanobis': MahalanobisDetector,
    'knn': KNNDetector,
    'kde': KDEDetector,
    'gmm': GMMDetector,
    'ocsvm': OCSVMDetector,
    'iforest': IsolationForestDetector,
}
```

This produces an 18-cell grid (6 detectors x 3 representations) per attack, plus aggregated averages. Output JSON grows from `{attack: {representation: metrics}}` to `{attack: {detector: {representation: metrics}}}`.

### Step 3: Add AUPR and FPR@95TPR metrics

`average_precision_score` is already imported (line 39) but never called. Add after the AUROC computation (line 390):

```python
aupr = average_precision_score(labels, scores)
fpr_at_95tpr = fpr_arr[np.searchsorted(tpr_arr, 0.95, side='left')] if len(tpr_arr) > 1 else 1.0
```

Every cell in the grid reports three numbers: AUROC, AUPR, FPR@95TPR. The main table uses AUROC; AUPR and FPR@95TPR go in supplementary tables.

### Step 4: Add SVD rank ablation mode

Add `--svd_ablation` flag to `compare_representations.py` argument parser (after line 548). When active, sweep `max_components` in `{16, 32, 64, 128, 256, 512}` for all detectors on all representations. Output a separate JSON file `svd_ablation.json` under `experiments/{experiment}/comparison/`.

This directly addresses BFn5's dimensionality confound. If KMs at rank 64 outperform penultimate features at rank 256, the advantage is structural.

### Step 5: Add computational cost measurement

Wrap each representation extraction call in `time.perf_counter()` and `torch.cuda.max_memory_allocated()`. Record:

- Wall-clock time per sample (seconds)
- Peak GPU memory (MB)
- Total extraction time for the full dataset

Store in `experiments/{experiment}/comparison/cost.json`. This feeds into a LaTeX cost table in Step 6.

### Step 6: Generate new LaTeX tables in `generate_latex_tables.py`

Extend `generate_latex_tables.py` (currently reads `grid_search.txt` and `baseline.txt`) to also read `representation_comparison.json`. Generate three new tables:

- **Table 1 (main result):** Representation x Detector AUROC grid, averaged across attacks. Rows = 3 representations, columns = 6 detectors. One table per experiment, plus a grand-average table.
- **Table 2:** Per-attack AUROC for the best detector on each representation.
- **Table 3:** Computational cost (time and memory per representation per architecture).

The ellipsoid results from `grid_search.txt` appear in a separate table in the "Exploiting KM Geometry" section.

### Step 7: Integrate as Step Gc in `run_experiment.sh`

Add a new Slurm step between Gb and H. Step Gc depends on A (weights), B (training matrices), C (adversarial examples), and F (adversarial matrices). It runs on GPU (for feature extraction) and produces the comparison JSON.

```bash
# Step Gc: Representation Comparison (depends on A + all B + C + all F)
python compare_representations.py --experiment $EXPERIMENT --temp_dir $SLURM_TMPDIR
```

Step H's dependency list expands from `Ga, Gb` to `Ga, Gb, Gc`.

### Step 8: Add unit tests for all detectors

Add `unit_test/test_detectors.py` alongside the existing tests in `unit_test/`. For each of the 6 detector classes:

- Test `fit()` on synthetic Gaussian data (100 samples, 20 features, 3 classes).
- Test `score()` returns array of correct length.
- Test that adversarial-like outlier points score higher than inlier points.
- Test SVD path triggers when `features.shape[1] > max_components`.

Run with `python -m pytest unit_test/test_detectors.py -v`.

---

## 5. Explicit Exclusions

The following items are deliberately **not** part of this plan:

- **Adaptive attacks.** The reframing from "defense" to "representation comparison" makes adaptive attack evaluation less critical. Tramer et al. (2020) will be discussed in the paper narrative: we argue that the relevant question is whether the representation advantage persists under stronger attacks (which the 17-attack suite with varying strengths already probes), not whether a specific detector can be bypassed. No new code.

- **LID detector (Ma et al., 2018).** Deferred to future work. LID requires per-layer intrinsic dimensionality estimation with careful minibatch handling. Adding it does not change the representation comparison story --- 6 detectors are already more than most papers use.

- **Multiple random seeds.** Full pipeline re-runs are too expensive on the Compute Canada allocation. Cross-attack standard deviation serves as a variability proxy. Stated as a limitation.

- **ImageNet experiments.** The `alexnet_imagenet` config exists in `constants.py` but has not been run. At ImageNet scale (k=1000, d=150,528), each knowledge matrix would be ~600 MB in float32. Stated as future work with an honest cost estimate.

- **Vision Transformers.** The `knowledgematrix` library (git+samueleblanc@0d26c7a) does not support attention layers. Stated as a limitation with a theoretical discussion of how quiver representations could extend to self-attention modules.

---

## 6. Verification Checklist

Each TMLR reviewer concern is mapped to its resolution:

| Reviewer Concern | Resolution | Implementation Step |
|---|---|---|
| Non-standard metrics (`good_defence - wrong_rejection`) | Report AUROC, AUPR, FPR@95TPR for every cell | **Step 3** --- `average_precision_score` already imported at line 39 of `compare_representations.py`; FPR@95TPR is a threshold lookup on existing `roc_curve` output |
| Unfair hyperparameter comparison (729 vs 3) | Same 6 detectors with default HPs on all representations; no per-detector tuning | **Step 2** --- `DETECTORS` dict with fixed default parameters |
| Dimensionality confound (KMs 8--75x larger) | SVD rank ablation across all representations | **Step 4** --- `--svd_ablation` flag sweeping rank in {16, 32, 64, 128, 256, 512} |
| Missing computational cost analysis | Wall-clock time + GPU memory per representation per architecture | **Step 5** --- `cost.json` output, fed into LaTeX table |
| No AUROC reported | AUROC is already computed in `compare_representations.py`; now supplemented with AUPR and FPR@95TPR | **Step 3** |
| Weak baselines (only ellipsoid) | 6 standard detectors (KNN, KDE, GMM, OCSVM, IsolationForest, Mahalanobis) on all 3 representations | **Steps 1--2** --- 5 new detector classes + modified `run_comparison()` loop |
| No baseline comparison framework | Central table becomes Representation x Detector AUROC grid | **Step 6** --- new LaTeX tables in `generate_latex_tables.py` |
| Reproducibility concerns | Unit tests for all detectors; deterministic SVD (`random_state=0`) | **Step 8** --- `unit_test/test_detectors.py` |

---

## Closing Remarks

> **Defender:** This plan turns the paper's greatest weakness --- the ellipsoid-centric evaluation --- into a strength. The 18-cell Representation x Detector grid is the cleanest possible experiment for our actual claim. The ellipsoid results are still there, in their own section, for readers who care about KM-specific geometry. The reframing is not spin; it is what the theory always supported.

> **Kv2C:** I am satisfied that standard metrics will be reported. My remaining concern --- adaptive attacks --- is addressed narratively, and I accept the argument that the representation framing shifts the burden. I would upgrade my recommendation to Accept if the 8 steps are executed cleanly.

> **BFn5:** The SVD ablation is the key experiment. If KMs at rank 64 outperform penultimate features at rank 256 across multiple detectors, I have no remaining objection to the scientific claim. I still want multi-seed runs, but I accept that cross-attack variance is a reasonable proxy given compute constraints.

> **Vj72:** The 6-detector x 3-representation design is what I asked for in the first debate. The `fit()`/`score()` API makes the code auditable, the SVD ablation controls the confound, and the cost table is honest about scalability. I look forward to seeing the results.
