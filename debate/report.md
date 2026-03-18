# Academic Debate Report: Strengthening "Hidden Activations Are Not Enough"

## Executive Summary

Four agents (three simulated reviewers + one paper defender) conducted deep research and debated the revised version of the paper. The consensus: **the revision substantially addresses the original rejection reasons, but several critical gaps remain that must be fixed before resubmission.**

### Verdict by Agent

| Agent | Original Verdict | Revised Verdict | Key Remaining Issue |
|-------|-----------------|-----------------|---------------------|
| Kv2C (Adversarial ML) | Reject | Borderline Accept | Non-standard metrics, no adaptive attack discussion |
| BFn5 (Experimentalist) | Reject | Major Revision | Unfair hyperparameter comparison, dimensionality confound |
| Vj72 (Theorist) | Weak Accept | Accept | Needs AUROC metrics and computational cost table |
| Defender | — | Strong Accept | Must execute the reframing cleanly |

### The Three Must-Fix Issues (All Agents Agree)

1. **Report AUROC, AUPR, and FPR@95TPR** — the codebase uses non-standard `good_defence - wrong_rejection`. Without standard metrics, the paper cannot be compared to any published baseline.

2. **Add computational cost analysis** — wall-clock time and memory for KM construction vs. penultimate feature extraction, per architecture. All reviewers flagged this.

3. **Reframe as representation comparison, not defense** — lead with "KMs are a better representation" rather than "we propose a detector." This neutralizes the adaptive attack concern and aligns theory with experiments.

---

## Part I: How Original Concerns Are Addressed

### Datasets (MNIST/FashionMNIST → CIFAR-10/100)

**Status: Substantially addressed.**

All agents agree that CIFAR-10/100 is a meaningful step up and is the standard benchmark for adversarial robustness papers (RobustBench uses CIFAR-10 as its primary dataset). ImageNet is desirable but not required for a theory paper — the `alexnet_imagenet` config exists in the codebase but experiments haven't been completed.

> **Kv2C:** "CIFAR-10/100 with multiple architectures may suffice if the theoretical contribution is strong."
>
> **Defender:** "CIFAR-10/100 is the standard for theory papers. Zhang et al. (ICLR 2017 Best Paper) used CIFAR-10. TMLR policy states work should not be rejected for not achieving SOTA."

### Architectures (MLPs → CNNs)

**Status: Substantially addressed.**

AlexNet, ResNet, VGG, LeNet cover the CNN design space (shallow, deep sequential, residual, compact). This demonstrates KM construction works with convolutions, pooling, batch norm, and skip connections.

**Gap: No Vision Transformers.** All reviewers note this. If ViTs can't be supported by the `knowledgematrix` library, this must be stated as a limitation with a theoretical discussion of extensibility.

> **BFn5:** "The paper should be honest about ViT applicability as a limitation rather than presenting the framework as universally applicable."

### Attack Diversity (Redundant PGD → 17 Diverse Attacks)

**Status: Fully addressed.**

The 17-attack suite includes the full AutoAttack ensemble (APGD-CE, APGD-T, FAB, Square), gradient-free attacks (Square, SPSA, Pixle), elastic-net attacks (EADL1, EADEN), and standard gradient attacks. This exceeds the evaluation breadth of most published detection papers.

> **Kv2C:** "This is an excellent attack suite... The inclusion of gradient-free attacks is particularly valuable."

**Important caveat:** The paper must explicitly state epsilon values and norm constraints for each attack, matching RobustBench conventions (L-inf 8/255 for CIFAR-10).

### Baseline Comparisons (None → 6 Standard Detectors)

**Status: Fully addressed with an elegant design.**

Running KNN, KDE, GMM, OCSVM, Isolation Forest, and Mahalanobis on BOTH penultimate features AND knowledge matrices is the gold standard for representation comparison. This eliminates detector-design confounds and directly tests the core claim.

> **Vj72:** "This is exactly what I had hoped for... the correct experimental design for the claim being made."
>
> **Defender:** "Running the same 6 detectors on both representations is the cleanest possible experiment. No confounds."

### Adaptive Attacks (Not Addressed → Reframing Defense)

**Status: Not directly addressed, but the reframing significantly mitigates the concern.**

No adaptive attacks are implemented. However, the paper's reframing from "defense" to "representation comparison" changes the relevance of this concern:

> **Defender:** "The adaptive attack concern applies to papers proposing defenses. KMs are a mathematical property of the forward pass — they cannot be 'fooled' separately from the network itself. By Theorem 4.5, any perturbation that preserves the KM also preserves the logits."
>
> **Kv2C:** "The reframing significantly mitigates this concern... the relevant question changes from 'can an adaptive adversary bypass this defense?' to 'does the representation quality advantage persist under stronger attacks?'"
>
> **BFn5 (dissenting):** "Despite the reframing, any claim that KMs enable better detection IS a defense claim. This is the single most serious issue."

**Recommendation:** Add a discussion paragraph explicitly addressing adaptive attacks, citing Tramer et al. (2020), and arguing why the representation framing makes this less critical. Optionally, show that the KM advantage is stable as attack strength increases (varying PGD steps/epsilon).

---

## Part II: New Concerns Raised by the Revision

### 1. Non-Standard Evaluation Metrics (ALL AGENTS — CRITICAL)

The codebase uses `good_defence` (TPR) minus `wrong_rejection` (FPR) as the selection criterion. The adversarial/OOD detection community universally reports **AUROC, AUPR, and FPR@95TPR**.

> **BFn5:** "The code does not compute AUROC anywhere in the main detection pipeline. This is a significant omission."
>
> **Vj72:** "Reporting these metrics — even if computed post hoc from existing results — would dramatically increase the paper's accessibility."

**Action required:** Implement AUROC computation from existing detection scores. This is straightforward: sweep the threshold and compute ROC curves from the per-sample detection scores.

### 2. Unfair Hyperparameter Comparison (BFn5 — CRITICAL)

The ellipsoid detector (Step Ga) searches over 729 parameter combinations (9 values each for t_epsilon, epsilon, epsilon_p), while each baseline detector has only 3 hyperparameter settings.

> **BFn5:** "This is an unfair comparison. With so many degrees of freedom, one can almost always find a parameter setting that 'works.'"

**Solutions (pick one):**
- Give baselines equal hyperparameter budget
- Use cross-validation to select KM detector parameters
- Fix KM parameters across experiments (same parameters for all architectures/datasets)
- Focus the main comparison on Step Gb (same detectors on both representations) rather than Step Ga vs baselines

### 3. Dimensionality Confound (BFn5, Vj72 — IMPORTANT)

Knowledge matrices are much higher-dimensional than penultimate features:
- AlexNet on CIFAR-10: KM = 10 × 3072 = 30,720 vs. penultimate = 4,096
- VGG on CIFAR-100: KM = 100 × 3072 = 307,200 vs. penultimate = 4,096

The code applies TruncatedSVD (capped at 256 dimensions) before running detectors on matrices.

> **BFn5:** "Any performance difference could be explained by the information-theoretic advantage of having more features, not by the specific mathematical properties of quiver representations."
>
> **Vj72:** "If the KMs must be projected to 256 dimensions for the detector to work, what advantage do they have over penultimate features already in a compact space?"

**Recommended experiments:**
1. **SVD rank ablation:** Sweep TruncatedSVD rank and report detection performance
2. **Dimensionality control:** Concatenate features from multiple intermediate layers to match KM dimensionality; if this performs similarly, the advantage is "more features" not "better features"
3. **Intrinsic dimensionality analysis:** Compare explained variance curves for KMs vs. penultimate features

### 4. Computational Cost (ALL AGENTS — IMPORTANT)

Knowledge matrix construction is substantially more expensive than penultimate feature extraction. The pipeline uses 8 parallel Slurm jobs on H100 GPUs for matrix computation, with GPU-calibrated batch sizes.

> **Vj72:** "For ImageNet (k=1000, d=150,528), each knowledge matrix would be ~150 million entries per sample — ~600 MB in float32."
>
> **Defender:** "The NTK requires O((NO)^2) memory and O((NO)^3) computation, yet is one of the most cited theoretical contributions. Computational cost and scientific validity are different questions."

**Action required:** Add a table reporting wall-clock time and memory for KM construction vs. feature extraction, per architecture. Be transparent about scalability limitations.

### 5. Model Accuracy (BFn5, Kv2C — MODERATE)

ResNet on CIFAR-100 achieves only 50.17% accuracy. State-of-the-art is 75-78%.

> **BFn5:** "A model that is wrong half the time raises serious questions about what 'adversarial' even means."

**Response:** The paper should discuss whether detection quality correlates with base model accuracy. If possible, improve the ResNet-CIFAR100 training or replace with a better-tuned model.

### 6. Theory-Experiment Bridge (BFn5, Vj72 — MODERATE)

The theoretical distance lower bound (Theorem 4.5) is not directly validated experimentally.

> **BFn5:** "The experiments show different performance but do not isolate WHY KMs work differently."
>
> **Vj72:** "I would like to see ablation studies on the tightness of the distance lower bound."

**Recommended experiment:** Compute actual KM-space distances vs. logit-space distances for clean and adversarial examples. Show distributions. This directly validates Theorem 4.5.

---

## Part III: The Reframing Strategy

### The Core Narrative (Defender's Recommendation)

> "Knowledge matrices are the unique neural network representation that is simultaneously information-complete, isomorphism-invariant, and geometrically well-structured. We show that when the same standard anomaly detectors are applied to knowledge matrices instead of penultimate-layer features, adversarial detection improves consistently across architectures and attacks."

### Why This Framing Works

All agents agree the reframing is stronger than the original:

1. **It is a more fundamental claim.** Representation quality is upstream of detector design — if KMs are better inputs, they improve ALL detectors simultaneously.

2. **It aligns with the literature.** Reiss et al. (2022) explicitly argue that "tackling the next generation of anomaly detection tasks requires improvements in representation learning." PatchCore (Roth et al., CVPR 2022) achieved 99.6% AUROC on MVTec by choosing better features, not a better detector.

3. **It neutralizes the adaptive attack concern.** The paper claims representation quality, not defense robustness. Even if an adaptive attacker could bypass KM-based detection, the representation comparison remains valid.

4. **It matches the theory.** The theorems prove properties of the representation (invariance, completeness, convexity, distance bounds), not properties of any particular detector.

### Recommended Paper Structure

1. **Abstract:** Lead with "knowledge matrices are a better representation than penultimate features for adversarial detection"
2. **Introduction:** Motivate the representation comparison angle; position within representation learning
3. **Background (Section 3):** Quiver representations, knowledge matrices (same as original)
4. **Theory (Section 4):** Theorems 4.1-4.5 (same as original)
5. **NEW Section 4.5: "Why More Information Should Help Detection"**
   - Forgetful functor: penultimate features lose information
   - Data Processing Inequality: I(adversarial; features) ≤ I(adversarial; KM)
   - KM as sufficient statistic for logits
   - Cite Tishby (2015), Reiss (2022), Bengio (2013)
6. **Experiments (Section 5):**
   - Table 1: Same 6 detectors on KMs vs. penultimate features (central result)
   - Table 2: Ellipsoid detector results (KM-specific geometry)
   - Table 3: Per-attack category breakdown
   - Table 4: Computational cost comparison
   - Appendix: Full per-attack, per-detector results
7. **Discussion:** Why KMs help, when they fail, scalability limitations, adaptive attack discussion
8. **Future Work:** ViTs, ImageNet, efficient KM computation, OOD detection

---

## Part IV: Strongest Arguments — Synthesis

### FOR Acceptance (All Agents Contributed)

1. **Unique mathematical contribution.** No other representation in the adversarial/OOD detection literature has provable isomorphism invariance, information completeness, and convex class regions. This is genuinely novel.

2. **Controlled experimental design.** Same 6 detectors on both representations is the gold standard for representation comparison. This eliminates confounds from detector design.

3. **Comprehensive attack diversity.** 17 attacks including full AutoAttack, gradient-free, and elastic-net attacks exceed the evaluation standard of most published detection papers.

4. **Meaningful scale-up.** CIFAR-10/100 with 4 CNN architectures (AlexNet, ResNet, VGG, LeNet) demonstrates the method works beyond toy settings.

5. **Theoretical guarantees are rare.** The proven distance lower bound (Theorem 4.5) provides formal justification for why KMs should be better for detection. Most adversarial detection work is purely empirical.

6. **The convexity result connects to emerging literature.** Tetkova et al. (Nature Communications 2025) independently found that approximate convexity is pervasive in neural representations. Theorem 4.4 proves exact convexity in KM space — a stronger result.

7. **Information-theoretic grounding.** The Data Processing Inequality argument (penultimate features = lossy compression of KMs) provides a principled explanation for why KMs should outperform features.

### AGAINST Acceptance (All Agents Contributed)

1. **Non-standard metrics.** Without AUROC/AUPR/FPR@95TPR, results cannot be compared to any published baseline. This is a dealbreaker for reviewers in the detection community.

2. **Unfair hyperparameter comparison.** 729 grid points for KM detector vs. 3 for baselines creates an asymmetry that could invalidate comparative claims.

3. **Dimensionality confound not controlled.** KMs being 8-75x higher-dimensional than penultimate features, then compressed via SVD, means the comparison may reflect "more features" rather than "better features."

4. **No adaptive attack evaluation or discussion.** While the reframing mitigates this, the absence of any discussion of Tramer et al. (2020) is conspicuous.

5. **Computational cost unreported.** If KMs are 100x more expensive than feature extraction, the practical value is limited regardless of detection quality.

6. **Below-standard model accuracy.** ResNet on CIFAR-100 at 50% accuracy weakens the generalizability of conclusions.

7. **No ViTs, no ImageNet.** In 2026, a paper about neural network representations without transformer experiments faces relevance questions.

---

## Part V: Prioritized Action Items

### Tier 1 — Must Do (blocks acceptance)

| # | Action | Effort | Agent Source |
|---|--------|--------|-------------|
| 1 | **Add AUROC/AUPR/FPR@95TPR metrics** — sweep thresholds on existing detection scores | Medium (code change) | All agents |
| 2 | **Reframe the paper narrative** — lead with "representation comparison," not "detector proposal" | Low (writing) | Defender |
| 3 | **Add computational cost table** — wall-clock time and memory per architecture | Low (benchmarking) | All agents |
| 4 | **Add adaptive attack discussion paragraph** — cite Tramer et al. (2020), explain why the representation framing mitigates the concern | Low (writing) | Kv2C, BFn5 |
| 5 | **Equalize or acknowledge hyperparameter asymmetry** — give baselines more HP budget OR focus comparison on Step Gb (same detectors) | Medium (code/analysis) | BFn5 |

### Tier 2 — Should Do (significantly strengthens paper)

| # | Action | Effort | Agent Source |
|---|--------|--------|-------------|
| 6 | **SVD rank ablation** — sweep TruncatedSVD dimensions for matrix-based baselines | Medium (experiment) | BFn5, Vj72 |
| 7 | **Dimensionality control experiment** — concatenate multi-layer features to match KM dimensionality | Medium (experiment) | BFn5 |
| 8 | **Validate Theorem 4.5 empirically** — compute KM distances vs logit distances, show distributions | Medium (analysis) | BFn5, Vj72 |
| 9 | **Specify threat model explicitly** — epsilon values, norm types, attacker knowledge | Low (writing) | Kv2C |
| 10 | **Report confidence intervals** — run key experiments with 3+ seeds or use cross-validation | High (compute) | BFn5 |
| 11 | **State ViT limitation explicitly** — discuss theoretical extensibility to attention | Low (writing) | Kv2C, BFn5 |

### Tier 3 — Nice to Have (polishes the paper)

| # | Action | Effort | Agent Source |
|---|--------|--------|-------------|
| 12 | **Add information-theoretic argument section** — DPI, sufficient statistics, forgetful functor | Low (writing) | Defender, Vj72 |
| 13 | **Add LID as a 7th baseline detector** | Medium (code) | Kv2C |
| 14 | **Improve ResNet-CIFAR100 accuracy** | High (retraining) | BFn5, Kv2C |
| 15 | **Run alexnet_imagenet experiment** — config already exists | High (compute) | Kv2C, BFn5 |
| 16 | **Connect convexity to Tetkova et al. (2025)** | Low (writing) | Vj72 |
| 17 | **Attack-strength sensitivity analysis** — vary PGD steps and epsilon | Medium (experiment) | Kv2C |
| 18 | **Add Lee et al. (2018) multi-layer Mahalanobis** as additional baseline | Medium (code) | Vj72 |

---

## Part VI: Key References (Deduplicated)

### Adversarial Evaluation Standards
- Croce & Hein (ICML 2020). "Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-Free Attacks" (AutoAttack). [arXiv:2003.01690](https://arxiv.org/abs/2003.01690)
- Croce et al. (NeurIPS 2021). "RobustBench: A Standardized Adversarial Robustness Benchmark." [robustbench.github.io](https://robustbench.github.io/)
- Tramer et al. (NeurIPS 2020). "On Adaptive Attacks to Adversarial Example Defenses." [arXiv:2002.08347](https://arxiv.org/abs/2002.08347)
- Carlini et al. (2019). "On Evaluating Adversarial Robustness." [arXiv:1902.06705](https://arxiv.org/abs/1902.06705)
- Carlini & Wagner (AISec 2017). "Adversarial Examples Are Not Easily Detected: Bypassing Ten Detection Methods." [arXiv:1705.07263](https://arxiv.org/abs/1705.07263)

### Adversarial/OOD Detection Baselines
- Lee et al. (NeurIPS 2018). "A Simple Unified Framework for Detecting OOD Samples and Adversarial Attacks." [arXiv:1807.03888](https://arxiv.org/abs/1807.03888)
- Ma et al. (ICLR 2018). "Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality." [arXiv:1801.02613](https://arxiv.org/abs/1801.02613)
- Papernot & McDaniel (2018). "Deep k-Nearest Neighbors." [arXiv:1803.04765](https://arxiv.org/abs/1803.04765)
- Feinman et al. (2017). "Detecting Adversarial Samples from Artifacts." [arXiv:1703.00410](https://arxiv.org/abs/1703.00410)
- Sun et al. (ICML 2022). "Out-of-Distribution Detection with Deep Nearest Neighbors."
- OpenOOD v1.5. "Enhanced Benchmark for OOD Detection." [arXiv:2306.09301](https://arxiv.org/abs/2306.09301)

### Representation Quality and Comparison
- Reiss et al. (ECCV 2022 Workshop). "Anomaly Detection Requires Better Representations." [arXiv:2210.10773](https://arxiv.org/abs/2210.10773)
- Bengio, Courville & Vincent (IEEE TPAMI 2013). "Representation Learning: A Review and New Perspectives." [arXiv:1206.5538](https://arxiv.org/abs/1206.5538)
- Kornblith et al. (ICML 2019). "Similarity of Neural Network Representations Revisited" (CKA). [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)
- Roth et al. (CVPR 2022). "Towards Total Recall in Industrial Anomaly Detection" (PatchCore). [arXiv:2106.08265](https://arxiv.org/abs/2106.08265)

### Information Theory and Neural Networks
- Tishby & Zaslavsky (2015). "Deep Learning and the Information Bottleneck Principle." [arXiv:1503.02406](https://arxiv.org/abs/1503.02406)
- Shwartz-Ziv & Tishby (2017). "Opening the Black Box of Deep Neural Networks via Information." [arXiv:1703.00810](https://arxiv.org/abs/1703.00810)
- Saxe et al. (ICLR 2018). "On the Information Bottleneck Theory of Deep Learning."

### Neural Network Symmetries and Invariance
- Dinh et al. (ICML 2017). "Sharp Minima Can Generalize For Deep Nets." [arXiv:1703.04933](https://arxiv.org/abs/1703.04933)
- Brea et al. (2019). "Weight-space symmetry in deep networks gives rise to permutation saddles." [arXiv:1907.02911](https://arxiv.org/abs/1907.02911)
- Navon et al. (ICML 2023). "Equivariant Architectures for Learning in Deep Weight Spaces."
- Zhou et al. (2023). "A Permutation-Invariant Representation of Neural Networks."

### Convexity in Neural Representations
- Tetkova et al. (Nature Communications 2025). "On Convex Decision Regions in Deep Network Representations." [arXiv:2305.17154](https://arxiv.org/abs/2305.17154)
- Mustafa et al. (ICCV 2019). "Adversarial Defense by Restricting the Hidden Space." [arXiv:1904.00887](https://arxiv.org/abs/1904.00887)
- Pfrommer et al. (2023). "Asymmetric Certified Robustness via Feature-Convex Neural Networks." [arXiv:2302.01961](https://arxiv.org/abs/2302.01961)

### Expensive-but-Valuable Representations (Precedents)
- Jacot et al. (NeurIPS 2018). "Neural Tangent Kernel." [arXiv:1806.07572](https://arxiv.org/abs/1806.07572)
- Mohamadi et al. (ICML 2023). "A Fast, Well-Founded Approximation to the Empirical NTK." [arXiv:2206.12543](https://arxiv.org/abs/2206.12543)

### Theory Papers with CIFAR-Scale Experiments (Precedents)
- Zhang et al. (ICLR 2017 Best Paper). "Understanding Deep Learning Requires Rethinking Generalization." [arXiv:1611.03530](https://arxiv.org/abs/1611.03530)

### Quiver Representation Theory
- Armenta & Jodoin (2021). "The Representation Theory of Neural Networks." [arXiv:2007.12213](https://arxiv.org/abs/2007.12213)
- Armenta et al. (2022). "Double framed moduli spaces of quiver representations." [arXiv:2109.14589](https://arxiv.org/abs/2109.14589)

### Venue Policy
- TMLR Editorial Policies. [jmlr.org/tmlr/editorial-policies.html](https://jmlr.org/tmlr/editorial-policies.html)
