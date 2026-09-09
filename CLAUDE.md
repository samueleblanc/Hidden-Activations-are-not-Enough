# Knowledge Matrices as Canonical Neural Network Representations

## Overview

Research implementation of "Hidden Activations Are Not Enough: A General Approach to Neural Network Predictions" (arXiv:2409.13163) by Samuel Leblanc, Aiky Rasolomanana, and Marco Armenta. Computes "knowledge matrices" via quiver representations of neural networks and studies them as function-determined (gauge-invariant, germ-complete) per-sample representations. Do NOT describe KMs as "superior to penultimate activations" — the repo's own detector bake-off refuted that framing (penultimate won 5/6 configurations; reported as an honest negative).

- **Language:** Python 3.11 + Bash (Slurm job scripts)
- **Cluster:** Compute Canada Alliance HPC (Rorqual, H100 GPUs)
- **License:** Apache 2.0
- **Key dependencies:** torch 2.2.2, torchvision 0.17.2, torchattacks 3.5.1, knowledgematrix (git+MarcoArmenta/knowledgematrix-cluster@fe64a13 — Phase-1 fork; pinned in `requirements-{slurm,local}.txt`), scikit-learn 1.3.2, scipy 1.10.1, neuralteleportation (for teleportation experiment)

## This is HAaNE I of a three-paper series (decided 2026-09-04) — read this first

| | Paper | Where |
|---|---|---|
| **HAaNE 0** | *Hidden Activations Are Not Enough* (arXiv:2409.13163) — defines the knowledge map, the KM and the row-sum identity | published; rejected by TMLR Nov 2024; being renamed 0 |
| **HAaNE I** | **this repository** — *Knowledge Matrices as Higher Representations* | `docs/Final-twist/paper/` |
| **HAaNE II** | *Do Independently Trained Networks Learn the Same Knowledge Matrix?* | the `Neural-Networks-Matrices` repo, branch `learning-mechanics` |

**The boundary rule.** HAaNE I is about **the object at one trained network**: `x` moves, `θ` is fixed —
germ identity, maximal invariance, completeness, the VJP equivalence, adversarial distance geometry,
teleportation tiers, and the alignment-free comparison on the pretrained zoo. HAaNE II is about **the
population statistic across training runs**. Do not let work drift across that line.

**Vocabulary ownership is enforced by greps, not by taste.** This paper never writes *universality*,
*seed floor* or *training controls* — those are II's. II never writes *higher representation* — that is
this paper's title. The gates are at the top of the status board.

> **`docs/Final-twist/paper/STATUS.md` is the living board.** It answers "where are we on paper I":
> what is ready, what is left in order, the open review flags, and the pre-submission checklist. Read it
> before planning anything, and update it in place when you change the paper. `CLUSTER-RUNS-STATUS.md`
> (repo root, local-only) remains the cluster record.

**State as of 2026-09-05.** All cluster work is complete and **nothing is blocked**; every remaining item
is a local decision, a local run, or writing. The paper builds clean at 48 pages and is **not submitted**.
Three things a session here must know:

- **The paper tree is now tracked.** `docs/` is ignored by this repo's `.gitignore`; commit `8b2f731`
  started tracking `docs/Final-twist/paper/` with `git add -f` and staged the ignore carve-out. Further
  paper commits still need `-f` for **new** files.
- **The vocabulary appendix is a mirror that has been deliberately FORKED (2026-09-07).**
  `docs/Final-twist/paper/sections/vocab/` was copied byte-for-byte from `docs/vocab/` in the HAaNE II
  repository (16 of its 31 entries; the rest carry II's reserved vocabulary). **It is no longer byte-equal.**
  On Marco's instruction the copy now uses this paper's notation: `f_\theta` → `\Psi(W\!,f)` for the
  network function (θ = parameters, W = weights, so θ = W), and `d_f` → `d_\Psi`. Nine entries changed:
  network-as-map, km, row-sum-invariant, penultimate-features, logits-softmax-cross-entropy,
  parameters-vs-architecture, activation-function, class-centring, invariances-of-measures.
  **Consequence: a naive `cp` re-sync from `docs/vocab/` will silently revert all of this.** Either port
  the notation upstream into HAaNE II first, or re-apply the rename after any future sync. Re-sync recipe:
  the last section of the status board.
- **The ordering tables are generated, not hand-written.** `scripts/regen_ordering_tables.py` emits
  `tables/s3_table.tex` from the Phase-1 reduce, with 80 enforced self-checks that abort the run and write
  nothing rather than print a caption that disagrees with its numbers. Regenerate; do not hand-edit.

## TMLR Resubmission Direction

Paper rejected by TMLR (Nov 2024). New direction: **"Knowledge Matrices as Canonical Neural Network Representations"** — dropping adversarial detection claims entirely. Three studies:

1. **Study 1 — Invariance.** CORRECTED 2026-06-11: the paper's Thms 4.1/4.2 cover nonzero per-neuron *rescalings* only — permutations are NOT in that group. Permutation invariance (and invariance under the entire function-stabilizer) is now covered by the new germ-identity/maximal-invariance theorems (proofs in `../resubmission-artifacts-2026-06-11/paper-drop-in/`). The uniqueness claim ("no other practically computable representation has this") is FALSE — gradient×input/FullGrad share the invariance for piecewise-linear nets; we own that via the equivalence theorem instead. Sub-studies: 1a (permutation, signal-relative framing), 1b (neural teleportation — **function-EXACT**, corrected 2026-09-07; see the dedicated section below), 1c (9-measure similarity panel).
2. **Study 2 — Distance geometry.** CORRECTED 2026-06-11: raw and RMS "amplification" are both unit artifacts of the row-sum constraint. The metric-invariant statistic is the coherence $A=(d_f/d_M)^2$, with theorem reference lines $A=1$ (one-pixel law) and $A\le d$ (within-region cap). The surviving empirical results: the attack-family ordering (Kendall $W=0.921$ across 6 archs) and the size-controlled crossing-mechanism pilot.
3. **Study 3 — Cross-architecture canonical comparison.** KMs are uniformly $1000 \times 150{,}529$ for any feedforward network on $224 \times 224$ ImageNet inputs, so KM Frobenius distance compares ResNet-152, DenseNet-121, and GoogLeNet directly without any alignment step. Includes a same-arch cross-recipe positioning experiment (Step E).

Phase 1 (added 2026-05-03) extends Studies 1, 2, and 3 with a 9-measure representation-similarity panel and Cui/Murphy controls (`docs/superpowers/specs/2026-05-03-cka-similarity-experiments-design.md`).

Additionally: testing how penultimate activation distances behave when increasing network size (using pretrained torchvision models directly, no training needed).

## Resubmission status — what is left (updated 2026-06-11)

**Read these first:** the adjudicated plan + proved theorems are in
`../HAaNE-Resubmission-Plan-and-Proofs-2026-06-11.pdf`; all paper-ready artifacts
(tables, theory drop-in sections, full proofs appendix, Fig. 1, cover letter, scripts,
demo results) are in `../resubmission-artifacts-2026-06-11/` (see its README). Every
theorem there was verified numerically at float64 by independent adversarial agents.

**Gate decision (spent).** Both gates cleared: the Phase-1 reduce is green (06-19, controls
06-21) and the Step-B question closed on 06-23 — see PATH B below. Step E was never a gate
and is now effectively dropped: "Step E" appears nowhere in the paper source. The remaining
sequence to submission is the ordered LEFT list on the status board.

> **Cluster-runs status board — `CLUSTER-RUNS-STATUS.md` (repo root).** The single
> source of truth for everything cluster-side: what has run, what is in flight, what is
> left (Steps A–E, Phase-1 S1–S3, reduce→tar, the D2 deployment), push state, gate
> files, result paths, standing facts, and risks. It is a **living document updated IN
> PLACE** — rewrite stale fields, never append logs. **It is LOCAL-ONLY: never commit or
> push it** (it is in `.git/info/exclude`); the status-doc edits in this CLAUDE.md are
> likewise kept uncommitted. It gets updated when Marco pastes cluster output into a
> local session (e.g., a `/cluster-debug` transcript, scheduler snapshots, or pulled
> result files): read it at session start, rewrite the rows the new information touches
> before the session ends. Same convention as `TRILLIUM-PIPELINE.md` in the
> Neural-Networks-Matrices repo.

### Writing actions (local)
1. Wire the drop-ins into `docs/Final-twist/paper/`: `paper-drop-in/{fig1_germ_identity,
   section_math_core, appendix_proofs, study_snippets}.tex` + `study2_tables/*.tex`
   (compile-tested together; see PREVIEW.pdf). Add bib entries: Shrikumar/Ancona (G×I),
   Srinivas–Fleuret (FullGrad), Balestriero–Baraniuk (spline/CPA), Lakshminarayanan–Singh
   (NPF/NPK), Mohan et al. (denoiser Jacobians), Novak et al., Phuong–Lampert,
   Rolnick–Kording, Grigsby–Lindsey, Flinth et al. 2026.
2. Apply the cut list (file:line table in the plan PDF, Part B §3). The five land-mines:
   `study1_invariance.tex:144-161` (γ⁻¹ upper bound — INVERTS Thm 4.5; cut),
   `:211-216` (theorem-asserted 0 — replace with measured drift), the 130×/5×/8–16×
   headlines, `mathematical_background.tex:48` vs `:55-60` (1_d vs 1_{d+1}),
   `study1_invariance.tex:23-31` (describe the wide_face permutation correctly).
3. Fill `docs/Final-twist/paper/tables/s{1,2,3}_table.tex` stubs when the reduce lands;
   regenerate theorem45/isomorphism tables in survivor metrics
   (`../resubmission-artifacts-2026-06-11/scripts/make_study2_tables.py`).
4. Cover letter: `paper-drop-in/cover_letter.md` — every bullet keyed to a verbatim
   reviewer/AE quote; includes the "what we deliberately do not claim" section.

### No f(0)=0 condition (2026-09-08)

The KM is defined for **every** activation; there is no condition on `f`, and in particular none on
`f(0)`. The hypothesis is `x in X_nz`: no hidden pre-activation vanishes. Sigmoid (`f(0)=0.5`) satisfies
the row-sum identity to 1e-14. At an exactly-zero pre-activation the identity fails by exactly `f(0)` and
**no guard can repair it** (`D*0 = 0` for any finite `D`; checked with guards 0, 1, 1e6) -- so `f(0)=0` is
just the condition making that exceptional set empty, a bonus not a prerequisite. `X_nz` is open, dense,
full measure for real-analytic `f`, and for the ReLU family equals the complement of Lemma A.1's
hyperplanes. Caveat to keep stating: when `f(0) != 0`, `D ~ f(0)/z` is unbounded near `{z=0}` (5.25 at
z=1e-1, 5e5 at z=1e-6 for sigmoid) -- exact, but not small. Under (LCS)+continuity `f(0)=0` is a
*consequence*, so `X_nz` is vacuous there.

### (LCS), not "PL", is the hypothesis (2026-09-07)

The germ results need **(LCS)**: the slope diagonal `D^(l)` is *locally constant* in `x`. This is
**strictly stronger than PL** and the difference is real: hard-tanh `clip(z,-1,1)` and `max(z-1,0)` are
piecewise linear, their networks are piecewise affine and have germs, **but the germ identity fails** --
at `z=2` hard-tanh has `f(z)/z = 1/2` vs `f'(z) = 0` (verified: `|J_sec - dPsi| = 1.67` end-to-end; `0`
for ReLU/LeakyReLU/abs). Prop 2.2 proves that for continuous `f` with `f(0)=0`, (LCS) holds iff
`f(z) = a+ max(z,0) + a- min(z,0)` -- each piece through the origin, so continuity forces the only break
to be at `z=0`, which is why Lemma A.1's walls are exactly the zero sets. Never write "for PL networks"
as the hypothesis of the germ identity, maximal invariance, or the gradient x input equivalence; write
"under (LCS)". "PL" is fine as descriptive shorthand for our ReLU architectures.

### The KM is defined for ANY activation (clarified 2026-09-07)

`D^(l)(x)` is the diagonal of activation-to-pre-activation quotients `f(z_q)/z_q` (guard `0/0 -> 0`), for
any activation with `f(0) = 0`. This is **the** definition — it is Armenta-Jodoin's induced thin quiver
representation `W^f_x`, and their **Theorem 6.4** gives the row-sum identity for any activation. The `0/1`
mask is the **ReLU specialisation**, not the definition; do not describe the secant form as an
"extension to non-PL activations".

What IS piecewise-linear: the **germ identity** (Thm 2.2) and everything downstream of it (maximal
invariance Thm 2.5, completeness Thm 2.7, and all of Section 3's region/wall/crossing geometry). For
non-PL `f` the quotient `f(z)/z` is not `f'(z)`, so `J` is not the Jacobian and the matrix is **not**
determined by the germ — verified: at `x0=1` a tanh unit and the affine map with the same germ
(0.419974, 0.761594) give KMs `[0.761594|0]` vs `[0.419974|0.341620]`; the ReLU control gives `[1|0]` for
both. Lemma A.1 likewise does not generalise: for smooth `f` the pattern is locally constant essentially
nowhere, so `X_reg` is generically empty. Whether weaker function-determination survives for non-PL is
**open** — do not claim it either way.

### Teleportation is EXACT (corrected 2026-09-07) — do not reintroduce "approximate"

Neural teleportation is an **exact** function-preserving isomorphism on these BatchNorm nets in eval
mode. `neuralteleportation` does not migrate BN running stats because it does not need to:
`layers/neuron.py:BatchNormMixin._forward` computes `base_BN(input / prev_cob)`, restoring the original
pre-BN activation, then scales `weight`/`bias` by `next_cob`. The measured drift is pure floating-point:
the same COB draw at fp32 vs fp64 gives a ratio of 3.3–7.9e8 against the roundoff prediction
`eps32/eps64 = 5.37e8` (a real function change would give ~1), with the fp64 residual at ~1e-15 relative;
the KM itself drifts 0.8–2.5e-15 relative at fp64 on all three archs. The cluster's 1e-2–1e-1 is **TF32
convolution arithmetic on the H100** (`job_teleportation.sh` uses `--gpus=h100:1`; PyTorch defaults
`cudnn.allow_tf32=True` and this repo never disables it; TF32 eps = 4.9e-4, 8192x coarser than fp32).
Secondary amplifier: `cob_range=1` samples tau on [0,2], so the smallest of ~2e4 draws is ~3e-5 and the
multiply/divide round trip passes through ~1e4.

Reproduce: `scripts/verify_teleportation_exactness.py`, `scripts/verify_teleportation_km_exactness.py`
(CPU, minutes, no cluster). Paper: `docs/Final-twist/paper/sections/appendix_teleport_exact.tex`.
Consequences: Study 1b is a genuine multi-arch invariance verification; **Limitation L1 withdrawn**; the
third honest negative (approximate teleportation) **withdrawn**; the "BN-aware teleportation" follow-up is
**closed**. The June PATH B finding is NOT refuted — it concerns the separate `load_cob_into_km` load
path, which does have a real bug and whose measured-drift column stays dropped.

### Claims discipline (dead list — never reintroduce)
Raw or RMS amplification headlines (both unit artifacts; use coherence A); "superior to
penultimate"; DPI arguments in either direction; "γ̂>0 validated"; "teleportation is
function-approximate" (it is EXACT — see above); theorem-asserted table
zeros; "identical to numerical precision" (float32 completeness residuals reach 0.15–0.30
on ResNet-152; float64 spot-checks reduce them to ~1e-10); "no other practically
computable representation has this property"; permutation invariance cited to Thms 4.1/4.2.

### Facts established 2026-06-11 (cite, don't rediscover)
- Germ identity: M(x) = [J·diag(x) | f−Jx] with J = ∂f/∂x a.e. for PL nets ⇒ M is
  computable by C VJPs: 50–150× cheaper than probing at ImageNet (measured 296× on CPU;
  library-probe vs autograd agreement 3.3e-17 max-abs on AlexNet, fp64, eval mode).
- Networks MUST be in eval mode for KM computation: active Dropout desynchronizes the
  saving/probe/autograd passes and the row-sum identity appears to fail at ~1e-2.
- VGG γ̂=0 diagnosed: attack-failure pairs in a single linear region (8 identical sample
  indices across attacks; d_M=0 is exact). Apply an attack-success filter (d_f ≥ 1).
- Mechanism pilot (VGG per-pair debug data): raw Spearman(H, A) is size-confounded;
  the size-controlled partial correlation is negative (−0.30…−0.34) on all three attacks
  — the conditional crossing mechanism is supported. D4 inherits this analysis plan.
- Signal-relative invariance: KM permutation noise ≈0.1–0.2% of its adversarial signal
  (worst tail 1.14×); penultimate ≤3.05× its own noise, below 1× in 16/24 cells.

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
