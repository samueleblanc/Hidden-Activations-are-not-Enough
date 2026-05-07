# Knowledge Matrices as Canonical Neural Network Representations

Implementation of "Hidden Activations Are Not Enough: A General Approach to Neural Network Predictions" (arXiv:2409.13163) by Samuel Leblanc, Aiky Rasolomanana, and Marco Armenta.

Given a neural network and a data sample, we compute a **knowledge matrix** (via quiver representations). These matrices capture the full linear behavior of the network at each input point. We show that knowledge matrices are superior to penultimate-layer activations as neural network representations through the following studies:

1. **Study 1 — Isomorphism Invariance.** Knowledge matrices are invariant under neuron permutations (quiver isomorphisms), while penultimate activations change arbitrarily. Verified via random neuron permutation (Study 1a) and neural teleportation (Study 1b); within-arch invariance under the 9-measure similarity panel (Study 1c).
2. **Study 2 — Distance Lower Bound (Theorem 4.5).** Knowledge matrix distances lower-bound logit distances: `||M(x) - M(x')|| >= gamma * ||f(x) - f(x')||`. Empirically, KMs amplify separations more than penultimate features.
3. **Study 3 — Cross-architecture canonical comparison.** Knowledge matrices are uniformly $1000 \times 150{,}529$ for any feedforward network on $224\times224$ ImageNet inputs, regardless of architecture. This means KM Frobenius distance compares ResNet-152, DenseNet-121, and GoogLeNet directly on the same inputs without any alignment step, while CKA / Procrustes / Bures all require ad-hoc dimension matching that either discards variance or introduces spurious agreement. Phase 1 (this codebase) demonstrates this empirically across the 3 architecture pairs on $N = 25{,}000$ ImageNet validation samples, against the 9-measure representation-similarity panel of [Klabunde et al. 2025 ReSi].

**Phase 1 experimental program** (added 2026-05-03): we extend Studies 1, 2, and 3 with a 9-measure representation-similarity panel (debiased linear / angular CKA, Procrustes shape distance, Bures similarity, soft-matching, RSA-Spearman, output JSD, Gromov-Wasserstein, distance correlation) computed at scale ($N = 25{,}000$ for invariance and cross-architecture; $N = 5{,}000$ adversarial pairs per attack for distance amplification). The panel translates the canonical-representation argument into the dominant similarity-measure language used by the rep-similarity community, complete with the random-network control of [Cui et al. 2022] and the shuffled-pair control of [Murphy et al. 2024]. Knowledge matrices are zero (within-arch invariance, by Theorem 4.1) or amplifying (cross-arch and adversarial-pair) where the standard measures either fail to detect drift or refuse to apply. See `docs/superpowers/specs/2026-05-03-cka-similarity-experiments-design.md` for the full design.

Additionally, we investigate how **penultimate activation distances behave when increasing network size** using pretrained torchvision models directly.

---

## Quick Start (Cluster)

All experiments use **pretrained torchvision models** on ImageNet — Studies 1, 2, and 3 evaluate on ResNet-152, DenseNet-121, and GoogLeNet (InceptionV1). No training step required.

```bash
# Run the full pipeline on Nibi (scan for existing results, submit only needed jobs)
bash run_pipeline.sh

# Dry run -- see what would be submitted without submitting
bash run_pipeline.sh --dry-run
```

The pipeline runs Studies 1, 2, and 3 in parallel:

| Step | Script | Experiments | Wall time per task | # tasks |
|------|--------|-------------|--------------------|---------|
| 0. Calibration | `job_calibrate.sh` | ResNet-152, DenseNet-121, GoogLeNet × ImageNet (3-tier 85/90/93% mem) | ~10 min | 3 |
| B. Teleportation (Study 1b) | `job_teleportation.sh` | ResNet-152, DenseNet-121, GoogLeNet × ImageNet | ~2 h | 3 |
| B′. S1 measure panel (Study 1c, NEW) | `job_phase1_s1.sh` | 9-measure panel on teleportation pairs (`--array=0-63`) | ~25 min | 64 |
| C. Theorem 4.5 (Study 2) | `job_theorem45.sh` | (RN-152, DN-121, GN) × (FGSM, PGD, CW, DeepFool, APGD, Square) × ImageNet | ~55 min | 18 |
| C-agg. Aggregation | `job_theorem45_agg.sh` | per experiment, writes `theorem45_results.json` | ~1 min | 3 |
| D. Adversarial scale-up (NEW) | `job_adv_scaleup.sh` | scales adversarial pairs from $N=200$ to $N=5{,}000$ per attack (`--array=0-17`) | ~6 h (Square dominates) | 18 |
| C′. S3 measure panel (Study 2 extension, NEW) | `job_phase1_s3.sh` | 9-measure panel on $N=5{,}000$/attack adversarial pairs (`--array=0-63`) | ~50 min | 64 |
| E. Cross-model same-arch (positioned vs. Study 3) | `job_cross_model.sh` | ResNet-152 (10 pairs) + DenseNet-121 (1 pair) cross-recipe | ~8 h | 11 |
| F. Cross-architecture (Study 3, NEW) | `job_phase1_s2.sh` | 9-measure panel + KM Frobenius across (RN-152, DN-121, GN) pairs (`--array=0-63`) | ~10 min | 64 |
| G. Reduce + sanity (NEW) | `job_phase1_reduce.sh` | aggregate chunk artifacts; bootstrap CI; permutation null; Cui/Murphy controls | ~30 min | 1 |
| H. Tar artifact (NEW) | `job_tar_artifacts.sh` | gated on G's `sanity_report.json` all-pass; produces `phase1-results-*.tar.gz` | ~5 min | 1 |

Phase 1 step labels (A1, A2, B1, B2, B3, C, D) used in the design spec map to the table above as: A1 = Calibration (Step 0); A2 = Adv scale-up (Step D); B1 = S1 (Step B′); B2 = S2 cross-arch (Step F); B3 = S3 (Step C′); C = Reduce (Step G); D = Tar (Step H).

The pipeline runs all steps in parallel where dependencies allow.
Phase-1 array chains use `--dependency=afterany:` rather than
`afterok:` so a single OOMing array task doesn't cancel the whole
cascade — Phase-1 workers gate on `calibration.json` existence
themselves and abort cleanly if it's missing. The C-aggregate step
likewise uses `afterany:` on the per-attack array; Step D's tar
gates internally on the sanity-report's `all_pass` flag.

A sentinel wrapper (`bin/sentinel.sh`, opt-in via `USE_SENTINEL=true bash run_pipeline.sh`)
post-processes each Phase-1 job and handles system-OOM (2× `--mem` retry),
timeout (2× `--time` retry), and CUDA-OOM (step EVERY architecture's
calibration tier down through 85/90/93%, fail loudly at the floor).
Sentinel resubmits inherit `--account` from the orchestrator.

The critical path is 0 → D → C′ → G → H ≈ 8 hours; B′ and F finish
much earlier and wait at G. Total cluster time at peak parallelism
is ~200 GPU-hours.

Each cell in Step C is one SLURM array task; the 18 attack jobs run independently. Step C-agg. is dependency-chained `afterany:` Step C so the aggregates are written as soon as each experiment's 6 per-attack files exist.

### Robustness features

- **Import smoke-test on preflight.** Before any `sbatch`, the orchestrator imports every entry point (`validate_theorem45`, `cross_model_experiment`, `teleportation_experiment`, `bin.calibrate`, and the three `cka_similarity.workers`). A `NameError` or missing module is caught at the login node, not after a SLURM allocation.
- **`squeue` idempotency check.** Aborts at the start if any of the orchestrator's known job names are already running, preventing duplicate submissions on accidental re-runs.
- **Phase-1 worker `.complete` sentinels.** Each of S1/S2/S3 writes `results/phase1/{s1,s2,s3}/.complete` when its full chunk × arch grid is on disk; the orchestrator skips re-submission for completed steps automatically.
- **Sparse `--array=` for Step D (adversarial scale-up).** The orchestrator enumerates all 3 archs × 6 attacks and submits only the missing combinations.
- **Atomic JSON writes everywhere.** Canonical headline files (theorem45, teleportation, `pipeline_state.json`) write via `tmp + os.replace` (`utils.atomic_io.atomic_json_dump`); a wallclock kill mid-write cannot corrupt them.
- **`--dry-run` is side-effect-free.** No `pipeline_state.json` is written on dry-run.
- **`set -euo pipefail`** in every `job_*.sh`, `bin/sentinel.sh`, and `bin/tar_artifacts.sh` — a Python crash followed by the trailing `echo "Task X completed"` no longer silently masks the failure.

**Changing the SLURM account.** Defaults to `def-amorales` (Nibi). Override via env var:

```bash
ACCOUNT=def-bruestle_gpu bash run_pipeline.sh
```

Each `job_*.sh` also carries `#SBATCH --account=def-amorales` so direct `sbatch job_X.sh` invocations work; the orchestrator's `--account="$ACCOUNT"` overrides the directive on every submit, making the env var the single source of truth for orchestrated runs.

---

## Phase 1: Canonical similarity-measure expansion

The Phase 1 expansion (added 2026-05-03) translates the canonical-representation
argument into the dominant representation-similarity language used by the
post-2019 vision interpretability literature. Three coordinated sub-studies,
all run by the extended `bash run_pipeline.sh`:

### Sub-study S1 — within-arch invariance under teleportation

For each of $T = 50$ random neural teleportations per architecture (ResNet-152,
DenseNet-121, GoogLeNet), compute the 9-measure representation-similarity panel
between $h_W(x)$ and $h_{\tilde W}(x)$ on $N = 25{,}000$ ImageNet validation
samples. Knowledge matrices are zero by Theorem 4.1 (verified by unit tests, not
recomputed). The panel reveals which similarity measures detect quiver-isomorphism
drift and which ad-hoc-quotient it out via their narrower invariance class.

### Sub-study S2 — cross-architecture comparison

For each of the 3 unordered architecture pairs (RN-152 ↔ DN-121, RN-152 ↔ GN,
DN-121 ↔ GN), compute the 9-measure panel + KM Frobenius distance on the same
$N = 25{,}000$ samples. Penultimate-feature dimension mismatch (RN-152 D=2048
vs DN-121/GN D=1024) means CKA, Procrustes, Bures, distance correlation
require dimension matching; we report the natively-cross-dim measures
(soft-matching, GW, RSA, JSD) in the main text and PCA-padded versions of
the dimension-restricted measures in the appendix. Knowledge matrices are
uniformly $1000 \times 150{,}529$ regardless of architecture and require no
post-processing.

### Sub-study S3 — within-arch distance amplification

Extends Study 2 (Theorem 4.5 amplification) by scaling from $N = 200$ to
$N = 5{,}000$ adversarial pairs per attack via Step D and computing the
9-measure panel on each pair. Translates the Frobenius-norm amplification
$d_M / d_f$ into the similarity-measure language used by the rep-similarity
community. Per-attack, per-architecture amplification factors with bootstrap
95\% CIs.

### The 9 measures

| Measure | Reference | Invariance class | n×n dual? |
|---------|-----------|------------------|-----------|
| Debiased linear CKA | [Kornblith 2019] / [Nguyen 2021] / [Murphy 2024] | orthogonal + isotropic scaling | yes |
| Angular CKA | [Williams 2021] | orthogonal + isotropic scaling | yes |
| Procrustes shape distance | [Williams 2021] | orthogonal | yes |
| Bures similarity | [Harvey 2023] | orthogonal | yes |
| Soft-matching | [Khosla 2024] | permutation only | n/a (OT) |
| RSA-Spearman | [Kriegeskorte 2008] | rotation + monotone-of-distance | yes |
| Output JSD (sqrt) | [Endres 2003] | none (functional) | n/a |
| Entropic Gromov-Wasserstein | [Mémoli 2011] / [Peyré 2016] | isometry | n/a (OT) |
| Distance correlation | [Székely 2007] | translation + orthogonal | yes |

For CKA we use the unbiased HSIC₁ estimator [Song 2012] as plugged into the
minibatch CKA framework [Nguyen 2021]; this is the methodologically correct
choice in the $n < p$ regime where the biased estimator carries the upward
bias documented by [Murphy 2024].

### Controls (always reported)

- **Cui 2022 random-network control**: untrained variants of each architecture
  (`pretrained=False`) compared to the trained variants. If random-network
  similarity is comparably high to teleported-pair similarity, the input-space
  population structure dominates and the measure is misleading.
- **Murphy 2024 shuffled-pair control**: sample alignment permuted between $X$
  and $Y$. Expected ≈ 0 for the debiased estimator; verifies the
  implementation.

### Statistics

- 95% bootstrap confidence intervals over $10{,}000$ resamples per cell.
- Permutation null over $1{,}000$ shuffles for S1/S2 (the δ-statistic tier).
- 3-tier KM batch-size calibration (85% / 90% / 93% memory). Sentinel
  steps down through tiers on CUDA-OOM.

### File layout

- `cka_similarity/` — per-chunk worker scripts and measure implementations.
- `bin/calibrate.py` — restored from `legacy/`, computes per-arch 3-tier batch sizes.
- `bin/sentinel.sh` — sbatch wrapper handling OOM/timeout retries.
- `experiments/calibration/{arch}_imagenet/calibration.json` — per-arch batch size tiers.
- `experiments/{arch}_imagenet/adversarial_pairs_N5000/{attack}/pairs.pth` — scaled-up adversarial pairs.
- `results/phase1/{s1,s2,s3}/` — per-chunk accumulators / per-pair distance lists.
- `results/phase1/aggregated/` — final tables, sanity report, controls.
- `results/phase1/artifact/phase1-results-*.tar.gz` — final downloadable bundle (Step H output).

### Reproducing Phase 1 from scratch

```bash
# 1. Submit the full pipeline. Calibration runs as Phase -1 automatically;
#    everything else is gated on what's already on disk. Re-running is
#    safe — the squeue idempotency check aborts if jobs are still pending.
bash run_pipeline.sh

# 2. Monitor
squeue -u $USER
tail -f slurm_out/*.out
cat pipeline_state.json | jq

# 3. After completion, the final artifact lives at
ls results/phase1/artifact/phase1-results-*.tar.gz

# 4. Download
scp $CLUSTER:$REPO_PATH/results/phase1/artifact/phase1-results-*.tar.gz .
tar xzf phase1-results-*.tar.gz
```

---

## Running individual experiments

### Isomorphism Invariance

Demonstrates that knowledge matrices are invariant under neuron permutations while penultimate activations are not. Architecture-set alignment (commit `5cf31fc`): Step A is run on `resnet152_imagenet` only — DenseNet-121 and GoogLeNet have concat-based topologies (dense connections / Inception parallel branches) that don't admit a simple post-pool neuron permutation. Study 1 evidence on those architectures comes from Step B (teleportation = Study 1b).

```bash
python isomorphism_experiment.py --experiment resnet152_imagenet
```

### Theorem 4.5 Validation

Empirical validation of the distance lower bound. Generates adversarial pairs on-the-fly, computes logit, penultimate, and KM distances, estimates gamma with bootstrap CI.

The attack set is `IMAGENET_ATTACKS` in `constants/constants.py`: FGSM, PGD, CW, DeepFool, APGD, Square.

Per-experiment attack hyperparameters are overridden in `ATTACK_OVERRIDES` (`validate_theorem45.py`). For pretrained ResNet-ImageNet, DeepFool uses `steps=200`, APGD uses `steps=50, loss='dlr'`, and Square uses `n_queries=20000` — the torchattacks defaults yield ~zero perturbations on these three attacks. The Phase-1 ImageNet archs ({resnet152, densenet121, googlenet}_imagenet) inherit the same overrides as a starting point; tune if smoke runs reveal `n_exact_zero` high.

A forward-pass diagnostic inside `generate_adversarial_pairs` prints `||adv - clean||` (L_inf and L_2) for every run, so it is immediately visible whether an attack silently noop'd. If all logit distances for an attack collapse to ~0 the result is written to `per_attack/{attack}_SKIPPED.json` rather than polluting the aggregate.

```bash
# Run one attack for one experiment (what each SLURM array task does)
python validate_theorem45.py --experiment resnet152_imagenet --attacks FGSM --num_samples 200

# Aggregate all 6 per-attack files for one experiment (Step C-agg.)
python validate_theorem45.py --experiment resnet152_imagenet --aggregate
```

### Teleportation Experiment

Demonstrates penultimate activation instability under neural teleportation (quiver isomorphism via the `neuralteleportation` library). Run per architecture:

```bash
python teleportation_experiment.py --architecture resnet152   --dataset imagenet --pretrained \
    --num_teleportations 100 --num_samples 500 --data_dir /datashare/imagenet/ILSVRC2012
python teleportation_experiment.py --architecture densenet121 --dataset imagenet --pretrained \
    --num_teleportations 100 --num_samples 500 --data_dir /datashare/imagenet/ILSVRC2012
python teleportation_experiment.py --architecture googlenet   --dataset imagenet --pretrained \
    --num_teleportations 100 --num_samples 500 --data_dir /datashare/imagenet/ILSVRC2012
```

### Generate LaTeX Tables

```bash
python generate_theorem45_tables.py --experiments resnet152_imagenet densenet121_imagenet googlenet_imagenet --output tables/
```

### Study 3: cross-architecture canonical comparison

The cross-model story has two coordinated experiments — a same-arch
cross-recipe comparison (Step E) positioned relative to the
cross-architecture comparison (Step F) that is Study 3 proper:

- **Same-arch cross-recipe representational comparison** (Step E,
  `cross_model_experiment.py`). Compares two independently-trained models of
  the *same* architecture (e.g., 10 ResNet-152 cross-recipe pairs from the
  torchvision recipe zoo + 1 DenseNet-121 pair). Demonstrates that KMs
  separate cross-recipe reps where the standard 9-measure panel reports
  near-identity. **In the resubmitted paper, positioned relative to
  Study 3.**
- **Study 3 — cross-architecture canonical comparison (NEW)** (Step F =
  Phase 1 sub-study S2, `job_phase1_s2.sh`). Compares ResNet-152 ↔ DenseNet-121,
  ResNet-152 ↔ GoogLeNet, DenseNet-121 ↔ GoogLeNet at $1000\times150{,}529$ on
  $N = 25{,}000$ ImageNet samples without any dimension matching. The other
  similarity measures either need PCA padding (CKA, Procrustes, Bures, dCor)
  or apply natively at coarser resolution (RSA, JSD, soft-matching, GW).
  **In the resubmitted paper.**

---

## Recently retired

The following pipeline step was dropped from the new direction. The script
is preserved for direct invocation and historical reference:

- **Step A — Random neuron permutation isomorphism** (dropped per commit
  `59c3d6d`, "feat: CKA + SD reporting + smoke flags + N scale-up for
  Study 1b/2"). The standalone Study-1a random-permutation isomorphism
  experiment was superseded by the teleportation-based Study 1b (Step B,
  which provides quiver-isomorphism evidence on all three Phase-1
  architectures, including the concat-topology ones — DenseNet-121 and
  GoogLeNet — that don't admit a simple post-pool neuron permutation).
  Original script: `isomorphism_experiment.py` (still present at the repo
  root for direct invocation; no longer in the orchestrator).

---

## Setup

**Local:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Nibi cluster (Compute Canada):**
```bash
module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
virtualenv env
source env/bin/activate
pip install -r requirements-slurm.txt
# requirements-slurm.txt pins the cluster knowledgematrix fork at
# MarcoArmenta/knowledgematrix-cluster@fe64a13 (Pillar-3 stack:
# extract_weff + densenet121 + googlenet + resnet152 PRs cherry-picked).
# Do not pip-install upstream samueleblanc/knowledgematrix on top — the
# wrapper APIs differ.

# neuralteleportation is not in requirements-slurm.txt — install it
# directly. The PyTorch-2.x compatibility patches and the GoogLeNetCOB
# drop-in are applied automatically by the first `bash run_pipeline.sh`
# (idempotent — re-applies are no-ops).
pip install git+https://github.com/vitalab/neuralteleportation.git
```

That's the entire one-time setup. From then on, `bash run_pipeline.sh` is the only command needed — see §"Pipeline Orchestration" for the preflight chain it runs (weight caching, patches, import smoke-test, queue idempotency check) before any `sbatch`.

---

## Pipeline Orchestration

`run_pipeline.sh` is the single entry point for running all experiments on the cluster. Every step runs on the login node before any `sbatch`, so a misconfigured environment fails fast rather than after a SLURM allocation.

**Phase 0 — login-node preflights (idempotent; all run on `--dry-run` too):**

1. **Queue idempotency check.** Aborts if any of the orchestrator's known job names are already running. Prevents duplicate submission on accidental re-runs.
2. **Module + venv activate.** `module load StdEnv/2023 python/3.11.5 scipy-stack/2025a`, then `source env/bin/activate`.
3. **`neuralteleportation` patches.** `bash patches/apply_neuralteleportation_patches.sh ./env`. Idempotent (`cmp -s` short-circuits on already-applied). Applying once on the shared-FS venv covers every Phase-1 array task.
4. **Import smoke-test.** Imports every entry point (`validate_theorem45`, `cross_model_experiment`, `teleportation_experiment`, `bin.calibrate`, the three `cka_similarity.workers`). A `NameError` or missing module surfaces here.
5. **Pretrained-weight cache.** `weights='DEFAULT'` for `resnet152` / `densenet121` / `googlenet` (covers every Phase-1 sub-study + Step E `tv_v1`). Plus the cross-model alternate recipes used by Step E's verify step: torchvision `IMAGENET1K_V2` for `resnet152`, and the timm RSB variants `resnet152.a1_in1k` / `a2_in1k` / `a3_in1k` and `densenet121.ra_in1k`. Compute nodes have no internet; this populates `~/.cache/torch/hub/checkpoints/` and `~/.cache/huggingface/hub/` on the login node where they do. ~30 s–2 min for a first-time download of all variants; ~milliseconds on subsequent runs.

**Phase -1 / Phase 1 — scan and submit:**

6. **Scans** for existing results and checkpoints across all active sub-steps (Step 0 calibration, Step B teleportation, Steps C/C-agg theorem 4.5, Step E cross-recipe pairs, Phase-1 Steps A2/B′/C′/F/G/H).
7. **Writes** `pipeline_state.json` (atomic via `mktemp` + `os.replace`; **skipped on `--dry-run`**) with the current state of each task (`done`, `in_progress`, or `pending`).
8. **Submits** calibration first (Phase −1), then everything else. Only the SLURM array tasks corresponding to missing outputs are submitted. Phase-1 array deps use `afterany:` so a single failing array task doesn't cancel the whole cascade.

The pipeline is **idempotent** -- safe to re-run after partial failures. For Theorem 4.5, it detects checkpoint files and resumes from where it left off. For the Phase-1 panel jobs, per-chunk artifacts in `results/phase1/{s1,s2,s3}/` are accumulated by the reduce step (Step G); each worker writes a `.complete` sentinel only when the FULL chunk × arch grid is on disk, so a re-run after partial completion submits only the missing chunks. For Step D (adversarial scale-up), the orchestrator enumerates all 3 archs × 6 attacks and submits only the missing combinations.

### Result files

| Step | Output |
|------|--------|
| Isomorphism | `experiments/{experiment}/isomorphism/isomorphism_results.json` |
| Teleportation | `results/teleportation/{architecture}_{dataset}_teleportation.json` |
| Theorem 4.5 (per attack) | `experiments/{experiment}/theorem45/per_attack/{ATTACK}.json` |
| Theorem 4.5 (per attack, zeroed) | `experiments/{experiment}/theorem45/per_attack/{ATTACK}_SKIPPED.json` |
| Theorem 4.5 (aggregate) | `experiments/{experiment}/theorem45/theorem45_results.json` |
| Cross-model same-arch (Step E) | `results/cross_model/{arch}/per_pair/{i}__{j}.json` |
| Phase 1 panels (Steps B′/C′/F + reduce) | `results/phase1/{s1,s2,s3}/...`, `results/phase1/aggregated/...`, `results/phase1/artifact/phase1-results-*.tar.gz` |
| Pipeline state | `pipeline_state.json` (overwritten on every non-dry-run scan) |

---

## Sanity Check

Before submitting a full pipeline run, verify (a) the orchestrator's preflight imports succeed, and (b) the Phase-1 architecture wrappers load and forward-pass.

```bash
# (a) Same import smoke-test the orchestrator runs. A NameError /
#     ImportError here is the same failure you'd see in slurm_err/.
python -c "
from validate_theorem45 import validate_theorem45
from cross_model_experiment import main as _xm
from teleportation_experiment import run_experiment as _tp
from bin.calibrate import main as _cal
from cka_similarity.workers import s1_within_arch_invariance, s2_cross_architecture, s3_distance_amplification
print('imports OK')
"

# (b) Phase-1 wrapper smoke: load each KM-wrapped model and forward-pass.
#     The wrapper exposes .layers / .input_shape (consumed by
#     KnowledgeMatrixComputer) and forwards as a regular nn.Module.
python -c "
import torch
from utils.km_models import build_model
for arch in ('resnet152', 'densenet121', 'googlenet'):
    m = build_model(arch, 'cpu')
    y = m(torch.randn(1, 3, 224, 224))
    assert y.shape == (1, 1000), (arch, y.shape)
    print(f'{arch}: layers={len(m.layers)}, input_shape={m.input_shape}, logits_ok')
"

# (c) Pretrained-accuracy spot-check on a small validation subset
#     (real test images, not random tensors). Expect mid-70s top-1 on
#     resnet152 / densenet121 and high-60s on googlenet (InceptionV1).
python -c "
import torch
from utils.km_models import build_model
from utils.utils import get_imagenet_val_dataset
_, val_set = get_imagenet_val_dataset()  # uses /datashare/imagenet/ILSVRC2012 by default
xs = torch.stack([val_set[i][0] for i in range(40)])
ys = torch.tensor([val_set[i][1] for i in range(40)])
m = build_model('resnet152', 'cpu')
with torch.no_grad():
    acc = (m(xs).argmax(1) == ys).float().mean().item()
print(f'resnet152 pretrained acc on 40 samples: {acc:.1%}')  # expect ~70%
"
```

Swap `resnet152` for `densenet121` / `googlenet` to check the other Phase-1 architectures.

---

## Troubleshooting: `knowledgematrix` pretrained ResNet18

`utils/utils.py:get_architecture()` applies three repairs to the pretrained ResNet18 path that are not present in the upstream `knowledgematrix` package at commit `0d26c7a`:

1. **`tv_model.eval()` before wrapping** (`utils.py:498`) — `knowledgematrix.NN.residual()` calls `shape_at_layer()` which runs a training-mode forward with a random probe input. Without the eval switch the probe overwrites each `BatchNorm2d`'s `running_mean` / `running_var` with random-input statistics, collapsing ImageNet accuracy to ~0%.
2. **Post-residual ReLU injection** (`utils.py:_inject_postresidual_relus`) — torchvision's `BasicBlock.forward` applies `out = self.relu(out + identity)`, but the upstream wrapper's `basic_block.children()` iteration yields only one shared `relu`, and `NN.apply_residual` does no activation. One `nn.ReLU()` is injected at each residual end index.
3. **Residual-start index shift `>= end`** — in the upstream layout, block *k*'s end index coincides with block *k+1*'s start index, so `NN.forward` saves block *k+1*'s identity BEFORE applying block *k*'s residual-add. Shifting start indices by +1 at each insertion separates them.

All three are gated on `architecture_index == -2 AND pretrained == True`. AlexNet and VGG11 (non-`_bn`) have no BatchNorm or residuals in their torchvision wrappers, so none of these repairs are needed for those architectures.

If you upgrade `knowledgematrix` past `0d26c7a`, revisit these — upstream may have fixed them.

---

## Repository Structure

```
.
├── run_pipeline.sh                # Pipeline orchestrator (scan + submit)
├── isomorphism_experiment.py      # Study 1a: KM invariance under neuron permutations
├── validate_theorem45.py          # Study 2: empirical distance lower bound
├── teleportation_experiment.py    # Penultimate activation instability under teleportation
├── generate_theorem45_tables.py   # LaTeX table generation for Theorem 4.5
├── debug_isomorphism.py           # Float32 vs float64 precision check for KM invariance
├── job_isomorphism.sh             # SLURM job: Step A (array 0-2)
├── job_teleportation.sh           # SLURM job: Step B (array 0-2)
├── job_theorem45.sh               # SLURM job: Step C per-attack (array 0-17)
├── job_theorem45_agg.sh           # SLURM job: Step C aggregation (array 0-2, dependency-chained)
├── job_debug_isomorphism.sh       # SLURM job: precision diagnostic (one-shot)
├── job_calibrate.sh               # SLURM job: Step A1 per-arch 3-tier batch-size calibration
├── job_adv_scaleup.sh             # SLURM job: Step A2 N=200 → N=5000 adversarial pairs
├── job_phase1_s1.sh               # SLURM job: Step B′ S1 panel (within-arch invariance)
├── job_phase1_s2.sh               # SLURM job: Step F  S2 panel (cross-architecture, Study 3)
├── job_phase1_s3.sh               # SLURM job: Step C′ S3 panel (adversarial-pair amplification)
├── job_phase1_reduce.sh           # SLURM job: Step G aggregate + bootstrap CI + sanity report
├── job_tar_artifacts.sh           # SLURM job: Step H final tar artifact
├── job_cross_model.sh             # SLURM job: Step E same-arch cross-recipe pairs
├── cross_model_experiment.py      # Step E driver
├── cka_similarity/                # Phase 1 worker scripts + 9-measure implementations
├── bin/
│   ├── calibrate.py               # Per-arch 3-tier batch-size calibration
│   └── sentinel.sh                # OOM/timeout retry sbatch wrapper
├── constants/
│   └── constants.py               # Experiment configs, architectures, attacks
├── utils/
│   ├── utils.py                   # Model loading, datasets, get_architecture with ResNet fixes
│   ├── features.py                # Penultimate feature extraction
│   └── atomic_io.py               # Atomic file writes
├── patches/                       # neuralteleportation PyTorch 2.x compatibility
├── experiments/                   # Per-experiment outputs (calibration + theorem45 results)
├── results/teleportation/         # Teleportation experiment outputs
├── results/cross_model/           # Step E cross-recipe per-pair results
├── results/phase1/                # Phase-1 panel chunks + aggregated tables + artifact
└── legacy/                        # Archived adversarial-detection pipeline + CIFAR training code
```

---

## Experiment × step coverage

The three experiment keys (defined in `constants/constants.py`) drive every step in the active pipeline. Step B uses the same architectures as Steps C/E/F via the `neuralteleportation` library's COB wrappers (with this paper's parallel-branch patch for the concat topologies of DenseNet-121 and GoogLeNet).

| Experiment key | Step B (Study 1b) | Step C (Study 2) | Step E (cross-recipe) | Step F (Study 3) | Architecture |
|----------------|:-----------------:|:----------------:|:---------------------:|:----------------:|--------------|
| `resnet152_imagenet`   | ✅ | ✅ | ✅ (10 pairs) | ✅ | torchvision `resnet152` |
| `densenet121_imagenet` | ✅ | ✅ | ✅ (1 pair)   | ✅ | torchvision `densenet121` |
| `googlenet_imagenet`   | ✅ | ✅ | —             | ✅ | torchvision `googlenet` (InceptionV1, **not** Inception_v3) |

All three torchvision weight variants (`resnet152`, `densenet121`, `googlenet`) must be pre-cached on a login node — see Setup above.

---

## License

Apache 2.0
