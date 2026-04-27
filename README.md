# Knowledge Matrices as Canonical Neural Network Representations

Implementation of "Hidden Activations Are Not Enough: A General Approach to Neural Network Predictions" (arXiv:2409.13163) by Samuel Leblanc, Aiky Rasolomanana, and Marco Armenta.

Given a neural network and a data sample, we compute a **knowledge matrix** (via quiver representations). These matrices capture the full linear behavior of the network at each input point. We show that knowledge matrices are superior to penultimate-layer activations as neural network representations through three pillars:

1. **Isomorphism Invariance** -- Knowledge matrices are invariant under neuron permutations (quiver isomorphisms), while penultimate activations change arbitrarily.
2. **Distance Lower Bound (Theorem 4.5)** -- Knowledge matrix distances lower-bound logit distances: `||M(x) - M(x')|| >= gamma * ||f(x) - f(x')||`. Empirically, KMs amplify separations more than penultimate features.
3. **Exact Image-Space Attribution** -- The per-class row of the per-input knowledge matrix `M(x)` is itself a saliency map, with properties no baseline (Grad-CAM, Integrated Gradients, SmoothGrad) achieves: it satisfies `Σ_j M(x)[c, j] = f_c(x)` exactly, requires no baseline image to be chosen, and is invariant under hidden-layer neuron permutations. Demonstrated on ResNet-152, DenseNet-121, and GoogLeNet (InceptionV1) — see the `km_feature_viz` pipeline below.

Additionally, we investigate how **penultimate activation distances behave when increasing network size** using pretrained torchvision models directly.

---

## Quick Start (Cluster)

All experiments use **pretrained torchvision models** on ImageNet — Pillars 1 & 2 evaluate on AlexNet, ResNet18, and VGG11; Pillar 3 (`km_feature_viz/`) evaluates on ResNet152, DenseNet121, and GoogLeNet (InceptionV1). No training step required.

```bash
# Run the full pipeline on Nibi (scan for existing results, submit only needed jobs)
bash run_pipeline.sh

# Dry run -- see what would be submitted without submitting
bash run_pipeline.sh --dry-run
```

The pipeline runs Pillars 1, 2, and 3 in parallel:

| Step | Script | Experiments | Wall time per task | # tasks |
|------|--------|-------------|--------------------|---------|
| A. Isomorphism (Pillar 1) | `job_isomorphism.sh` | AlexNet, ResNet18, VGG11 × ImageNet | up to 4 h | 3 |
| B. Teleportation (Pillar 1) | `job_teleportation.sh` | ResNet18, VGG11-BN, ResNet50 × ImageNet | ~2 h | 3 |
| C. Theorem 4.5 (Pillar 2) | `job_theorem45.sh` | (AlexNet, ResNet18, VGG11) × (FGSM, PGD, CW, DeepFool, APGD, Square) × ImageNet | ~55 min (DeepFool ~1 h 45) | 18 |
| C-agg. Aggregation | `job_theorem45_agg.sh` | per experiment, writes `theorem45_results.json` | ~1 min | 3 |
| D1. KM compute (Pillar 3) | `job_kmfv_kms.sh`         | ResNet152, DenseNet121, GoogLeNet × ImageNet (`--array=0-2`) | ~1 h            | 3 |
| D2. Baselines (Pillar 3)  | `job_kmfv_baselines.sh`   | Grad-CAM / IG / SmoothGrad / feature-maps / PGD across all 3 archs           | ~1 h            | 5 (per method) |
| D3. DeepDream (Pillar 3)  | `job_kmfv_deepdream.sh`   | ResNet152, DenseNet121, GoogLeNet × ImageNet (`--array=0-2`)                 | ~1 h            | 3 |
| D4. Formulations (Pillar 3) | `job_kmfv_formulations.sh` | counterfactual-LP + Jacobian-sensitivity (per arch)                        | ~30 min         | 2 |
| D5. Bundle (Pillar 3)     | `job_kmfv_bundle.sh`      | tar `km-feature-viz.tar` of `results/km-feature-viz/`                        | ~5 min          | 1 |

Each cell in Step C is one SLURM array task; the 18 attack jobs run independently. Step C-agg. is dependency-chained `afterany:` Step C so the aggregates are written as soon as each experiment's 6 per-attack files exist.

D1 and D3 are SLURM array jobs (`--array=0-2`) — one task per architecture (`resnet152`, `densenet121`, `googlenet`). D5 is dependency-chained on D1–D4. Per-step state files live under `results/km-feature-viz/state/` and the orchestrator skips any sub-step whose state file is already complete.

**Changing the SLURM account.** A single variable at `run_pipeline.sh:230` — `ACCOUNT="def-bruestle_gpu"`. Every `sbatch` in the orchestrator picks it up; the individual `job_*.sh` files do not hard-code an account.

---

## Running individual experiments

### Isomorphism Invariance

Demonstrates that knowledge matrices are invariant under neuron permutations while penultimate activations are not.

```bash
python isomorphism_experiment.py --experiment alexnet_imagenet
python isomorphism_experiment.py --experiment resnet_imagenet
python isomorphism_experiment.py --experiment vgg_imagenet
```

### Theorem 4.5 Validation

Empirical validation of the distance lower bound. Generates adversarial pairs on-the-fly, computes logit, penultimate, and KM distances, estimates gamma with bootstrap CI.

The attack set is `IMAGENET_ATTACKS` in `constants/constants.py`: FGSM, PGD, CW, DeepFool, APGD, Square.

Per-experiment attack hyperparameters are overridden in `ATTACK_OVERRIDES` (`validate_theorem45.py:56`). For pretrained ResNet-ImageNet, DeepFool uses `steps=200`, APGD uses `steps=50, loss='dlr'`, and Square uses `n_queries=20000` — the torchattacks defaults yield ~zero perturbations on these three attacks.

A forward-pass diagnostic inside `generate_adversarial_pairs` prints `||adv - clean||` (L_inf and L_2) for every run, so it is immediately visible whether an attack silently noop'd. If all logit distances for an attack collapse to ~0 the result is written to `per_attack/{attack}_SKIPPED.json` rather than polluting the aggregate.

```bash
# Run one attack for one experiment (what each SLURM array task does)
python validate_theorem45.py --experiment resnet_imagenet --attacks FGSM --num_samples 200

# Aggregate all 6 per-attack files for one experiment (Step C-agg.)
python validate_theorem45.py --experiment resnet_imagenet --aggregate
```

### Teleportation Experiment

Demonstrates penultimate activation instability under neural teleportation (quiver isomorphism via the `neuralteleportation` library). Run per architecture:

```bash
python teleportation_experiment.py --architecture resnet18  --dataset imagenet --pretrained \
    --num_teleportations 100 --num_samples 500 --data_dir /datashare/imagenet/ILSVRC2012
python teleportation_experiment.py --architecture vgg11_bn  --dataset imagenet --pretrained \
    --num_teleportations 100 --num_samples 500 --data_dir /datashare/imagenet/ILSVRC2012
python teleportation_experiment.py --architecture resnet50  --dataset imagenet --pretrained \
    --num_teleportations 100 --num_samples 500 --data_dir /datashare/imagenet/ILSVRC2012
```

### Generate LaTeX Tables

```bash
python generate_theorem45_tables.py --experiments alexnet_imagenet resnet_imagenet vgg_imagenet --output tables/
```

### Pillar 3: KM as canonical saliency (`km_feature_viz/`)

Demonstrates that the per-class row of the per-input knowledge matrix `M(x)` is itself a saliency map, head-to-head against Grad-CAM, Integrated Gradients, SmoothGrad, and a PGD adversarial-perturbation column. Run end-to-end on the cluster via `run_pipeline.sh` (D1–D5 above).

**Architectures** (from `km_feature_viz/manifest.py:TIER_A_MODELS`):

| Arch | Display name | Family | Input | DeepDream neuron selection |
| ---- | ------------ | ------ | ----- | --------------------------- |
| `resnet152`   | ResNet-152              | residual               | 224×224 | class-conditional Grad-CAM channel ranking (Selvaraju et al. 2017, ch-wise variant) |
| `densenet121` | DenseNet-121            | dense connectivity     | 224×224 | class-conditional Grad-CAM channel ranking (Selvaraju et al. 2017, ch-wise variant) |
| `googlenet`   | GoogLeNet (InceptionV1) | multi-branch inception | 224×224 | catalogued from Distill *Circuits Thread* + OpenAI Microscope (Olah et al. 2017; Cammarata et al. 2020) |

**Image budget.** 20 ImageNet validation images (7 + 7 + 6) across 3 classes — golden retriever (207), tiger cat (282), zebra (340). The class set + per-class counts are pinned in `km_feature_viz/manifest.py:TIER_A_IMAGES_PER_CLASS` with seed `20260426`.

**Storage.** All tensors are stored at **fp32** (the previous fp16 storage caused inf overflow on the residual path). Knowledge matrices are `(1000, 150529)` per `(arch, class, image)` — 1000 ImageNet output classes × (3·224·224 + 1 bias). Bundle: `km-feature-viz.tar` at the repo root.

**DeepDream is NOT a column in the headline panel.** Per the round-3 Interp/KM debate, DeepDream is a neuron prototype, not a class-conditional explanation; including it next to KM/Grad-CAM/IG would invite reading the columns as parallel. It's rendered as an opt-in supplementary plate (`scripts/render_km_viz.py --include-deepdream`), with the per-arch neuron-selection methodology and per-tile citations always visible.

**Per-arch neuron-selection metadata** is written by a separate sub-pipeline to `results/km-feature-viz/state/03a_neuron_selection_<arch>.json` with schema `{method: "gradcam_class_conditional" | "catalogued_distill", channels: {<layer>: [{channel, rank, mean_alpha?, label?, citation?}, ...]}}` — consumed by `scripts/render_km_viz.py` to format DeepDream tile captions and panel-level citation footers.

**Render the gallery** (after `tar -xf km-feature-viz.tar -C results/km-feature-viz-cluster --strip-components=1`):

```bash
# Headline render: KM + Grad-CAM + IG + SmoothGrad + PGD per (arch, class, image)
python scripts/render_km_viz.py --per-class 7

# With DeepDream supplementary section surfaced as an open top-level section
python scripts/render_km_viz.py --per-class 7 --include-deepdream

open results/km-feature-viz-cluster/_viewable/index.html
```

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
pip install git+https://github.com/samueleblanc/knowledgematrix.git

# Cache pretrained weights on login node (no internet on compute nodes)
python -c "
import torchvision.models as m
m.alexnet(weights='DEFAULT')      # Step A, C   (Pillars 1, 2)
m.resnet18(weights='DEFAULT')     # Step A, B, C (Pillars 1, 2)
m.vgg11(weights='DEFAULT')        # Step A, C   (Pillars 1, 2)
m.vgg11_bn(weights='DEFAULT')     # Step B      (Pillar 1)
m.resnet50(weights='DEFAULT')     # Step B      (Pillar 1)
m.resnet152(weights='DEFAULT')    # Step D1-D5  (Pillar 3)
m.densenet121(weights='DEFAULT')  # Step D1-D5  (Pillar 3)
m.googlenet(weights='DEFAULT')    # Step D1-D5  (Pillar 3, InceptionV1)
print('All weights cached.')
"
```

For the teleportation experiment, install the patched `neuralteleportation`:
```bash
pip install git+https://github.com/vitalab/neuralteleportation.git
bash patches/apply_neuralteleportation_patches.sh
```

---

## Pipeline Orchestration

`run_pipeline.sh` is the single entry point for running all experiments on the cluster:

1. **Scans** for existing results and checkpoints across all 14 sub-steps (Pillars 1 & 2: A1–A3, B1–B3, C1–C18 + agg.; Pillar 3: D1–D5)
2. **Writes** `pipeline_state.json` with the current state of each task (`done`, `in_progress`, or `pending`)
3. **Submits** only the needed SLURM array jobs, skipping completed tasks

The pipeline is **idempotent** -- safe to re-run after partial failures. For Theorem 4.5, it detects checkpoint files and resumes from where it left off. For Pillar 3, per-arch state files in `results/km-feature-viz/state/` track D1–D5 completion (see "Result files" below).

### Result files

| Step | Output |
|------|--------|
| Isomorphism | `experiments/{experiment}/isomorphism/isomorphism_results.json` |
| Teleportation | `results/teleportation/{architecture}_{dataset}_teleportation.json` |
| Theorem 4.5 (per attack) | `experiments/{experiment}/theorem45/per_attack/{ATTACK}.json` |
| Theorem 4.5 (per attack, zeroed) | `experiments/{experiment}/theorem45/per_attack/{ATTACK}_SKIPPED.json` |
| Theorem 4.5 (aggregate) | `experiments/{experiment}/theorem45/theorem45_results.json` |
| KM feature-viz raw output | `results/km-feature-viz/{images,kms,baselines,deepdream}/...` (per-`(arch, class, image)` `.pt` files at fp32) |
| KM feature-viz state | `results/km-feature-viz/state/01_compute_kms_<arch>.json`, `02_<method>.json`, `03_deepdream_<arch>.json`, `03a_neuron_selection_<arch>.json`, `05_counterfactual_lp.json`, `06_jacobian_sensitivity.json` |
| KM feature-viz bundle | `km-feature-viz.tar` (single tarball at repo root, written by D5) |
| Pipeline state | `pipeline_state.json` (overwritten on every `run_pipeline.sh` scan) |

---

## Sanity Check

Before submitting a full pipeline run, verify pretrained accuracy matches torchvision baselines. Quick smoke test:

```python
import torch
from utils.utils import get_architecture, get_dataset, subset, get_input_shape, _move_residuals_to_device, get_device
from constants.constants import DEFAULT_EXPERIMENTS
cfg = DEFAULT_EXPERIMENTS['resnet_imagenet']
ish = get_input_shape(cfg['dataset'])
device = get_device()
m = get_architecture(architecture_index=cfg['architecture_index'], input_shape=ish,
                    num_classes=1000, pretrained=True, freeze_features=False).to(device)
_move_residuals_to_device(m, device); m.eval()
_, test_set = get_dataset(cfg['dataset'], data_loader=False)
data, labels = subset(test_set, 20, ish)
with torch.no_grad():
    acc = (m(data.to(device).float()).argmax(1) == labels.to(device)).float().mean().item()
print(f"ResNet18 pretrained acc on 20 samples: {acc:.1%}")  # expect 70-80%
```

Swap `resnet_imagenet` for `alexnet_imagenet` / `vgg_imagenet` to check those too (AlexNet ≈ 55%, VGG11 ≈ 69%).

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
├── isomorphism_experiment.py      # Pillar 1: KM invariance under neuron permutations
├── validate_theorem45.py          # Pillar 2: empirical distance lower bound
├── teleportation_experiment.py    # Penultimate activation instability under teleportation
├── generate_theorem45_tables.py   # LaTeX table generation for Theorem 4.5
├── debug_isomorphism.py           # Float32 vs float64 precision check for KM invariance
├── job_isomorphism.sh             # SLURM job: Step A (array 0-2)
├── job_teleportation.sh           # SLURM job: Step B (array 0-2)
├── job_theorem45.sh               # SLURM job: Step C per-attack (array 0-17)
├── job_theorem45_agg.sh           # SLURM job: Step C aggregation (array 0-2, dependency-chained)
├── job_debug_isomorphism.sh       # SLURM job: precision diagnostic (one-shot)
├── job_kmfv_kms.sh                # SLURM job: D1 KM compute (array 0-2 over archs)
├── job_kmfv_baselines.sh          # SLURM job: D2 baselines (Grad-CAM/IG/SmoothGrad/feature-maps/PGD)
├── job_kmfv_deepdream.sh          # SLURM job: D3 DeepDream (array 0-2 over archs)
├── job_kmfv_formulations.sh       # SLURM job: D4 counterfactual-LP + Jacobian-sensitivity
├── job_kmfv_bundle.sh             # SLURM job: D5 tar bundle
├── constants/
│   └── constants.py               # Experiment configs, architectures, attacks
├── km_feature_viz/                # Pillar 3: KM as canonical saliency
│   ├── manifest.py                # TIER_A_MODELS = [resnet152, densenet121, googlenet]
│   ├── compute_kms.py             # D1: chunked KM computation per arch
│   ├── compute_baselines.py       # D2: Grad-CAM, IG, SmoothGrad, feature-maps, PGD
│   ├── compute_deepdream.py       # D3: noise+jitter DeepDream per (arch, layer, channel)
│   ├── jacobian_sensitivity.py    # D4: Jacobian-based sensitivity baseline
│   └── dictionary.py              # KM dictionary helpers (drops bias column)
├── scripts/
│   └── render_km_viz.py           # Render results/km-feature-viz-cluster/ → PNG gallery
├── utils/
│   ├── utils.py                   # Model loading, datasets, get_architecture with ResNet fixes
│   ├── features.py                # Penultimate feature extraction
│   └── atomic_io.py               # Atomic file writes
├── patches/                       # neuralteleportation PyTorch 2.x compatibility
├── experiments/                   # Per-experiment outputs (isomorphism + theorem45 results)
├── results/teleportation/         # Teleportation experiment outputs
├── results/km-feature-viz/        # Pillar 3 raw .pt outputs + state/ files
└── legacy/                        # Archived adversarial-detection pipeline + CIFAR training code
```

---

## Experiment × step coverage

The three experiment keys (defined in `constants/constants.py`) drive Steps A and C. Step B uses a parallel set of raw torchvision architectures because `neuralteleportation` ships its own COB models, not `knowledgematrix` wrappers. Step D (Pillar 3) uses a third arch list — heavier ImageNet models that span the dominant CNN family lines.

| Experiment key | Step A (isomorphism) | Step C (theorem 4.5) | Architecture |
|----------------|:--------------------:|:---------------------:|--------------|
| `alexnet_imagenet` | ✅ | ✅ | torchvision `alexnet` |
| `resnet_imagenet`  | ✅ | ✅ | torchvision `resnet18` |
| `vgg_imagenet`     | ✅ | ✅ | torchvision `vgg11` |

| Step B architecture (teleportation) | Source |
|-------------------------------------|--------|
| `resnet18`  | `neuralteleportation.models.model_zoo.resnetcob` |
| `vgg11_bn`  | `neuralteleportation.models.model_zoo.vggcob` |
| `resnet50`  | `neuralteleportation.models.model_zoo.resnetcob` |

| Step D architecture (km-feature-viz, Pillar 3) | Family | Source |
|------------------------------------------------|--------|--------|
| `resnet152`   | residual               | torchvision `resnet152` |
| `densenet121` | dense connectivity     | torchvision `densenet121` |
| `googlenet`   | multi-branch inception | torchvision `googlenet` (InceptionV1, **not** Inception_v3) |

All eight torchvision weight variants (`alexnet`, `resnet18`, `vgg11`, `vgg11_bn`, `resnet50`, `resnet152`, `densenet121`, `googlenet`) must be pre-cached on a login node — see Setup above.

---

## License

Apache 2.0
