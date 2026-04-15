# Knowledge Matrices as Canonical Neural Network Representations

Implementation of "Hidden Activations Are Not Enough: A General Approach to Neural Network Predictions" (arXiv:2409.13163) by Samuel Leblanc, Aiky Rasolomanana, and Marco Armenta.

Given a neural network and a data sample, we compute a **knowledge matrix** (via quiver representations). These matrices capture the full linear behavior of the network at each input point. We show that knowledge matrices are superior to penultimate-layer activations as neural network representations through two pillars:

1. **Isomorphism Invariance** -- Knowledge matrices are invariant under neuron permutations (quiver isomorphisms), while penultimate activations change arbitrarily.
2. **Distance Lower Bound (Theorem 4.5)** -- Knowledge matrix distances lower-bound logit distances: `||M(x) - M(x')|| >= gamma * ||f(x) - f(x')||`. Empirically, KMs amplify separations more than penultimate features.

Additionally, we investigate how **penultimate activation distances behave when increasing network size** using pretrained torchvision models directly.

---

## Quick Start (Cluster)

All experiments use **pretrained torchvision models** (AlexNet, ResNet18, VGG11) on ImageNet. No training step required.

```bash
# Run the full pipeline on Nibi (scan for existing results, submit only needed jobs)
bash run_pipeline.sh

# Dry run -- see what would be submitted without submitting
bash run_pipeline.sh --dry-run
```

The pipeline runs 3 independent steps in parallel:

| Step | Script | Experiments | Wall time |
|------|--------|-------------|-----------|
| A. Isomorphism | `job_isomorphism.sh` | AlexNet, ResNet18, VGG11 × ImageNet | ~1h each |
| B. Teleportation | `job_teleportation.sh` | ResNet18, VGG11-BN, ResNet50 × ImageNet | ~2h each |
| C. Theorem 4.5 | `job_theorem45.sh` | AlexNet, ResNet18, VGG11 × ImageNet | ~2h each |

---

## Experiments

### Isomorphism Invariance

Demonstrates that knowledge matrices are invariant under neuron permutations while penultimate activations are not.

```bash
# Pretrained model (no weights file needed)
python isomorphism_experiment.py --experiment alexnet_imagenet

# Trained model (loads from local weights)
python isomorphism_experiment.py --experiment alexnet_cifar10
python isomorphism_experiment.py --experiment alexnet_cifar10 --num_permutations 5 --num_samples 500
```

### Theorem 4.5 Validation

Empirical validation of the distance lower bound. Generates adversarial pairs on-the-fly, computes logit, penultimate, and KM distances, estimates gamma with bootstrap CI. Uses a reduced attack set (6 attacks) for ImageNet experiments.

```bash
python validate_theorem45.py --experiment alexnet_imagenet
python validate_theorem45.py --experiment alexnet_cifar10 --temp_dir $SLURM_TMPDIR
```

### Teleportation Experiment

Demonstrates penultimate activation instability under neural teleportation (quiver isomorphism via the `neuralteleportation` library).

```bash
# Pretrained model
python teleportation_experiment.py --architecture resnet18 --dataset imagenet --pretrained \
    --data_dir /datashare/imagenet/ILSVRC2012

# From saved weights
python teleportation_experiment.py --architecture resnet18 --dataset cifar10 \
    --weights_path path/to/weights.pth
```

### Generate LaTeX Tables

```bash
python generate_theorem45_tables.py --experiments alexnet_imagenet resnet_imagenet vgg_imagenet --output tables/
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
m.alexnet(weights='DEFAULT')
m.resnet18(weights='DEFAULT')
m.vgg11(weights='DEFAULT')
m.vgg11_bn(weights='DEFAULT')
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

1. **Scans** for existing results and checkpoints across all 9 tasks (3 steps × 3 architectures)
2. **Writes** `pipeline_state.json` with the current state of each task (`done`, `in_progress`, or `pending`)
3. **Submits** only the needed SLURM array jobs, skipping completed tasks

The pipeline is **idempotent** -- safe to re-run after partial failures. For Theorem 4.5, it detects checkpoint files and resumes from where it left off.

### Result files

| Step | Output |
|------|--------|
| Isomorphism | `experiments/{experiment}/isomorphism/isomorphism_results.json` |
| Teleportation | `results/teleportation/{architecture}_{dataset}_teleportation.json` |
| Theorem 4.5 | `experiments/{experiment}/theorem45/theorem45_results.json` |

---

## Training Models (optional)

For experiments on CIFAR-10/100 (not needed for the pretrained ImageNet pipeline):

```bash
python training.py --experiment_name alexnet_cifar10 --temp_dir $SLURM_TMPDIR
```

Outputs weights to `experiments/{experiment}/weights/epoch_{N}.pth`.

---

## Repository Structure

```
.
├── run_pipeline.sh                # Pipeline orchestrator (scan + submit)
├── isomorphism_experiment.py      # Pillar 1: KM invariance under neuron permutations
├── validate_theorem45.py          # Pillar 2: empirical distance lower bound
├── teleportation_experiment.py    # Penultimate activation instability under teleportation
├── training.py                    # Model training (optional, for CIFAR experiments)
├── generate_matrices.py           # Knowledge matrix computation (chunked)
├── generate_theorem45_tables.py   # LaTeX table generation for Theorem 4.5
├── plot_graphs.py                 # Training curve visualization
├── job_isomorphism.sh             # SLURM job: Step A
├── job_teleportation.sh           # SLURM job: Step B
├── job_theorem45.sh               # SLURM job: Step C
├── job_training.sh                # SLURM job: model training (optional)
├── job_matrices.sh                # SLURM job: matrix computation (optional)
├── constants/
│   └── constants.py               # Experiment configs, architectures, attacks
├── utils/
│   ├── utils.py                   # Model loading, datasets, utilities
│   ├── features.py                # Penultimate feature extraction
│   └── atomic_io.py               # Atomic file writes
├── matrix_construction/
│   ├── parallel.py                # Parallel matrix computation
│   └── matrix_computation.py      # Core matrix computation logic
├── model_zoo/                     # Local model implementations (AlexNet, ResNet, VGG, CNN, MLP)
├── patches/                       # neuralteleportation PyTorch 2.x compatibility
├── unit_test/                     # Unit tests
├── docs/
│   ├── new_direction.tex          # New paper direction strategy document
│   └── rebuttal/                  # LaTeX paper sources
├── experiments/                   # Per-experiment outputs (weights, matrices, results)
└── legacy/                        # Old adversarial detection pipeline (archived)
```

---

## Available Experiments

Defined in `constants/constants.py`:

| Name | Architecture | Dataset | Pretrained | Epochs |
|------|-------------|---------|------------|--------|
| `alexnet_imagenet` | AlexNet | ImageNet | Yes | 0 |
| `resnet_imagenet` | ResNet18 | ImageNet | Yes | 0 |
| `vgg_imagenet` | VGG11 | ImageNet | Yes | 0 |
| `alexnet_cifar10` | AlexNet | CIFAR-10 | Yes (fine-tuned) | 70 |
| `resnet_cifar10` | ResNet18 | CIFAR-10 | No | 100 |
| `resnet_cifar100` | ResNet18 | CIFAR-100 | No | 100 |
| `vgg_cifar100` | VGG11 | CIFAR-100 | No | 150 |

---

## License

Apache 2.0
