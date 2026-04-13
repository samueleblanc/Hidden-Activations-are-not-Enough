# Knowledge Matrices as Canonical Neural Network Representations

Implementation of "Hidden Activations Are Not Enough: A General Approach to Neural Network Predictions" (arXiv:2409.13163) by Samuel Leblanc, Aiky Rasolomanana, and Marco Armenta.

Given a neural network and a data sample, we compute a **knowledge matrix** (via quiver representations). These matrices capture the full linear behavior of the network at each input point. We show that knowledge matrices are superior to penultimate-layer activations as neural network representations through two pillars:

1. **Isomorphism Invariance** -- Knowledge matrices are invariant under neuron permutations (quiver isomorphisms), while penultimate activations change arbitrarily.
2. **Distance Lower Bound (Theorem 4.5)** -- Knowledge matrix distances lower-bound logit distances: `||M(x) - M(x')|| >= gamma * ||f(x) - f(x')||`. Empirically, KMs amplify separations more than penultimate features.

Additionally, we investigate how **penultimate activation distances behave when increasing network size** using pretrained torchvision models.

---

## Experiments

### Isomorphism Invariance

Demonstrates that knowledge matrices are invariant under neuron permutations while penultimate activations are not.

```bash
python isomorphism_experiment.py --experiment alexnet_cifar10
python isomorphism_experiment.py --experiment alexnet_cifar10 --num_permutations 5 --num_samples 500
```

### Theorem 4.5 Validation

Empirical validation of the distance lower bound. Generates adversarial pairs on-the-fly, computes logit, penultimate, and KM distances, estimates gamma with bootstrap CI.

```bash
python validate_theorem45.py --experiment alexnet_cifar10 --temp_dir $SLURM_TMPDIR
```

### Teleportation Experiment

Demonstrates penultimate activation instability under neural teleportation (quiver isomorphism via the `neuralteleportation` library).

```bash
python teleportation_experiment.py --arch resnet18 --dataset cifar10 --num_teleportations 10
```

### Generate LaTeX Tables

```bash
python generate_theorem45_tables.py --experiments alexnet_cifar10 resnet_cifar10 --output tables/
```

---

## Setup

**Local:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Compute Canada / Alliance cluster:**
```bash
module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
virtualenv env
source env/bin/activate
pip install -r requirements-slurm.txt
pip install git+https://github.com/samueleblanc/knowledgematrix.git
```

For the teleportation experiment, install the patched `neuralteleportation`:
```bash
pip install git+https://github.com/GRAAL-Research/neuralteleportation.git
bash patches/apply_neuralteleportation_patches.sh
```

---

## Training Models

```bash
python training.py --experiment_name alexnet_cifar10 --temp_dir $SLURM_TMPDIR
```

Outputs weights to `experiments/{experiment}/weights/epoch_{N}.pth`.

## Computing Knowledge Matrices

```bash
python generate_matrices.py --experiment alexnet_cifar10 --chunk_id 0 --total_chunks 8 --batch_size 1800
```

---

## Repository Structure

```
.
├── isomorphism_experiment.py      # Pillar 1: KM invariance under neuron permutations
├── validate_theorem45.py          # Pillar 2: empirical distance lower bound
├── teleportation_experiment.py    # Penultimate activation instability under teleportation
├── training.py                    # Model training
├── generate_matrices.py           # Knowledge matrix computation (chunked)
├── generate_theorem45_tables.py   # LaTeX table generation for Theorem 4.5
├── plot_graphs.py                 # Training curve visualization
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
├── job_training.sh                # SLURM job: model training
├── job_matrices.sh                # SLURM job: matrix computation
├── experiments/                   # Per-experiment outputs (weights, matrices, results)
└── legacy/                        # Old adversarial detection pipeline (archived)
```

---

## Available Experiments

Defined in `constants/constants.py`:

| Name | Architecture | Dataset | Epochs |
|------|-------------|---------|--------|
| `alexnet_cifar10` | AlexNet (pretrained) | CIFAR-10 | 70 |
| `resnet_cifar10` | ResNet18 | CIFAR-10 | 100 |
| `resnet_cifar100` | ResNet18 | CIFAR-100 | 100 |
| `vgg_cifar100` | VGG11 | CIFAR-100 | 150 |

---

## License

Apache 2.0
