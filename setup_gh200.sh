#!/bin/bash
# ==============================================================
# setup_gh200.sh — One-time setup for GH200 / IQ HPC (gh-aria)
#
# Run this on the IQ HPC login node (or via salloc -p gh-aria)
# BEFORE submitting any pipeline jobs. It creates a Python venv,
# installs dependencies with ARM64 CUDA wheels, downloads
# pretrained weights & datasets, and verifies CUDA availability.
#
# Usage:
#   ssh aria  # or salloc -p gh-aria --gres=gpu:1 --mem=32G
#   bash setup_gh200.sh
# ==============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_NAME="${VENV_NAME:-gh_env}"
VENV_PATH="$SCRIPT_DIR/$VENV_NAME"
INDEX_URL="https://download.pytorch.org/whl/cu128"

echo "=============================================================="
echo "  GH200 Setup — Hidden Activations Pipeline"
echo "=============================================================="
echo ""
echo "  Project dir: $SCRIPT_DIR"
echo "  Venv:        $VENV_PATH"
echo "  Architecture: $(uname -m)"
echo ""

# --- Check architecture ---
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo "WARNING: Expected aarch64, got $ARCH."
    echo "This setup script is designed for the GH200 ARM64 node."
    echo "Proceeding anyway..."
fi

# --- Create venv ---
if [ -d "$VENV_PATH" ]; then
    echo "Venv already exists at $VENV_PATH"
    echo "To recreate, delete it first: rm -rf $VENV_PATH"
else
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_PATH"
    echo "  Created: $VENV_PATH"
fi

source "$VENV_PATH/bin/activate"
echo "Activated venv: $(which python3)"
echo ""

# --- Install PyTorch (ARM64 CUDA wheels) ---
echo "Installing PyTorch + torchvision (ARM64 CUDA)..."
pip install --upgrade pip
pip install torch torchvision --index-url "$INDEX_URL"
echo ""

# --- Install remaining dependencies ---
echo "Installing project dependencies..."
pip install torchattacks==3.5.1 \
    "scikit-learn>=1.3.2" \
    "scipy>=1.10.1" \
    "numpy<2.0" \
    matplotlib pillow tqdm \
    "optuna>=4.1.0" \
    "sqlalchemy>=2.0"

# --- Install knowledgematrix ---
echo ""
echo "Installing knowledgematrix..."
pip install git+https://github.com/samueleblanc/knowledgematrix.git@0d26c7a2959b8845bd370874c195a2b30c553688

echo ""

# --- Verify CUDA ---
echo "Verifying CUDA availability..."
python3 -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:             {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'  GPU memory:      {props.total_memory / 1e9:.1f} GB')
    print(f'  Compute cap:     {props.major}.{props.minor}')
else:
    print('  WARNING: CUDA not available! Check PyTorch installation.')
    print('  This may be expected if running on a login node without GPU.')
"
echo ""

# --- Download pretrained weights ---
echo "Downloading pretrained weights (if needed)..."
cd "$SCRIPT_DIR"
python3 << 'WEIGHTS_EOF'
import os, sys
sys.path.insert(0, '.')
from constants.constants import DEFAULT_EXPERIMENTS

arch_map = {-3: 'alexnet', -2: 'resnet', -1: 'vgg'}
weight_paths = {
    'alexnet': 'experiments/alexnet_imagenet/weights/pretrained-weights.pth',
    'resnet': 'experiments/resnet_imagenet/weights/pretrained-weights.pth',
    'vgg': 'experiments/vgg_imagenet/weights/pretrained-weights.pth',
}

needed = set()
for exp, cfg in DEFAULT_EXPERIMENTS.items():
    idx = cfg.get('architecture_index', 0)
    if idx in arch_map:
        needed.add(arch_map[idx])

for arch in needed:
    path = weight_paths[arch]
    if os.path.exists(path):
        print(f'  OK: {arch} ({path})')
    else:
        print(f'  Downloading: {arch}...')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import torch
        if arch == 'alexnet':
            from torchvision.models import alexnet, AlexNet_Weights
            model = alexnet(weights=AlexNet_Weights.DEFAULT)
            torch.save(model.state_dict(), path)
        elif arch == 'resnet':
            from torchvision.models import resnet18, ResNet18_Weights
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            torch.save(model.state_dict(), path)
        elif arch == 'vgg':
            from torchvision.models import vgg11, VGG11_Weights
            model = vgg11(weights=VGG11_Weights.DEFAULT)
            torch.save(model.state_dict(), path)
        print(f'  Saved: {path}')

print('Pretrained weights ready.')
WEIGHTS_EOF

echo ""

# --- Download datasets ---
echo "Downloading datasets (if needed)..."
python3 << 'DATASETS_EOF'
import os, sys
sys.path.insert(0, '.')
from constants.constants import DEFAULT_EXPERIMENTS

datasets_needed = set()
for exp, cfg in DEFAULT_EXPERIMENTS.items():
    datasets_needed.add(cfg.get('dataset', 'cifar10'))

dataset_dirs = {
    'cifar10': 'data/cifar-10-batches-py',
    'cifar100': 'data/cifar-100-python',
    'mnist': 'data/MNIST',
    'fashion': 'data/FashionMNIST',
}

for ds in datasets_needed:
    dir_path = dataset_dirs.get(ds)
    if dir_path and os.path.isdir(dir_path):
        print(f'  OK: {ds} ({dir_path})')
    elif ds == 'imagenet':
        print(f'  SKIP: imagenet (must be pre-staged)')
    elif dir_path:
        print(f'  Downloading: {ds}...')
        if ds == 'cifar10':
            from torchvision.datasets import CIFAR10
            CIFAR10(root='./data', train=True, download=True)
            CIFAR10(root='./data', train=False, download=True)
        elif ds == 'cifar100':
            from torchvision.datasets import CIFAR100
            CIFAR100(root='./data', train=True, download=True)
            CIFAR100(root='./data', train=False, download=True)
        elif ds == 'mnist':
            import torchvision
            torchvision.datasets.MNIST(root='./data', train=True, download=True)
        elif ds == 'fashion':
            import torchvision
            torchvision.datasets.FashionMNIST(root='./data', train=True, download=True)
        print(f'  Downloaded: {ds}')

print('Datasets ready.')
DATASETS_EOF

echo ""
echo "=============================================================="
echo "  Setup Complete"
echo "=============================================================="
echo ""
echo "  Venv: $VENV_PATH"
echo "  Activate with: source $VENV_PATH/bin/activate"
echo ""
echo "  Next steps:"
echo "    1. bash run_experiment.sh --dry-run --test --skip-audit alexnet_cifar10"
echo "    2. Inspect generated scripts in experiments/alexnet_cifar10/orchestrator_jobs/"
echo "    3. bash run_experiment.sh --test --skip-audit alexnet_cifar10"
echo "=============================================================="
