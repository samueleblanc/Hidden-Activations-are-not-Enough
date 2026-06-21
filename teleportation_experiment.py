"""
Pillar 1: Penultimate Activation Instability under Neural Teleportation.

Demonstrates that penultimate-layer activations change substantially under
neural teleportation (quiver isomorphism), even though the network computes
the same function. This validates that penultimate features are NOT invariant
under weight-space symmetries.

Usage:
    python teleportation_experiment.py \
        --architecture resnet18 \
        --dataset cifar10 \
        --weights_path path/to/weights.pth \
        --num_teleportations 100 \
        --num_samples 500
"""

import argparse
import copy
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pathlib import Path

from utils.atomic_io import atomic_json_dump
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.models.model_zoo.resnetcob import (
    resnet18COB, resnet34COB, resnet50COB, resnet101COB, resnet152COB,
)
from neuralteleportation.models.model_zoo.vggcob import (
    vgg11COB, vgg13COB, vgg16COB, vgg19COB,
    vgg11_bnCOB, vgg13_bnCOB, vgg16_bnCOB, vgg19_bnCOB,
)
from neuralteleportation.models.model_zoo.densenetcob import (
    densenet121COB, densenet161COB, densenet169COB, densenet201COB,
)
# GoogLeNet COB is supplied by the local patch. It is installed into the
# neuralteleportation package by patches/apply_neuralteleportation_patches.sh
# (see patches/googlenetcob.py for the source). Wrap the import in try/except
# so a clear error surfaces if patches were not applied.
try:
    from neuralteleportation.models.model_zoo.googlenetcob import GoogLeNetCOB
except ImportError as e:
    GoogLeNetCOB = None
    _GOOGLENETCOB_IMPORT_ERROR = e


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

ARCHITECTURES = {
    'resnet18':    {'factory': resnet18COB,    'penultimate_dim': 512,  'family': 'resnet'},
    'resnet34':    {'factory': resnet34COB,    'penultimate_dim': 512,  'family': 'resnet'},
    'resnet50':    {'factory': resnet50COB,    'penultimate_dim': 2048, 'family': 'resnet'},
    'resnet101':   {'factory': resnet101COB,   'penultimate_dim': 2048, 'family': 'resnet'},
    'resnet152':   {'factory': resnet152COB,   'penultimate_dim': 2048, 'family': 'resnet'},
    'vgg11':       {'factory': vgg11COB,       'penultimate_dim': 4096, 'family': 'vgg'},
    'vgg13':       {'factory': vgg13COB,       'penultimate_dim': 4096, 'family': 'vgg'},
    'vgg16':       {'factory': vgg16COB,       'penultimate_dim': 4096, 'family': 'vgg'},
    'vgg19':       {'factory': vgg19COB,       'penultimate_dim': 4096, 'family': 'vgg'},
    'vgg11_bn':    {'factory': vgg11_bnCOB,    'penultimate_dim': 4096, 'family': 'vgg'},
    'vgg13_bn':    {'factory': vgg13_bnCOB,    'penultimate_dim': 4096, 'family': 'vgg'},
    'vgg16_bn':    {'factory': vgg16_bnCOB,    'penultimate_dim': 4096, 'family': 'vgg'},
    'vgg19_bn':    {'factory': vgg19_bnCOB,    'penultimate_dim': 4096, 'family': 'vgg'},
    'densenet121': {'factory': densenet121COB, 'penultimate_dim': 1024, 'family': 'densenet'},
    'densenet161': {'factory': densenet161COB, 'penultimate_dim': 2208, 'family': 'densenet'},
    'densenet169': {'factory': densenet169COB, 'penultimate_dim': 1664, 'family': 'densenet'},
    'densenet201': {'factory': densenet201COB, 'penultimate_dim': 1920, 'family': 'densenet'},
    'googlenet':   {'factory': GoogLeNetCOB,   'penultimate_dim': 1024, 'family': 'googlenet'},
}

NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'tiny_imagenet': 200, 'imagenet': 1000}


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def create_model(arch_name, num_classes):
    """Create a COB model compatible with neuralteleportation.

    COB models have the same architecture as torchvision models but use
    COB layer types. Torchvision-trained state_dicts load directly.

    Args:
        arch_name: Key into ARCHITECTURES (e.g. 'resnet18', 'vgg11_bn').
        num_classes: Number of output classes for the final FC layer.

    Returns:
        nn.Module: A COB model instance.

    Raises:
        KeyError: If arch_name is not in ARCHITECTURES.
        RuntimeError: If GoogLeNetCOB was requested but the patch is not
            installed (patches/apply_neuralteleportation_patches.sh).
    """
    config = ARCHITECTURES[arch_name]
    factory = config['factory']
    if factory is None and arch_name == 'googlenet':
        raise RuntimeError(
            "GoogLeNetCOB is not available — run "
            "`bash patches/apply_neuralteleportation_patches.sh` to install "
            "the local GoogLeNet COB module."
        ) from globals().get('_GOOGLENETCOB_IMPORT_ERROR', None)
    if config['family'] == 'googlenet':
        # GoogLeNetCOB takes (num_classes=, init_weights=) — match torchvision
        # state-dict layout by skipping the random init since pretrained
        # weights are loaded right after.
        return factory(num_classes=num_classes, init_weights=False)
    return factory(pretrained=False, num_classes=num_classes)


# ---------------------------------------------------------------------------
# Offline-safe pretrained COB loader (single source of truth)
# ---------------------------------------------------------------------------

# torchvision factory + Weights-enum attribute name per supported arch. The
# Weights enum's ``DEFAULT.url`` basename is the file torch.hub caches, and is
# exactly what `run_pipeline.sh` Phase-0d pre-fetches with `weights='DEFAULT'`.
_TORCHVISION_WEIGHTS_ENUM = {
    'resnet18': 'ResNet18_Weights', 'resnet34': 'ResNet34_Weights',
    'resnet50': 'ResNet50_Weights', 'resnet101': 'ResNet101_Weights',
    'resnet152': 'ResNet152_Weights',
    'vgg11': 'VGG11_Weights', 'vgg13': 'VGG13_Weights',
    'vgg16': 'VGG16_Weights', 'vgg19': 'VGG19_Weights',
    'vgg11_bn': 'VGG11_BN_Weights', 'vgg13_bn': 'VGG13_BN_Weights',
    'vgg16_bn': 'VGG16_BN_Weights', 'vgg19_bn': 'VGG19_BN_Weights',
    'densenet121': 'DenseNet121_Weights', 'densenet161': 'DenseNet161_Weights',
    'densenet169': 'DenseNet169_Weights', 'densenet201': 'DenseNet201_Weights',
    'googlenet': 'GoogLeNet_Weights',
}


def expected_default_weight_basename(arch):
    """Return the hub-cache filename of torchvision's DEFAULT weights for ``arch``.

    Derived live from ``torchvision.models.<Weights>.DEFAULT.url`` so it always
    matches whatever ``weights='DEFAULT'`` resolves to on the installed
    torchvision (V1 on the cluster's 0.17.2, V2 for ResNet-152 on newer local
    builds). This is the exact basename torch.hub keys its checkpoint cache on.
    """
    import os
    import torchvision.models as tv_models
    enum_name = _TORCHVISION_WEIGHTS_ENUM[arch]
    weights_enum = getattr(tv_models, enum_name)
    return os.path.basename(weights_enum.DEFAULT.url)


def assert_default_weights_cached(arch):
    """Fail loudly (on the login node) if DEFAULT weights are not hub-cached.

    Compute nodes have no internet, so a missing checkpoint would otherwise
    only surface mid-job as a silent torch.hub download attempt that hangs or
    crashes. Raising here — derived from the *same* DEFAULT URL the loader will
    request — means a future torchvision URL drift breaks before allocation.
    """
    import os
    import torch
    basename = expected_default_weight_basename(arch)
    ckpt_dir = os.path.join(torch.hub.get_dir(), 'checkpoints')
    path = os.path.join(ckpt_dir, basename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"torchvision DEFAULT weights for {arch!r} not cached: expected "
            f"{path!r} (basename derived from "
            f"{_TORCHVISION_WEIGHTS_ENUM[arch]}.DEFAULT.url). Compute nodes have "
            f"no internet — pre-cache on a login node (run_pipeline.sh Phase-0d, "
            f"or `python -c \"import torchvision.models as m; "
            f"m.{arch}(weights='DEFAULT')\"`)."
        )
    return path


def load_pretrained_cob(arch, device, num_classes=1000):
    """Build a COB model and load torchvision DEFAULT (ImageNet) weights — offline-safe.

    Single source of truth for the pretrained-COB load shared by Step B
    (`teleportation_experiment.py`) and D2 (`teleportation_km_drift.py`).

    Why not ``factory(pretrained=True)``: the `neuralteleportation` COB factories
    download via their own legacy ``model_urls`` (e.g. resnet152 →
    ``resnet152-b121ed2d.pth``, the orphan IMAGENET1K_V1). That basename differs
    from the torchvision DEFAULT (V2 ``resnet152-f82ba261.pth``) that Phase-0d
    pre-caches, so the COB path would miss the cache and try to download on a
    no-internet compute node → crash. Building with ``pretrained=False`` and
    loading ``torchvision.<arch>(weights='DEFAULT').state_dict()`` instead reuses
    the pre-cached DEFAULT file. (Bonus: this aligns D2's measured checkpoint with
    Steps B/C — all three now use the torchvision DEFAULT, V2 f82ba261 for
    ResNet-152, rather than the orphan V1 the old COB path pulled.)

    Args:
        arch: Architecture key into ARCHITECTURES.
        device: torch device / device string to place the model on.
        num_classes: ImageNet class count (1000); kept explicit for parity with
            create_model's signature.

    Returns:
        nn.Module: COB model with DEFAULT weights loaded, in eval mode, on device.
    """
    import torchvision.models as tv_models

    # Fail on the login node if the DEFAULT checkpoint is not hub-cached
    # (offline compute nodes cannot download it).
    assert_default_weights_cached(arch)

    model = create_model(arch, num_classes)
    tv_factory = getattr(tv_models, arch)
    if ARCHITECTURES[arch]['family'] == 'googlenet':
        # torchvision insists aux_logits=True with pretrained weights; set
        # transform_input=False to match this repo's dataset normalization, then
        # strip the aux classifiers and filter their keys before loading into the
        # GoogLeNetCOB (which never has aux classifiers).
        tv_model = tv_factory(weights='DEFAULT', transform_input=False)
        tv_model.aux_logits = False
        tv_model.aux1 = None
        tv_model.aux2 = None
        sd = {k: v for k, v in tv_model.state_dict().items()
              if not k.startswith('aux1.') and not k.startswith('aux2.')}
        model.load_state_dict(sd, strict=True)
    else:
        # ResNet/VGG/DenseNet COB state-dicts share torchvision's key layout
        # exactly (verified: identical key order), so a plain strict load works.
        tv_model = tv_factory(weights='DEFAULT')
        model.load_state_dict(tv_model.state_dict())
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Penultimate-layer feature extraction
# ---------------------------------------------------------------------------

class PenultimateExtractor:
    """Extract penultimate-layer activations via a forward hook.

    ResNets:    hooks on avgpool -> flatten -> (batch, channels)
    VGGs:       hooks on classifier[4] (ReLU after 2nd-to-last Linear) -> (batch, 4096)
    DenseNets:  hooks on adaptive_avg_pool2d -> flatten -> (batch, num_features)
    GoogLeNet:  hooks on avgpool -> flatten -> (batch, 1024)
    """

    def __init__(self, model, arch_name):
        self._features = None
        config = ARCHITECTURES[arch_name]
        if config['family'] == 'resnet':
            target = model.avgpool
        elif config['family'] == 'vgg':
            target = model.classifier[4]
        elif config['family'] == 'densenet':
            # DenseNetCOB exposes the post-pool layer as `adaptive_avg_pool2d`
            target = model.adaptive_avg_pool2d
        elif config['family'] == 'googlenet':
            # GoogLeNetCOB exposes `avgpool` (AdaptiveAvgPool2dCOB((1,1)))
            target = model.avgpool
        else:
            raise ValueError(f"Unknown architecture family: {config['family']}")
        self._hook = target.register_forward_hook(self._capture)

    def _capture(self, module, input, output):
        self._features = output.detach()

    def extract(self, model, data, batch_size=64):
        """Extract penultimate features for all samples in data.

        Args:
            model: The model to run forward passes on.
            data: Tensor of shape (N, C, H, W).
            batch_size: Number of samples per forward pass.

        Returns:
            Tensor of shape (N, penultimate_dim).
        """
        model.eval()
        all_feats = []
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                model(batch)
                feat = self._features
                if feat.dim() > 2:
                    feat = feat.flatten(1)
                all_feats.append(feat.cpu())
        return torch.cat(all_feats, dim=0)

    def remove(self):
        """Remove the forward hook."""
        self._hook.remove()


# ---------------------------------------------------------------------------
# Teleportation wrapper
# ---------------------------------------------------------------------------

def teleport_model(model, input_shape, seed, cob_range=1.0):
    """Create a random teleportation of the model.

    Returns a new model with different weights but identical function.
    The original model is not modified.

    Args:
        model: nn.Module (COB model) to teleport.
        input_shape: tuple, e.g. (1, 3, 224, 224) for JIT tracing.
        seed: random seed for reproducibility.
        cob_range: change-of-basis magnitude. Default 1.0 (unchanged for Step-B
            and the shallower D2 archs). A smaller value keeps the per-path COB
            product within fp32 range when the knowledge matrix is computed on a
            very deep teleported net (resnet152's 152-layer product overflows to
            NaN/inf at cob_range=1; see teleportation_km_drift.py and
            docs/Final-twist/km-notes.md). The teleportation stays
            function-preserving at any range.

    Returns:
        nn.Module: teleported model (deep copy with modified weights).
    """
    model_copy = copy.deepcopy(model)
    torch.manual_seed(seed)
    np.random.seed(seed)  # COB generation uses np.random
    tp = NeuralTeleportationModel(model_copy, input_shape=input_shape)
    tp.random_teleport(cob_range=cob_range)
    return model_copy


# ---------------------------------------------------------------------------
# Distance computation
# ---------------------------------------------------------------------------

def compute_normalized_distances(feats_orig, feats_teleported):
    """Compute ||y - y'|| / sqrt(dim) per sample.

    Args:
        feats_orig: Tensor of shape (N, D) -- original penultimate features.
        feats_teleported: Tensor of shape (N, D) -- teleported features.

    Returns:
        numpy array of shape (N,) with normalized L2 distances.
    """
    diff = feats_orig - feats_teleported
    l2_per_sample = torch.norm(diff, dim=1)
    dim = feats_orig.shape[1]
    return (l2_per_sample / (dim ** 0.5)).numpy()


def linear_cka(X, Y):
    """Linear Centered Kernel Alignment (Kornblith et al. 2019), Gram form.

    Returns a scalar in [0, 1]. CKA = 1 iff the two representations are equal
    up to an orthogonal transformation. Under `neuralteleportation`'s positive-
    diagonal COB, CKA < 1 (the COB is not orthogonal), so this column captures
    the gap between L2 drift (huge) and the strongest pre-baseline metric in
    the rep-similarity literature.
    """
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)
    cross = Xc.T @ Yc
    num = (cross * cross).sum()
    den = ((Xc.T @ Xc).norm() * (Yc.T @ Yc).norm()).clamp_min(1e-30)
    return float((num / den).item())


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_dataset(dataset_name, split, num_samples, data_dir='data', seed=42):
    """Load a random subset of a dataset as a (N, 3, 224, 224) tensor.

    Args:
        dataset_name: One of 'cifar10', 'cifar100', 'tiny_imagenet'.
        split: 'train' or 'test'.
        num_samples: Number of samples to load.
        data_dir: Root directory for datasets.
        seed: Random seed for reproducible subset selection.

    Returns:
        Tensor of shape (num_samples, 3, 224, 224).
    """
    is_train = (split == 'train')
    if dataset_name == 'cifar10':
        ds = datasets.CIFAR10(data_dir, train=is_train, download=False,
                              transform=IMAGENET_TRANSFORM)
    elif dataset_name == 'cifar100':
        ds = datasets.CIFAR100(data_dir, train=is_train, download=False,
                               transform=IMAGENET_TRANSFORM)
    elif dataset_name == 'tiny_imagenet':
        subdir = 'train' if is_train else 'val'
        path = Path(data_dir) / 'tiny-imagenet-200' / subdir
        ds = datasets.ImageFolder(str(path), transform=IMAGENET_TRANSFORM)
    elif dataset_name == 'imagenet':
        # Use validation set (split into train/test halves for consistency)
        val_dir = Path(data_dir) / 'validation'
        if not val_dir.exists():
            val_dir = Path(data_dir) / 'val'
        ds = datasets.ImageFolder(str(val_dir), transform=IMAGENET_TRANSFORM)
        # First half for 'train', second half for 'test'
        half = len(ds) // 2
        if is_train:
            indices = list(range(half))
        else:
            indices = list(range(half, len(ds)))
        ds = torch.utils.data.Subset(ds, indices)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(ds), generator=g)[:num_samples]
    images = torch.stack([ds[int(i)][0] for i in indices])
    return images


def generate_random_inputs(num_samples, shape=(3, 224, 224), seed=42):
    """Generate i.i.d. N(0,1) random inputs.

    Args:
        num_samples: Number of random samples to generate.
        shape: Shape of each sample (default: (3, 224, 224)).
        seed: Random seed for reproducibility.

    Returns:
        Tensor of shape (num_samples, *shape).
    """
    g = torch.Generator().manual_seed(seed)
    return torch.randn(num_samples, *shape, generator=g)


# ---------------------------------------------------------------------------
# Output equivalence verification
# ---------------------------------------------------------------------------

def verify_equivalence(model_orig, model_teleported, data, batch_size=64):
    """Check that two models produce the same outputs.

    Args:
        model_orig: Original model.
        model_teleported: Teleported model.
        data: Tensor of shape (N, C, H, W).
        batch_size: Number of samples per forward pass.

    Returns:
        dict with 'max_logit_diff' (float) and 'all_predictions_match' (bool).
    """
    model_orig.eval()
    model_teleported.eval()
    max_diff = 0.0
    all_match = True
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            out_orig = model_orig(batch)
            out_tp = model_teleported(batch)
            diff = (out_orig - out_tp).abs().max().item()
            max_diff = max(max_diff, diff)
            if not torch.equal(out_orig.argmax(dim=1), out_tp.argmax(dim=1)):
                all_match = False
    return {'max_logit_diff': float(max_diff), 'all_predictions_match': all_match}


# ---------------------------------------------------------------------------
# Single teleportation run
# ---------------------------------------------------------------------------

def run_single_teleportation(model, arch_name, input_shape, splits,
                             orig_feats, tp_seed, device):
    """Run one teleportation and measure penultimate activation distances.

    Args:
        model: original nn.Module (not modified).
        arch_name: architecture name string.
        input_shape: (1, 3, 224, 224) for JIT tracing.
        splits: dict of {split_name: (N, C, H, W) tensor on device}.
        orig_feats: dict of {split_name: (N, D) CPU tensor} pre-extracted features.
        tp_seed: random seed for this teleportation.
        device: torch device.

    Returns:
        dict with per-split distance statistics.
    """
    model_tp = teleport_model(model, input_shape, tp_seed)
    model_tp = model_tp.to(device)
    model_tp.eval()

    # Sanity check: outputs should match
    first_split_data = next(iter(splits.values()))
    equiv = verify_equivalence(model, model_tp, first_split_data[:64], batch_size=64)

    extractor = PenultimateExtractor(model_tp, arch_name)
    result = {
        'teleportation_seed': tp_seed,
        'output_equivalence': equiv,
    }

    for split_name, data in splits.items():
        feats_tp = extractor.extract(model_tp, data)
        distances = compute_normalized_distances(orig_feats[split_name], feats_tp)
        cka = linear_cka(orig_feats[split_name].double(), feats_tp.double())
        result[split_name] = {
            'mean': float(np.mean(distances)),
            'std': float(np.std(distances)),
            'min': float(np.min(distances)),
            'max': float(np.max(distances)),
            'per_sample': distances.tolist(),
            'cka_linear': cka,
        }

    extractor.remove()
    del model_tp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------

def run_experiment(args):
    """Run the full teleportation experiment."""
    arch_name = args.architecture
    dataset = args.dataset
    num_classes = NUM_CLASSES[dataset]
    penultimate_dim = ARCHITECTURES[arch_name]['penultimate_dim']
    input_shape = (1, 3, 224, 224)

    print(f"Teleportation Experiment: {arch_name} / {dataset}", flush=True)
    print(f"  Penultimate dim: {penultimate_dim}", flush=True)
    print(f"  Teleportations: {args.num_teleportations}", flush=True)
    print(f"  Samples per split: {args.num_samples}", flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}", flush=True)

    # Load model
    if args.pretrained:
        # Offline-safe shared loader (single source of truth, also used by
        # teleportation_km_drift.py): builds the COB model and loads the
        # torchvision DEFAULT ImageNet state-dict from the pre-cached hub file.
        model = load_pretrained_cob(arch_name, device, num_classes=num_classes)
        print(f"  Using pretrained torchvision weights for {arch_name}",
              flush=True)
    else:
        model = create_model(arch_name, num_classes)
        state_dict = torch.load(args.weights_path, map_location='cpu',
                                weights_only=True)
        model.load_state_dict(state_dict)
        print(f"  Weights loaded from: {args.weights_path}", flush=True)
        model = model.to(device)
        model.eval()

    # Load data
    print("Loading datasets...", flush=True)
    train_data = load_dataset(dataset, 'train', args.num_samples,
                              data_dir=args.data_dir, seed=args.seed)
    test_data = load_dataset(dataset, 'test', args.num_samples,
                             data_dir=args.data_dir, seed=args.seed)
    random_data = generate_random_inputs(args.num_samples, seed=args.seed)

    splits = {
        'train': train_data.to(device),
        'test': test_data.to(device),
        'random': random_data.to(device),
    }

    # Extract original features (done once)
    print("Extracting original penultimate features...", flush=True)
    extractor = PenultimateExtractor(model, arch_name)
    orig_feats = {}
    for split_name, data in splits.items():
        orig_feats[split_name] = extractor.extract(model, data)
    extractor.remove()

    # Run teleportations
    print(f"\nRunning {args.num_teleportations} teleportations...", flush=True)
    per_teleportation = []

    for t in range(args.num_teleportations):
        tp_seed = args.seed + t
        t0 = time.perf_counter()

        result = run_single_teleportation(
            model, arch_name, input_shape, splits, orig_feats, tp_seed, device
        )
        per_teleportation.append(result)

        elapsed = time.perf_counter() - t0
        equiv = result['output_equivalence']
        equiv_ok = "OK" if equiv['all_predictions_match'] else "MISMATCH"
        if equiv['max_logit_diff'] > 1e-3:
            equiv_ok = f"WARN(diff={equiv['max_logit_diff']:.2e})"
        print(
            f"  [{t+1:3d}/{args.num_teleportations}] seed={tp_seed} "
            f"train={result['train']['mean']:.4f} "
            f"test={result['test']['mean']:.4f} "
            f"random={result['random']['mean']:.4f} "
            f"equiv={equiv_ok} [{elapsed:.1f}s]",
            flush=True,
        )

    # Aggregate
    aggregate = {}
    for split_name in ['train', 'test', 'random']:
        means = [r[split_name]['mean'] for r in per_teleportation]
        ckas = [r[split_name].get('cka_linear', float('nan'))
                for r in per_teleportation]
        cka_dist = [1.0 - c for c in ckas]
        all_samples = []
        for r in per_teleportation:
            all_samples.extend(r[split_name]['per_sample'])
        aggregate[split_name] = {
            'mean_of_means': float(np.mean(means)),
            'std_of_means': float(np.std(means)),
            'overall_mean': float(np.mean(all_samples)),
            'overall_std': float(np.std(all_samples)),
            'cka_linear_mean': float(np.mean(ckas)),
            'cka_linear_std': float(np.std(ckas)),
            'cka_linear_1m_mean': float(np.mean(cka_dist)),
            'cka_linear_1m_std': float(np.std(cka_dist)),
        }

    # Print summary
    print(f"\n{'=' * 60}", flush=True)
    print(f"SUMMARY: {arch_name} / {dataset}", flush=True)
    print(f"{'=' * 60}", flush=True)
    for split_name in ['train', 'test', 'random']:
        agg = aggregate[split_name]
        print(
            f"  {split_name:8s}: mean={agg['mean_of_means']:.4f} "
            f"+/- {agg['std_of_means']:.4f}  "
            f"(overall std={agg['overall_std']:.4f})",
            flush=True,
        )

    # Save results
    results = {
        'architecture': arch_name,
        'dataset': dataset,
        'penultimate_dim': penultimate_dim,
        'num_teleportations': args.num_teleportations,
        'num_samples_per_split': args.num_samples,
        'seed': args.seed,
        'metric': 'l2_norm_divided_by_sqrt_dim',
        'per_teleportation': per_teleportation,
        'aggregate': aggregate,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{arch_name}_{dataset}_teleportation.json'
    atomic_json_dump(output_file, results)
    print(f"\nResults saved to {output_file}", flush=True)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pillar 1: Penultimate activation instability under teleportation."
    )
    parser.add_argument('--architecture', required=True,
                        choices=sorted(ARCHITECTURES.keys()))
    parser.add_argument('--dataset', required=True,
                        choices=sorted(NUM_CLASSES.keys()))
    weights_group = parser.add_mutually_exclusive_group(required=True)
    weights_group.add_argument('--weights_path', type=str,
                               help="Path to saved state_dict.")
    weights_group.add_argument('--pretrained', action='store_true',
                               help="Use torchvision pretrained weights.")
    parser.add_argument('--num_teleportations', type=int, default=100)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results/teleportation')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--smoke', action='store_true',
                        help='Smoke-test mode: 3 teleportations, 10 samples/split. '
                             'Verifies CKA + SD aggregation paths execute (need >=3 '
                             'teleports for SD to be non-trivial).')
    args = parser.parse_args()
    if args.smoke:
        args.num_teleportations = 3
        args.num_samples = 10
    return args


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)
