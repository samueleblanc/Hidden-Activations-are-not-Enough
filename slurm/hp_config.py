"""Shared configuration for the hyperparameter tuning pipeline.

Contains architecture/dataset definitions, MIG tier specifications,
data augmentation transforms, and Optuna search space definitions.
"""

from torchvision import transforms

# ---------------------------------------------------------------------------
# Architectures & Datasets
# ---------------------------------------------------------------------------

ARCHITECTURES = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "vgg11_bn",
    "vgg13_bn",
    "vgg16_bn",
    "vgg19_bn",
]

DATASETS = ["cifar10", "cifar100", "tiny_imagenet"]

DATASET_NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "tiny_imagenet": 200,
}

# Cartesian product of architectures x datasets
ALL_CONFIGS = [f"{arch}_{ds}" for arch in ARCHITECTURES for ds in DATASETS]

# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------

EXIT_CUDA_OOM = 42

# ---------------------------------------------------------------------------
# Config name parsing
# ---------------------------------------------------------------------------


def arch_dataset_from_config(config_name):
    """Parse a config name into (architecture, dataset).

    Handles compound names by matching the longest dataset suffix first.
    Examples:
        "resnet18_cifar10"       -> ("resnet18", "cifar10")
        "vgg16_bn_cifar100"      -> ("vgg16_bn", "cifar100")
        "resnet152_tiny_imagenet" -> ("resnet152", "tiny_imagenet")
    """
    # Sort datasets by length descending so we match longest suffix first
    for ds in sorted(DATASETS, key=len, reverse=True):
        suffix = f"_{ds}"
        if config_name.endswith(suffix):
            arch = config_name[: -len(suffix)]
            return arch, ds
    raise ValueError(f"Cannot parse config name: {config_name}")


# ---------------------------------------------------------------------------
# MIG Tiers
# ---------------------------------------------------------------------------

MIG_TIERS = [
    {
        "name": "H100-1g.10gb",
        "gres": "gpu:h100:1g.10gb:1",
        "gpu_mem_gb": 10,
        "cpus": 2,
        "mem": "15G",
    },
    {
        "name": "H100-2g.20gb",
        "gres": "gpu:h100:2g.20gb:1",
        "gpu_mem_gb": 20,
        "cpus": 4,
        "mem": "31G",
    },
    {
        "name": "H100-3g.40gb",
        "gres": "gpu:h100:3g.40gb:1",
        "gpu_mem_gb": 40,
        "cpus": 8,
        "mem": "62G",
    },
    {
        "name": "H100-80gb",
        "gres": "gpu:h100:1",
        "gpu_mem_gb": 80,
        "cpus": 16,
        "mem": "124G",
    },
]

_TIER_INDEX = {tier["name"]: i for i, tier in enumerate(MIG_TIERS)}


def next_mig_tier(current):
    """Return the name of the next MIG tier, or None if already at max."""
    idx = _TIER_INDEX.get(current)
    if idx is None or idx >= len(MIG_TIERS) - 1:
        return None
    return MIG_TIERS[idx + 1]["name"]


def mig_resources(tier_name):
    """Return the tier dict for the given tier name, or None if unknown."""
    idx = _TIER_INDEX.get(tier_name)
    if idx is None:
        return None
    return MIG_TIERS[idx]


# ---------------------------------------------------------------------------
# Data Augmentation Transforms
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(dataset):
    """Return training data augmentation transform for the given dataset.

    All datasets resize to 256 then crop to 224 (ImageNet-compatible).
    CIFAR-100 adds RandomErasing(p=0.1).
    Tiny ImageNet uses wider crop scale and RandomErasing(p=0.15).
    """
    if dataset == "cifar10":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
    elif dataset == "cifar100":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            transforms.RandomErasing(p=0.1),
        ])
    elif dataset == "tiny_imagenet":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            transforms.RandomErasing(p=0.15),
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_test_transform(dataset):
    """Return test/evaluation transform for the given dataset.

    Same for all datasets: Resize(256) + CenterCrop(224) + ToTensor + Normalize.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Optuna Search Space
# ---------------------------------------------------------------------------


def sample_hyperparameters(trial):
    """Sample hyperparameters from the Optuna search space.

    Returns a dict with Phase 1 (feature-extraction) and Phase 2
    (fine-tuning) hyperparameters.
    """
    params = {}

    # Phase 1: feature extraction (frozen backbone)
    params["p1_optimizer"] = trial.suggest_categorical(
        "p1_optimizer", ["adam", "sgd"]
    )
    params["p1_lr"] = trial.suggest_float("p1_lr", 1e-4, 1e-2, log=True)
    params["p1_epochs"] = trial.suggest_int("p1_epochs", 5, 20)
    params["p1_weight_decay"] = trial.suggest_float(
        "p1_weight_decay", 1e-6, 1e-3, log=True
    )
    if params["p1_optimizer"] == "sgd":
        params["p1_momentum"] = trial.suggest_float("p1_momentum", 0.8, 0.99)

    # Phase 2: fine-tuning (unfrozen backbone)
    params["p2_optimizer"] = trial.suggest_categorical(
        "p2_optimizer", ["sgd", "adam", "adamw"]
    )
    params["p2_lr"] = trial.suggest_float("p2_lr", 1e-6, 1e-3, log=True)
    params["p2_weight_decay"] = trial.suggest_float(
        "p2_weight_decay", 1e-6, 1e-2, log=True
    )
    params["p2_scheduler"] = trial.suggest_categorical(
        "p2_scheduler",
        ["cosine", "cosine_warm_restarts", "one_cycle", "step"],
    )
    params["p2_max_epochs"] = trial.suggest_int("p2_max_epochs", 20, 80)
    params["p2_patience"] = trial.suggest_int("p2_patience", 5, 15)
    if params["p2_optimizer"] == "sgd":
        params["p2_momentum"] = trial.suggest_float("p2_momentum", 0.8, 0.99)

    return params
