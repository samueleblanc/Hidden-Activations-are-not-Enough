"""Optuna-aware training trial script with 2-phase transfer learning.

Each Slurm GPU job runs this script once to execute one Optuna trial.
Phase 1: frozen backbone, train head only.
Phase 2: unfrozen backbone, full fine-tuning with pruning + early stopping.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# Add project root to path for direct execution (python slurm/train_trial.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset

try:
    import optuna
    from optuna.exceptions import TrialPruned
except ImportError:
    optuna = None
    TrialPruned = Exception

from slurm.hp_config import (
    EXIT_CUDA_OOM,
    DATASET_NUM_CLASSES,
    arch_dataset_from_config,
    get_train_transform,
    get_test_transform,
    sample_hyperparameters,
)
from utils.atomic_io import atomic_json_dump

# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY = {
    "resnet18": torchvision.models.resnet18,
    "resnet34": torchvision.models.resnet34,
    "resnet50": torchvision.models.resnet50,
    "resnet101": torchvision.models.resnet101,
    "resnet152": torchvision.models.resnet152,
    "vgg11_bn": torchvision.models.vgg11_bn,
    "vgg13_bn": torchvision.models.vgg13_bn,
    "vgg16_bn": torchvision.models.vgg16_bn,
    "vgg19_bn": torchvision.models.vgg19_bn,
}


# ---------------------------------------------------------------------------
# Model loading & freezing
# ---------------------------------------------------------------------------


def load_pretrained_model(arch, num_classes):
    """Load a torchvision model with pretrained weights and replace the head.

    For ResNets: replaces model.fc with a new Linear layer.
    For VGGs: replaces model.classifier[-1] with a new Linear layer.

    Args:
        arch: Architecture name (e.g. 'resnet18', 'vgg11_bn').
        num_classes: Number of output classes for the new head.

    Returns:
        The modified model with pretrained backbone and fresh head.
    """
    if arch not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {arch}")

    model_fn = _MODEL_REGISTRY[arch]
    model = model_fn(weights="DEFAULT")

    if arch.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif arch.startswith("vgg"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture family: {arch}")

    return model


def freeze_backbone(model, arch):
    """Freeze all parameters, then unfreeze the classification head only.

    For ResNets: unfreezes model.fc.
    For VGGs: unfreezes model.classifier[-1].
    """
    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze head
    if arch.startswith("resnet"):
        for param in model.fc.parameters():
            param.requires_grad = True
    elif arch.startswith("vgg"):
        for param in model.classifier[-1].parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unsupported architecture family: {arch}")


def unfreeze_all(model):
    """Set requires_grad=True for all model parameters."""
    for param in model.parameters():
        param.requires_grad = True


# ---------------------------------------------------------------------------
# Optimizer & Scheduler builders
# ---------------------------------------------------------------------------


def build_optimizer(model, optimizer_name, lr, weight_decay, momentum=None):
    """Build an optimizer from trainable parameters only.

    Args:
        model: The model (or nn.Module) whose trainable params to optimize.
        optimizer_name: One of 'adam', 'sgd', 'adamw'.
        lr: Learning rate.
        weight_decay: Weight decay.
        momentum: Momentum (only used for SGD).

    Returns:
        A torch.optim.Optimizer instance.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_name == "adam":
        return torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum if momentum is not None else 0.0,
        )
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def build_scheduler(optimizer, scheduler_name, max_epochs, lr=None):
    """Build a learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule.
        scheduler_name: One of 'cosine', 'cosine_warm_restarts', 'one_cycle', 'step'.
        max_epochs: Total number of epochs for this phase.
        lr: Current learning rate (needed for one_cycle). If None, reads from optimizer.

    Returns:
        A torch.optim.lr_scheduler instance.
    """
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs
        )
    elif scheduler_name == "cosine_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, max_epochs // 3)
        )
    elif scheduler_name == "one_cycle":
        if lr is None:
            lr = optimizer.defaults["lr"]
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr * 10,
            total_steps=max_epochs,
            epochs=max_epochs,
            steps_per_epoch=1,
        )
    elif scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, max_epochs // 3), gamma=0.1
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(dataset_name, data_dir, batch_size=64):
    """Load dataset and split train into train(90%)/val(10%) with seed=42.

    Args:
        dataset_name: One of 'cifar10', 'cifar100', 'tiny_imagenet'.
        data_dir: Root directory for datasets.
        batch_size: Batch size for data loaders.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_transform = get_train_transform(dataset_name)
    test_transform = get_test_transform(dataset_name)

    if dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True, download=False, transform=train_transform
        )
        # Need a separate dataset with test transforms for validation
        val_base_dataset = datasets.CIFAR10(
            root=data_dir, train=True, download=False, transform=test_transform
        )
        test_dataset = datasets.CIFAR10(
            root=data_dir, train=False, download=False, transform=test_transform
        )
    elif dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=data_dir, train=True, download=False, transform=train_transform
        )
        val_base_dataset = datasets.CIFAR100(
            root=data_dir, train=True, download=False, transform=test_transform
        )
        test_dataset = datasets.CIFAR100(
            root=data_dir, train=False, download=False, transform=test_transform
        )
    elif dataset_name == "tiny_imagenet":
        train_dir = os.path.join(data_dir, "tiny-imagenet-200", "train")
        val_dir = os.path.join(data_dir, "tiny-imagenet-200", "val")
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_base_dataset = datasets.ImageFolder(train_dir, transform=test_transform)
        test_dataset = datasets.ImageFolder(val_dir, transform=test_transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Split train into train(90%) / val(10%) with fixed seed
    n_total = len(train_dataset)
    indices = list(range(n_total))
    rng = random.Random(42)
    rng.shuffle(indices)

    n_val = n_total // 10
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_base_dataset, val_indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Training & evaluation loops
# ---------------------------------------------------------------------------


def train_one_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    """Standard training loop for one epoch.

    Args:
        model: The model to train.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: torch device.
        scheduler: Optional LR scheduler (stepped per epoch after this call if provided).

    Returns:
        (avg_loss, accuracy) for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    if scheduler is not None:
        scheduler.step()

    avg_loss = running_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate model on a dataset.

    Args:
        model: The model to evaluate.
        loader: DataLoader for evaluation.
        criterion: Loss function.
        device: torch device.

    Returns:
        (avg_loss, accuracy) on the dataset.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Trial checkpoint saving
# ---------------------------------------------------------------------------


def _save_trial_checkpoint(
    args, trial_number, params, best_val_acc, train_acc, val_acc, test_acc, status
):
    """Write a JSON checkpoint for this trial.

    Args:
        args: Parsed CLI arguments (needs output_dir, config).
        trial_number: Optuna trial number.
        params: Dict of hyperparameters used.
        best_val_acc: Best validation accuracy during training.
        train_acc: Final training accuracy.
        val_acc: Final validation accuracy.
        test_acc: Final test accuracy.
        status: Trial status string ('complete', 'pruned', 'failed').
    """
    checkpoint = {
        "trial_number": trial_number,
        "config": args.config,
        "params": params,
        "best_val_acc": best_val_acc,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "status": status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    checkpoint_dir = Path(args.output_dir) / "trials"
    checkpoint_path = checkpoint_dir / f"trial_{trial_number}.json"
    atomic_json_dump(checkpoint, checkpoint_path)


# ---------------------------------------------------------------------------
# Seed setting
# ---------------------------------------------------------------------------


def _set_global_seed(seed):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Main trial runner
# ---------------------------------------------------------------------------


def run_trial(args):
    """Execute one Optuna trial of 2-phase transfer learning.

    Phase 1: Freeze backbone, train classification head with Phase 1 HPs.
    Phase 2: Unfreeze all, fine-tune with Phase 2 HPs, pruning + early stopping.

    Args:
        args: Parsed CLI arguments with config, study_db, output_dir, data_dir, seed.
    """
    if optuna is None:
        print("ERROR: optuna is not installed")
        sys.exit(1)

    _set_global_seed(args.seed)

    arch, dataset_name = arch_dataset_from_config(args.config)
    num_classes = DATASET_NUM_CLASSES[dataset_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Connect to Optuna study
    storage = f"sqlite:///{args.study_db}"
    study = optuna.load_study(study_name=args.config, storage=storage)
    trial = study.ask()
    trial_number = trial.number

    print(f"Trial {trial_number} | config={args.config} | arch={arch} | dataset={dataset_name}")

    params = None
    best_val_acc = 0.0

    try:
        # Sample hyperparameters
        params = sample_hyperparameters(trial)
        print(f"Trial {trial_number} | params={json.dumps(params, indent=2)}")

        # Load model and data
        model = load_pretrained_model(arch, num_classes)
        model = model.to(device)
        train_loader, val_loader, test_loader = load_dataset(
            dataset_name, args.data_dir
        )
        criterion = nn.CrossEntropyLoss()

        # ---------------------------------------------------------------
        # Phase 1: frozen backbone, train head only
        # ---------------------------------------------------------------
        freeze_backbone(model, arch)
        p1_optimizer = build_optimizer(
            model,
            params["p1_optimizer"],
            lr=params["p1_lr"],
            weight_decay=params["p1_weight_decay"],
            momentum=params.get("p1_momentum"),
        )

        print(f"Trial {trial_number} | Phase 1: {params['p1_epochs']} epochs")
        for epoch in range(params["p1_epochs"]):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, p1_optimizer, device
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            best_val_acc = max(best_val_acc, val_acc)
            print(
                f"  P1 epoch {epoch+1}/{params['p1_epochs']} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
            # Report val_acc for logging (no pruning in Phase 1)
            trial.report(val_acc, epoch)

        # ---------------------------------------------------------------
        # Phase 2: unfreeze all, fine-tune
        # ---------------------------------------------------------------
        unfreeze_all(model)
        p2_optimizer = build_optimizer(
            model,
            params["p2_optimizer"],
            lr=params["p2_lr"],
            weight_decay=params["p2_weight_decay"],
            momentum=params.get("p2_momentum"),
        )
        p2_scheduler = build_scheduler(
            p2_optimizer,
            params["p2_scheduler"],
            max_epochs=params["p2_max_epochs"],
            lr=params["p2_lr"],
        )

        patience_counter = 0
        p1_total_epochs = params["p1_epochs"]

        print(f"Trial {trial_number} | Phase 2: up to {params['p2_max_epochs']} epochs")
        for epoch in range(params["p2_max_epochs"]):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, p2_optimizer, device, scheduler=p2_scheduler
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            global_epoch = p1_total_epochs + epoch
            trial.report(val_acc, global_epoch)

            print(
                f"  P2 epoch {epoch+1}/{params['p2_max_epochs']} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model weights
                best_weights_path = (
                    Path(args.output_dir) / "weights" / f"trial_{trial_number}_best.pth"
                )
                best_weights_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), best_weights_path)
            else:
                patience_counter += 1

            # Check pruning (prune if below median)
            if trial.should_prune():
                print(f"Trial {trial_number} | Pruned at P2 epoch {epoch+1}")
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                _save_trial_checkpoint(
                    args, trial_number, params, best_val_acc,
                    train_acc, val_acc, 0.0, "pruned"
                )
                sys.exit(0)

            # Early stopping
            if patience_counter >= params["p2_patience"]:
                print(
                    f"Trial {trial_number} | Early stopping at P2 epoch {epoch+1} "
                    f"(patience={params['p2_patience']})"
                )
                break

        # ---------------------------------------------------------------
        # Final evaluation on train/val/test
        # ---------------------------------------------------------------
        # Load best weights if they were saved
        best_weights_path = (
            Path(args.output_dir) / "weights" / f"trial_{trial_number}_best.pth"
        )
        if best_weights_path.exists():
            model.load_state_dict(torch.load(best_weights_path, map_location=device))

        _, final_train_acc = evaluate(model, train_loader, criterion, device)
        _, final_val_acc = evaluate(model, val_loader, criterion, device)
        _, final_test_acc = evaluate(model, test_loader, criterion, device)

        print(
            f"Trial {trial_number} | Final: "
            f"train_acc={final_train_acc:.4f} val_acc={final_val_acc:.4f} "
            f"test_acc={final_test_acc:.4f}"
        )

        # Tell Optuna the result
        study.tell(trial, best_val_acc)

        _save_trial_checkpoint(
            args, trial_number, params, best_val_acc,
            final_train_acc, final_val_acc, final_test_acc, "complete"
        )

    except torch.cuda.OutOfMemoryError:
        print(f"Trial {trial_number} | CUDA OOM — exiting with code {EXIT_CUDA_OOM}")
        try:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
        except Exception:
            pass
        sys.exit(EXIT_CUDA_OOM)

    except TrialPruned:
        print(f"Trial {trial_number} | Pruned (via exception)")
        if params is not None:
            _save_trial_checkpoint(
                args, trial_number, params, best_val_acc,
                0.0, 0.0, 0.0, "pruned"
            )
        sys.exit(0)

    except Exception as e:
        print(f"Trial {trial_number} | ERROR: {e}")
        try:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
        except Exception:
            pass
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run one Optuna trial of 2-phase transfer learning."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config name, e.g. resnet18_cifar10",
    )
    parser.add_argument(
        "--study-db",
        type=str,
        required=True,
        help="Path to SQLite database for Optuna study",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for weights and trial checkpoints",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory containing datasets",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Entry point for the training trial script."""
    args = _parse_args(argv)
    run_trial(args)


if __name__ == "__main__":
    main()
