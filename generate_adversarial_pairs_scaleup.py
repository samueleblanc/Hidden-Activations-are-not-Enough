"""Step A2 — Adversarial pair scale-up to N=5000/attack.

Generates 4800 new pairs per (arch, attack) by extending the existing
200-pair Study 2 sets. Output:
experiments/{arch}_imagenet/adversarial_pairs_N5000/{attack}/pairs.pth.

Resumable via the n_done field of the saved tensor dict.

Usage:
    python generate_adversarial_pairs_scaleup.py \\
        --arch resnet152 --attack pgd --target_n 5000 --temp_dir $SLURM_TMPDIR
"""
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch

from utils.utils import get_device, get_imagenet_val_dataset
from utils.atomic_io import atomic_torch_save


def resume_state(path):
    """Read a saved partial pairs.pth; return None if missing."""
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu")


def get_attack(name, model, kwargs):
    """Map attack name to a torchattacks instance with the agreed hyperparameters."""
    import torchattacks
    cls_map = {
        "fgsm":     ("FGSM",     {"eps": 8/255}),
        "pgd":      ("PGD",      {"eps": 8/255, "alpha": 2/255, "steps": 7}),
        "cw":       ("CW",       {"c": 1.0, "kappa": 0, "steps": 100, "lr": 0.01}),
        "deepfool": ("DeepFool", {"steps": 100, "overshoot": 0.02}),
        "apgd":     ("APGD",     {"norm": "Linf", "eps": 8/255, "steps": 50, "loss": "dlr"}),
        "square":   ("Square",   {"eps": 8/255, "n_queries": 20000}),
    }
    cls_name, default_kwargs = cls_map[name]
    cls = getattr(torchattacks, cls_name, None)
    if cls is None:
        raise RuntimeError(f"torchattacks lacks {cls_name} (version mismatch?)")
    final_kwargs = {**default_kwargs, **(kwargs or {})}
    attack = cls(model, **final_kwargs)
    # Tell the attack about ImageNet normalization so it operates in [0,1]
    # space and returns adversarial examples in the same normalized space
    # as the inputs (matches Study 2 / validate_theorem45.py convention).
    attack.set_normalization_used(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    # Stash kwargs on the instance for later persistence in pairs.pth.
    attack.kwargs = final_kwargs
    return attack


def main():
    parser = ArgumentParser()
    parser.add_argument("--arch", required=True, choices=["resnet152", "densenet121", "googlenet"])
    parser.add_argument("--attack", required=True, choices=["fgsm", "pgd", "cw", "deepfool", "apgd", "square"])
    parser.add_argument("--target_n", type=int, default=5000)
    parser.add_argument("--temp_dir", default=None)
    parser.add_argument("--data_dir", default="/datashare/imagenet/ILSVRC2012")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    args = parser.parse_args()

    device = get_device()

    # Load arch
    import torchvision.models as tvm
    arch_loader = {
        "resnet152": tvm.resnet152,
        "densenet121": tvm.densenet121,
        "googlenet": tvm.googlenet,
    }[args.arch]
    model = arch_loader(weights="DEFAULT").to(device).eval()

    # Output path
    out_dir = Path("experiments") / f"{args.arch}_imagenet" / "adversarial_pairs_N5000" / args.attack
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pairs.pth"

    # Resume?
    state = resume_state(str(out_path))
    if state is not None:
        n_done = state["n_done"]
        x_clean_all = state["x_clean"]
        x_adv_all   = state["x_adv"]
        y_clean_all = state["y_clean"]
        y_adv_all   = state["y_adv"]
        if n_done >= args.target_n:
            print(f"Already done: n_done={n_done} >= target_n={args.target_n}")
            return
        print(f"Resuming from n_done={n_done}", flush=True)
    else:
        n_done = 0
        x_clean_all = torch.empty(0, 3, 224, 224)
        x_adv_all   = torch.empty(0, 3, 224, 224)
        y_clean_all = torch.empty(0, dtype=torch.long)
        y_adv_all   = torch.empty(0, dtype=torch.long)

    # Build dataset — use the deterministically-ordered ImageNet val set
    # (NOT get_dataset(), which does a random_split). We need stable indexing
    # so that resume picks up where we left off.
    data_root = args.temp_dir if args.temp_dir else args.data_dir
    # If temp_dir was used, expect /data/ILSVRC2012 inside it (matches the
    # job script which copies val/ to $SLURM_TMPDIR/data/ILSVRC2012/val).
    if args.temp_dir is not None:
        candidate = Path(args.temp_dir) / "data" / "ILSVRC2012"
        if candidate.is_dir():
            data_root = str(candidate)
        else:
            data_root = args.data_dir
    _, val_set = get_imagenet_val_dataset(data_root)

    attack = get_attack(args.attack, model, {})

    needed = args.target_n - n_done
    cursor = n_done   # ImageNet val index

    while needed > 0:
        # Batch up to checkpoint_every samples
        batch_size = min(args.batch_size, needed)
        batch_x = []
        batch_y = []
        for i in range(batch_size):
            x_i, y_i = val_set[cursor + i]
            batch_x.append(x_i.unsqueeze(0))
            batch_y.append(y_i)
        x = torch.cat(batch_x, dim=0).to(device)
        y = torch.tensor(batch_y).to(device)

        x_adv = attack(x, y)

        # Predicted class on adversarial
        with torch.no_grad():
            y_adv_pred = model(x_adv).argmax(1)

        x_clean_all = torch.cat([x_clean_all, x.cpu()], dim=0)
        x_adv_all   = torch.cat([x_adv_all, x_adv.cpu()], dim=0)
        y_clean_all = torch.cat([y_clean_all, y.cpu()], dim=0)
        y_adv_all   = torch.cat([y_adv_all, y_adv_pred.cpu()], dim=0)
        n_done += batch_size
        cursor += batch_size
        needed -= batch_size

        # Atomic checkpoint
        if n_done % args.checkpoint_every < args.batch_size or needed <= 0:
            atomic_torch_save(str(out_path), {
                "x_clean": x_clean_all,
                "x_adv":   x_adv_all,
                "y_clean": y_clean_all,
                "y_adv":   y_adv_all,
                "n_done":  n_done,
                "attack":  args.attack,
                "arch":    args.arch,
                "attack_kwargs": getattr(attack, "kwargs", {}),
            })
            print(f"  n_done={n_done}/{args.target_n}", flush=True)

    print(f"DONE: {out_path}")


if __name__ == "__main__":
    main()
