from utils.unified_memory import init_unified_memory
init_unified_memory()

import sys
import torch
import torchattacks
from torch.utils.data import TensorDataset, DataLoader
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Union

from utils.utils import get_model, get_dataset, subset, get_num_classes, get_input_shape, get_device
from constants.constants import DEFAULT_EXPERIMENTS, ATTACKS
from utils.atomic_io import atomic_torch_save


def parse_args(
        parser:Union[ArgumentParser, None] = None
    ) -> Namespace:
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument(
        "--experiment_name", "--experiment",
        type = str,
        default = 'alexnet_cifar10',
        dest = "experiment_name",
        help = "Name of experiment <<network>>_<<dataset>>"
    )
    parser.add_argument(
        "--test_size",
        type = int,
        default = -1,
        help = "Size of subset of test data from where to generate adversarial examples. "
              "As default -1 takes 10k test samples"
    )
    parser.add_argument(
        "--temp_dir",
        type = str,
        help = "Temporary directory for reading data when using clusters."
    )
    parser.add_argument(
        "--attacks",
        nargs = "+",
        default = None,
        help = "Subset of attacks to run (e.g. --attacks FGSM PGD CW). "
               "If not specified, all attacks from ATTACKS are run."
    )
    parser.add_argument(
        "--no_auto_test",
        action = "store_true",
        default = False,
        help = "Skip automatic prepending of 'test' (VANILA) attack. "
               "Use when running individual attacks in parallel Slurm jobs."
    )
    parser.add_argument(
        "--num_adv_examples",
        type = int,
        default = 500,
        help = "Number of successful adversarial examples to collect per attack. "
               "For 'test', collects this many correctly-classified clean samples."
    )
    parser.add_argument(
        "--batch_size",
        type = int,
        default = 512,
        help = "Batch size for processing test samples through attacks."
    )
    return parser.parse_args()


def apply_attack(
        attack_name: str,
        data: torch.Tensor,
        labels: torch.Tensor,
        weights_path: Path,
        architecture_index: int,
        path_adv_examples: Path,
        input_shape,
        num_classes: int,
        num_adv_examples: int = 500,
        batch_size: int = 512,
    ):
    device = get_device()

    attack_save_path = path_adv_examples / f'{attack_name}/adversarial_examples.pth'
    attack_save_path.parent.mkdir(parents=True, exist_ok=True)

    if attack_save_path.exists():
        print(f"Attack {attack_name} exists.")
        return

    print(f"Attacking with {attack_name}", flush=True)
    model = get_model(
        path = weights_path,
        architecture_index = architecture_index,
        input_shape = input_shape,
        num_classes = num_classes,
        device = device
    )
    model.eval()

    # Build attack instance lazily (handles missing attacks in different torchattacks versions)
    attack_map = {
        "test": "VANILA", "GN": "GN", "FGSM": "FGSM", "PGD": "PGD",
        "EOTPGD": "EOTPGD", "MIFGSM": "MIFGSM", "VMIFGSM": "VMIFGSM",
        "CW": "CW", "DeepFool": "DeepFool", "Pixle": "Pixle",
        "APGD": "APGD", "APGDT": "APGDT", "FAB": "FAB", "Square": "Square",
        "SPSA": "SPSA", "EADL1": "EADL1", "EADEN": "EADEN",
    }
    attack_cls_name = attack_map.get(attack_name)
    if attack_cls_name is None:
        print(f"Unknown attack {attack_name}")
        return
    attack_cls = getattr(torchattacks, attack_cls_name, None)
    if attack_cls is None:
        print(f"WARNING: Attack {attack_name} ({attack_cls_name}) not available in torchattacks {torchattacks.__version__}. Skipping.", flush=True)
        return
    attack_instance = attack_cls(model)

    ds = TensorDataset(data, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    needed = num_adv_examples
    total_processed = 0

    if attack_name == "test":
        # Collect correctly-classified clean samples
        collected = []
        collected_labels = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            with torch.no_grad():
                preds = torch.argmax(model(xb), dim=1)
            correct_mask = (preds == yb)
            total_processed += xb.size(0)
            if correct_mask.any():
                collected.append(xb[correct_mask].cpu())
                collected_labels.append(yb[correct_mask].cpu())
            if sum(t.size(0) for t in collected) >= needed:
                break

        if len(collected) == 0:
            print(f"  WARNING: test produced 0 correctly-classified samples.", flush=True)
            return

        result = torch.cat(collected)[:needed]
        result_labels = torch.cat(collected_labels)[:needed]
        atomic_torch_save(result, attack_save_path)
        atomic_torch_save(result_labels, path_adv_examples / f'{attack_name}/labels.pth')
        print(f"Attack: test. Collected {result.size(0)}/{needed} correctly-classified samples from {total_processed} processed.", flush=True)
        del collected, collected_labels, result, result_labels
        return

    # Real attacks: collect samples that are correctly classified originally
    # but misclassified after the attack
    collected_adv = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        total_processed += xb.size(0)

        # Step 1: filter to correctly-classified originals
        with torch.no_grad():
            orig_preds = torch.argmax(model(xb), dim=1)
        correct_mask = (orig_preds == yb)
        if not correct_mask.any():
            continue
        xb_correct = xb[correct_mask]
        yb_correct = yb[correct_mask]

        # Step 2: run attack on correctly-classified subset
        try:
            attacked = attack_instance(xb_correct, yb_correct)
        except Exception as e:
            print(f"Error applying attack {attack_name} on batch: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        # Step 3: keep only samples where the attack succeeded (misclassified)
        with torch.no_grad():
            adv_preds = torch.argmax(model(attacked), dim=1)
        success_mask = (adv_preds != yb_correct)
        if success_mask.any():
            collected_adv.append(attacked[success_mask].cpu())

        del xb, yb, xb_correct, yb_correct, attacked, adv_preds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if sum(t.size(0) for t in collected_adv) >= needed:
            break

    if len(collected_adv) > 0:
        result = torch.cat(collected_adv)[:needed]
        print(f"Attack: {attack_name}. Collected {result.size(0)}/{needed} adversarial examples from {total_processed} processed.", flush=True)
        atomic_torch_save(result, attack_save_path)
        del result
    else:
        print(f"  WARNING: {attack_name} produced 0 successful adversarial examples after processing {total_processed} samples. Skipping save.", flush=True)
        save_dir = path_adv_examples / f'{attack_name}'
        save_dir.mkdir(parents=True, exist_ok=True)
        marker_path = save_dir / 'zero_misclassifications.txt'
        with open(marker_path, 'w') as f:
            f.write(f"Attack {attack_name} produced 0 successful adversarial examples from {total_processed} samples\n")

    # cleanup
    del collected_adv, model, attack_instance
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_adversarial_examples(
        exp_dataset_test: torch.Tensor,
        exp_labels_test: torch.Tensor,
        weights_path: Path,
        architecture_index: int,
        experiment_name: str,
        input_shape,
        num_classes: int,
        attacks: list = None,
        no_auto_test: bool = False,
        num_adv_examples: int = 500,
        batch_size: int = 512,
    ) -> None:

    experiment_dir = Path(f'experiments/{experiment_name}/adversarial_examples')
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print("Generating adversarial examples...", flush=True)

    exp_dataset_test = exp_dataset_test.detach().clone()
    exp_labels_test = exp_labels_test.detach().clone()

    attack_list = attacks if attacks is not None else ATTACKS
    run_list = attack_list if no_auto_test else ["test"] + attack_list
    failed_attacks = 0
    for attack_name in run_list:
        try:
            apply_attack(attack_name,
                         exp_dataset_test,
                         exp_labels_test,
                         weights_path,
                         architecture_index,
                         experiment_dir,
                         input_shape,
                         num_classes,
                         num_adv_examples=num_adv_examples,
                         batch_size=batch_size)
        except Exception as e:
            print(f'ERROR: Attack {attack_name} failed entirely: {type(e).__name__}: {e}', flush=True)
            failed_attacks += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    total_attacks = len(run_list)
    if failed_attacks == total_attacks:
        print(f"FATAL: All {total_attacks} attacks failed.", flush=True)
        sys.exit(1)
    elif failed_attacks > 0:
        print(f"WARNING: {failed_attacks}/{total_attacks} attacks failed.", flush=True)


def main() -> None:
    args = parse_args()
    if args.experiment_name is None:
        raise ValueError("Default index not specified in constants/constants.py")

    experiment = args.experiment_name
    architecture_index = DEFAULT_EXPERIMENTS[experiment]['architecture_index']
    dataset = DEFAULT_EXPERIMENTS[experiment]['dataset']
    epoch = DEFAULT_EXPERIMENTS[experiment]['epochs']

    print("Experiment: ", experiment)

    if args.temp_dir is not None:
        weights_path = Path(f'{args.temp_dir}/experiments/{experiment}/weights/epoch_{epoch}.pth')
    else:
        weights_path = Path(f'experiments/{experiment}/weights/epoch_{epoch}.pth')

    if not weights_path.exists():
        raise ValueError(f"Couldn't find weights at {weights_path}")

    input_shape = get_input_shape(dataset)
    num_classes = get_num_classes(dataset)
    _, test_set = get_dataset(dataset, data_loader=False, data_path=args.temp_dir)
    test_size = len(test_set) if args.test_size == -1 else args.test_size
    exp_dataset_test, exp_labels_test = subset(test_set, test_size, input_shape=input_shape)

    # Quick accuracy check before running attacks
    device = get_device()
    model = get_model(
        path=weights_path,
        architecture_index=architecture_index,
        input_shape=input_shape,
        num_classes=num_classes,
        device=device,
    )
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data, batch_labels in DataLoader(
            TensorDataset(exp_dataset_test, exp_labels_test), batch_size=64, shuffle=False
        ):
            if total >= 500:
                break
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data)
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    accuracy = correct / total if total > 0 else 0
    print(f"  Model accuracy on {total} test samples: {accuracy:.4f}", flush=True)
    if accuracy < 0.1:
        print(f"  WARNING: Model accuracy is very low ({accuracy:.4f}). "
              f"Adversarial examples may not be meaningful.", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    generate_adversarial_examples(
        exp_dataset_test = exp_dataset_test,
        exp_labels_test = exp_labels_test,
        weights_path = weights_path,
        architecture_index = architecture_index,
        experiment_name = experiment,
        input_shape = input_shape,
        num_classes = num_classes,
        attacks = args.attacks,
        no_auto_test = args.no_auto_test,
        num_adv_examples = args.num_adv_examples,
        batch_size = args.batch_size,
    )


if __name__ == "__main__":
    main()
