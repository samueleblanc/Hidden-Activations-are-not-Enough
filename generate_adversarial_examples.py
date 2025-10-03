import torch
import torchattacks
from torch.utils.data import TensorDataset, DataLoader
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Union

from utils.utils import get_model, get_dataset, subset, get_num_classes, get_input_shape, get_device
from constants.constants import DEFAULT_EXPERIMENTS, ATTACKS


def parse_args(
        parser:Union[ArgumentParser, None] = None
    ) -> Namespace:
    """
        Args:
            parser: the parser to use.
        Returns:
            The parsed arguments.
    """
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument(
        "--experiment_name",
        type = str,
        default = 'alexnet_cifar10',
        help = "Name of experiment <<network>>_<<dataset>>"
    )
    parser.add_argument(
        "--test_size",
        type = int,
        default = -1,
        help = "Size of subset of test data from where to generate adversarial examples."
              "As default -1 takes 10k test samples"
    )
    parser.add_argument(
        "--temp_dir",
        type = str,
        help = "Temporary directory for reading data when using clusters."
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
        batch_size: int = 8,
    ):
    device = get_device()

    attack_save_path = path_adv_examples / f'{attack_name}/adversarial_examples.pth'
    wrong_pred_save_path = path_adv_examples / f'{attack_name}/wrong_predictions.pth'
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

    # don't move the whole dataset to device (that causes OOM)
    # data = data.to(device)
    # labels = labels.to(device)

    # prepare DataLoader to iterate in small batches
    ds = TensorDataset(data, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    # build attack instance once (some attacks accept keyword args to reduce iterations/steps)
    attacks_classes = dict(
        zip(
            ["test"] + ATTACKS,
            [torchattacks.VANILA(model),
             torchattacks.GN(model),
             torchattacks.FGSM(model),
             torchattacks.PGD(model),
             torchattacks.EOTPGD(model),
             torchattacks.MIFGSM(model),
             torchattacks.VMIFGSM(model),
             torchattacks.CW(model),
             torchattacks.DeepFool(model),
             torchattacks.Pixle(model),
             torchattacks.APGD(model),
             torchattacks.APGDT(model),
             torchattacks.FAB(model),
             torchattacks.Square(model),
             torchattacks.SPSA(model),
             #torchattacks.JSMA(model),
             torchattacks.EADL1(model),
             torchattacks.EADEN(model)
            ]
        )
    )

    attack_instance = attacks_classes.get(attack_name)
    if attack_instance is None:
        print(f"Unknown attack {attack_name}")
        return

    if attack_name == "test":
        # run on entire dataset in batches but save everything
        adv_list = []
        labels_list = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            with torch.no_grad():
                attacked = attack_instance(xb, yb)
            adv_list.append(attacked.cpu())
            labels_list.append(yb.cpu())
            del xb, yb, attacked
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        torch.save(torch.cat(adv_list), attack_save_path)
        torch.save(torch.cat(labels_list), path_adv_examples / f'{attack_name}/labels.pth')
        del adv_list, labels_list
        return

    # For real attacks: store only misclassified adversarial examples to save RAM
    adv_saved = []
    wrong_preds_saved = []
    total = 0
    misclassified = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        try:
            attacked_batch = attack_instance(xb, yb)  # most attacks operate batchwise
        except Exception as e:
            print(f"Error applying attack {attack_name} on a batch: {e}")
            # free and continue to next batch / or break depending on severity
            del xb, yb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        with torch.no_grad():
            preds = torch.argmax(model(attacked_batch), dim=1)

        mis_idx = (yb != preds)
        miscount_batch = mis_idx.sum().item()
        misclassified += miscount_batch
        total += xb.size(0)

        if miscount_batch > 0:
            adv_saved.append(attacked_batch[mis_idx].cpu())
            wrong_preds_saved.append(preds[mis_idx].cpu())

        # free GPU memory from this batch
        del xb, yb, attacked_batch, preds, mis_idx
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Attack: {attack_name}. Misclassified after attack: {misclassified} out of {total}.", flush=True)

    if len(adv_saved) > 0:
        torch.save(torch.cat(adv_saved), attack_save_path)
        torch.save(torch.cat(wrong_preds_saved), wrong_pred_save_path)
    else:
        raise ValueError(f'Non successful attack method: {attack_name}')

    # cleanup
    del adv_saved, wrong_preds_saved, model, attack_instance
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
    ) -> None:
    """
        Args:
            exp_dataset_test: the test set.
            exp_labels_test: the labels of the test set.
            weights_path: the path to the weights.
            architecture_index: the index of the architecture (See constants/constants.py).
            experiment_name: the name of the experiment (See constants/constants.py).
            input_shape: the shape of the input.
    """

    experiment_dir = Path(f'experiments/{experiment_name}/adversarial_examples')
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print("Generating adversarial examples...", flush=True)

    exp_dataset_test = exp_dataset_test.detach().clone()
    exp_labels_test = exp_labels_test.detach().clone()

    for attack_name in ["test"] + ATTACKS:
        apply_attack(attack_name,
                     exp_dataset_test,
                     exp_labels_test,
                     weights_path,
                     architecture_index,
                     experiment_dir,
                     input_shape,
                     num_classes)


def main() -> None:
    """
        Main function to generate adversarial examples.
    """
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

    generate_adversarial_examples(
        exp_dataset_test = exp_dataset_test,
        exp_labels_test = exp_labels_test,
        weights_path = weights_path,
        architecture_index = architecture_index,
        experiment_name = experiment,
        input_shape = input_shape,
        num_classes = num_classes,
    )


if __name__ == "__main__":
    main()
