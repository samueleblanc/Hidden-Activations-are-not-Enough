import torch
import torchattacks
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
        default = 0,
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
    ):
    """
        Applies an attack to the data and saves the adversarial examples.

        Args:
            attack_name: the name of the attack.
            data: the data to attack.
            labels: the labels of the data.
            weights_path: the path to the weights.
            architecture_index: the index of the architecture (See constants/constants.py).
            path_adv_examples: the path to save the adversarial examples.
            input_shape: the shape of the input.
            num_classes: the number of classes.
        Returns:
            The name of the attack and the adversarial examples (that are misclassified).
    """
    device = get_device()
    print(f"Using device: {device}", flush=True)

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

    data = data.to(device)
    labels = labels.to(device)

    attacks_classes = dict(
        zip(
            ["test"] + ATTACKS,
            [torchattacks.VANILA(model),
            torchattacks.GN(model),
            torchattacks.FGSM(model),
            # torchattacks.RFGSM(model),
            torchattacks.PGD(model),
            torchattacks.EOTPGD(model),
            # torchattacks.FFGSM(model),
            # torchattacks.TPGD(model),
            torchattacks.MIFGSM(model),
            # torchattacks.UPGD(model),
            # torchattacks.DIFGSM(model),
            # torchattacks.Jitter(model),
            # torchattacks.NIFGSM(model),
            # torchattacks.PGDRS(model),
            torchattacks.VMIFGSM(model),
            # torchattacks.VNIFGSM(model),
            torchattacks.CW(model),
            # torchattacks.PGDL2(model),
            # torchattacks.PGDRSL2(model),
            torchattacks.DeepFool(model),
            # torchattacks.SparseFool(model),
            # torchattacks.OnePixel(model),
            torchattacks.Pixle(model),
            torchattacks.APGD(model),
            torchattacks.APGDT(model),
            torchattacks.FAB(model),
            torchattacks.Square(model),
            torchattacks.SPSA(model),
            torchattacks.JSMA(model),
            torchattacks.EADL1(model),
            torchattacks.EADEN(model)
            ]
        )
    )
    try:
        attacked_data = attacks_classes[attack_name](data, labels)
    except Exception as e:
        print(f"Error applying attack {attack_name}: {e}")
        return

    if attack_name == "test":
        torch.save(attacked_data.cpu(), attack_save_path)
        torch.save(labels.cpu(), path_adv_examples / f'{attack_name}/labels.pth')
        return

    with torch.no_grad():
        predictions = torch.argmax(model(attacked_data), dim=1)

    misclassified = (labels != predictions).sum().item()
    total = data.size(0)

    print(f"Attack: {attack_name}. Misclassified after attack: {misclassified} out of {total}.", flush=True)

    # Filter only the attacked images where labels != attacked_predictions
    misclassified_indexes = labels != predictions
    misclassified_images = attacked_data[misclassified_indexes]

    torch.save(misclassified_images.cpu(), attack_save_path)
    torch.save(predictions.cpu(), wrong_pred_save_path)


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
    epoch = DEFAULT_EXPERIMENTS[experiment]['epochs']-1

    print("Experiment: ", experiment)

    if args.temp_dir is not None:
        weights_path = Path(f'{args.temp_dir}/experiments/{experiment}/weights/epoch_{epoch}.pth')
    else:
        weights_path = Path(f'experiments/{experiment}/weights/epoch_{epoch}.pth')

    if not weights_path.exists():
        raise ValueError(f"Experiment needs to be trained")

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
