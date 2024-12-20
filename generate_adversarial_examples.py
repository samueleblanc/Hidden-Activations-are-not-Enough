"""
    Creates adversarial examples
"""
import torch
import torchattacks
import argparse
from multiprocessing import Pool
from pathlib import Path

from utils.utils import get_model, get_dataset, subset
from constants.constants import DEFAULT_EXPERIMENTS, ATTACKS


def parse_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        "--default_index",
        type=int,
        default=0,
        help="Index of default trained network.",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=-1,
        help="Size of subset of test data from where to generate adversarial examples. As default -1 takes 10k test samples",
    )
    parser.add_argument(
        "--nb_workers",
        type=int,
        default=8,
        help="How many processes in parallel for adversarial examples computations and their matrices.",
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        help="Temporary directory for reading data when using clusters.")
    return parser.parse_args()


def apply_attack(attack_name, data, labels, weights_path, architecture_index, path_adv_examples, residual, input_shape, dropout):
    attack_save_path = path_adv_examples / f'{attack_name}/adversarial_examples.pth'

    if attack_save_path.exists():
        print(f"Loading attack {attack_name}")
        misclassified_images = torch.load(attack_save_path)
        return attack_name, misclassified_images

    print(f"Attacking with {attack_name}", flush=True)
    model = get_model(weights_path, architecture_index, residual, input_shape, dropout)

    attacks_classes = dict(zip(["test"] + ATTACKS,
                               [torchattacks.VANILA(model),
                                torchattacks.GN(model),
                                torchattacks.FGSM(model),
                                torchattacks.RFGSM(model),
                                torchattacks.PGD(model),
                                torchattacks.EOTPGD(model),
                                torchattacks.FFGSM(model),
                                # torchattacks.TPGD(model),
                                torchattacks.MIFGSM(model),
                                torchattacks.UPGD(model),
                                torchattacks.DIFGSM(model),
                                torchattacks.Jitter(model),
                                # torchattacks.NIFGSM(model),
                                torchattacks.PGDRS(model),
                                # torchattacks.VMIFGSM(model),
                                # torchattacks.VNIFGSM(model),
                                torchattacks.CW(model),
                                # torchattacks.PGDL2(model),
                                # torchattacks.PGDRSL2(model),
                                torchattacks.DeepFool(model),
                                torchattacks.SparseFool(model),
                                torchattacks.OnePixel(model),
                                torchattacks.Pixle(model),
                                torchattacks.APGD(model),
                                torchattacks.APGDT(model),
                                torchattacks.FAB(model),
                                torchattacks.Square(model),
                                torchattacks.SPSA(model),
                                torchattacks.JSMA(model),
                                torchattacks.EADL1(model),
                                torchattacks.EADEN(model)
                                ]))
    try:
        attacked_data = attacks_classes[attack_name](data, labels)
    except:
        return None, None

    attack_save_path.parent.mkdir(parents=True, exist_ok=True)

    if attack_name == "test":
        torch.save(attacked_data, attack_save_path)
        torch.save(labels, path_adv_examples / f'{attack_name}/labels.pth')
        return attack_name, attacked_data

    attacked_predictions = torch.argmax(model(attacked_data), dim=1)
    misclassified = (labels != attacked_predictions).sum().item()
    total = data.size(0)

    print(f"Attack: {attack_name}. Misclassified after attack: {misclassified} out of {total}.", flush=True)

    # Filter only the attacked images where labels != attacked_predictions
    misclassified_indexes = labels != attacked_predictions
    misclassified_images = attacked_data[misclassified_indexes]

    torch.save(misclassified_images, attack_save_path)

    return attack_name, misclassified_images


def generate_adversarial_examples(exp_dataset_test: torch.Tensor,
                                  exp_labels_test: torch.Tensor,
                                  weights_path,
                                  architecture_index,
                                  default_index,
                                  nb_workers,
                                  residual,
                                  input_shape,
                                  dropout
                                  ):

    experiment_dir = Path(f'experiments/{default_index}/adversarial_examples')
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print("Generating adversarial examples...", flush=True)

    exp_dataset_test = exp_dataset_test.detach().clone()
    exp_labels_test = exp_labels_test.detach().clone()

    arguments = [(attack_name,
                  exp_dataset_test,
                  exp_labels_test,
                  weights_path,
                  architecture_index,
                  experiment_dir,
                  residual,
                  input_shape,
                  dropout)
                 for attack_name in ["test"] + ATTACKS]

    with Pool(processes=nb_workers) as pool:
        pool.starmap(apply_attack, arguments)


def main():
    args = parse_args()
    if args.default_index is not None:
        try:
            experiment = DEFAULT_EXPERIMENTS[f'experiment_{args.default_index}']

            architecture_index = experiment['architecture_index']
            residual = experiment['residual']
            dropout = experiment['dropout']
            dataset = experiment['dataset']
            epoch = experiment['epoch'] - 1

        except KeyError:
            print(f"Error: Default index {args.default_index} does not exist.")
            print(f"When computing adversarial examples of new model, add the experiment to constants.constants.py inside DEFAULT_EXPERIMENTS"
                  f"and provide the corresponding --default_index when running this script.")
            return -1

    else:
        raise ValueError("Default index not specified in constants/constants.py")

    print("Experiment: ", args.default_index)

    if args.temp_dir is not None:
        weights_path = Path(f'{args.temp_dir}/experiments/{args.default_index}/weights/epoch_{epoch}.pth')
    else:
        weights_path = Path(f'experiments/{args.default_index}/weights/epoch_{epoch}.pth')

    if not weights_path.exists():
        raise ValueError(f"Experiment needs to be trained")

    input_shape = (3, 32, 32) if dataset == 'cifar10' else (1, 28, 28)
    _, test_set = get_dataset(dataset, data_loader=False, data_path=args.temp_dir)
    test_size = len(test_set) if args.test_size == -1 else args.test_size
    exp_dataset_test, exp_labels_test = subset(test_set, test_size, input_shape=input_shape)

    generate_adversarial_examples(exp_dataset_test,
                                  exp_labels_test,
                                  weights_path,
                                  architecture_index,
                                  default_index=args.default_index,
                                  nb_workers=args.nb_workers,
                                  residual=residual,
                                  input_shape=input_shape,
                                  dropout=dropout)


if __name__ == "__main__":
    main()
