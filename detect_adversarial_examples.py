import os
import json
import torch
from argparse import ArgumentParser, Namespace
from pathlib import Path

from utils.utils import get_ellipsoid_data, zero_std, get_model
from constants.constants import DEFAULT_EXPERIMENTS, ATTACKS


def parse_args(
        parser:ArgumentParser|None = None
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
        "--default_index",
        type=int,
        default=0,
        help="Index of default trained networks.",
    )
    parser.add_argument(
        "--t_epsilon",
        type=float,
        default=1,
        help="This times the standard deviation gives a margin for rejection level.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Determines how small should the standard deviation be per coordinate on matrix statistics.",
    )
    parser.add_argument(
        "--epsilon_p",
        type=float,
        default=0.1,
        help="Determines how small should the standard deviation be per coordinate when detecting.",
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default=None,
        help="Temporary directory to read data for computations from, such as weights, matrix statistics and adversarial matrices."
             "Useful on clusters but not on local experiments.",
    )

    return parser.parse_args()


def reject_predicted_attacks(
        default_index: int,
        weights_path: str,
        architecture_index: int,
        residual: bool,
        input_shape: tuple[int,int,int],
        num_classes: int,
        dropout: bool,
        ellipsoids: dict,
        t_epsilon:float = 2,
        epsilon:float = 0.1,
        epsilon_p:float = 0.1,
        verbose:bool = True,
        temp_dir:str|None = None
    ) -> None:
    """
        Goes over the dataset and predicts if it is an adversarial example or not.

        Args:
            default_index: experiment index (See constants/constants.py).
            weights_path: the path to the weights.
            architecture_index: the index of the architecture (See constants/constants.py).
            residual: whether the model has residual connections.
            input_shape: the shape of the input.
            num_classes: the number of classes.
            dropout: whether the model has dropout layers.
            ellipsoids: the ellipsoids.
            t_epsilon: t^epsilon from the paper.
            epsilon: the threshold for matrix statistics.
            epsilon_p: the threshold for detection.
            verbose: whether to print the results.
            temp_dir: the temporary directory.
    """
    if temp_dir is not None:
        reject_path = f'{temp_dir}/experiments/{default_index}/rejection_levels/reject_at_{t_epsilon}_{epsilon}.json'
        Path(f'{temp_dir}/experiments/{default_index}/rejection_levels/').mkdir(parents=True, exist_ok=True)
    else:
        reject_path = f'experiments/{default_index}/rejection_levels/reject_at_{t_epsilon}_{epsilon}.json'
        Path(f'experiments/{default_index}/rejection_levels/').mkdir(parents=True, exist_ok=True)

    model = get_model(
        path = weights_path,
        architecture_index = architecture_index,
        residual = residual,
        input_shape = input_shape,
        num_classes = num_classes,
        dropout = dropout
    )

    if os.path.exists(reject_path):
        print("Loading rejection level...", flush=True)
        file = open(reject_path)
        reject_at = json.load(file)[0]
    else:
        print(f"File does not exists: {reject_path}", flush=True)
        return

    if reject_at <= 0:
        print(f"Rejection level too small: {reject_at}", flush=True)
        return

    print(f"Will reject when 'zero dims' < {reject_at}.", flush=True)
    adv_succes = {attack: [] for attack in ["test"]+ATTACKS}  # Save adversarial examples that were not detected
    results = []  # (Rejected, Was attacked)
    # For test counts how many were trusted, and for attacks how many where detected
    counts = {key: {'not_rejected_and_attacked': 0,
                    'not_rejected_and_not_attacked': 0,
                    'rejected_and_attacked': 0,
                    'rejected_and_not_attacked': 0} for key in ["test"] + ATTACKS}

    if temp_dir is not None:
        path_adv_matrices = f'{temp_dir}/experiments/{default_index}/adversarial_matrices/'
        test_labels = torch.load(f'{temp_dir}/experiments/{default_index}/adversarial_examples/test/labels.pth')
    else:
        path_adv_matrices = f'experiments/{default_index}/adversarial_matrices/'
        test_labels = torch.load(f'experiments/{default_index}/adversarial_examples/test/labels.pth')

    test_acc = 0

    for a in ["test"]+ATTACKS:
        print(f"Trying for {a}", flush=True)
        try:
            if temp_dir is not None:
                attacked_dataset = torch.load(f'{temp_dir}/experiments/{default_index}/adversarial_examples/{a}/adversarial_examples.pth')
            else:
                attacked_dataset = torch.load(f'experiments/{default_index}/adversarial_examples/{a}/adversarial_examples.pth')
        except:
            print(f"Attack {a} not found.", flush=True)
            continue
        not_rejected_and_attacked = 0
        not_rejected_and_not_attacked = 0
        rejected_and_attacked = 0
        rejected_and_not_attacked = 0

        for i in range(len(attacked_dataset)):
            current_matrix_path = f"{path_adv_matrices}{a}/{i}/matrix.pth"
            im = attacked_dataset[i]
            pred = torch.argmax(model.forward(im))
            mat = torch.load(current_matrix_path)

            b = get_ellipsoid_data(ellipsoids, pred, "std")
            c = zero_std(mat, b, epsilon_p).item()

            res = ((reject_at > c), (a != "test"))

            # if not rejected and it was an attack
            # so detected adversarial example
            if not res[0] and a != "test":
                not_rejected_and_attacked += 1
                counts[a]['not_rejected_and_attacked'] += 1
                if len(adv_succes[a]) < 10:
                    adv_succes[a].append(im)

            # if rejected and it was an attack
            if res[0] and a != 'test':
                rejected_and_attacked += 1
                counts[a]['rejected_and_attacked'] += 1

            # if rejected and it was test data
            # so wrong rejection of natural data
            if res[0] and a == "test":
                rejected_and_not_attacked += 1
                counts[a]['rejected_and_not_attacked'] += 1

            # if not rejected and it was test data
            if not res[0] and a == "test":
                not_rejected_and_not_attacked += 1
                counts[a]['not_rejected_and_not_attacked'] += 1
                if pred == test_labels[i]:
                    test_acc += 1

            results.append(res)

        if verbose:
            print("Attack method: ", a, flush=True)
            if a == 'test':
                print(f'Wrongly rejected test data : {rejected_and_not_attacked} out of {len(attacked_dataset)}', flush=True)
                print(f'Trusted test data : {not_rejected_and_not_attacked} out of {len(attacked_dataset)}', flush=True)

                if counts['test']['not_rejected_and_not_attacked'] == 0:
                    test_acc = 0
                else:
                    test_acc = test_acc / counts['test']['not_rejected_and_not_attacked']

                print("Accuracy on test data that was not rejected: ",
                      test_acc, flush=True)

            else:
                print(f'Detected adversarial examples : {rejected_and_attacked} out of {len(attacked_dataset)}', flush=True)
                print(f'Successful adversarial examples : {not_rejected_and_attacked} out of {len(attacked_dataset)}', flush=True)

    counts_file = f'experiments/{default_index}/counts_per_attack/counts_per_attack_{t_epsilon}_{epsilon}_{epsilon_p}.json'
    Path(f'experiments/{default_index}/counts_per_attack/').mkdir(parents=True, exist_ok=True)
    with open(counts_file, 'w') as json_file:
        json.dump(counts, json_file, indent=4)

    test_accuracy = f'experiments/{default_index}/counts_per_attack/test_accuracy_{t_epsilon}_{epsilon}_{epsilon_p}.json'
    with open(test_accuracy, 'w') as json_file:
        json.dump([test_acc], json_file, indent=4)

    good_defence = 0
    wrongly_rejected = 0
    num_att = 0
    for rej, att in results:
        if att:
            good_defence += int(rej)
            num_att += 1
        else:
            wrongly_rejected += int(rej)
    print(f"Percentage of good defences: {good_defence/num_att}", flush=True)
    print(f"Percentage of wrong rejections: {wrongly_rejected/(len(results)-num_att)}", flush=True)


def main() -> None:
    """
        Main function to detect adversarial examples.
    """
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
            return -1

    else:
        raise ValueError("Default index not specified in constants/constants.py")

    print("Detecting adversarial examples for Experiment: ", args.default_index, flush=True)

    input_shape = (3, 32, 32) if dataset == 'cifar10' or dataset == 'cifar100' else (1, 28, 28)
    num_classes = 10  # TODO: This should be easily changed

    if args.temp_dir is not None:
        weights_path = Path(f'{args.temp_dir}/experiments/{args.default_index}/weights') / f'epoch_{epoch}.pth'
        matrices_path = Path(f'{args.temp_dir}/experiments/{args.default_index}/matrices/matrix_statistics.json')
        ellipsoids_file = open(f"{args.temp_dir}/experiments/{args.default_index}/matrices/matrix_statistics.json")
    else:
        weights_path = Path(f'experiments/{args.default_index}/weights') / f'epoch_{epoch}.pth'
        matrices_path = Path(f'experiments/{args.default_index}/matrices/matrix_statistics.json')
        ellipsoids_file = open(f"experiments/{args.default_index}/matrices/matrix_statistics.json")

    if not weights_path.exists():
        raise ValueError(f"Experiment needs to be trained")

    if not matrices_path.exists():
        raise ValueError(f"Matrix statistics have to be computed")

    ellipsoids = json.load(ellipsoids_file)

    reject_predicted_attacks(
        default_index = args.default_index,
        weights_path = weights_path,
        architecture_index = architecture_index,
        residual = residual,
        input_shape = input_shape,
        num_classes = num_classes,
        dropout = dropout,
        ellipsoids = ellipsoids,
        t_epsilon = args.t_epsilon,
        epsilon = args.epsilon,
        epsilon_p = args.epsilon_p,
        verbose = True,
        temp_dir = args.temp_dir
    )


if __name__ == '__main__':
    main()
