from utils.utils import get_ellipsoid_data, zero_std, get_model
from constants.constants import DEFAULT_EXPERIMENTS, ATTACKS
from pathlib import Path
import argparse
import torch
import json
import os


def parse_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        "--default_index",
        type=int,
        default=0,
        help="Index of default trained networks.",
    )
    parser.add_argument(
        "--std",
        type=float,
        default=1,
        help="This times the standard deviation gives a margin for rejection level.",
    )
    parser.add_argument(
        "--d1",
        type=float,
        default=0.1,
        help="Determines how small should the standard deviation be per coordinate on matrix statistics.",
    )
    parser.add_argument(
        "--d2",
        type=float,
        default=0.1,
        help="Determines how small should the standard deviation be per coordinate when detecting.",
    )

    return parser.parse_args()


def reject_predicted_attacks(default_index,
                             weights_path,
                             architecture_index,
                             residual,
                             input_shape,
                             dropout,
                             ellipsoids: dict,
                             std: float = 2,
                             d1: float = 0.1,
                             d2: float = 0.1,
                             verbose: bool = True) -> None:

    # Compute mean and std of number of (almost) zero dims
    reject_path = f'experiments/{default_index}/rejection_levels/reject_at_{std}_{d1}.json'
    Path(f'experiments/{default_index}/rejection_levels/').mkdir(parents=True, exist_ok=True)
    model = get_model(weights_path, architecture_index, residual, input_shape, dropout)

    if os.path.exists(reject_path):
        print("Loading rejection level...")
        file = open(reject_path)
        reject_at = json.load(file)[0]
    else:
        raise ValueError(f"File does not exists: {reject_path}")

    if reject_at <= 0:
        raise ValueError(f"Rejection level is {reject_at}")

    print(f"Will reject when 'zero dims' < {reject_at}.")
    adv_succes = {attack: [] for attack in ["test"]+ATTACKS}  # Save adversarial examples that were not detected
    results = []  # (Rejected, Was attacked)
    # For test counts how many were trusted, and for attacks how many where detected
    counts = {key: {'not_rejected_and_attacked': 0,
                    'not_rejected_and_not_attacked': 0,
                    'rejected_and_attacked': 0,
                    'rejected_and_not_attacked': 0} for key in ["test"] + ATTACKS}

    path_adv_matrices = f'experiments/{default_index}/adversarial_matrices/'
    attacked_dataset = torch.load(f'experiments/{default_index}/adversarial_examples/adversarial_examples.pth')
    test_labels = torch.load(f'experiments/{default_index}/adversarial_examples/test/labels.pth')
    test_acc = 0

    for a in ["test"]+ATTACKS:
        not_rejected_and_attacked = 0
        not_rejected_and_not_attacked = 0
        rejected_and_attacked = 0
        rejected_and_not_attacked = 0

        for i in range(len(attacked_dataset[a])):
            current_matrix_path = path_adv_matrices + f"/{a}/{i}/matrix.pth"
            im = attacked_dataset[a][i]
            pred = torch.argmax(model.forward(im))
            mat = torch.load(current_matrix_path)

            b = get_ellipsoid_data(ellipsoids, pred, "std")
            c = zero_std(mat, b, d2).item()

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
            print("Attack method: ", a)
            if a == 'test':
                print(f'Wrongly rejected test data : {rejected_and_not_attacked} out of {len(attacked_dataset[a])}', flush=True)
                print(f'Trusted test data : {not_rejected_and_not_attacked} out of {len(attacked_dataset[a])}', flush=True)
                print("Accuracy on test data that was not rejected: ",
                      test_acc / counts['test']['not_rejected_and_not_attacked'])

            else:
                print(f'Detected adversarial examples : {rejected_and_attacked} out of {len(attacked_dataset[a])}', flush=True)
                print(f'Successful adversarial examples : {not_rejected_and_attacked} out of {len(attacked_dataset[a])}', flush=True)

    counts_file = f'experiments/{default_index}/counts_per_attack/counts_per_attack_{std}_{d1}.json'
    Path(f'experiments/{default_index}/counts_per_attack/').mkdir(parents=True, exist_ok=True)
    with open(counts_file, 'w') as json_file:
        json.dump(counts, json_file, indent=4)

    test_accuracy = f'experiments/{default_index}/counts_per_attack/test_accuracy_{std}_{d1}.json'
    with open(test_accuracy, 'w') as json_file:
        json.dump([test_acc / counts['test']['not_rejected_and_not_attacked']], json_file, indent=4)

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

    counts_tensor = torch.tensor([counts[key]['not_rejected_and_attacked'] for key in ATTACKS], dtype=torch.float)
    num_attacked_samples = torch.tensor([len(attacked_dataset[key]) for key in ATTACKS], dtype=torch.float)
    normalized_counts = counts_tensor / num_attacked_samples
    probabilities = {key: normalized_counts[i].item() for i, key in enumerate(ATTACKS)}

    Path(f'experiments/{default_index}/adversarial_examples/probabilities_adv_success/').mkdir(parents=True, exist_ok=True)
    probs = f'experiments/{default_index}/adversarial_examples/probabilities_adv_success/' \
            f'prob-adv-success-per-attack_{std}_{d1}_{d2}.json'
    with open(probs, 'w') as json_file:
        json.dump(probabilities, json_file, indent=4)

    Path(f'experiments/{default_index}/adversarial_examples/adversarial_success').mkdir(parents=True, exist_ok=True)
    torch.save(adv_succes,
               f'experiments/{default_index}/adversarial_examples/adversarial_success/'
               f'adv_success_{std}_{d1}_{d2}.pth')


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
            return -1

    else:
        raise ValueError("Default index not specified in constants/constants.py")

    print("Experiment: ", args.default_index)

    weights_path = Path(f'experiments/{args.default_index}/weights') / f'epoch_{epoch}.pth'
    if not weights_path.exists():
        raise ValueError(f"Experiment needs to be trained")

    matrices_path = Path(f'experiments/{args.default_index}/matrices/matrix_statistics.json')
    if not matrices_path.exists():
        raise ValueError(f"Matrix statistics have to be computed")

    input_shape = (3, 32, 32) if dataset == 'cifar10' else (1, 28, 28)

    ellipsoids_file = open(f"experiments/{args.default_index}/matrices/matrix_statistics.json")
    ellipsoids = json.load(ellipsoids_file)

    reject_predicted_attacks(args.default_index,
                             weights_path,
                             architecture_index,
                             residual,
                             input_shape,
                             dropout,
                             ellipsoids,
                             args.std,
                             args.d1,
                             args.d2,
                             verbose=True)


if __name__ == '__main__':
    main()
