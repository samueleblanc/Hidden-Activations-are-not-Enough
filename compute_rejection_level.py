import os
import json
import torch
from argparse import Namespace
from pathlib import Path
from typing import Union

from utils.utils import get_ellipsoid_data, zero_std


def process_sample(
        ellipsoids: dict, 
        epsilon: float, 
        default_index: int, 
        i: int, 
        temp_dir:Union[str, None]
    ) -> Union[torch.Tensor, None]:
    """
        Args:
            ellipsoids: the ellipsoids.
            epsilon: the threshold.
            default_index: experiment index (See constants/constants.py).
            i: the index of the sample.
            temp_dir: the temporary directory.
        Returns:
            The rejection level.
    """
    if temp_dir is not None:
        path_experiment_matrix = Path(f'{temp_dir}/experiments/{default_index}/rejection_levels/matrices/{i}/matrix.pth')
        path_prediction = Path(f'{temp_dir}/experiments/{default_index}/rejection_levels/matrices/{i}/prediction.pth')
    else:
        path_experiment_matrix = Path(f'experiments/{default_index}/rejection_levels/matrices/{i}/matrix.pth')
        path_prediction = Path(f'experiments/{default_index}/rejection_levels/matrices/{i}/prediction.pth')

    if os.path.exists(path_experiment_matrix):
        mat = torch.load(path_experiment_matrix)
        pred = torch.load(path_prediction)
        a = get_ellipsoid_data(ellipsoids, pred, "std")
        return zero_std(mat, a, epsilon).expand([1])
    else:
        return None


def compute_rejection_level(
        exp_dataset_train: torch.Tensor,
        default_index: int,
        ellipsoids: dict,
        t_epsilon:float = 2,
        epsilon:float = 0.1,
        temp_dir:Union[str, None] = None
    ) -> None:
    """
        Args:
            exp_dataset_train: the training set.
            default_index: experiment index (See constants/constants.py).
            ellipsoids: the ellipsoids.
            t_epsilon: t^epsilon from the paper.
            epsilon: the threshold.
            temp_dir: the temporary directory.
    """
    # Compute mean and std of number of (almost) zero dims
    reject_path = f'experiments/{default_index}/rejection_levels/reject_at_{t_epsilon}_{epsilon}.json'
    Path(f'experiments/{default_index}/rejection_levels/').mkdir(parents=True, exist_ok=True)
    print("Computing rejection level...", flush=True)

    results = []
    for i in range(len(exp_dataset_train)):
        results.append(process_sample(ellipsoids, epsilon, default_index, i, temp_dir))

    zeros = torch.cat([result for result in results if result is not None]).float()

    reject_at = zeros.mean().item() - t_epsilon*zeros.std().item()

    print(f"Rejection level: {reject_at}", flush=True)

    with open(reject_path, 'w') as json_file:
        json.dump([reject_at], json_file, indent=4)


def main(
        default_index:Union[int, None] = None,
        t_epsilon:Union[float, None] = None, 
        epsilon:Union[float, None] = None, 
        temp_dir:Union[str, None] = None
    ) -> None:
    """
        Main function to compute the rejection level.
        Args:
            t_epsilon: t^epsilon from the paper.
            epsilon: the threshold.
            temp_dir: the temporary directory.
    """
    args = Namespace(
        default_index = default_index, 
        t_epsilon = t_epsilon, 
        epsilon = epsilon, 
        temp_dir = temp_dir
    )
    print("Experiment: ", args.default_index, flush=True)
    if args.temp_dir is not None:
        matrices_path = Path(f'{args.temp_dir}/experiments/{args.default_index}/matrices/matrix_statistics.json')
        exp_dataset_train = torch.load(f'{args.temp_dir}/experiments/{args.default_index}/rejection_levels/exp_dataset_train.pth')
        ellipsoids_file = open(f"{args.temp_dir}/experiments/{args.default_index}/matrices/matrix_statistics.json")
    else:
        matrices_path = Path(f'experiments/{args.default_index}/matrices/matrix_statistics.json')
        exp_dataset_train = torch.load(f'experiments/{args.default_index}/rejection_levels/exp_dataset_train.pth')
        ellipsoids_file = open(f"experiments/{args.default_index}/matrices/matrix_statistics.json")

    if not matrices_path.exists():
        raise ValueError(f"Matrix statistics have to be computed: {matrices_path}")

    ellipsoids = json.load(ellipsoids_file)

    compute_rejection_level(
        exp_dataset_train = exp_dataset_train,
        default_index = args.default_index,
        ellipsoids = ellipsoids,
        t_epsilon = args.t_epsilon,
        epsilon = args.epsilon,
        temp_dir = args.temp_dir
    )


if __name__ == '__main__':
    main()
