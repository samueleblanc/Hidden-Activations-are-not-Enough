"""
    This script computes matrices for a subset of a dataset for a neural network trained with specific hyper parameters.
"""
import os
from argparse import ArgumentParser, Namespace
from multiprocessing import Pool

from matrix_construction.parallel import ParallelMatrixConstruction
from constants.constants import DEFAULT_EXPERIMENTS


def parse_args() -> Namespace:
    """
        Returns:
            The parsed arguments.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--default_index", 
        type = int, 
        default = 0, 
        help = "The index for default experiment"
    )
    parser.add_argument(
        "--num_samples_per_class",
        type = int, 
        default = 1000,
        help = "Number of data samples per class to compute matrices."
    )
    parser.add_argument(
        "--nb_workers", 
        type = int, 
        default = 8, 
        help = "Number of threads for parallel computation"
    )
    parser.add_argument(
        "--temp_dir", 
        type = str, 
        help = "Temporary directory for data. Useful when using clusters."
    )
    return parser.parse_args()


def compute_matrices(
        mat_constructer: ParallelMatrixConstruction, 
        chunk_id: int
    ) -> None:
    """
        Args:
            mat_constructer: the matrix constructor (See matrix_construction/parallel.py).
            chunk_id: the id of the chunk.
    """
    mat_constructer.values_on_epoch(chunk_id=chunk_id)


def main() -> None:
    """
        Main function to compute matrices.
    """
    args = parse_args()
    if args.default_index is not None:
        try:
            experiment = DEFAULT_EXPERIMENTS[f'experiment_{args.default_index}']

            epoch = experiment['epoch'] - 1
            dataset = experiment['dataset']
            architecture_index = experiment['architecture_index']
            residual = experiment['residual']
            dropout = experiment['dropout']
            num_samples = args.num_samples_per_class

        except KeyError:
            print(f"Error: Default index {args.default_index} does not exist.")
            print(f"When computing matrices of new model add the experiment to constants.constants.py"
                  f" inside DEFAULT_EXPERIMENTS and provide the corresponding --default_index when running this script.")
            return -1
    else:
        raise ValueError("Default index not specified in constants/constants.py")

    chunk_size = num_samples // args.nb_workers

    if args.temp_dir is not None:
        weights_path = f'{args.temp_dir}/experiments/{args.default_index}/weights/'
    else:
        weights_path = f'experiments/{args.default_index}/weights/'

    if not os.path.exists(weights_path):
        ValueError(f"Model needs to be trained first")

    save_path = f'experiments/{args.default_index}/matrices'

    dict_exp = {
        "epochs": epoch,
        "weights_path": weights_path,
        "save_path": save_path,
        "data_name": dataset,
        'num_samples': num_samples,
        'chunk_size': chunk_size,
        'architecture_index': architecture_index,
        'residual': residual,
        'dropout': dropout,
    }

    mat_constructer = ParallelMatrixConstruction(dict_exp)
    chunks = list(range(num_samples // chunk_size))
    arguments = list(zip([mat_constructer for _ in range(len(chunks))], chunks))

    print(f"Computing matrices...", flush=True)
    with Pool(processes=args.nb_workers) as pool:
        pool.starmap(compute_matrices, arguments)
    print("Done!", flush=True)


if __name__ == '__main__':
    main()
