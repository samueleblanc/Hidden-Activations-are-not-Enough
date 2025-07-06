import os
import torch
from argparse import ArgumentParser, Namespace
from pathlib import Path
from multiprocessing import Pool
from typing import Union

from model_zoo.mlp import MLP
from model_zoo.cnn import CNN_2D
from model_zoo.alex_net import AlexNet
from model_zoo.res_net import ResNet
from model_zoo.vgg import VGG
from constants.constants import DEFAULT_EXPERIMENTS
from utils.utils import get_model, subset, get_dataset, zip_and_cleanup, get_num_classes, get_input_shape
from matrix_construction.matrix_computation import MlpRepresentation, ConvRepresentation_2D


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
        "--default_index",
        type = int,
        default = 0,
        help = "Index of default trained networks.",
    )
    parser.add_argument(
        "--num_samples_rejection_level",
        type = int,
        default = 10000,
        help = "Number of train samples to compute rejection level.",
    )
    parser.add_argument(
        "--nb_workers",
        type = int,
        default = 8,
        help = "How many processes in parallel for adversarial examples computations.",
    )
    parser.add_argument(
        "--temp_dir",
        type = str,
        default = None,
        help = "Temporary directory to save and read data. Useful when using clusters."
    )
    return parser.parse_args()


def compute_one_matrix(args: tuple) -> None:
    """
        Args:
            args: the arguments (see below).
    """
    (
        im, 
        label, 
        weights_path, 
        architecture_index, 
        residual, 
        input_shape, 
        num_classes,
        default_index, 
        dropout, 
        i, 
        temp_dir
    ) = args

    model = get_model(
        path = weights_path,
        architecture_index = architecture_index,
        residual = residual,
        input_shape = input_shape,
        num_classes = num_classes,
        dropout = dropout
    )
    if isinstance(model, MLP):
        matrix_computer = MlpRepresentation(model)
    elif isinstance(model, (CNN_2D, AlexNet, VGG, ResNet)):
        matrix_computer = ConvRepresentation_2D(model, batch_size=16)
    else:
        raise NotImplementedError(f"Model {type(model)} not supported")
    pred = torch.argmax(model.forward(im))
    if temp_dir is not None:
        path_experiment_matrix = Path(f'{temp_dir}/experiments/{default_index}/rejection_levels/matrices/{i}/matrix.pth')
    else:
        path_experiment_matrix = Path(f'experiments/{default_index}/rejection_levels/matrices/{i}/matrix.pth')
    # if it is not correctly classified, do not use it for rejection level
    if pred != label:
        return

    if os.path.exists(path_experiment_matrix):
        return
    mat = matrix_computer.forward(im)
    if temp_dir is not None:
        path_prediction = Path(f'{temp_dir}/experiments/{default_index}/rejection_levels/matrices/{i}/prediction.pth')
        Path(f'{temp_dir}/experiments/{default_index}/rejection_levels/matrices/{i}/').mkdir(parents=True, exist_ok=True)
    else:
        path_prediction = Path(f'experiments/{default_index}/rejection_levels/matrices/{i}/prediction.pth')
        Path(f'experiments/{default_index}/rejection_levels/matrices/{i}/').mkdir(parents=True, exist_ok=True)

    torch.save(pred, path_prediction)
    torch.save(mat, path_experiment_matrix)


def compute_matrices_for_rejection_level(
        exp_dataset_train: torch.Tensor,
        exp_dataset_labels: torch.Tensor,
        default_index: int,
        weights_path: str,
        architecture_index: int,
        residual: bool,
        input_shape: tuple[int,int,int],
        num_classes: int,
        dropout: bool,
        nb_workers: int = 8,
        temp_dir:Union[str, None] = None
    ) -> None:
    """
        Args:
            exp_dataset_train: the training set.
            exp_dataset_labels: the labels of the training set.
            default_index: the index of the default experiment (See constants/constants.py).
            weights_path: the path to the weights.
            architecture_index: the index of the architecture (See constants/constants.py).
            residual: whether the model has residual connections.
            input_shape: the shape of the input.
            dropout: whether the model has dropout layers.
            nb_workers: the number of workers.
            temp_dir: the temporary directory.
    """
    Path(f'experiments/{default_index}/rejection_levels/').mkdir(parents=True, exist_ok=True)
    print("Computing matrices for rejection level...", flush=True)

    with Pool(processes=nb_workers) as pool:
        args = [(exp_dataset_train[i],
                 exp_dataset_labels[i],
                 weights_path,
                 architecture_index,
                 residual,
                 input_shape,
                 num_classes,
                 default_index,
                 dropout,
                 i,
                 temp_dir) for i in range(len(exp_dataset_train))]
        pool.map(compute_one_matrix, args)


def main() -> None:
    """
        Main function to compute the matrices for the rejection level.
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

    print("Computing matrices for rejection level for Experiment: ", args.default_index,flush=True)

    if args.temp_dir is not None:
        weights_path = Path(f'{args.temp_dir}/experiments/{args.default_index}/weights') / f'epoch_{epoch}.pth'
    else:
        weights_path = Path(f'experiments/{args.default_index}/weights') / f'epoch_{epoch}.pth'

    if not weights_path.exists():
        raise ValueError(f"Experiment needs to be trained")

    input_shape = get_input_shape(dataset)
    num_classes = get_num_classes(dataset)
    train_set, _ = get_dataset(
        data_set = dataset,
        data_loader = False
    )
    exp_dataset_train, exp_dataset_labels = subset(
        train_set = train_set,
        length = args.num_samples_rejection_level,
        input_shape = input_shape
    )

    Path(f'experiments/{args.default_index}/rejection_levels/').mkdir(parents=True, exist_ok=True)
    torch.save(
        obj = exp_dataset_train, 
        f = f'experiments/{args.default_index}/rejection_levels/exp_dataset_train.pth'
    )

    compute_matrices_for_rejection_level(
        exp_dataset_train = exp_dataset_train,
        exp_dataset_labels = exp_dataset_labels,
        default_index = args.default_index,
        weights_path = weights_path,
        architecture_index = architecture_index,
        residual = residual,
        input_shape = input_shape,
        num_classes = num_classes,
        dropout = dropout,
        nb_workers = args.nb_workers,
        temp_dir = args.temp_dir
    )

    if args.temp_dir is not None:
        zip_and_cleanup(
            src_directory = f'{args.temp_dir}/experiments/{args.default_index}/rejection_levels/matrices/',
            zip_filename = f'experiments/{args.default_index}/rejection_levels/matrices/matrices',
            clean = False
        )


if __name__ == '__main__':
    main()
