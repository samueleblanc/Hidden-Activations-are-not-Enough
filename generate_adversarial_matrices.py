import torch
from argparse import ArgumentParser, Namespace
from multiprocessing import Pool
from pathlib import Path
from typing import Union

from model_zoo.mlp import MLP
from model_zoo.cnn import CNN_2D
from model_zoo.alex_net import AlexNet
from model_zoo.res_net import ResNet
from model_zoo.vgg import VGG
from matrix_construction.matrix_computation import MlpRepresentation, ConvRepresentation_2D
from utils.utils import get_model, get_num_classes, get_input_shape, zip_and_cleanup
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
        "--default_index",
        type = int,
        default = 0,
        help = "Index of default trained network."
    )
    parser.add_argument(
        "--nb_workers",
        type = int,
        default = 8,
        help = "How many processes in parallel for adversarial examples computations and their matrices."
    )
    parser.add_argument(
        "--temp_dir",
        type = str,
        default = None,
        help = "Temporary directory to save and read data. Useful when using clusters."
    )

    return parser.parse_args()


def save_one_matrix(
        im: torch.Tensor, 
        attack: str, 
        i: int, 
        default_index: int, 
        weights_path: str, 
        architecture_index: int, 
        residual: bool, 
        input_shape: tuple[int,int,int], 
        num_classes: int, 
        dropout: bool, 
        temp_dir: Union[str, None]
    ) -> None:
    """
        Args:
            im: the image to save the matrix of.
            attack: the attack to save the matrix of.
            i: the index of the image.
            default_index: the index of the default experiment (See constants/constants.py).
            weights_path: the path to the weights.
            architecture_index: the index of the architecture (See constants/constants.py).
            residual: whether the model has residual connections.
            input_shape: the shape of the input.
            num_classes: the number of classes.
            dropout: whether the model has dropout layers.
            temp_dir: the temporary directory to save the matrix.
    """
    model = get_model(
        path = weights_path, 
        architecture_index = architecture_index, 
        residual = residual, 
        input_shape = input_shape, 
        num_classes = num_classes, 
        dropout = dropout
    )
    if isinstance(model, MLP):
        representation = MlpRepresentation(model)
    elif isinstance(model, (CNN_2D, AlexNet, VGG, ResNet)):
        representation = ConvRepresentation_2D(model, batch_size=16)
    else:
        raise NotImplementedError(f"Model {type(model)} not implemented.")
    if temp_dir is not None:
        matrix_save_path = Path(f'{temp_dir}/experiments/{default_index}/adversarial_matrices') / f'{attack}' / f'{i}/matrix.pth'
    else:
        matrix_save_path = Path(f'experiments/{default_index}/adversarial_matrices') / f'{attack}' / f'{i}/matrix.pth'

    matrix_save_path.parent.mkdir(parents=True, exist_ok=True)
    if not matrix_save_path.exists():
        mat = representation.forward(im)
        torch.save(mat, matrix_save_path)


def generate_matrices_for_attacks(
        default_index: int,
        temp_dir: Union[str, None],
        weights_path: str,
        architecture_index: int,
        residual: bool,
        input_shape: tuple[int,int,int],
        num_classes: int,
        dropout: bool,
        nb_workers: int
    ) -> None:
    """
        Calls the save_one_matrix function for each adversarial example.

        Args:
            default_index: the index of the default experiment (See constants/constants.py).
            temp_dir: the temporary directory to save the matrices.
            weights_path: the path to the weights.
            architecture_index: the index of the architecture (See constants/constants.py).
            residual: whether the model has residual connections.
            input_shape: the shape of the input.
            dropout: whether the model has dropout layers.
            nb_workers: the number of workers.
    """
    for attack in ['test'] + ATTACKS:
        if temp_dir is not None:
            path_adv_examples = Path(temp_dir) / f'experiments/{default_index}/adversarial_examples' / f"{attack}/adversarial_examples.pth"
        else:
            path_adv_examples = Path(f'experiments/{default_index}/adversarial_examples') / f"{attack}/adversarial_examples.pth"
        if not path_adv_examples.exists():
            print(f'Attak {attack} does NOT exists.', flush=True)
            continue
        attacked_dataset = torch.load(path_adv_examples)

        print(f"Generating matrices for attack {attack}.", flush=True)
        arguments = [(attacked_dataset[i].detach(),
                      attack,
                      i,
                      default_index,
                      weights_path,
                      architecture_index,
                      residual,
                      input_shape,
                      num_classes,
                      dropout, 
                      temp_dir)
                     for i in range(len(attacked_dataset))]

        with Pool(processes=nb_workers) as pool:
            pool.starmap(save_one_matrix, arguments)


def main() -> None:
    """
        Main function to generate adversarial matrices.
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
            print(f"When computing adversarial examples of new model, add the experiment to constants.constants.py inside DEFAULT_EXPERIMENTS"
                  f"and provide the corresponding --default_index N when running this script.")
            return -1
    else:
        raise ValueError("Default index not specified in constants/constants.py")

    print("Experiment: ", args.default_index, flush=True)

    if args.temp_dir is not None:
        weights_path = Path(f'{args.temp_dir}/experiments/{args.default_index}/weights/epoch_{epoch}.pth')
    else:
        weights_path = Path(f'experiments/{args.default_index}/weights/epoch_{epoch}.pth')

    if not weights_path.exists():
        raise ValueError(f"Experiment needs to be trained")

    input_shape = get_input_shape(dataset)
    num_classes = get_num_classes(dataset)

    generate_matrices_for_attacks(
        default_index = args.default_index,
        temp_dir = args.temp_dir,
        weights_path = weights_path,
        architecture_index = architecture_index,
        residual = residual,
        input_shape = input_shape,
        num_classes = num_classes,
        dropout = dropout,
        nb_workers = args.nb_workers
    )

    if args.temp_dir is not None:
        zip_and_cleanup(f'{args.temp_dir}/experiments/{args.default_index}/adversarial_matrices/',
                        f'experiments/{args.default_index}/adversarial_matrices/adversarial_matrices', 
                        clean = False)


if __name__ == "__main__":
    main()
