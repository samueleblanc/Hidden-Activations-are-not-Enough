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
from utils.utils import get_model, get_num_classes, get_input_shape, zip_and_cleanup, get_device
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
        default = None,
        help = "Name of experiment."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of colums in matrix to process at the same time."
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
        experiment_name: str,
        representation,
        temp_dir: Union[str, None],
        device
    ) -> None:
    """
        Args:
            im: the image to save the matrix of.
            attack: the attack to save the matrix of.
            i: the index of the image.
            default_index: the index of the default experiment (See constants/constants.py).

    """

    if temp_dir is not None:
        matrix_save_path = Path(f'{temp_dir}/experiments/{experiment_name}/adversarial_matrices') / f'{attack}' / f'{i}/matrix.pth'
    else:
        matrix_save_path = Path(f'experiments/{experiment_name}/adversarial_matrices') / f'{attack}' / f'{i}/matrix.pth'

    matrix_save_path.parent.mkdir(parents=True, exist_ok=True)
    if not matrix_save_path.exists():
        im = im.to(device)
        mat = representation.forward(im)
        torch.save(mat.cpu(), matrix_save_path)


def generate_matrices_for_attacks(
        experiment_name: str,
        temp_dir: Union[str, None],
        weights_path: Path,
        architecture_index: int,
        input_shape,
        num_classes: int,
        batch_size: int,
        device,
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
    model = get_model(
        path=weights_path,
        architecture_index=architecture_index,
        input_shape=input_shape,
        num_classes=num_classes,
        device=device
    ).to(device)
    for attack in ['test'] + ATTACKS:
        if temp_dir is not None:
            path_adv_examples = Path(temp_dir) / f'experiments/{experiment_name}/adversarial_examples' / f"{attack}/adversarial_examples.pth"
        else:
            path_adv_examples = Path(f'experiments/{experiment_name}/adversarial_examples') / f"{attack}/adversarial_examples.pth"
        if not path_adv_examples.exists():
            print(f'Attak {attack} does NOT exists.', flush=True)
            continue
        attacked_dataset = torch.load(path_adv_examples)
        attacked_dataset = attacked_dataset.to(device)

        print(f"Generating matrices for attack {attack}.", flush=True)

        #model = get_model(
        #    path=weights_path,
        #    architecture_index=architecture_index,
        #    input_shape=input_shape,
        #    num_classes=num_classes,
        #    device=device
        #)

        if isinstance(model, MLP):
            representation = MlpRepresentation(model)
        elif isinstance(model, (CNN_2D, AlexNet, VGG, ResNet)):
            representation = ConvRepresentation_2D(model, batch_size=batch_size, device=device)
        else:
            raise NotImplementedError(f"Model {type(model)} not implemented.")

        for i in range(len(attacked_dataset)):
            save_one_matrix(attacked_dataset[i],
                            attack,
                            i,
                            experiment_name,
                            representation,
                            temp_dir, device)


def main() -> None:
    """
        Main function to generate adversarial matrices.
    """
    args = parse_args()
    if args.experiment_name is not None:
        experiment = DEFAULT_EXPERIMENTS[f'{args.experiment_name}']
        architecture_index = experiment['architecture_index']
        dataset = experiment['dataset']
        epoch = experiment['epochs'] - 1

    else:
        raise ValueError("Default experiment not specified.")

    print("Experiment: ", args.experiment_name, flush=True)

    if args.temp_dir is not None:
        weights_path = Path(f'{args.temp_dir}/experiments/{args.experiment_name}/weights/epoch_{epoch}.pth')
    else:
        weights_path = Path(f'experiments/{args.experiment_name}/weights/epoch_{epoch}.pth')

    if not weights_path.exists():
        raise ValueError(f"Experiment needs to be trained")

    input_shape = get_input_shape(dataset)
    num_classes = get_num_classes(dataset)
    device = get_device()

    generate_matrices_for_attacks(
        experiment_name = args.experiment_name,
        temp_dir = args.temp_dir,
        weights_path = weights_path,
        architecture_index = architecture_index,
        input_shape = input_shape,
        num_classes = num_classes,
        batch_size = args.batch_size,
        device=device
    )

    #if args.temp_dir is not None:
    #    zip_and_cleanup(f'{args.temp_dir}/experiments/{args.experiment_name}/adversarial_matrices/',
    #                    f'experiments/{args.experiment_name}/adversarial_matrices/adversarial_matrices',
    #                    clean = False)


if __name__ == "__main__":
    main()
