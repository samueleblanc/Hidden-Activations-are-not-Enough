import os
import torch
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Union

from model_zoo.mlp import MLP
from model_zoo.cnn import CNN_2D
from knowledgematrix.models.alexnet import AlexNet
from knowledgematrix.models.vgg11 import VGG11
from knowledgematrix.models.resnet18 import ResNet18
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from constants.constants import DEFAULT_EXPERIMENTS
from utils.utils import get_model, subset, get_dataset, get_num_classes, get_input_shape, get_device
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
        "--experiment_name",
        type = str,
        default = None,
        help = "Name of experiment. Check constants/constants.py.",
    )
    parser.add_argument(
        "--num_samples_rejection_level",
        type = int,
        default = 10000,
        help = "Number of train samples to compute rejection level.",
    )
    parser.add_argument(
        "--batch_size",
        type = int,
        default = 18816, # for h100 gpu
        help = "Temporary directory to save and read data. Useful when using clusters."
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
        experiment_name,
        i,
        temp_dir,
        batch_size,
        matrix_computer,
        pred
    ) = args

    if temp_dir is not None:
        path_experiment_matrix = Path(f'{temp_dir}/experiments/{experiment_name}/rejection_levels/matrices/{i}/matrix.pth')
        path_prediction = Path(f'{temp_dir}/experiments/{experiment_name}/rejection_levels/matrices/{i}/prediction.pth')
        Path(f'{temp_dir}/experiments/{experiment_name}/rejection_levels/matrices/{i}/').mkdir(parents=True, exist_ok=True)
    else:
        path_experiment_matrix = Path(f'experiments/{experiment_name}/rejection_levels/matrices/{i}/matrix.pth')
        path_prediction = Path(f'experiments/{experiment_name}/rejection_levels/matrices/{i}/prediction.pth')
        Path(f'experiments/{experiment_name}/rejection_levels/matrices/{i}/').mkdir(parents=True, exist_ok=True)

    # if it is not correctly classified, do not use it for rejection level
    if pred != label:
        return

    if os.path.exists(path_experiment_matrix):
        return

    mat = matrix_computer.forward(im.unsqueeze(0))

    torch.save(pred.cpu().detach(), path_prediction)
    torch.save(mat.cpu().detach(), path_experiment_matrix)


def compute_matrices_for_rejection_level(
        exp_dataset_train: torch.Tensor,
        exp_dataset_labels: torch.Tensor,
        experiment_name: str,
        weights_path: Path,
        architecture_index: int,
        input_shape,
        num_classes: int,
        temp_dir:Union[str, None] = None,
        device: torch.device = 'cpu',
        batch_size: int = 18816,
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
    Path(f'experiments/{experiment_name}/rejection_levels/').mkdir(parents=True, exist_ok=True)
    print("Computing matrices for rejection level...", flush=True)

    model = get_model(path=weights_path,
                      architecture_index=architecture_index,
                      input_shape=input_shape,
                      num_classes=num_classes,
                      device=device)

    matrix_computer = KnowledgeMatrixComputer(model, batch_size=batch_size, device=device)

    for i in range(len(exp_dataset_train)):
        model.eval()
        pred = torch.argmax(model.forward(exp_dataset_train[i].unsqueeze(0).to(device)))

        args = (exp_dataset_train[i].to(device),
                exp_dataset_labels[i].to(device),
                experiment_name,
                i,
                temp_dir,
                batch_size,
                matrix_computer,
                pred
                )
        print(f'Matrix {i}/{len(exp_dataset_train)}', flush=True)
        compute_one_matrix(args)


def main() -> None:
    """
        Main function to compute the matrices for the rejection level.
    """
    args = parse_args()
    if args.experiment_name is not None:
        experiment = DEFAULT_EXPERIMENTS[f'{args.experiment_name}']
        architecture_index = experiment['architecture_index']
        dataset = experiment['dataset']
        epoch = experiment['epochs']

    else:
        raise ValueError("Default index not specified in constants/constants.py")

    print("Computing matrices for rejection level for Experiment: ", args.experiment_name,flush=True)

    device = get_device()

    if args.temp_dir is not None:
        weights_path = Path(f'{args.temp_dir}/experiments/{args.experiment_name}/weights') / f'epoch_{epoch}.pth'
    else:
        weights_path = Path(f'experiments/{args.experiment_name}/weights') / f'epoch_{epoch}.pth'

    if not weights_path.exists():
        raise ValueError(f"Experiment needs to be trained")

    input_shape = get_input_shape(dataset)
    num_classes = get_num_classes(dataset)

    Path(f'experiments/{args.experiment_name}/rejection_levels/').mkdir(parents=True, exist_ok=True)
    exp_dataset_train_file = Path(f'experiments/{args.experiment_name}/rejection_levels/exp_dataset_train.pth')
    exp_dataset_labels_file = Path(f'experiments/{args.experiment_name}/rejection_levels/exp_dataset_labels.pth')

    if exp_dataset_train_file.exists() and exp_dataset_labels_file.exists():
        exp_dataset_train = torch.load(exp_dataset_train_file)
        exp_dataset_labels = torch.load(exp_dataset_labels_file)

    else:
        train_set, _ = get_dataset(
            data_set=dataset,
            data_loader=False
        )
        exp_dataset_train, exp_dataset_labels = subset(
            train_set = train_set,
            length = args.num_samples_rejection_level,
            input_shape = input_shape
        )

        torch.save(
            obj = exp_dataset_train,
            f = f'experiments/{args.experiment_name}/rejection_levels/exp_dataset_train.pth'
        )
        torch.save(
            obj = exp_dataset_labels,
            f = f'experiments/{args.experiment_name}/rejection_levels/exp_dataset_labels.pth'
        )

    compute_matrices_for_rejection_level(
        exp_dataset_train = exp_dataset_train,
        exp_dataset_labels = exp_dataset_labels,
        experiment_name = args.experiment_name,
        weights_path = weights_path,
        architecture_index = architecture_index,
        input_shape = input_shape,
        num_classes = num_classes,
        temp_dir = args.temp_dir,
        device=device,
        batch_size = args.batch_size,
    )
    print("---ALL MATRICES COMPUTED----", flush=True)

    #if args.temp_dir is not None:
    #    zip_and_cleanup(
    #        src_directory = f'{args.temp_dir}/experiments/{experiment}/rejection_levels/matrices/',
    #        zip_filename = f'experiments/{experiment}/rejection_levels/matrices/matrices',
    #        clean = False
    #    )


if __name__ == '__main__':
    main()
