import torch
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Union
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer

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
        default=18816,
        help="Number of colums in matrix to process at the same time."
    )
    parser.add_argument(
        "--temp_dir",
        type = str,
        default = None,
        help = "Temporary directory to save and read data. Useful when using clusters."
    )
    parser.add_argument(
        "--chunk_id",
        type = int,
        default = 0,
        help = "Current chunk id or slurm task id"
    )
    parser.add_argument(
        "--total_chunks",
        type = int,
        default = 4,
        help = "Temporary directory to save and read data. Useful when using clusters."
    )

    return parser.parse_args()


def save_one_matrix(
        im: torch.Tensor, 
        attack: str, 
        i: int, 
        experiment_name: str,
        matrix_computer,
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

    if not matrix_save_path.exists():
        mat = matrix_computer.forward(im.unsqueeze(0).to(device))
        matrix_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(mat.cpu(), matrix_save_path)


def generate_matrices_for_attacks(
        experiment_name: str,
        temp_dir: Union[str, None],
        weights_path: Path,
        architecture_index: int,
        input_shape,
        num_classes: int,
        device,
        batch_size: int = 18816,
        chunk_id: int = 0,
        total_chunks: int = 1,
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
    )
    matrix_computer = KnowledgeMatrixComputer(model, batch_size=batch_size, device=device)
    for attack in ['test'] + ATTACKS:
        if temp_dir is not None:
            path_adv_examples = Path(temp_dir) / f'experiments/{experiment_name}/adversarial_examples' / f"{attack}/adversarial_examples.pth"
        else:
            path_adv_examples = Path(f'experiments/{experiment_name}/adversarial_examples') / f"{attack}/adversarial_examples.pth"
        if not path_adv_examples.exists():
            print(f'Attak {attack} does NOT exists.', flush=True)
            continue
        attacked_dataset = torch.load(path_adv_examples)

        print(f"Generating matrices for attack {attack}.", flush=True)

        N = attacked_dataset.shape[0]
        base_chunk = N // total_chunks
        remainder = N % total_chunks
        # distribute the remainder among the first `remainder` chunks
        if chunk_id < remainder:
            start = chunk_id * (base_chunk + 1)
            end = start + (base_chunk + 1)
        else:
            start = chunk_id * base_chunk + remainder
            end = start + base_chunk

        # Bound check
        start = max(0, start)
        end = min(N, end)

        print(f"Worker chunk_id={chunk_id} handling indices [{start}, {end}) out of {N}", flush=True)

        # iterate only over the slice for this chunk
        model.eval()
        for i in range(start, end):
            print(f'Chunk {chunk_id} - Matrix {i}/{N}', flush=True)
            save_one_matrix(attacked_dataset[i].to(device),
                            attack,
                            i,
                            experiment_name,
                            matrix_computer,
                            temp_dir,
                            device)




def main() -> None:
    """
        Main function to generate adversarial matrices.
    """
    args = parse_args()
    if args.experiment_name is not None:
        experiment = DEFAULT_EXPERIMENTS[f'{args.experiment_name}']
        architecture_index = experiment['architecture_index']
        dataset = experiment['dataset']
        epoch = experiment['epochs']

    else:
        raise ValueError("Experiment not specified. Use --experiment_name")

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
    print("----ALL MATRICES OF ADVERSARIAL EXAMPLES COMPUTED----", flush=True)


if __name__ == "__main__":
    main()
