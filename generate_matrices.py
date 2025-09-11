"""
    This script computes matrices for a subset of a dataset for a neural network trained with specific hyper parameters.
"""
import os
import torch
from argparse import ArgumentParser, Namespace

from matrix_construction.parallel import ParallelMatrixConstruction
from constants.constants import DEFAULT_EXPERIMENTS


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default=None, help="The index for default experiment")
    parser.add_argument("--num_samples_per_class", type=int, default=1000, help="Number of data samples per class")
    parser.add_argument("--temp_dir", default=None, type=str, help="Temporary directory for data")
    parser.add_argument("--chunk_id", type=int, default=None, help="Chunk ID to process (set by Slurm task ID)")
    parser.add_argument("--total_chunks", type=int, default=1, help="Total number of chunks")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of samples to process at a time per GPU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.experiment_name is None:
        raise ValueError("Default index not specified in constants/constants.py")

    experiment = args.experiment_name

    dataset = DEFAULT_EXPERIMENTS[experiment]['dataset']
    epochs = DEFAULT_EXPERIMENTS[experiment]['epochs']-1 if dataset != 'imagenet' else None
    architecture_index = DEFAULT_EXPERIMENTS[experiment]['architecture_index']
    num_samples = args.num_samples_per_class

    chunk_id = int(os.getenv('SLURM_ARRAY_TASK_ID', args.chunk_id))
    if chunk_id is None:
        raise ValueError("chunk_id must be provided or set via SLURM_ARRAY_TASK_ID")

    chunk_size = num_samples // args.total_chunks

    if args.temp_dir is not None:
        weights_path = f'{args.temp_dir}/experiments/{experiment}/weights/'
        save_path = f'{args.temp_dir}/experiments/{experiment}/matrices'
    else:
        weights_path = f'experiments/{experiment}/weights/'
        save_path = f'experiments/{experiment}/matrices'

    if not os.path.exists(weights_path):
        raise ValueError(f"Model needs to be trained first")

    dict_exp = {
        "epochs": epochs,
        "weights_path": weights_path,
        "save_path": save_path,
        "data_name": dataset,
        'num_samples': num_samples,
        'chunk_size': chunk_size,
        'architecture_index': architecture_index,
        'batch_size': args.batch_size,
    }

    # Set device to the GPU assigned to this task
    gpu_id = chunk_id % torch.cuda.device_count()  # Cycle through available GPUs
    torch.cuda.set_device(gpu_id)
    print(f"Processing chunk {chunk_id} on GPU {gpu_id}", flush=True)

    mat_constructer = ParallelMatrixConstruction(dict_exp)
    mat_constructer.values_on_epoch(chunk_id=chunk_id)

    done_file = os.path.join(save_path, f"done_chunk_{args.chunk_id}.txt")
    with open(done_file, 'w') as f:
        f.write("done")

    print(f"Chunk {args.chunk_id} completed and saved to {save_path}", flush=True)
    print(f"Chunk {chunk_id} completed!", flush=True)


if __name__ == '__main__':
    main()
