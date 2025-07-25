"""
    This script contains functions for computing several matrices from neural networks in parallel.
"""

import os
import torch
from torch.utils.data import DataLoader, Subset
from typing import Union

from model_zoo.mlp import MLP
from model_zoo.cnn import CNN_2D
from model_zoo.alex_net import AlexNet
from model_zoo.res_net import ResNet
from model_zoo.vgg import VGG
from matrix_construction.matrix_computation import MlpRepresentation, ConvRepresentation_2D
from utils.utils import get_architecture, get_dataset, get_num_classes, get_input_shape


class ParallelMatrixConstruction:
    """
    A class for computing neural network matrices in parallel across multiple samples.
    Takes a dictionary of experiment parameters and handles parallel matrix computation
    for different model architectures (MLP, CNN, etc) and datasets.

    Args:
        dict_exp (dict): Dictionary containing experiment parameters including:
            - epochs: Number of training epochs
            - num_samples: Number of samples to process
            - data_name: Name of dataset
            - weights_path: Path to model weights
            - chunk_size: Size of parallel processing chunks
            - save_path: Where to save computed matrices
            - architecture_index: Index of model architecture
    """
    def __init__(self, dict_exp: dict) -> None:
        self.epoch: int = dict_exp["epochs"]
        self.num_samples: int = dict_exp["num_samples"]
        self.dataname: str = dict_exp["data_name"].lower()
        self.weights_path: str = dict_exp["weights_path"]
        self.chunk_size: int = dict_exp['chunk_size']
        self.save_path: str = dict_exp['save_path']
        self.architecture_index: int = dict_exp['architecture_index']
        self.device = 'cuda'
        self.batch_size = dict_exp['batch_size']
        self.num_classes: int = get_num_classes(self.dataname)

        self.data = get_dataset(self.dataname, data_loader=False)[0]

    def compute_matrices_on_dataset(
            self,
            model: Union[MLP, CNN_2D, AlexNet, ResNet, VGG],
            chunk_id: int
    ) -> None:
        """
        Computes matrices for all classes in the dataset in parallel.
        """
        if isinstance(model, MLP):
            matrix_computer = MlpRepresentation(model=model)
        elif isinstance(model, (CNN_2D, AlexNet, ResNet, VGG)):
            matrix_computer = ConvRepresentation_2D(model=model, batch_size=self.batch_size)
        else:
            raise ValueError(f"Architecture not supported: {model}. Expects MLP, CNN_2D, AlexNet, ResNet, or VGG.")

        for i in range(self.num_classes):
            if self.dataname == 'mnist1d':
                x_train = [self.data[idx][0] for idx, (_, target) in enumerate(self.data) if target in [i]]
            else:
                train_indices = [idx for idx, target in enumerate(self.data.targets) if target in [i]]
                sub_train_dataloader = DataLoader(
                                            Subset(self.data, train_indices),
                                            batch_size=int(self.num_samples),
                                            drop_last=True
                                        )

                x_train = next(iter(sub_train_dataloader))[0] # 0 for input and 1 for label

            self.compute_chunk_of_matrices(
                data = x_train,
                matrix_computer = matrix_computer,
                out_class = i,
                chunk_id = chunk_id
            )

    def values_on_epoch(self, chunk_id: int) -> None:
        """
        Loads the model state dictionary and computes matrices for all classes in the dataset.
        """
        path = os.getcwd()
        directory = f"{self.weights_path}"
        new_path = os.path.join(path, directory)
        model_file = f"epoch_{self.epoch}.pth"
        model_path = os.path.join(new_path, model_file)
        state_dict = torch.load(model_path, map_location=self.device)

        model = get_architecture(
                    input_shape = get_input_shape(self.dataname),
                    num_classes = self.num_classes,
                    architecture_index = self.architecture_index
                )
        model.to(self.device)
        model.load_state_dict(state_dict)
        self.compute_matrices_on_dataset(model, chunk_id=chunk_id)
    
    def compute_chunk_of_matrices(
            self,
            data: torch.Tensor,
            matrix_computer: Union[MlpRepresentation, ConvRepresentation_2D],
            out_class: int,
            chunk_id: int = 0
    ) -> None:
        """
        Computes and saves matrices induced by a neural network for a chunk of data samples from a given class.

        Args:
            data (torch.Tensor): Input data tensor containing samples from a single class
            matrix_computer (MlpRepresentation|ConvRepresentation_2D): Object that computes the matrix representation
                of the neural network at each input point
            out_class (int): The class label for this chunk of data
            chunk_id (int, optional): Index of the chunk being processed. Defaults to 0.

        The matrices are saved to disk in the following structure:
        save_path/class_label/sample_index/matrix.pt
        """
        directory = f"{self.save_path if self.save_path is not None else ''}/{out_class}/"
        os.makedirs(directory, exist_ok=True)

        data = data[chunk_id*self.chunk_size:(chunk_id+1)*self.chunk_size].to(self.device)

        for i, d in enumerate(data):
            idx = chunk_id*self.chunk_size+i
            root = os.path.join(directory, f"{idx}")
            matrix_path = os.path.join(root, "matrix.pt")
            
            if os.path.exists(matrix_path):
                # if matrix was already computed, pass to next sample of data
                continue

            '''
            if not os.path.exists(root):
                # if the path has not been created, then no one is working on this sample
                os.makedirs(root)
            else:
                # if the path has been created, someone else is already computing the matrix
                continue
            '''

            matrix = matrix_computer.forward(d)
            os.makedirs(root)
            torch.save(matrix, matrix_path)
