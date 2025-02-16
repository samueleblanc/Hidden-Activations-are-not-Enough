"""
    This script contains functions for computing several matrices from neural networks in parallel.
"""
import os
import torch
from torch.utils.data import DataLoader, Subset

from model_zoo.mlp import MLP
from model_zoo.cnn import CNN_2D
from model_zoo.alex_net import AlexNet
from model_zoo.res_net import ResNet
from model_zoo.vgg import VGG
from matrix_construction.matrix_computation import MlpRepresentation, ConvRepresentation_2D
from utils.utils import get_architecture, get_dataset


class ParallelMatrixConstruction:
    def __init__(self, dict_exp: dict) -> None:
        self.epoch: int = dict_exp["epochs"]
        self.num_samples: int = dict_exp["num_samples"]
        self.dataname: str = dict_exp["data_name"].lower()
        self.weights_path: str = dict_exp["weights_path"]
        self.chunk_size: int = dict_exp['chunk_size']
        self.save_path: str = dict_exp['save_path']
        self.architecture_index: int = dict_exp['architecture_index']
        self.residual: bool = dict_exp['residual']
        self.dropout: bool = dict_exp['dropout']

        self.num_classes: int = 10  # TODO: This should not be fixed
        self.data = get_dataset(self.dataname, data_loader=False)[0]

    def compute_matrices_on_dataset(
            self,
            model: MLP|CNN_2D|AlexNet|ResNet|VGG,
            chunk_id: int
    ) -> None:
        if isinstance(model, MLP):
            matrix_computer = MlpRepresentation(model=model)
        elif isinstance(model, (CNN_2D, AlexNet, ResNet, VGG)):
            matrix_computer = ConvRepresentation_2D(model=model)
        else:
            raise ValueError(f"Architecture not supported: {model}. Expects MLP, CNN_2D, AlexNet, ResNet, or VGG.")

        for i in range(self.num_classes):
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
                save_path = self.save_path,
                chunk_id = chunk_id,
                chunk_size = self.chunk_size
            )

    def values_on_epoch(self, chunk_id: int) -> None:
        path = os.getcwd()
        directory = f"{self.weights_path}"
        new_path = os.path.join(path, directory)
        model_file = f"epoch_{self.epoch}.pth"
        model_path = os.path.join(new_path, model_file)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        # TODO: input_shape should be more general than that.
        input_shape = (3, 32, 32) if self.dataname == 'cifar10' or self.dataname == 'cifar100' else (1, 28, 28)
        model = get_architecture(
                    architecture_index = self.architecture_index,
                    residual = self.residual,
                    input_shape = input_shape,
                    dropout = self.dropout
                )

        model.load_state_dict(state_dict)
        self.compute_matrices_on_dataset(model, chunk_id=chunk_id)
    
    def compute_chunk_of_matrices(
            self,
            data: torch.Tensor,
            matrix_computer: MlpRepresentation|ConvRepresentation_2D,
            out_class: int,
            chunk_id: int = 0
    ) -> None:
        """
        Given a subset of data of a class out_class and an MlpRepresentation or ConvRepresentation_2D (matrix_computer), 
        the function computes and saves accordingly the induced matrices in the corresponding chunk of samples in data.
        """
        directory = f"{self.save_path if self.save_path is not None else ''}/{out_class}/"
        os.makedirs(directory, exist_ok=True)

        data = data[chunk_id*self.chunk_size:(chunk_id+1)*self.chunk_size]

        for i, d in enumerate(data):
            idx = chunk_id*self.chunk_size+i
            
            if os.path.exists(f"{directory}{idx}/matrix.pt"):
                # if matrix was already computed, pass to next sample of data
                continue
            if not os.path.exists(f"{directory}{idx}/"):
                # if the path has not been created, then no one is working on this sample
                os.makedirs(f"{directory}{idx}/")
            else:
                # if the path has been created, someone else is already computing the matrix
                continue

            matrix = matrix_computer.forward(d)
            torch.save(matrix, f"{directory}{idx}/matrix.pt")
