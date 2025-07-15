import os
import torch
import json
import random
import torchvision
from torchvision import transforms
import shutil
from typing import Union
from mnist1d.data import make_dataset, get_dataset_args

from model_zoo.mlp import MLP
from model_zoo.cnn import CNN_2D
from model_zoo.res_net import ResNet
from model_zoo.alex_net import AlexNet
from model_zoo.vgg import VGG
from constants.constants import ARCHITECTURES


def get_device() -> torch.device:
    """
        Returns:
            The device to use.
    """
    if torch.cuda.is_available():
        print("DEVICE: cuda")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("DEVICE: mps")
        return torch.device("mps")
    else:
        print("DEVICE: cpu")
        return torch.device("cpu")


def get_architecture(
        input_shape = (1, 28, 28),
        num_classes:int = 10,
        architecture_index:int = 0,
        residual:bool = False,
        dropout:bool = False
    ) -> Union[MLP, CNN_2D, ResNet, AlexNet, VGG]:
    """
        Args:
            input_shape: The shape of the input data.
            num_classes: The number of classes in the dataset.
            architecture_index: The index of the architecture to use (See constants/constants.py).
            residual: Whether to use residual connections.
            dropout: Whether to use dropout.
        Returns:
            The architecture to use.
    """
    if architecture_index <= 7 and architecture_index >= 0:
        model = MLP(
            input_shape = input_shape,
            num_classes = num_classes,
            hidden_sizes = ARCHITECTURES[architecture_index],
            residual = residual,
            bias = True,
            dropout = dropout,
        )
    elif architecture_index == -3:
        model = AlexNet(
            input_shape = input_shape,
            num_classes = num_classes
        )
    elif architecture_index == -2:
        model = ResNet(
            input_shape = input_shape,
            num_classes = num_classes
        )
    elif architecture_index == -1:
        model = VGG(
            input_shape = input_shape,
            num_classes = num_classes
        )
    else:
        model = CNN_2D(
            input_shape = input_shape,
            num_classes = num_classes,
            channels = ARCHITECTURES[architecture_index][0],
            fc = ARCHITECTURES[architecture_index][1]
        )
    return model


def get_model(
        path: str, 
        architecture_index: int, 
        residual: bool, 
        input_shape,
        num_classes: int, 
        dropout: bool
    ) -> Union[MLP, CNN_2D, ResNet, AlexNet, VGG]:
    """ 
        Args:
            path: The path to the model weights.
            architecture_index: The index of the architecture to use (See constants/constants.py).
            residual: Whether to use residual connections.
            input_shape: The shape of the input data.
            num_classes: The number of classes in the dataset.
            dropout: Whether to use dropout.
        Returns:
            The model to use.
    """
    weight_path = torch.load(str(path), map_location=torch.device('cpu'))
    model = get_architecture(
                architecture_index = architecture_index,
                residual = residual,
                input_shape = input_shape,
                num_classes = num_classes,
                dropout = dropout
            )
    model.load_state_dict(weight_path)
    return model


def get_input_shape(
        data_set: str
    ) -> tuple[int]:
    """
        Args:
            data_set: The dataset to use.
        Returns:
            The input shape.
    """
    if data_set == 'mnist' or data_set == 'fashion':
        return (1, 28, 28)
    elif data_set == 'mnist1d':
        return (1, 1, 40)
    elif data_set == 'cifar10' or data_set == 'cifar100':
        return (3, 32, 32)
    else:
        raise ValueError("Unsupported dataset.")


def get_num_classes(
        data_set: str
    ) -> int:
    """
        Args:
            data_set: The dataset to use.
        Returns:
            The number of classes.
    """
    if data_set == 'cifar100':
        return 100
    else:
        return 10


def get_dataset(
        data_set: str, 
        batch_size:int = 32,
        data_loader:bool = True,
        data_path:Union[str, None] = None
    ) -> tuple:
    """
        Args:
            data_set: The dataset to use.
            batch_size: The batch size.
            data_loader: Whether to use a data loader.
            data_path: The path to the data.
        Returns:
            The dataset to use.
    """
    transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.5,), (0.5,))
    ])
    if data_path is None:
        data_path = './data'
    else:
        data_path = data_path + '/data'

    if data_set == 'mnist':
        train_set = torchvision.datasets.MNIST(
            root = data_path, 
            train = True, 
            transform = transform, 
            download = True
        )
        test_set = torchvision.datasets.MNIST(
            root = data_path, 
            train = False, 
            transform = transform, 
            download = True
        )
    elif data_set == 'fashion':
        train_set = torchvision.datasets.FashionMNIST(
            root = data_path, 
            train = True, 
            transform = transform, 
            download = True
        )
        test_set = torchvision.datasets.FashionMNIST(
            root = data_path, 
            train = False, 
            transform = transform, 
            download = True
        )
    elif data_set == 'cifar10':
        # Use data augmentation for CIFAR-10
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            # mean=[0.4914, 0.4822, 0.4465],
            # std=[0.2470, 0.2435, 0.2616]
        )
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            # transforms.Resize((32,32)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(7),
            # transforms.RandomAffine(0, shear=6, scale=(0.9,1.1)),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            normalize
        ])
        train_set = torchvision.datasets.CIFAR10(
            root=data_path,
            train=True,
            transform=transform_train,
            download=False
        )
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        test_set = torchvision.datasets.CIFAR10(
            root=data_path,
            train=False,
            transform=test_transforms,
            download=False
        )
    elif data_set == 'cifar100':
        # Use data augmentation for CIFAR-100
        normalize = transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
        transform_train = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(7),
            transforms.RandomAffine(0, shear=6, scale=(0.9,1.1)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            normalize
        ])
        train_set = torchvision.datasets.CIFAR100(
            root = data_path, 
            train = True, 
            transform = transform_train, 
            download = True
        )
        test_set = torchvision.datasets.CIFAR100(
            root = data_path, 
            train = False, 
            transform = transform, 
            download = True
        )
    elif data_set == 'imagenette':
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_set = torchvision.datasets.Imagenette(
            root = data_path, 
            train = True, 
            transform = preprocess, 
            download = True
        )
        test_set = torchvision.datasets.Imagenette(
            root = data_path, 
            train = False, 
            transform = preprocess, 
            download = True
        )
    elif data_set == "mnist1d":
        defaults = get_dataset_args()
        data = make_dataset(defaults)
        train_set = torch.from_numpy(data['x']).reshape(-1, 1, 1, 40).float()
        test_set = torch.from_numpy(data['x_test']).reshape(-1, 1, 1, 40).float()
        train_set = list(zip(train_set, torch.from_numpy(data['y'])))
        test_set = list(zip(test_set, torch.from_numpy(data['y_test'])))
    else:
        print(f"Dataset {data_set} not supported...")
        exit(1)

    if data_loader:
        train_loader = torch.utils.data.DataLoader(
            dataset = train_set, 
            batch_size = batch_size, 
            shuffle = True
        )
        test_loader = torch.utils.data.DataLoader(
            dataset = test_set, 
            batch_size = batch_size, 
            shuffle = False
        )
        return train_loader, test_loader
    else:
        return train_set, test_set


def find_matrices(base_dir: str):
    """
        Finds the matrices for the given base directory.

        Args:
            base_dir: The base directory to search for matrices.
        Returns:
            A dictionary with the keys being the class indices and the values 
            being the paths to the matrices.
    """
    matrix_paths = {}
    for j in range(10):  # Considering subfolders '0' to '9'
        matrices_path = os.path.join(base_dir, str(j))  # Only use training data
        if os.path.exists(matrices_path):
            for i in os.listdir(matrices_path):  # Iterating through each 'i' subdirectory
                matrix_file_path = os.path.join(matrices_path, i, 'matrix.pt')
                if os.path.isfile(matrix_file_path):  # Check if matrix.pt exists
                    if j not in matrix_paths:
                        matrix_paths[j] = [matrix_file_path]
                    else:
                        matrix_paths[j].append(matrix_file_path)
    return matrix_paths


def compute_statistics(
        matrix_paths
    ):
    """
        Computes the statistics for the given matrix paths.

        Args:
            matrix_paths: A dictionary with the keys being the class indices and 
            the values being the paths to the matrices.
        Returns:
            A dictionary with the keys being the class indices and the values 
            being the statistics (mean and std).
    """
    statistics = {}
    for j, paths in matrix_paths.items():
        matrices = [torch.load(path) for path in paths]
        # Stack all matrices to compute statistics across all matrices in a subfolder
        stacked_matrices = torch.stack(matrices)
        # Compute mean and std across the stacked matrices
        mean_matrix = torch.mean(stacked_matrices, dim=0)
        std_matrix = torch.std(stacked_matrices, dim=0)
        # Store the computed statistics
        statistics[j] = {'mean': mean_matrix, 'std': std_matrix}

    return statistics


def compute_train_statistics(
        default_index:int = 0, 
        path:Union[str, None] = None
    ) -> None:
    """
        Computes the statistics for the given path.

        Args:
            default_index: The index of the experiment.
            path: The path to the matrices.
    """
    if path is not None:
        original_matrices_path = f'{path}/experiments/{default_index}/matrices/'
    else:
        original_matrices_path = f'experiments/{default_index}/matrices/'
    original_matrices_paths = find_matrices(original_matrices_path)

    statistics = compute_statistics(original_matrices_paths)

    # Convert tensors to lists (or numbers) for JSON serialization
    for subfolder, stats in statistics.items():
        for key, tensor in stats.items():
            if tensor.numel() == 1:  # If the tensor has only one element, convert to a Python scalar
                stats[key] = tensor.item()
            else:  # Otherwise, convert to a list
                stats[key] = tensor.tolist()

    with open(f'experiments/{default_index}/matrices/matrix_statistics.json', 'w') as json_file:
        json.dump(statistics, json_file, indent=4)


def get_ellipsoid_data(
        ellipsoids: dict, 
        result: torch.Tensor, 
        param: str
    ) -> torch.Tensor:
    """
        Args:
            ellipsoids: matrix statistics dictionary with keys the classes and mean and std
            result: predicted class by the model
            param: the parameter to get from the ellipsoid statistics
        Returns:
            The boundary of the ellipsoid.
    """
    return torch.Tensor(ellipsoids[str(result.item())][param])


def is_in_ellipsoid(
        matrix: torch.Tensor,
        ellipsoid_mean: torch.Tensor,
        ellipsoid_std: torch.Tensor,
        std: float = 2
    ) -> torch.LongTensor:
    """
        Args:
            matrix: the matrix to check.
            ellipsoid_mean: the mean of the ellipsoid.
            ellipsoid_std: the std of the ellipsoid.
            std: increase the size of the ellipsoid by this factor.
        Returns:
            The number of elements in the ellipsoid.
    """
    low_bound = torch.le(ellipsoid_mean-std*ellipsoid_std, matrix)
    up_bound = torch.le(matrix, ellipsoid_mean+std*ellipsoid_std)
    return torch.count_nonzero(torch.logical_and(low_bound, up_bound))


def zero_std(
        matrix: torch.Tensor,
        ellipsoid_std: torch.Tensor,
        epsilon: float = 0
    ) -> torch.LongTensor:
    """
        Args:
            matrix: the matrix to check.
            ellipsoid_std: the std of the ellipsoid.
            epsilon: the threshold.
        Returns:
            The number of elements in the ellipsoid.
    """
    return torch.count_nonzero(torch.logical_and((ellipsoid_std <= epsilon), (matrix > epsilon)))

def subset(
        train_set, 
        length: int, 
        input_shape = (1, 28, 28)
    ):
    """
        Make a random subset of the training set of the given length. 
        If the length is greater or equal to the length of the training set, 
        this function will shuffle the training set.
        Args:
            train_set: the training set (MNIST, CIFAR-10, etc.).
            length: the length of the subset.
            input_shape: the shape of the input.
        Returns:
            A random subset of the training set of the given length.
    """
    if length > len(train_set):
        length = len(train_set)
    idx = random.sample(range(len(train_set)), length)
    exp_dataset = torch.zeros([length, input_shape[0], input_shape[1], input_shape[2]])
    exp_labels = torch.zeros([length], dtype=torch.long)
    for i, j in enumerate(idx):
        exp_dataset[i] = train_set[j][0]
        exp_labels[i] = train_set[j][1]
    return exp_dataset, exp_labels


def zip_and_cleanup(
        src_directory: str, 
        zip_filename: str, 
        clean:bool = True
    ) -> None:
    """
        Args:
            src_directory: the source directory.
            zip_filename: the filename of the zip file.
            clean: whether to clean the source directory.
    """
    # Create a zip archive
    print("Zipping", flush=True)
    shutil.make_archive(zip_filename, 'zip', src_directory)

    # Walk the directory tree and remove files and subdirectories
    if clean:
        print("Cleaning", flush=True)
        for root, dirs, files in os.walk(src_directory, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
