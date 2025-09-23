import os
import torch
import json
import random
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
import shutil
from pathlib import Path
from typing import Union
from mnist1d.data import make_dataset, get_dataset_args

from model_zoo.mlp import MLP
from model_zoo.cnn import CNN_2D
from knowledgematrix.models.alexnet import AlexNet
from knowledgematrix.models.resnet18 import ResNet18
from knowledgematrix.models.vgg11 import VGG11
from constants.constants import ARCHITECTURES


from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from typing import Tuple, Optional, Callable


class ImageNetVal(Dataset):
    """
    Custom Dataset for ImageNet validation set, loading images and ground truth labels.

    Args:
        root (str): Path to validation images (e.g., /datashare/imagenet/ILSVRC2012/val/).
        gt_path (str): Path to ILSVRC2012_validation_ground_truth.txt.
        transform (Callable, optional): Transforms for images (e.g., Resize, ToTensor, Normalize).
        target_transform (Callable, optional): Transforms for labels.

    Loads 50,000 validation images with labels (0-999, matching ILSVRC2012_ID - 1).
    Images are sorted alphabetically to match ground truth order.
    """
    def __init__(
        self,
        root: str,
        gt_path: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.gt_path = gt_path

        # Load ground truth labels (50,000 lines, ILSVRC2012_ID 1-1000)
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Ground truth file not found at {gt_path}")
        with open(gt_path, 'r') as f:
            self.labels = [int(line.strip()) - 1 for line in f.readlines()]  # Convert to 0-indexed (0-999)

        if len(self.labels) != 50000:
            raise ValueError(f"Expected 50,000 labels, got {len(self.labels)} in {gt_path}")

        # Get sorted image paths (alphabetical order to match ground truth)
        self.image_paths = sorted(
            [os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(('.jpeg', '.jpg'))]
        )

        if len(self.image_paths) != 50000:
            raise ValueError(f"Expected 50,000 validation images, got {len(self.image_paths)} in {root}")

        # Verify image filenames (e.g., ILSVRC2012_val_00000001.JPEG)
        for i, path in enumerate(self.image_paths[:5], 1):
            expected = f"ILSVRC2012_val_{str(i).zfill(8)}.JPEG"
            if os.path.basename(path) != expected:
                print(f"Warning: Image {path} does not match expected {expected}. Labels may misalign.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        try:
            image = Image.open(self.image_paths[index]).convert('RGB')
        except Exception as e:
            print(f"Error loading image {self.image_paths[index]}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))  # Fallback black image

        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label


def get_imagenet_val_dataset(
    data_path: str = '/datashare/imagenet/ILSVRC2012',
    transform: Optional[transforms.Compose] = None,
    batch_size: int = 32
) -> Tuple[DataLoader, ImageNetVal]:
    """
    Loads ImageNet validation dataset as a proxy for test set with real labels.

    Args:
        data_path (str): Base path (val in data_path/val/, gt in data_path/ILSVRC2012_devkit_t12/data/).
        transform (transforms.Compose, optional): Image transforms.
        batch_size (int): Batch size for DataLoader.

    Returns:
        Tuple[DataLoader, ImageNetVal]: Validation DataLoader and dataset.
    """
    if transform is None:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),  # Standard ImageNet preprocessing
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    val_root = os.path.join(data_path, 'val')
    gt_path = os.path.join(data_path, 'ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt')

    val_set = ImageNetVal(
        root=val_root,
        gt_path=gt_path,
        transform=transform
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  # Adjust for Nibi cluster I/O
        pin_memory=True  # Faster GPU transfers
    )

    return val_loader, val_set


def get_device(trial_number: int = 1, gpu_count: int = 1, verbose=True) -> torch.device:
    """
        Returns:
            The device to use.
    """
    if gpu_count == 0:
        return torch.device("cpu")
        # Assign GPU based on trial number (e.g., trial 0 -> cuda:0, trial 1 -> cuda:1)

    if torch.cuda.is_available():
        if verbose:
            print("DEVICE: cuda")
        gpu_id = trial_number % gpu_count
        return torch.device(f"cuda:{gpu_id}")
    elif torch.backends.mps.is_available():
        if verbose:
            print("DEVICE: mps")
        return torch.device("mps")
    else:
        if verbose:
            print("DEVICE: cpu")
        return torch.device("cpu")


def get_architecture(
        input_shape = (1, 28, 28),
        num_classes:int = 10,
        architecture_index:int = 0,
        pretrained = False,
        freeze_features = False
    ) -> Union[MLP, CNN_2D, ResNet18, AlexNet, VGG11]:
    """
        Args:
            input_shape: The shape of the input data.
            num_classes: The number of classes in the dataset.
            architecture_index: The index of the architecture to use (See constants/constants.py).
            residual: Whether to use residual connections.
            dropout: Whether to use dropout.
        Returns:
            The architecture to use.

    if architecture_index <= 7 and architecture_index >= 0:
        model = MLP(
            input_shape = input_shape,
            num_classes = num_classes,
            hidden_sizes = ARCHITECTURES[architecture_index],
            residual = residual,
            bias = True,
            dropout = dropout,
        )
    """
    if architecture_index == -4:
        print("Lenet LOADED", flush=True)
        model = CNN_2D(input_shape=input_shape,
                       num_classes=num_classes,
                       channels=(6, 16),
                       padding=((2, 2), (0, 0)),
                       fc=(784, 84),
                       kernel_size=((5, 5), (5, 5)),
                       bias=False,
                       activation="relu",
                       pooling="avg")
    elif architecture_index == -3:
        model = AlexNet(
            input_shape, num_classes, pretrained=pretrained, freeze_features=freeze_features
        )
    elif architecture_index == -2:
        model = ResNet18(
            input_shape, num_classes, pretrained=pretrained, freeze_features=freeze_features
        )
    elif architecture_index == -1:
        model = VGG11(
            input_shape, num_classes, pretrained=pretrained, freeze_features=freeze_features
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
        path: Path,
        architecture_index: int,
        input_shape,
        num_classes: int,
        device: torch.device = torch.device('cpu'),
    ) -> Union[MLP, CNN_2D, ResNet18, AlexNet, VGG11]:
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
    weight_path = torch.load(str(path), map_location=torch.device(device))
    model = get_architecture(
                architecture_index = architecture_index,
                input_shape = input_shape,
                num_classes = num_classes,
            ).to(device)
    model.load_state_dict(weight_path)
    return model


def get_input_shape(
        data_set: str
    ):
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
    elif data_set == 'imagenet':
        return (3, 224, 224)
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
    elif data_set == 'imagenet':
        return 1000
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
        # Use ImageNet normalization for pretrained models
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose([
            transforms.Resize(224),  # Resize to match AlexNet input
            transforms.RandomHorizontalFlip(),  # Optional augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        test_transform = transforms.Compose([
            transforms.Resize(224),  # Resize to match AlexNet input
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        train_set = CIFAR10(root=data_path or './data', train=True, download=True, transform=train_transform)
        test_set = CIFAR10(root=data_path or './data', train=False, download=True, transform=test_transform)

        '''
        # Use data augmentation for CIFAR-10
        transform_train = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(7),
            transforms.RandomAffine(0, shear=6, scale=(0.9,1.1)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = torchvision.datasets.CIFAR10(
            root = data_path, 
            train = True, 
            transform = transform_train, 
            download = True
        )
        test_set = torchvision.datasets.CIFAR10(
            root = data_path, 
            train = False, 
            transform = transform, 
            download = True
        )
        '''
    elif data_set == 'cifar100':
        # Use ImageNet normalization for pretrained models
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose([
            transforms.Resize(224),  # Resize to match AlexNet input
            transforms.RandomHorizontalFlip(),  # Optional augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        test_transform = transforms.Compose([
            transforms.Resize(224),  # Resize to match AlexNet input
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        train_set = CIFAR100(root=data_path or './data', train=True, download=True, transform=train_transform)
        test_set = CIFAR100(root=data_path or './data', train=False, download=True, transform=test_transform)
        '''
        # Use data augmentation for CIFAR-100
        transform_train = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(7),
            transforms.RandomAffine(0, shear=6, scale=(0.9,1.1)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
        '''
    elif data_set == 'imagenet':
        val_loader, val_set = get_imagenet_val_dataset(data_path or '/datashare/imagenet/ILSVRC2012', batch_size=batch_size)
        if data_loader:
            return None, val_loader  # No train loader, return val as test
        else:
            return None, val_set  # No train set, return val as test
        '''
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
        '''
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
        matrices = [torch.load(path, map_location=torch.device('cpu')) for path in paths]
        #matrices = [torch.load(path).cpu() for path in paths]
        # Stack all matrices to compute statistics across all matrices in a subfolder
        stacked_matrices = torch.stack(matrices)
        # Compute mean and std across the stacked matrices
        mean_matrix = torch.mean(stacked_matrices, dim=0)
        std_matrix = torch.std(stacked_matrices, dim=0)
        # Store the computed statistics
        statistics[j] = {'mean': mean_matrix, 'std': std_matrix}

    return statistics


def compute_train_statistics(
        experiment_name:str = None,
        path = None
    ) -> None:
    """
        Computes the statistics for the given path.

        Args:
            default_index: The index of the experiment.
            path: The path to the matrices.
    """
    if path is not None:
        original_matrices_path = f'{path}/experiments/{experiment_name}/matrices/'
    else:
        original_matrices_path = f'experiments/{experiment_name}/matrices/'
    original_matrices_paths = find_matrices(original_matrices_path)

    statistics = compute_statistics(original_matrices_paths)

    # Convert tensors to lists (or numbers) for JSON serialization
    for subfolder, stats in statistics.items():
        for key, tensor in stats.items():
            if tensor.numel() == 1:  # If the tensor has only one element, convert to a Python scalar
                stats[key] = tensor.item()
            else:  # Otherwise, convert to a list
                stats[key] = tensor.tolist()

    with open(f'experiments/{experiment_name}/matrices/matrix_statistics.json', 'w') as json_file:
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
    return torch.count_nonzero(torch.logical_and((ellipsoid_std.detach().cpu() <= epsilon), (matrix.detach().cpu() > epsilon)))

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

def get_parameters_baseline(dataset):
    param_sets = {
        'mnist': {
            'knn': [3, 5, 7],
            'kde': [0.5, 1, 1.5],
            'gmm': [10, 20, 30],
            'ocsvm': [0.01, 0.05, 0.1],
            'iforest': [50, 100, 150],
            'softmax': [0.9, 0.95, 0.99],
            'mahalanobis': [0.9, 0.95, 0.99]
        },
        'cifar10': {
            'knn': [5, 10, 15],
            'kde': [1, 2, 3],
            'gmm': [10, 20, 30],
            'ocsvm': [0.01, 0.05, 0.1],
            'iforest': [100, 150, 200],
            'softmax': [0.5, 0.7, 0.9],
            'mahalanobis': [0.9, 0.95, 0.99]
        },
        'cifar100': {
            'knn': [10, 15, 20],
            'kde': [1.5, 2.5, 3.5],
            'gmm': [50, 100, 150],
            'ocsvm': [0.05, 0.1, 0.2],
            'iforest': [150, 200, 250],
            'softmax': [0.3, 0.5, 0.7],
            'mahalanobis': [0.9, 0.95, 0.99]
        }
    }

    return param_sets[dataset]
