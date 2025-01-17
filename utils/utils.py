import os
import torch
import json
import torchvision
from torchvision import transforms
import shutil

from model_zoo.mlp import MLP
from model_zoo.cnn import CNN_2D
from constants.constants import ARCHITECTURES


def get_device():
    if torch.cuda.is_available():
        print("DEVICE: cuda")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("DEVICE: mps")
        return torch.device("mps")
    else:
        print("DEVICE: cpu")
        return torch.device("cpu")


def get_architecture(input_shape=(1, 28, 28), num_classes=10, architecture_index=0, residual=False, dropout=False) -> MLP|CNN_2D:
    if architecture_index <= 7:
        model = MLP(input_shape=input_shape,
                    num_classes=num_classes,
                    hidden_sizes=ARCHITECTURES[architecture_index],
                    residual=residual,
                    bias=True,
                    dropout=dropout,
                    )
    else:
        model = CNN_2D(input_shape=input_shape,
                       num_classes=num_classes,
                       channels=ARCHITECTURES[architecture_index][0],
                       fc=ARCHITECTURES[architecture_index][1]
                       )
    return model


def get_model(path, architecture_index, residual, input_shape, dropout):
    weight_path = torch.load(str(path), map_location=torch.device('cpu'))
    model = get_architecture(architecture_index=architecture_index,
                             residual=residual,
                             input_shape=input_shape,
                             dropout=dropout)
    model.load_state_dict(weight_path)
    return model


def get_dataset(data_set, batch_size=32, data_loader=True, data_path=None):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    if data_path is None:
        data_path = './data'
    else:
        data_path = data_path + '/data'

    if data_set == 'mnist':
        train_set = torchvision.datasets.MNIST(root=data_path, train=True, transform=transform, download=True)
        test_set = torchvision.datasets.MNIST(root=data_path, train=False, transform=transform, download=True)
        if data_loader:
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
            return train_loader, test_loader
        else:
            return train_set, test_set
    elif data_set == 'fashion':
        train_set = torchvision.datasets.FashionMNIST(root=data_path, train=True, transform=transform, download=True)
        test_set = torchvision.datasets.FashionMNIST(root=data_path, train=False, transform=transform, download=True)
        if data_loader:
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
            return train_loader, test_loader
        else:
            return train_set, test_set
    elif data_set == 'cifar10':
        # Use data augmentation for CIFAR-10
        transform_train = transforms.Compose([transforms.Resize((32,32)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(7),
                                      transforms.RandomAffine(0, shear=6, scale=(0.9,1.1)),
                                      transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])
        train_set = torchvision.datasets.CIFAR10(root=data_path, train=True, transform=transform_train, download=True)
        test_set = torchvision.datasets.CIFAR10(root=data_path, train=False, transform=transform, download=True)
        if data_loader:
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
            return train_loader, test_loader
        else:
            return train_set, test_set
    else:
        print(f"Dataset {data_set} not supported...")
        exit(1)


# Function to accurately locate matrix.pt files for training data
def find_matrices(base_dir):
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


# Function to load matrices and compute statistics
def compute_statistics(matrix_paths):
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


def compute_train_statistics(default_index=0, path=None):
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


def interpolate(data1: torch.Tensor, data2: torch.Tensor, alpha: float=0.5) -> torch.Tensor:
    return (1-alpha)*data1 + alpha*data2


def rotate(data: torch.Tensor, angle: float) -> torch.Tensor:
    return transforms.functional.rotate(data, angle)


def get_ellipsoid_data(ellipsoids: dict, result: torch.Tensor, param: str) -> torch.Tensor:
    """

    :param: ellipsoids: matrix statistics dictionary with keys the classes and mean and std
    :param: result:     predicted class by the model
    :param: ellipsoids: ellipsoids per class

    """
    return torch.Tensor(ellipsoids[str(result.item())][param])


def is_in_ellipsoid(matrix: torch.Tensor,
                    ellipsoid_mean: torch.Tensor,
                    ellipsoid_std: torch.Tensor,
                    std: float=2) -> torch.LongTensor:
    # Gives the opposite of what would be intuitive (ie more non zero if attacked)
    low_bound = torch.le(ellipsoid_mean-std*ellipsoid_std, matrix)
    up_bound = torch.le(matrix, ellipsoid_mean+std*ellipsoid_std)
    return torch.count_nonzero(torch.logical_and(low_bound, up_bound))


def zero_std(matrix: torch.Tensor,
             ellipsoid_std: torch.Tensor,
             d1: float=0) -> torch.LongTensor:
    return torch.count_nonzero(torch.logical_and((ellipsoid_std <= d1), (matrix > d1)))  # ¬(P => Q) <==> P ∧ ¬Q


def subset(train_set, length: int, input_shape=(1, 28, 28)):
    idx = torch.randint(low=0, high=len(train_set), size=[length], generator=torch.Generator("cpu"))
    exp_dataset = torch.zeros([length, input_shape[0], input_shape[1], input_shape[2]])
    exp_labels = torch.zeros([length], dtype=torch.long)
    for i, j in enumerate(idx):
        exp_dataset[i] = train_set[j][0]
        exp_labels[i] = train_set.targets[j]
    return exp_dataset, exp_labels


def zip_and_cleanup(src_directory, zip_filename, clean=True):
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
