import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import Union

from model_zoo.alex_net import AlexNet
from model_zoo.cnn import CNN_2D
from model_zoo.mlp import MLP
from model_zoo.res_net import ResNet
from model_zoo.vgg import VGG
from constants.constants import DEFAULT_EXPERIMENTS
from utils.utils import get_architecture, get_dataset, get_device, get_input_shape


def parse_args() -> Namespace:
    """
        Return:
            The parsed arguments.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_name",
        type = str,
        default = None, # TODO: add the rest of experiments here and to other scripts
        help = "resnet_cifar100, resnet_cifar10, alexnet_cifar10, alexnet_imagenet, mlp_mnist, ..."
    )
    parser.add_argument(
        "--from_checkpoint", 
        action = 'store_true', 
        help = "Resume training from the last checkpoint."
    )
    parser.add_argument(
        "--temp_dir", 
        type = str, 
        default = None, 
        help = "Temporary path on compute node where the dataset is saved."
    )
    return parser.parse_args()


def train_one_epoch(
        model: Union[AlexNet, CNN_2D, MLP, ResNet, VGG], 
        train_loader, 
        criterion: nn.CrossEntropyLoss, 
        optimizer: Union[optim.SGD, optim.Adam], 
        device: torch.device
    ) -> None:
    """
        Args:
            model: the model to train.
            train_loader: the training loader.
            criterion: the criterion.
            optimizer: the optimizer.
            device: the device to train on.
    """
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)


def evaluate_model(
        model: Union[AlexNet, CNN_2D, MLP, ResNet, VGG], 
        data_loader, 
        criterion: nn.CrossEntropyLoss, 
        device: torch.device
    ):
    """
        Args:
            model: the model to evaluate.
            data_loader: the data loader.
            criterion: the criterion.
            device: the device to evaluate on.
        Returns:
            The loss and the accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(data_loader), correct / total


def main() -> None:
    """
        Main function to train the model.
    """
    print("Start main")
    args = parse_args()
    if args.experiment_name is not None:
        #try:
        experiment = args.experiment_name
        dataset = DEFAULT_EXPERIMENTS[experiment]['dataset']
        optimizer_name = DEFAULT_EXPERIMENTS[experiment]['optimizer']
        lr = DEFAULT_EXPERIMENTS[experiment]['lr']
        batch_size = DEFAULT_EXPERIMENTS[experiment]['batch_size']
        epochs = DEFAULT_EXPERIMENTS[experiment]['epochs']
        momentum = DEFAULT_EXPERIMENTS[experiment]['momentum']
        weight_decay = DEFAULT_EXPERIMENTS[experiment]['weight_decay']

        #except KeyError:
        #    print(f"Experiment {args.experiment_name} does not exist or has incorrect hyper parameters.")
        #    return
    else:
        raise ValueError("Default index not specified in constants/constants.py")

    device = get_device()
    train_loader, test_loader = get_dataset(
        data_set = dataset,
        batch_size = batch_size,
        data_loader = True,
        data_path = args.temp_dir
    )

    input_shape = get_input_shape(dataset)

    architecture_index = 0
    num_classes = 0

    if experiment[:7] == 'alexnet':
        architecture_index = -3
    elif experiment[:6] == 'resnet':
        architecture_index = -2
    elif experiment[:3] == 'vgg':
        architecture_index = -1

    if experiment[-7:] == 'cifar10':
        num_classes = 10
    elif experiment[-5:] == 'mnist':
        num_classes = 10
    elif experiment[-8:] == 'cifar100':
        num_classes = 100
    elif experiment[-9:] == 'imagenet':
        num_classes = 1000

    model = get_architecture(
        architecture_index = architecture_index,
        input_shape = input_shape,
        num_classes = num_classes,
        residual = residual,
        input_shape = get_input_shape(dataset),
        dropout = dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "sgd":
        optimizer = optim.SGD(
            params = model.parameters(), 
            lr = lr, 
            weight_decay = weight_decay,
            momentum = momentum
        )
    elif optimizer_name == "adam":
        optimizer = optim.Adam(
            params = model.parameters(), 
            lr = lr, 
            weight_decay = weight_decay
        )
    else:
        raise ValueError("Unsupported optimizer. Add it manually at line 221 on training.py")

    step_size = epochs-10 if epochs-10>0 else 10
    scheduler = StepLR(
        optimizer = optimizer,
        step_size = step_size,
        gamma = 0.1
    )

    start_epoch = 0
    if args.from_checkpoint:
        checkpoints_path = Path(f'experiments/{args.experiment}/weights/')
        if checkpoints_path.exists():
            checkpoints = list(checkpoints_path.glob('epoch_*.pth'))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda cp: int(cp.stem.split('_')[1]))
                model.load_state_dict(torch.load(latest_checkpoint))
                start_epoch = int(latest_checkpoint.stem.split('_')[1]) + 1
                print(f"Resuming training from epoch {start_epoch}")

    history = {'train_acc':[],
               'test_acc':[],
               'train_loss':[],
               'test_loss':[]}

    print("Training...", flush=True)
    for epoch in range(start_epoch, epochs):
        train_one_epoch(
            model = model, 
            train_loader = train_loader, 
            criterion = criterion, 
            optimizer = optimizer, 
            device = device
        )
        train_loss, train_accuracy = evaluate_model(
            model = model, 
            data_loader = train_loader, 
            criterion = criterion, 
            device = device
        )
        test_loss, test_accuracy = evaluate_model(
            model = model, 
            data_loader = test_loader, 
            criterion = criterion, 
            device = device
        )

        history['train_acc'].append(train_accuracy)
        history['test_acc'].append(test_accuracy)
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)

        print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}", flush=True)

        scheduler.step()

        #if epoch % save_every_epochs == 0 or epoch == epochs - 1:
        #    os.makedirs(f'experiments/{args.default_index}/weights/', exist_ok=True)
        #    torch.save(model.state_dict(), f'experiments/{args.default_index}/weights/epoch_{epoch}.pth')
        os.makedirs(f'experiments/{experiment}/weights/', exist_ok=True)
        with open(f'experiments/{experiment}/weights/history.json', 'w') as json_file:
            json.dump(history, json_file, indent=4)


if __name__ == "__main__":
    main()
