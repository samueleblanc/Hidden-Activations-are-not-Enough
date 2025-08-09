import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from model_zoo.cnn import CNN_2D
from utils.utils import get_dataset, get_device, get_num_classes
from torch.optim.lr_scheduler import StepLR
import joblib
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

MODEL = 'vgg'
DATASET = 'cifar10'
VERSION = '02'


def train_model(trial):
    # Define hyperparameters to tune
    #batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.1)
    opt = trial.suggest_categorical("optimizer", ['adam', 'sgd'])
    mom = trial.suggest_float('momentum', 0, 0.99)
    wd = trial.suggest_float('weight_decay', 0, 0.99)

    data_dir = os.path.join(os.getcwd(), 'data')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    train_loader, test_loader = get_dataset(
        data_set=DATASET,
        batch_size=batch_size,
        data_loader=True,
        data_path=None
    )

    device = get_device()
    num_classes = get_num_classes(DATASET)

    if MODEL == 'lenet':
        model = CNN_2D(input_shape=(3, 32, 32),
                       num_classes=num_classes,
                       channels=(6, 16),
                       padding=((2, 2), (0, 0)),
                       fc=(784, 84),
                       kernel_size=((5, 5), (5, 5)),
                       bias=False,
                       activation="relu",
                       pooling="avg").to(device)
        epochs = 200

    elif MODEL == 'resnet':
        # Define the ResNet18-like architecture
        #           1    2   3   4  5    6   7    8    9    10   11   12   13   14   15   16   17   18   19    20
        channels = [64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512,
                    512]  # 20 conv layers
        residual = [(1, 3), (3, 5), (6, 8), (8, 10), (11, 13), (13, 15), (16, 18),
                    (18, 20)]  # Residual connections within blocks where channels match
        epochs = 100
        # Instantiate the CNN_2D model
        model = CNN_2D(
            input_shape=(3, 32, 32),  # CIFAR-10 input shape
            num_classes=num_classes,  # 10 classes in CIFAR-10
            channels=tuple(channels),  # Conv layer output channels
            kernel_size=((3, 3),) * 17,  # 3x3 kernels for all conv layers
            padding=((1, 1),) * 17,  # Padding to maintain 32x32 spatial size
            fc=(100,),
            residual=residual,  # Residual connections
            batch_norm=True,  # Use batch normalization as in ResNet
            dropout=False,  # No dropout in standard ResNet
            activation="relu",  # ReLU activation
            pooling="max",  # Average pooling (4x4, stride 4) reduces 32x32 to 8x8
            bias=False,  # No bias in conv layers; FC layers may use bias
            save=False  # No intermediate outputs saved
        ).to(device)

    elif MODEL == 'vgg':
        model = CNN_2D(
            input_shape=(3, 32, 32),  # CIFAR-10: 3 channels, 32x32 images
            num_classes=num_classes,  # 10 classes in CIFAR-10
            channels=(64, 128, 256, 256, 512, 512, 512, 512),  # VGG11 conv layers
            padding=((1, 1),) * 8,  # Padding 1 to maintain spatial size
            fc=(4096, 4096),  # Two FC layers as in VGG11
            kernel_size=((3, 3),) * 8,  # 3x3 filters for all conv layers
            bias=True,  # Bias for FC layers
            residual=[],  # No residual connections in VGG11
            batch_norm=False,  # VGG11 doesn't use batch norm
            dropout=True,  # Dropout as in VGG11
            activation="relu",  # ReLU activation as in VGG11
            pooling="max",  # Max pooling (single layer in base class)
            save=False  # No need to save activations
        ).to(device)
        epochs = 120

    else:
        raise ValueError(f'Model {MODEL} is not supported.')

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wd, momentum=mom)

    scheduler = StepLR(
        optimizer=optimizer,
        step_size=90,
        gamma=0.1
    )

    model.train()
    # Train the model
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            #torch.cuda.synchronize()
            scheduler.step()

    # Evaluate the model on the validation set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


def save_study(study, trial):
    # /lustre07/ -> narval
    # /lustre04/ -> beluga
    # /          -> graham & nibi
    #study_dir = "/scratch/armenta/"
    #if not os.path.exists(study_dir):
    #    os.makedirs(study_dir)
    joblib.dump(study, f"{MODEL}_{DATASET}_hypersearch_{VERSION}.pkl")


if __name__ == '__main__':
    # Create a study object and specify the direction
    storage = JournalStorage(JournalFileBackend(f'{MODEL}_{DATASET}_journal_{VERSION}.log'))
    study = optuna.create_study(direction="maximize",
                                study_name=f"{MODEL}_{DATASET}_{VERSION}",
                                storage=storage,
                                load_if_exists=True)

    study.optimize(train_model,
                   n_trials=100,
                   #n_jobs=torch.cuda.device_count(),
                   callbacks=[save_study])

    # Print the best hyperparameters and accuracy
    print("Best hyperparameters:", study.best_params)
    print("Best accuracy:", study.best_value)
