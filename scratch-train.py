import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from model_zoo.cnn import CNN_2D
from utils.utils import get_dataset, get_device, get_num_classes, get_input_shape
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ExponentialLR, MultiStepLR, CyclicLR
import joblib
import os
from argparse import ArgumentParser, Namespace


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from knowledgematrix.models.alexnet import AlexNet
from knowledgematrix.models.resnet18 import ResNet18
from knowledgematrix.models.vgg11 import VGG11

def parse_args() -> Namespace:
    """
        Return:
            The parsed arguments.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--temp_dir",
        type = str,
        default = '/',
        help = "Temporary path on compute node where the dataset is saved."
    )
    parser.add_argument(
        "--model",
        type = str,
        default = 'resnet',
        help = "Temporary path on compute node where the dataset is saved."
    )
    parser.add_argument(
        "--dataset",
        type = str,
        default = 'cifar10',
        help = "Temporary path on compute node where the dataset is saved."
    )
    parser.add_argument(
        "--version",
        type = str,
        default = '01',
        help = "Temporary path on compute node where the dataset is saved."
    )
    return parser.parse_args()


def train_model(trial):
    # Define hyperparameters to tune
    trial_number = trial.number
    gpu_count = torch.cuda.device_count()
    print(f"Trial {trial_number}: Using {gpu_count} GPUs, assigned to {get_device(trial_number, gpu_count)}")
    args = parse_args()
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512, 1024])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1.0, log=True)
    opt = trial.suggest_categorical("optimizer", ['adam', 'sgd'])
    mom = trial.suggest_float('momentum', 0.0, 0.99) if opt == 'sgd' else 0.0
    wd = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    sched = trial.suggest_categorical("scheduler", ['step', 'cosine', 'exp', 'multi', 'cyclic'])

    data_dir = os.path.join(os.getcwd(), 'data')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    train_loader, test_loader = get_dataset(
        data_set=args.dataset,
        batch_size=batch_size,
        data_loader=True,
        data_path=None
    )

    device = get_device()
    num_classes = get_num_classes(args.dataset)
    input_shape = get_input_shape(args.dataset)

    if args.model == 'alexnet':
        model = AlexNet(input_shape, num_classes).to(device)

    elif args.model == 'resnet':
        model = ResNet18(input_shape, num_classes).to(device)

    elif args.model == 'vgg':
        model = VGG11(input_shape, num_classes).to(device)

    else:
        raise ValueError(f'Model {args.model} is not supported.')

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wd, momentum=mom)

    if sched == 'step':
        scheduler = StepLR(optimizer=optimizer, step_size=30, gamma=0.1)

    elif sched == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=120)

    elif sched == 'exp':
        scheduler = ExponentialLR(optimizer, gamma=0.95)

    elif sched == 'multi':
        scheduler = MultiStepLR(optimizer, milestones=[60, 90], gamma=0.1)

    else:
        scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=1)


    model.train()
    # Train the model
    for epoch in range(120):
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
            if sched == 'cyclic':
                scheduler.step()

        if sched != 'cyclic':
            scheduler.step()

    model.eval()
    test_loss = 0.0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            total += images.size(0)

    test_loss = test_loss / total
    return test_loss


def save_study(study, trial):
    # /lustre07/ -> narval
    # /lustre04/ -> beluga
    # /          -> graham & nibi
    study_dir = "/scratch/armenta/"
    if not os.path.exists(study_dir):
        os.makedirs(study_dir)
    args = parse_args()
    joblib.dump(study, f"{args.model}_{args.dataset}_{args.version}.pkl")


if __name__ == '__main__':
    # Create a study object and specify the direction
    args = parse_args()
    storage = JournalStorage(JournalFileBackend(f'{args.model}_{args.dataset}_{args.version}.log'))
    study = optuna.create_study(direction="minimize",
                                study_name=f"{args.model}_{args.dataset}_{args.version}_",
                                storage=storage,
                                load_if_exists=True)

    study.optimize(train_model,
                   n_trials=100,
                   #n_jobs=torch.cuda.device_count(),
                   callbacks=[save_study])

    # Print the best hyperparameters and accuracy
    print("Best hyperparameters:", study.best_params)
    print("Best accuracy:", study.best_value)
