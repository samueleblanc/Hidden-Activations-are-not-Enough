import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model_zoo.res_net import ResNet
import joblib
#import certifi
import os
#from optuna.storages import JournalStorage, JournalFileBackend

from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
#os.environ['SSL_CERT_FILE'] = certifi.where()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = 'cuda' if torch.cuda.is_available() else device
print("Device: ", device, flush=True)
num_gpus = torch.cuda.device_count()

def train_model(trial):
    # Define hyperparameters to tune
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.1)
    epochs = trial.suggest_categorical("epochs", [90, 100, 150])
    opt = trial.suggest_categorical("optimizer",['adam','sgd'])
    mom = trial.suggest_float('momentum',0,0.99)
    wd = trial.suggest_float('weight_decay',0,0.99)
    # Define data transforms
    #transform = transforms.Compose([
    #    transforms.RandomCrop(32, padding=4),  # Random crop with padding
    #    transforms.RandomHorizontalFlip(),     # Random horizontal flip (50% chance)
    #    transforms.ToTensor(),                 # Convert PIL Image to tensor
    #    transforms.Normalize(
    #        mean=[0.4914, 0.4822, 0.4465],    # CIFAR-10 dataset-specific mean
    #        std=[0.2023, 0.1994, 0.2010]      # CIFAR-10 dataset-specific std
    #        )
    #])

    data_dir = os.path.join(os.getcwd(), 'data')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    # Load CIFAR10 dataset
    # Load CIFAR10 dataset
    # Define normalization transform
    #normalize = transforms.Normalize(
    #    mean=[0.4914, 0.4822, 0.4465],
    #    std=[0.2470, 0.2435, 0.2616]
    #)

    normalize = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )

    # Define transforms for training and testing
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    trainset = datasets.CIFAR100(data_dir, download=False, train=True, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)

    testset = datasets.CIFAR100(data_dir, download=False, train=False, transform=test_transforms)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)
    #print("Data ready",flush=True)
    #return
    # Initialize VGG model
    model = ResNet(input_shape=(3, 32, 32), num_classes=100, pretrained=False).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wd, momentum=mom)


    model.train()
    # Train the model
    for epoch in range(epochs):
        #print("Epoch: ", epoch)
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

    # Evaluate the model on the validation set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    #print("Accuracy: ", accuracy)
    return accuracy

def save_study(study, trial):
    study_dir = "/lustre07/scratch/armenta"
    if not os.path.exists(study_dir):
        os.makedirs(study_dir)
    joblib.dump(study, "resnet_cifar100_hyperparameter_tuning_study_01.pkl")


if __name__ == '__main__':
    # Create a study object and specify the direction
    storage = JournalStorage(JournalFileBackend('/lustre07/scratch/armenta/resnet_cifar100_journal_01.log'))
    study = optuna.create_study(direction="maximize",
                                study_name="resnet_cifar100_journal_01",
                                storage=storage,
                                load_if_exists=True)

    # Perform hyperparameter tuning using Optuna
    study.optimize(train_model, n_trials=100,
                                n_jobs=num_gpus, callbacks=[save_study])
    #study.optimize(train_model, n_trials=50)

    # Print the best hyperparameters and accuracy
    print("Best hyperparameters:", study.best_params)
    print("Best accuracy:", study.best_value)
