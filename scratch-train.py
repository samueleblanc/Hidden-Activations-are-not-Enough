import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model_zoo.vgg import VGG
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

def train_model(trial):
    # Define hyperparameters to tune
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1)
    epochs = trial.suggest_categorical("epochs", [40, 50, 60])

    # Define data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_dir = os.path.join(os.getcwd(), 'data')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    # Load CIFAR10 dataset
    trainset = datasets.CIFAR10(data_dir, download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = datasets.CIFAR10(data_dir, download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    #print("Data ready",flush=True)
    #return
    # Initialize VGG model
    model = VGG(input_shape=(3, 32, 32), num_classes=10, pretrained=False).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate the model on the validation set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print("Accuracy: ", accuracy)
    return accuracy

def save_study(study, trial):
    joblib.dump(study, "vgg_hyperparameter_tuning_study.pkl")


if __name__ == '__main__':
    # Create a study object and specify the direction
    storage = JournalStorage(JournalFileBackend('/lustre07/scratch/armenta/vgg_journal.log'))
    study = optuna.create_study(direction="maximize",
                                study_name="vgg_journal",
                                storage=storage,
                                load_if_exists=True)

    # Perform hyperparameter tuning using Optuna
    study.optimize(train_model, n_trials=50,
                                n_jobs=2, callbacks=[save_study])
    #study.optimize(train_model, n_trials=50)

    # Print the best hyperparameters and accuracy
    print("Best hyperparameters:", study.best_params)
    print("Best accuracy:", study.best_value)
