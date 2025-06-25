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
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
#os.environ['SSL_CERT_FILE'] = certifi.where()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = 'cuda' if torch.cuda.is_available() else device
print("Device: ", device, flush=True)
num_gpus = torch.cuda.device_count()

def train_model(trial):
    # Define hyperparameters to tune
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1.5)
    epochs = trial.suggest_categorical("epochs", [70, 80, 90])

    # Define data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_dir = os.path.join(os.getcwd(), 'data')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    # Load CIFAR10 dataset
    trainset = datasets.CIFAR10(data_dir, download=False, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)

    testset = datasets.CIFAR10(data_dir, download=False, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)
    #print("Data ready",flush=True)
    #return
    # Initialize VGG model
    model = VGG(input_shape=(3, 32, 32), num_classes=10, pretrained=False).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
    print("Accuracy: ", accuracy)
    return accuracy

def save_study(study, trial):
    study_dir = "/lustre04/scratch/armenta"
    if not os.path.exists(study_dir):
        os.makedirs(study_dir)
    joblib.dump(study, "vgg_hyperparameter_tuning_study_01.pkl")


if __name__ == '__main__':
    # Create a study object and specify the direction
    storage = JournalStorage(JournalFileBackend('/lustre04/scratch/armenta/vgg_journal_01.log'))
    study = optuna.create_study(direction="maximize",
                                study_name="vgg_journal_01",
                                storage=storage,
                                load_if_exists=True)

    # Perform hyperparameter tuning using Optuna
    study.optimize(train_model, n_trials=50,
                                n_jobs=num_gpus, callbacks=[save_study])
    #study.optimize(train_model, n_trials=50)

    # Print the best hyperparameters and accuracy
    print("Best hyperparameters:", study.best_params)
    print("Best accuracy:", study.best_value)
