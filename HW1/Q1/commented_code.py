# ## Import Necessary Packages
# Import essential libraries for data manipulation, deep learning, visualization, and evaluation.

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
from time import time
from sklearn.metrics import confusion_matrix
import seaborn as sns


# ## Download The Dataset & Define The Transforms
# Function to load and transform the MNIST dataset for training and validation.

def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
    valset = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
    return trainloader, valloader


# ## Exploring The Data
# Display the first batch of images and visualize 60 sample images from the training set.

def explore_data(trainloader):
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')

    figure = plt.figure()
    num_of_images = 60
    for index in range(1, num_of_images + 1):
        plt.subplot(6, 10, index)
        plt.axis('off')
        plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
    plt.show()


# ## Defining The Neural Network
# Creates a simple feed-forward neural network with two hidden layers and an output layer for classification.

def create_model():
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10
    model = nn.Sequential(
        nn.Linear(input_size, hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[1], output_size),
        nn.LogSoftmax(dim=1)
    )
    return model


# ## Training the Model
# Trains the model on the training dataset and evaluates on the validation dataset across 15 epochs.

def train_model(model, trainloader, valloader):
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    criterion = nn.NLLLoss()
    time0 = time()
    epochs = 15
    train_losses, val_losses = []

    for e in range(epochs):
        train_loss = 0
        val_loss = 0
        
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        with torch.no_grad():
            for images, labels in valloader:
                images = images.view(images.shape[0], -1)
                output = model(images)
                loss = criterion(output, labels)
                val_loss += loss.item()
        
        train_losses.append(train_loss / len(trainloader))
        val_losses.append(val_loss / len(valloader))
        print(f"Epoch {e+1}/{epochs} - Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}")

    plot_losses(train_losses, val_losses)


# ## Plot Training and Validation Losses
# Plots the training and validation loss across all epochs.

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('n_losses.png')


# ## Model Evaluation
# Evaluates the trained model on the validation dataset and prints the accuracy.

def evaluate_model(model, valloader):
    correct_count, all_count = 0, 0
    for images, labels in valloader:
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            with torch.no_grad():
                logps = model(img)
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if (true_label == pred_label):
                correct_count += 1
            all_count += 1

    print("Number Of Images Tested =", all_count)
    print("Model Accuracy =", (correct_count/all_count))


# ## Plot Confusion Matrix
# Generates and displays a confusion matrix for the model's predictions on the validation set.

def plot_confusion_matrix(model, valloader):
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels in valloader:
            images = images.view(images.shape[0], -1)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('n_confusion_matrix.png')


# ## Main Execution
# Executes the full pipeline: loading data, training the model, and evaluating it.

if __name__ == "__main__":
    trainloader, valloader = get_data()
    explore_data(trainloader)
    model = create_model()
    train_model(model, trainloader, valloader)
    evaluate_model(model, valloader)
    plot_confusion_matrix(model, valloader)
