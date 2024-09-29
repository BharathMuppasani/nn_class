# ## Import Necessary Packages
# Importing essential libraries for data manipulation, neural network creation, optimization, and visualization.
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
import matplotlib.pyplot as plt
from time import time


# ## Data Loading Function
# Function to load and preprocess the MNIST dataset. The dataset is normalized and converted to tensors.
# Returns DataLoaders for both training and validation sets.
def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize the images to [-1, 1] range
    ])
    # Download and load training and validation datasets
    trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
    valset = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
    return trainloader, valloader


# ## Model Creation Function with Variable Hidden Layer Sizes
# Function to create a feedforward neural network with a customizable number of hidden layers.
# The number of hidden layers is controlled by the input 'hidden_sizes' list.
def create_model(hidden_sizes):
    input_size = 784  # MNIST images are 28x28 pixels, flattened to a vector of size 784
    output_size = 10  # There are 10 output classes (digits 0-9)
    
    # Define the network structure starting with input-to-hidden layer, followed by ReLU activations
    layers = [nn.Linear(input_size, hidden_sizes[0]), nn.ReLU()]
    for i in range(len(hidden_sizes) - 1):  # Create intermediate hidden layers
        layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        layers.append(nn.ReLU())
    
    # Add the final output layer and LogSoftmax for the output activation
    layers.append(nn.Linear(hidden_sizes[-1], output_size))
    layers.append(nn.LogSoftmax(dim=1))
    
    # Return the sequential model
    model = nn.Sequential(*layers)
    return model


# ## Training and Evaluation Function
# Function to train and evaluate the model. It computes the training loss and validation loss after every epoch.
def train_and_evaluate_model(model, trainloader, valloader):
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)  # Using SGD optimizer with momentum
    criterion = nn.NLLLoss()  # Negative Log Likelihood loss function
    epochs = 15  # Number of training epochs
    train_losses, val_losses = [], []  # Lists to store training and validation losses

    # Training loop for each epoch
    for e in range(epochs):
        running_loss = 0  # Initialize running loss for training
        for images, labels in trainloader:  # Iterate over training batches
            images = images.view(images.shape[0], -1)  # Flatten MNIST images to a 784-length vector
            optimizer.zero_grad()  # Reset gradients
            output = model(images)  # Forward pass
            loss = criterion(output, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            running_loss += loss.item()  # Accumulate training loss
        
        # Validation loop (done after each epoch)
        val_loss = 0
        with torch.no_grad():  # Disable gradient tracking for validation
            model.eval()  # Set model to evaluation mode
            for images, labels in valloader:
                images = images.view(images.shape[0], -1)  # Flatten validation images
                output = model(images)  # Forward pass
                loss = criterion(output, labels)  # Compute validation loss
                val_loss += loss.item()  # Accumulate validation loss
        
        model.train()  # Switch back to training mode
        train_losses.append(running_loss / len(trainloader))  # Average training loss
        val_losses.append(val_loss / len(valloader))  # Average validation loss
    
    return train_losses, val_losses  # Return the losses for plotting


# Define different hidden layer sizes to experiment with
hidden_sizes_list = [[128, 64], [256, 128], [512, 256], [1024, 512]]
results = {}

# Load the MNIST data
trainloader, valloader = get_data()

# Train models with different hidden layer sizes and collect their losses
for hidden_sizes in hidden_sizes_list:
    model = create_model(hidden_sizes)  # Create model with specific hidden layer sizes
    train_losses, val_losses = train_and_evaluate_model(model, trainloader, valloader)  # Train and evaluate model
    results[str(hidden_sizes)] = (train_losses, val_losses)  # Store the results


# ## Plot Results
# Plot the training and validation losses for different hidden layer sizes over the epochs.
plt.figure(figsize=(14, 8))
for hidden_sizes, (train_losses, val_losses) in results.items():
    plt.plot(train_losses, label=f'Train Loss - {hidden_sizes}')
    plt.plot(val_losses, label=f'Val Loss - {hidden_sizes}', linestyle='--')

plt.title('Training and Validation Loss by Hidden Layer Sizes')  # Title of the plot
plt.xlabel('Epochs')  # X-axis label
plt.ylabel('Loss')  # Y-axis label
plt.legend()  # Show legend
plt.grid(True)  # Enable grid
plt.savefig('results_increase_neurons.png')  # Save the plot as an image
plt.show()  # Display the plot
