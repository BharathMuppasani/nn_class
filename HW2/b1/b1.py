import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Seed for reproducibility
np.random.seed(42)

# Step 1: Define the input data X and the target outputs Y
X = np.array([
    [0.1, 1.2],
    [0.7, 1.8],
    [0.8, 1.6],
    [0.8, 0.6],
    [1.0, 0.8],
    [0.3, 0.5],
    [0.0, 0.2],
    [-0.3, 0.8],
    [-0.5, -1.5],
    [-1.5, -1.3]
]).T  # Shape: (2, 10)

# Target outputs
Y = np.array([
    [1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
])  # Shape: (2, 10)

# Step 2: Add bias term to input data
bias = np.ones((1, X.shape[1]))  # Shape: (1, 10)
X_bias = np.vstack([bias, X])     # Shape: (3, 10)

# Step 3: Initialize weights randomly
input_size = X_bias.shape[0]  # 3 inputs (including bias)
output_size = Y.shape[0]      # 2 outputs
W = np.random.uniform(-0.5, 0.5, (output_size, input_size))  # Shape: (2, 3)

# Step 4: Define activation function and its derivative
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    # z is the output of the sigmoid function
    return z * (1 - z)

# Step 5: Training parameters
epochs = 10000
learning_rate = 0.1

# To store training error over epochs
error_history = []

# For plotting decision boundaries after specified epochs
epochs_to_plot = [3, 10, 100, 1000, 10000]
W_history = {}

# Step 6: Training loop
for epoch in range(1, epochs + 1):
    # Forward pass
    Z = np.dot(W, X_bias)        # Net input: Shape (2, 10)
    Y_pred = sigmoid(Z)          # Output: Shape (2, 10)

    # Compute error
    error = Y_pred - Y           # Shape: (2, 10)
    SSE = 0.5 * np.sum(error ** 2)
    error_history.append(SSE)

    # Backward pass
    delta = error * sigmoid_derivative(Y_pred)  # Shape: (2, 10)

    # Gradient calculation
    dW = np.dot(delta, X_bias.T) / X_bias.shape[1]  # Averaged over samples

    # Update weights
    W -= learning_rate * dW

    # Store weights for specified epochs
    if epoch in epochs_to_plot:
        W_history[epoch] = W.copy()

    # Print training progress
    # print(f"Epoch {epoch}/{epochs}, Error: {SSE:.4f}")

# Step 7: Plot training error vs. epoch number and save the plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), error_history, marker='o')
plt.title('Training Error vs Epoch Number')
plt.xlabel('Epoch')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.grid(True)
plt.savefig('training_error.png')  # Save the plot as 'training_error.png'
plt.show()

# Step 8: Plot decision boundaries and data points after specified epochs
# Define the function to plot decision boundaries
def plot_decision_boundary(W, epoch):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.01),
        np.arange(y_min, y_max, 0.01)
    )

    # Prepare the grid input
    grid_bias = np.ones((xx.ravel().shape[0], 1))
    grid_points = np.vstack([grid_bias.T, xx.ravel(), yy.ravel()])

    # Compute outputs over the grid
    Z = np.dot(W, grid_points)
    outputs = sigmoid(Z)
    # Decide class labels based on thresholds (0.5)
    labels = (outputs > 0.5).astype(int)


    # Combine the two outputs to get class labels (from binary codes)
    class_labels = labels[0, :] * 2 + labels[1, :]  # Unique label for each class

    # Plotting
    plt.figure(figsize=(10, 6))
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFFFAA'])
    cmap_bold = ['r', 'g', 'b', 'y']

    plt.contourf(xx, yy, class_labels.reshape(xx.shape), cmap=cmap_light, alpha=0.5)

    # Plot data points
    for idx, (x, y) in enumerate(zip(X.T, Y.T)):
        # Determine class label from target output
        class_label = int(y[0]) * 2 + int(y[1])
        plt.scatter(
            x[0], x[1],
            color=cmap_bold[class_label],
            edgecolor='k',
            s=100,
            label=f'Group {class_label + 1}' if epoch == epochs_to_plot[0] and idx == 0 else ""
        )

    plt.title(f'Decision Boundary and Data Points after {epoch} Epochs')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'decision_boundary_epoch_{epoch}.png')  # Save the plot
    plt.show()

# Plot decision boundaries for specified epochs and save the plots
for epoch in epochs_to_plot:
    W_epoch = W_history[epoch]
    plot_decision_boundary(W_epoch, epoch)
