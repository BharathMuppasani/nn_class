import numpy as np
import matplotlib.pyplot as plt

# Seed for reproducibility
np.random.seed(42)

# Step 1: Prepare the data
X = np.array([-1, -0.9, -0.8, -0.7, -0.6, -0.5,
              -0.4, -0.3, -0.2, -0.1, 0, 0.1,
              0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
              0.8, 0.9, 1])  # Shape: (21,)
Y = np.array([-0.96, -0.577, -0.073, 0.377, 0.641, 0.66,
              0.461, 0.134, -0.201, -0.434, -0.5, -0.393,
              -0.165, 0.099, 0.307, 0.396, 0.345, 0.182,
              -0.031, -0.219, -0.321])  # Shape: (21,)

# Reshape X and Y for matrix operations
X = X.reshape(-1, 1)  # Shape: (21, 1)
Y = Y.reshape(-1, 1)  # Shape: (21, 1)

# Step 2: Define network architecture
input_size = 1        # One input feature
hidden_size = 10      # Number of neurons in hidden layer
output_size = 1       # One output

# Step 3: Initialize weights
# Weights from input to hidden layer
W1 = np.random.uniform(-0.5, 0.5, (hidden_size, input_size + 1))  # Shape: (10, 2)
# Weights from hidden to output layer
W2 = np.random.uniform(-0.5, 0.5, (output_size, hidden_size + 1))  # Shape: (1, 11)

# Step 4: Define activation functions and derivatives
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    # z is the output of the sigmoid function
    return z * (1 - z)

def linear(z):
    return z

def linear_derivative(z):
    return np.ones_like(z)

# Step 5: Training parameters
epochs = 8000
learning_rate = 0.1

# To store training error over epochs
error_history = []

# For plotting after specific epochs
epochs_to_plot = [10, 100, 200, 400, 1000, 2000, 5000, 8000]
predictions_history = {}

# Training loop
for epoch in range(1, epochs + 1):
    total_error = 0
    for i in range(len(X)):
        # Step 6: Forward pass

        # Input layer to hidden layer
        x_i = X[i]
        y_i = Y[i]

        # Add bias to input
        x_bias = np.concatenate(([1], x_i))  # Shape: (2,)

        # Compute net input to hidden layer
        z1 = np.dot(W1, x_bias)  # Shape: (10,)
        # Activation of hidden layer
        h = sigmoid(z1)  # Shape: (10,)

        # Add bias to hidden layer activation
        h_bias = np.concatenate(([1], h))  # Shape: (11,)

        # Hidden layer to output layer
        z2 = np.dot(W2, h_bias)  # Shape: (1,)
        # Output activation
        y_pred = linear(z2)  # Shape: (1,)

        # Step 7: Compute error
        error = y_pred - y_i  # Shape: (1,)
        total_error += 0.5 * error ** 2

        # Step 8: Backward pass

        # Output layer error signal
        delta2 = error * linear_derivative(y_pred)  # Shape: (1,)

        # Gradient w.r.t W2
        dW2 = delta2 * h_bias.reshape(1, -1)  # Shape: (1, 11)

        # Hidden layer error signal
        delta1 = (W2[:, 1:].T @ delta2) * sigmoid_derivative(h)  # Shape: (10,)

        # Gradient w.r.t W1
        dW1 = delta1.reshape(-1, 1) @ x_bias.reshape(1, -1)  # Shape: (10, 2)

        # Step 9: Update weights
        W2 -= learning_rate * dW2
        W1 -= learning_rate * dW1

    # Record the total error for this epoch
    error_history.append(total_error[0])

    # Store predictions for specified epochs
    if epoch in epochs_to_plot:
        # Compute predictions for all data points
        predictions = []
        for i in range(len(X)):
            x_i = X[i]
            # Forward pass
            x_bias = np.concatenate(([1], x_i))  # Shape: (2,)
            z1 = np.dot(W1, x_bias)  # Shape: (10,)
            h = sigmoid(z1)  # Shape: (10,)
            h_bias = np.concatenate(([1], h))  # Shape: (11,)
            z2 = np.dot(W2, h_bias)  # Shape: (1,)
            y_pred = linear(z2)  # Shape: (1,)
            predictions.append(y_pred[0])
        predictions_history[epoch] = np.array(predictions)

    # Print training progress
    # Uncomment the next line to see progress
    # print(f"Epoch {epoch}/{epochs}, Error: {total_error[0]:.6f}")

# Step 10: Plot training error vs. epoch number
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), error_history, marker='o')
plt.title('Training Error vs Epoch Number')
plt.xlabel('Epoch')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.grid(True)
plt.savefig('training_error_function_approx.png')
plt.show()

# Step 11: Plot actual vs. approximate functions after specified epochs
for epoch in epochs_to_plot:
    predictions = predictions_history[epoch]
    plt.figure(figsize=(10, 6))
    plt.plot(X.flatten(), Y.flatten(), 'ro', label='Actual Function')
    plt.plot(X.flatten(), predictions, 'b-', label='NN Approximation')
    plt.title(f'Function Approximation after {epoch} Epochs')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'function_approx_epoch_{epoch}.png')
    plt.show()
