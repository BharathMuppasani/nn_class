import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Define the two-layer neural network function
def two_layer_nn(x1, x2, activation_func):
    # Inputs as a column vector
    x = np.array([x1, x2])
    
    # Weights and biases
    V = np.array([[-2.69, -3.39],
                  [-2.80, -4.56]])
    bv = np.array([-2.21, 4.76])
    W = np.array([-4.91, 4.95])
    bw = -2.28
    
    # Hidden layer calculation
    z = V.T @ x + bv
    if activation_func == 'sigmoid':
        h = sigmoid(z)
    elif activation_func == 'hard_limit':
        h = hard_limit(z)
    elif activation_func == 'radial_basis':
        h = radial_basis(z)
    
    # Output layer calculation
    y = W @ h + bw
    return y

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def hard_limit(z):
    return np.where(z >= 0, 1, 0)

def radial_basis(z):
    return np.exp(-(z**2))

# Generate grid of points in the domain
def generate_grid(num_points):
    x1 = np.linspace(-2, 2, num_points)
    x2 = np.linspace(-2, 2, num_points)
    X1, X2 = np.meshgrid(x1, x2)
    return X1, X2

# Plotting function
def plot_surface(X1, X2, activation_func, num_points):
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = two_layer_nn(X1[i, j], X2[i, j], activation_func)
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X1, X2, Z, cmap='viridis')
    plt.title(f'{activation_func.capitalize()} Activation with {num_points} points')
    plt.xlabel('x1')
    plt.ylabel('x2')
    ax.set_zlabel('y')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    # plt.show()
    plt.savefig(f'eq2_{activation_func}_{num_points}.png')

# Sample points and plot for each activation function and number of points
for num_points in [100, 5000, 10000]:
    X1, X2 = generate_grid(int(np.sqrt(num_points)))
    for activation_func in ['sigmoid', 'hard_limit', 'radial_basis']:
        plot_surface(X1, X2, activation_func, num_points)