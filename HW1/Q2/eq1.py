import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Define the perceptron function
def perceptron(x1, x2):
    return -4.79 * x1 + 5.90 * x2 - 0.93

# Define activation functions
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
    Z = perceptron(X1, X2)
    return X1, X2, Z

# Plotting function
def plot_surface(X1, X2, Y, title, num_points, activation_func):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X1, X2, Y, cmap='viridis')
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    ax.set_zlabel('y')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    # plt.show()
    plt.savefig(f'eq1_{num_points}_{activation_func}.png')

# Sample points and plot for each activation function and number of points
for num_points in [100, 5000, 10000]:
    X1, X2, Z = generate_grid(int(np.sqrt(num_points)))
    plot_surface(X1, X2, sigmoid(Z), f'Sigmoid Activation with {num_points} points', num_points, 'sigmoid')
    plot_surface(X1, X2, hard_limit(Z), f'Hard Limit Activation with {num_points} points', num_points, 'hard_limit')
    plot_surface(X1, X2, radial_basis(Z), f'Radial Basis Function Activation with {num_points} points', num_points, 'radial_basis')