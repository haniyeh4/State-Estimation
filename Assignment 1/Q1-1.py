import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
# Define parameters
np.random.seed(42)
x = np.array([1, np.radians(0.5)])  # Convert degrees to radians
covariances = [
    np.array([[0.01, 0], [0, 0.005]]),
    np.array([[0.01, 0], [0, 0.1]]),
    np.array([[0.01, 0], [0, 0.5]]),
    np.array([[0.01, 0], [0, 1]])
]
# Create a figure with subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Uncertainty Ellipses for Different Covariances')
# Loop through each covariance matrix
for i, cov in enumerate(covariances):
    # Monte Carlo simulation
    num_samples = 1000
    samples = np.random.multivariate_normal(x, cov, num_samples)
    samples_cartesian = np.column_stack((samples[:, 0] * np.cos(samples[:, 1]),
                                         samples[:, 0] * np.sin(samples[:, 1])))
    # Create scatter plot on the current subplot
    ax = axs[i // 2, i % 2]
    ax.scatter(samples_cartesian[:, 0], samples_cartesian[:, 1], s=10, label='Simulated Points')
    # Calculate and overlay uncertainty ellipse
    s = np.sqrt(5.991)
    eigvals, eigvecs = np.linalg.eig(cov)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    ellipse = Ellipse(x, 2 * s * np.sqrt(eigvals[0]), 2 * s * np.sqrt(eigvals[1]),
                      angle=angle, edgecolor='r', facecolor='none', label='Uncertainty Ellipse')
    ax.add_patch(ellipse)
    # Add labels, legend, etc.
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])
    ax.set_title(f'Covariance Matrix for equation {i+6}')
    ax.legend()
    ax.grid()
# Adjust layout and show the plot
plt.tight_layout()
plt.show()
