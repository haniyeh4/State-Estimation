import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Set the parameters
r = 1
theta_deg = 0.5
mean_x = r * np.cos(np.deg2rad(theta_deg))
mean_y = r * np.sin(np.deg2rad(theta_deg))
cov_y_1 = np.zeros((2, 2))
cov_y_2 = np.zeros((2, 2))
cov_y_3 = np.zeros((2, 2))
cov_y_4 = np.zeros((2, 2))

# Define different values for cov_x
cov_x_1 = np.array([[0.01, 0], [0, 0.005]])
cov_x_2 = np.array([[0.02, 0], [0, 0.1]])
cov_x_3 = np.array([[0.03, 0], [0, 0.5]])
cov_x_4 = np.array([[0.04, 0], [0, 1]])

# Calculate the covariance matrix for transformed samples
theta_rad = np.deg2rad(theta_deg)
G = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                   [np.sin(theta_rad), np.cos(theta_rad)]])
GT= np.transpose(G)
cov_y_1 = np.dot(np.dot(G, cov_x_1), GT)
cov_y_2 = np.matmul(np.matmul(G, cov_x_2), GT)
cov_y_3 = np.matmul(np.matmul(G, cov_x_3), GT)
cov_y_4 = np.matmul(np.matmul(G, cov_x_4), GT)


# Generate samples from the distribution
n_samples = 1000
samples_1 = np.random.multivariate_normal([mean_x, mean_y], cov_y_1, size=n_samples)
samples_2 = np.random.multivariate_normal([mean_x, mean_y], cov_y_2, size=n_samples)
samples_3 = np.random.multivariate_normal([mean_x, mean_y], cov_y_3, size=n_samples)
samples_4 = np.random.multivariate_normal([mean_x, mean_y], cov_y_4, size=n_samples)


# Calculate eigenvalues and eigenvectors of the covariance matrix
eigenvalues1, eigenvectors1 = np.linalg.eig(cov_y_1)
angle_rad1 = np.arctan2(eigenvectors1[0, 1], eigenvectors1[0, 0])

eigenvalues2, eigenvectors2 = np.linalg.eig(cov_y_2)
angle_rad2 = np.arctan2(eigenvectors2[0, 1], eigenvectors2[0, 0])

eigenvalues3, eigenvectors3 = np.linalg.eig(cov_y_3)
angle_rad3 = np.arctan2(eigenvectors3[0, 1], eigenvectors3[0, 0])

eigenvalues4, eigenvectors4 = np.linalg.eig(cov_y_4)
angle_rad4 = np.arctan2(eigenvectors4[0, 1], eigenvectors4[0, 0])

# Set the confidence level and ellipse parameters
confidence_level = 0.95
chi_square_value = np.sqrt(5.991)
width1 = 2 * np.sqrt(eigenvalues1[0]) * chi_square_value
height1 = 2 * np.sqrt(eigenvalues1[1]) * chi_square_value

width2 = 2 * np.sqrt(eigenvalues2[0]) * chi_square_value
height2 = 2 * np.sqrt(eigenvalues2[1]) * chi_square_value

width3 = 2 * np.sqrt(eigenvalues3[0]) * chi_square_value
height3 = 2 * np.sqrt(eigenvalues3[1]) * chi_square_value

width4 = 2 * np.sqrt(eigenvalues4[0]) * chi_square_value
height4 = 2 * np.sqrt(eigenvalues4[1]) * chi_square_value

# Create a figure and subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Plot the first sub-figure
ax = axes[0, 0]
ax.scatter(samples_1[:, 0], samples_1[:, 1], marker='.', color='blue')
ax.set_title('Value 1')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim(0.3,1.6)
ax.set_ylim(-3,3)
ax.set_title('Transformed Points and Ellipse for eq 6')
ax.grid(True)


# Add the ellipse to the plot
ellipse = Ellipse((mean_x, mean_y), width1, height1, color='blue', angle=angle_rad1, alpha=0.2)
ax.add_patch(ellipse)

# Plot the second sub-figure
ax = axes[0, 1]
ax.scatter(samples_2[:, 0], samples_2[:, 1], marker='.', color='blue')
ax.set_title('Value 2')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim(0.3,1.6)
ax.set_ylim(-3,3)
ax.set_title('Transformed Points and Ellipse for eq 7')
ax.grid(True)

# Add the ellipse to the plot
ellipse = Ellipse((mean_x, mean_y), width2, height2, color='blue', angle=angle_rad2, alpha=0.2)
ax.add_patch(ellipse)

# Plot the third sub-figure
ax = axes[1, 0]
ax.scatter(samples_3[:, 0], samples_3[:, 1], marker='.', color='blue')
ax.set_title('Value 3')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim(0.3,1.6)
ax.set_ylim(-3,3)
ax.set_title('Transformed Points and Ellipse for eq 8')
ax.grid(True)

# Add the ellipse to the plot
ellipse = Ellipse((mean_x, mean_y), width3, height3, color='blue', angle=angle_rad3, alpha=0.2)
ax.add_patch(ellipse)

# Plot the fourth sub-figure
ax = axes[1, 1]
ax.scatter(samples_4[:, 0], samples_4[:, 1], marker='.', color='blue')
ax.set_title('Value 4')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim(0.3,1.6)
ax.set_ylim(-3,3)
ax.set_title('Transformed Points and Ellipse for eq 9')
ax.grid(True)

# Add the ellipse to the plot
ellipse = Ellipse((mean_x, mean_y), width4, height4, color='blue', angle=angle_rad4, alpha=0.2)
ax.add_patch(ellipse)

# Adjust the spacing between subplots
plt.tight_layout()

# Display the figure
plt.show()
