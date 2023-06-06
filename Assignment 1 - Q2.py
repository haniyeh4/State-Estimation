import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Set the parameters
r = 1
theta_deg = 0.5
mean_x = r * np.cos(np.deg2rad(theta_deg))
mean_y = r * np.sin(np.deg2rad(theta_deg))
cov_x = np.array([[0.01, 0], [0, 0.005]])
cov_y = np.zeros((2, 2))

# Generate samples from the distribution
n_samples = 1000
samples = np.random.multivariate_normal([mean_x, mean_y], cov_x, size=n_samples)

# Transform the samples
transformed_samples = np.column_stack([
    samples[:, 0] * np.cos(samples[:, 1] - r * samples[:, 1] * np.sin(samples[:, 1])),
    samples[:, 0] * np.sin(samples[:, 1]) + r * samples[:, 1] * np.cos(samples[:, 1])
])

# Calculate the covariance matrix for transformed samples
theta_rad = np.deg2rad(theta_deg)
cov_y[0, 0] = 9.9996 * (10 ** -3)
cov_y[0, 1] = 4.3631 * (10 ** -3)
cov_y[1, 0] = cov_y[0, 1]
cov_y[1, 1] = 5.0004 * (10 ** -3)

# Calculate eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_y)
angle_rad = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])

# Plot the transformed points and ellipse
fig, ax = plt.subplots()
ax.scatter(transformed_samples[:, 0], transformed_samples[:, 1], marker='.', color='blue')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Transformed Points and Ellipse')
ax.grid(True)

# Set the confidence level and ellipse parameters
confidence_level = 0.95
chi_square_value = np.sqrt(5.991)
width = 2 * np.sqrt(eigenvalues[0]) * chi_square_value
height = 2 * np.sqrt(eigenvalues[1]) * chi_square_value

# Add the ellipse to the plot
ellipse = Ellipse((mean_x, mean_y), width, height, color='blue', angle=np.rad2deg(angle_rad), alpha=0.2)
ax.add_patch(ellipse)

# Show the plot
plt.show()
