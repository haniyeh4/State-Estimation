import numpy as np
import random
import matplotlib.pyplot as plt

source_points = np.array([[1.90659, 2.51737], [2.20896, 1.1542], [2.37878, 2.15422], [1.98784, 1.44557], [2.83467, 3.41243], [9.12775, 8.60163], [4.31247, 5.57856], [6.50957, 5.65667], [3.20486, 2.67803], [6.60663, 3.80709], [8.40191, 3.41115], [2.41345, 5.71343], [1.04413, 5.29942], [3.68784, 3.54342], [1.41243, 2.6001]])
destination_points = np.array([[5.0513, 1.14083], [1.61414, 0.92223], [1.95854, 1.05193], [1.62637, 0.93347], [2.4199, 1.22036], [5.58934, 3.60356], [3.18642, 1.48918], [3.42369, 1.54875], [3.65167, 3.73654], [3.09629, 1.41874], [5.55153, 1.73183], [2.94418, 1.43583], [6.8175, 0.01906], [2.62637, 1.28191], [1.78841, 1.0149]])

def find_inliers(source_points, destination_points, threshold):
    num_points = source_points.shape[0]
    best_inliers = []
    
    for _ in range(1000):  # Repeat for a sufficient number of iterations
        indices = random.sample(range(num_points), 4)  # Randomly select 4 points
        
        # Compute homography using the selected points
        A = []
        for i in indices:
            x, y = source_points[i]
            x_p, y_p = destination_points[i]
            A.append([-x, -y, -1, 0, 0, 0, x * x_p, y * x_p, x_p])
            A.append([0, 0, 0, -x, -y, -1, x * y_p, y * y_p, y_p])
        A = np.array(A)
        
        _, _, V = np.linalg.svd(A)
        H = V[-1, :].reshape(3, 3)
        
        # Compute the re-projection error for each point
        errors = []
        for i in range(num_points):
            x, y = source_points[i]
            x_p, y_p = destination_points[i]
            p = np.array([x, y, 1]).reshape(3, 1)
            p_projected = H @ p
            p_projected /= p_projected[2, 0]
            error = np.sqrt((p_projected[0, 0] - x_p) ** 2 + (p_projected[1, 0] - y_p) ** 2)
            errors.append(error)
        
        # Check if the error is below the threshold and update the best set of inliers
        inliers = [i for i, error in enumerate(errors) if error < threshold]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
    
    return best_inliers

inliers = find_inliers(source_points, destination_points, threshold=0.005)

def compute_homography(source_points, destination_points, inliers):
    num_inliers = len(inliers)
    A = np.zeros((2 * num_inliers, 9))
    
    for i, inlier in enumerate(inliers):
        x, y = source_points[inlier]
        x_p, y_p = destination_points[inlier]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, x * x_p, y * x_p, x_p]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, x * y_p, y * y_p, y_p]
    
    _, _, V = np.linalg.svd(A)
    homography_matrix = V[-1, :].reshape(3, 3)
    
    return homography_matrix

homography = compute_homography(source_points, destination_points, inliers)


def plot_scatter(source_points, destination_points, inliers):
    plt.scatter(source_points[:, 0], source_points[:, 1], marker='x', label='Outliers')
    plt.scatter(destination_points[:, 0], destination_points[:, 1], marker='x')
    plt.scatter(source_points[inliers, 0], source_points[inliers, 1], marker='o', label='Inliers')
    plt.scatter(destination_points[inliers, 0], destination_points[inliers, 1], marker='o')
    
    # Connect inliers with lines
    for i in inliers:
        plt.plot([source_points[i, 0], destination_points[i, 0]], [source_points[i, 1], destination_points[i, 1]], 'k-')
    
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot of Inliers and Outliers')
    plt.show()

plot_scatter(source_points, destination_points, inliers)
