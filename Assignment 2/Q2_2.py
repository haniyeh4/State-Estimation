import pygame
from pygame.locals import *
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import math

initial_pixel = np.array([500, 500])
scaling_factor = 1000
display_width = 1000
display_height = 1000

# Initial state of the robot
robot_pose = np.array([[0], [0], [0]])
ground_truth_pose = np.array([[0], [0], [0]])
range_bearing_measurement = np.array([[0], [0], [0]])

# Range/bearing pose
range_bearing_pose = np.array([[0], [0], [0]])

mean_vector = np.zeros((2))
covariance_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]).astype(float)

poses = [(initial_pixel[0], initial_pixel[1])]
ground_truth_poses = [(initial_pixel[0], initial_pixel[1])]
measurement_poses = [(initial_pixel[0], initial_pixel[1])]
kalman_filtered_poses = [(initial_pixel[0], initial_pixel[1])]

center = np.array([[10], [10]])

def cartesian_to_polar(point):
    rho = np.sqrt((point[0, 0] - center[0, 0])**2 + (point[1, 0] - center[1, 0])**2)
    theta = np.arctan2(point[1, 0] - center[1, 0], point[0, 0] - center[0, 0])
    polar_point = np.array([[rho], [theta]])
    return polar_point

def convert_to_pixel(position, scaling=1):
    pixel_location = position * scaling_factor / scaling + initial_pixel
    return [int(pixel_location[0]), int(pixel_location[1])]

def update_covariance_matrix():
    global covariance_matrix
    T = 1 / 8
    A = np.eye(3)
    Rt = np.array([[0.01, 0, 0], [0, 0.1, 0], [0, 0, 0]]) * T
    covariance_matrix = np.matmul(A, np.matmul(covariance_matrix, np.transpose(A))) + Rt

def apply_kalman_filter():
    global covariance_matrix, robot_pose

    # Jacobian
    H = np.zeros((2, 3))

    # Denominator of Jacobian
    denominator = np.sqrt((robot_pose[0, 0] - center[0, 0])**2 + (robot_pose[1, 0] - center[1, 0])**2)

    H[0, 0] = (robot_pose[0, 0] - center[0, 0]) / denominator
    H[0, 1] = (robot_pose[1, 0] - center[1, 0]) / denominator

    H[1, 0] = -1 * (robot_pose[1, 0] - center[1, 0]) / (denominator**2)
    H[1, 1] = (robot_pose[0, 0] - center[0, 0]) / (denominator**2)
    H[1, 2] = -1

    z = range_bearing_measurement

    Qt = np.array([[0.1, 0], [0, 0.01]])
    temp = np.matmul(H, np.matmul(covariance_matrix, np.transpose(H))) + Qt

    k = np.matmul(np.matmul(covariance_matrix, np.transpose(H)), np.linalg.inv(temp))
    k[np.isnan(k)] = 0

    temp2 = z - np.matmul(H, robot_pose)
    robot_pose = robot_pose + np.matmul(k, temp2)
    covariance_matrix = np.matmul(np.eye(3) - np.matmul(k, H), covariance_matrix)

def update_belief():
    global robot_pose
    global ground_truth_pose
    global range_bearing_pose


    T = 1 / 8
    r = 0.1
    L = 0.3

    if (math.dist([robot_pose[0, 0], robot_pose[1, 0]], center) < 9):
        ur = 0.1
        ul = 0
    if (math.dist([robot_pose[0, 0], robot_pose[1, 0]], center) > 10):
        ur = 0
        ul = 0.1

    A = np.eye(3)

    B = np.array([[r * T * np.cos(robot_pose[2, 0]), 0], [r * T * np.sin(robot_pose[2, 0]), 0], [0, T * r / L]])
    B_gt = np.array([[r * T * np.cos(robot_pose[2, 0]), 0], [r * T * np.sin(robot_pose[2, 0]), 0], [0, T * r / L]])

    u = np.array([[(ur + ul) / 2], [ur - ul]])

    robot_pose = np.matmul(A, robot_pose) + np.matmul(B, u) + T * np.array([[np.random.normal(0, 0.01)], [np.random.normal(0, 0.1)], [0]])
    ground_truth_pose = np.matmul(A, ground_truth_pose) + np.matmul(B_gt, u)

    range_bearing_pose = cartesian_to_polar(robot_pose)

def update_observation():
    global range_bearing_measurement, robot_pose, ground_truth_pose

    noise = np.array([[np.random.normal(0, 0.001)], [np.random.normal(0, 0.001)]])
    observation = cartesian_to_polar(ground_truth_pose) + noise
    range_bearing_measurement = observation

if __name__ == "__main__":
    pygame.init()
    pygame.display

    gameDisplay = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption("Python Simulation")

    white = (255, 255, 255)
    blue = (0, 0, 255)
    red = (255, 0, 0)
    green = (0, 255, 0)
    black = (0, 0, 0)

    gameDisplay.fill(white)

    clock = pygame.time.Clock()
    crashed = False
    t = 0

    while not crashed:
        update_belief()
        if t % 8 == 0:
            update_observation()
            apply_kalman_filter()

        pose_ = np.array([center[0, 0] + [range_bearing_pose[0, 0] * np.cos(range_bearing_pose[1, 0])], center[1, 0] + [range_bearing_pose[0, 0] * np.sin(range_bearing_pose[1, 0])]])
        poses.append(convert_to_pixel(np.array([pose_[0, 0], pose_[1, 0]])))
        pygame.draw.lines(gameDisplay, red, False, poses, 5)

        ground_truth_poses.append(convert_to_pixel(np.array([ground_truth_pose[0, 0], ground_truth_pose[1, 0]])))
        pygame.draw.lines(gameDisplay, blue, False, ground_truth_poses, 5)

        pose_measure = np.array([center[0, 0] + [range_bearing_measurement[0, 0] * np.cos(range_bearing_measurement[1, 0])], center[1, 0] + [range_bearing_measurement[0, 0] * np.sin(range_bearing_measurement[1, 0])]])
        measurement_poses.append(convert_to_pixel(np.array([pose_measure[0, 0], pose_measure[1, 0]])))
        pygame.draw.lines(gameDisplay, green, False, measurement_poses, 5)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True

        pygame.display.update()
        gameDisplay.fill(white)
        clock.tick(8)
        t += 1
