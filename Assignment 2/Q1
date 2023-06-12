import pygame
import numpy as np
import matplotlib.pyplot as plt

pygame.init()

display_width = 1000
display_height = 1000

gameDisplay = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption('Robot Trajectory')

black = (0, 0, 0)
white = (255, 255, 255)
blue = (0, 0, 255)
red = (255, 0, 0)
green = (0, 255, 0)

clock = pygame.time.Clock()
crashed = False
points = []
points.append([0, 0])
points_measure = []
points_gt = []
points_gt.append([0, 0])
points_measure.append([0, 0])
measured_positions_x = []
measured_positions_y = []
predicted_positions_x = []
predicted_positions_y = []
cov_width = []
cov_height = []
position = np.array([[0], [0]])
position_measure = position
p_0 = np.array([[0, 0], [0, 0]])
teta = -1
point_prev = [[0], [0]]


def draw_robot(x, y):
    pygame.draw.rect(gameDisplay, black, (x, y, 10, 10))


def estimate_pose(position):
    global position_new_true
    A = np.array([[1, 0], [0, 1]])
    r = 0.1
    delta_t = 1 / 8
    B = np.array([[r / 2 * delta_t, r / 2 * delta_t], [r / 2 * delta_t, r / 2 * delta_t]])
    u = np.array([[1], [1]])

    position_new = np.matmul(A, position) + np.matmul(B, u) + np.array([[np.random.normal(0, 0.1)], [np.random.normal(0, 0.15)]]) * delta_t
    position_new_true = np.matmul(A, position) + np.matmul(B, u)

    return position_new


def update():
    global p_0
    delta_t = 1 / 8
    A = np.array([[1, 0], [0, 1]])
    temp = np.matmul(A, p_0)
    R = np.array([[0.1, 0], [0, 0.15]]) * delta_t
    p_new = np.matmul(temp, A.transpose()) + R
    p_0 = p_new


def measurement():
    global position_measure
    C = np.array([[1, 0], [0, 2]])

    Z = np.matmul(C, position_new_true) + np.array([[np.random.normal(0, 0.05)], [np.random.normal(0, 0.075)]])

    position_measure = Z


def correction():
    C = np.array([[1, 0], [0, 2]])
    Q = np.array([[0.05, 0], [0, 0.075]])
    temp1 = np.matmul(p_0, C.transpose())
    temp2 = np.matmul(np.matmul(C, p_0), C.transpose()) + Q

    K = np.matmul(p_0, C.transpose()) / (np.matmul(np.matmul(C, p_0), C.transpose()) + Q)
    K[np.isnan(K)] = 0

    return C, K


def update_final(C, K):
    global p_0
    global position
    C = np.array([[1, 0], [0, 2]])
    position = position + np.matmul(K, (position_measure - np.matmul(C, position)))
    p_0 = np.matmul((np.identity(2) - np.matmul(K, C)), p_0)


t = 1
while not crashed:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            crashed = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                x_change = -5
            elif event.key == pygame.K_RIGHT:
                x_change = 5
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                x_change = 0

    position = estimate_pose(position)
    update()

    if t % 8 == 0:
        measurement()
        C_new, K_new = correction()
        update_final(C_new, K_new)

    gameDisplay.fill(white)
    red = (180, 50, 50)
    size = (position[0, 0] * 1000 + 50 - (p_0[0, 0] * 2000) / 2, position[1, 0] * 1000 + 50 - (2000 * p_0[1, 1]) / 2,
            p_0[0, 0] * 2000, 2000 * p_0[1, 1])
    pygame.draw.ellipse(gameDisplay, red, size, 1)
    draw_robot(position[0, 0] * 1000 + 50, position[1, 0] * 1000 + 50)

    points.append([position[0, 0] * 1000 + 50, position[1, 0] * 1000 + 50])
    points_gt.append([position_new_true[0, 0] * 1000 + 50, position_new_true[1, 0] * 1000 + 50])
    points_measure.append([position_measure[0, 0] * 1000 + 50, (position_measure[1, 0] / 2) * 1000 + 50])
    pygame.draw.lines(gameDisplay, blue, False, points, 5)
    pygame.draw.lines(gameDisplay, green, False, points_gt, 5)
    pygame.draw.lines(gameDisplay, red, False, points_measure, 5)

    if t % 8 == 0:
        pygame.draw.rect(gameDisplay, red, (position_measure[0, 0] * 1000 + 50, (position_measure[1, 0] / 2) * 1000 + 50, 10, 10))
        pygame.draw.rect(gameDisplay, green, (position_new_true[0, 0] * 1000 + 50, (position_new_true[1, 0] / 2) * 1000 + 50, 10, 10))

    pygame.display.update()
    clock.tick(8)

    t += 1

pygame.image.save(gameDisplay, 'final_display.png')
pygame.quit()
quit()
