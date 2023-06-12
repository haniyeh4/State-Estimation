import pygame
import numpy as np
from math import *
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

from pygame.locals import *
from hashlib import new
from re import U

pygame.init()

display_width = 1000
display_height = 1000

gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('Robot Trajectory')

black = (0,0,0)
white = (255,255,255)

clock = pygame.time.Clock()
crashed = False

measured_positions_x=[]
measured_positions_y=[]
predicted_positions_x=[]
predicted_positions_y=[]
cov_width=[]
cov_hight=[]
center=np.array([[10],[10]])
points=[]
points.append([400,400])
points_gt=[]
points_gt.append([400,400])
points_measure=[]
points_measure.append([400,400])


position=np.array([[0],[0],[0]])
position_measure=position
p_0=np.array([[0,0,0],[0,0,0],[0,0,0]])
position_new_true = np.array([[0,0,0],[0,0,0],[0,0,0]])

def estimate_pose(position):
    global position_new_true
    A=np.array([[1,0,0],[0,1,0],[0,0,1]])

    r=0.1
    l=0.3
    delta_t=1/8
    u_r=u_l=1
    if (dist([position[0,0],position[1,0]],center)<10):
        u_r=1
        u_l=0
    if (dist([position[0,0],position[1,0]],center)>10):
        u_r=0
        u_l=1
    G=np.array([[r*delta_t*cos(position[2,0]),0],[r*delta_t*sin(position[2,0]),0],[0,delta_t*r/l]])
    u=np.array([[(u_r+u_l)/2],[u_r-u_l]])
    
    position_new=np.matmul(A,position)+np.matmul(G,u) + delta_t*np.array([[np.random.normal(0,0.1)],[np.random.normal(0,0.1)],[np.random.normal(0,0.01)]])
    position_new_true = np.matmul(A,position_new_true) + np.matmul(G,u)
     
    return position_new

def update():
    global p_0 
    A=np.array([[1,0,0],[0,1,0],[0,0,1]])
    temp=np.matmul(A,p_0)
    Q=np.array([[0.1,0,0],[0,0.1,0],[0,0,0.01]])*1/8
    p_new=np.matmul(temp,A.transpose())+Q

    p_0=p_new
    
def measurement():
    global position_measure
    global position
    C=np.array([[1,0,0],[0,2,0],[0,0,1]])

    Q=np.array([[np.random.normal(0,0.05)],[np.random.normal(0,0.075)],[0]])
    Z = np.matmul(C,position_new_true) + Q
    position_measure = Z
    
def correction():
    global p_0
    C=np.array([[1,0,0],[0,2,0],[0,0,1]])
    Q=np.array([[0.05,0,0],[0,0.075,0],[0,0,0]])

    temp1=np.matmul(p_0,C.transpose())
    temp2=np.matmul(np.matmul(C,p_0),C.transpose())+Q

    K=temp1/temp2
    K[np.isnan(K)] = 0
  
    return C,K

def update_final(C,K):
    global p_0
    global position
    p_0=np.matmul((np.identity(3)-np.matmul(K,C)),p_0)
    C = np.array([[1,0,0],[0,2,0],[0,0,1]])
    position = position + np.matmul( K , (position_measure - np.matmul(C,position)) )

t=1   
while not crashed:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            crashed = True

        ######################
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                x_change = -5
            elif event.key == pygame.K_RIGHT:
                x_change = 5
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                x_change = 0
        ######################

    position=estimate_pose(position)

    update()

    if(t%8==0):
        measurement()
        C,K_new=correction()
        update_final(C,K_new)
        

    w=(255,255,255)
    b=(0,0,255)
    r=(255,0,0)
    g=(0,255,0)

    
    gameDisplay.fill(w)
    
    surface = pygame.Surface((320, 240))
    pygame.draw.polygon(gameDisplay, b,
                        [[position[0,0]*1000+400,position[1,0]*1000+400],[position[0,0]*1000+390,position[1,0]*1000+390] ,
                        [position[0,0]*1000+400,position[1,0]*1000+410]])

    size = (position[0,0]*1000+400-(p_0[0,0]*2000)/2, position[1,0]*1000+400-(2000*p_0[1,1])/2, p_0[0,0]*2000, 2000*p_0[1,1])
    pygame.draw.ellipse(gameDisplay, r, size,1)
    points.append([position[0,0]*1000+400,position[1,0]*1000+400])
    points_gt.append([position_new_true[0,0]*1000+400,position_new_true[1,0]*1000+400])
    points_measure.append([position_measure[0,0]*1000+400,(position_measure[1,0]/2)*1000+400])
    pygame.draw.lines(gameDisplay,b,False,points,5)
    pygame.draw.lines(gameDisplay,g,False,points_gt,5)
    pygame.draw.lines(gameDisplay,r,False,points_measure,5)
    
    if(t%8==0):
        pygame.draw.rect(gameDisplay,g,(position_new_true[0,0]*1000+400,(position_new_true[1,0]/2)*1000+400,10,10))
        pygame.draw.rect(gameDisplay,r,(position_measure[0,0]*1000+400,(position_measure[1,0]/2)*1000+400,10,10))
        predicted_positions_x.append(position[0,0]*1000+400)
        predicted_positions_y.append(position[1,0]*1000+400)
        measured_positions_x.append(position_measure[0,0]*1000+400)
        measured_positions_y.append((position_measure[1,0]/2)*1000+400)
        cov_width.append(2*p_0[0,0]*1000)
        cov_hight.append(2*p_0[1,1]*1000)
        for ellipse in zip(predicted_positions_x, predicted_positions_y, cov_width, cov_hight):
            pygame.draw.ellipse(gameDisplay, (0,255,0), ellipse, 2)
        for ellipse in zip(measured_positions_x, measured_positions_y, cov_width, cov_hight):
            pygame.draw.ellipse(gameDisplay, (255,0,0), ellipse, 2)

    pygame.display.update()
    clock.tick(8)
    t+=1


pygame.quit()
