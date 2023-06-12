from hashlib import new
from re import U
import pygame
import numpy as np
from pygame.locals import *
from math import *

###### elipse
from matplotlib.patches import Ellipse
import multiprocessing
import matplotlib.pyplot as plt
import random
import math 
######

pygame.init()

display_width = 1000
display_height = 1000

gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('Robot Trajectory')

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
points_measure=[]
points_measure.append([400,400])
points_gt=[]
points_gt.append([400,400])


position=np.array([[0],[0],[0]])
position_measure=position
cov=np.array([[0,0,0],[0,0,0],[0,0,0]])
position_new_true = np.array([[0],[0],[0]])
position_particle=np.array([[0],[0],[0]])

def compute_cov(x,y):
    x=np.array(x)
    y=np.array(y)
    
    return np.mean(x),np.std(x),np.mean(y),np.std(y)

def particle_generation():
    global position
    global  position_measure
    N_particles=100
    particles=[]
    particles_x=np.random.normal(position[0][0],0.0005,N_particles)
    particles_y=np.random.normal(position[1][0],0.0075,N_particles)
    particles_theta=np.random.normal(position[1][0],0,N_particles)

    for i in range(0,N_particles):

        particles.append(np.array([[particles_x[i]],[particles_y[i]],[particles_theta[i]]]))  
    
    return particles



def measurement_prob(x,y):
    dist = (1.0/(2*np.pi*0.005*0.075))*np.exp(-(((x - position_measure[0][0])**2/(2*0.005*0.005))+((y - position_measure[1][0])**2/(2*0.075*0.075))))
    dist = dist + 1e-9 
    return dist

def compute_weight(particles):

        global position_new_true
        dist=np.zeros((1,len(particles)))
        for i in range(0,len(particles)):
            dist[0,i]=measurement_prob(particles[i][0][0],particles[i][1][0])
        dist=dist/np.sum(dist)
        new_x = 0
        new_y = 0

        for i in range(0, len(particles)):	
            new_x = new_x + dist[0,i] * particles[i][0][0]
            new_y = new_y + dist[0,i] * particles[i][1][0]

        pose=np.array([[new_x],[new_y],[0]])
        return pose,dist    
def resample(particles, dist):
    N=len(particles)
    new_particles = []
    index = int(random.random() * N)
    beta = 0.0
    
    for i in range(0,N):

        dist[0,i]=measurement_prob(particles[i][0][0],particles[i][1][0])

        
    dist=dist/np.sum(dist)
    mw = dist.max()
    for i in range(N):
        beta += random.random() * 2.0 * mw
        while beta > dist[0][index]:
            beta -= dist[0][index]
            index = (index + 1) % N
        new_particles.append(np.array([[particles[index][0][0]],[particles[index][1][0]],[particles[index][2][0]]]))
        

    return new_particles    
def estimate_pose(position,particles):
    global position_new_true
    F=np.array([[1,0,0],[0,1,0],[0,0,1]])
    x=[]
    y=[]
    r=0.1
    l=0.3
    delta_t=1/8
    u_r=u_l=1
    if (math.dist([position[0,0],position[1,0]],center)<10):
        u_r=1
        u_l=0
    if (math.dist([position[0,0],position[1,0]],center)>11):
        u_r=0
        u_l=1
    G=np.array([[r*delta_t*cos(position[2,0]),0],[r*delta_t*sin(position[2,0]),0],[0,delta_t*r/l]])
    G_true = np.array([[r*delta_t*cos(position_new_true[2,0]),0],[r*delta_t*sin(position_new_true[2,0]),0],[0,delta_t*r/l]])

    u=np.array([[(u_r+u_l)/2],[u_r-u_l]])
    
    position_new=np.matmul(F,position)+np.matmul(G,u) + delta_t*np.array([[np.random.normal(0,0.01)],[np.random.normal(0,0.1)],[0]])
    position_new_true = np.matmul(F,position_new_true) + np.matmul(G_true,u)
    for i in range(0,len(particles)):
            if (math.dist([particles[i][0][0],particles[i][1][0]],center)<10):
                u_r=1
                u_l=0
            if (math.dist([particles[i][0][0],particles[i][1][0]],center)>11):
                u_r=0
                u_l=1

            u=np.array([[(u_r+u_l)/2],[u_r-u_l]])
            
            temp=np.matmul(F,particles[i])+np.matmul(G,u) + delta_t*np.array([[np.random.normal(0,0.01)],[np.random.normal(0,0.1)],[0]])
            particles[i]=temp
        
            x.append(temp[0][0])
            y.append(temp[1][0])
    return position_new,particles,x,y

x_change = 0



def update():
    global cov 
    delta_t=1/8
    A=np.array([[1,0,0],[0,1,0],[0,0,1]])
    temp=np.matmul(A,cov)
    R=np.array([[0.1,0,0],[0,0.1,0],[0,0,0.01]])*delta_t
    p_new=np.matmul(temp,A.transpose())+R

    cov=p_new
    

def measurement():
    global position_measure
    global position
    C=np.array([[1,0,0],[0,2,0],[0,0,1]])
    R=np.array([[np.random.normal(0,0.05)],[np.random.normal(0,0.075)],[0]])
    Z = np.matmul(C,position_new_true) + R
    position_measure = Z


t=0   
first_flag=0
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
       

    # 
    if first_flag==0:
        particles=particle_generation()
        first_flag=1

    

    position,particles,x,y=estimate_pose(position,particles)
    mean_x,std_x,mean_y,std_y=compute_cov(x,y)
    update()
    

     
    if(t%8==0):
        measurement()
        position_particle,dist1=compute_weight(particles)
        particles=resample(particles, dist1) 
    

    w=(255,255,255)
    b=(0,0,255)
    g=(0,255,0)
    r=(255,0,0)
    yellow=(255, 255, 0)
    gameDisplay.fill(w)
    for p in particles:
        pygame.draw.rect(gameDisplay,r,(p[0][0]*1000+400,(p[1][0]/2)*1000+400,2,2))        
    surface = pygame.Surface((320, 240))
    red = (180, 50, 50)
 
    pygame.draw.polygon(gameDisplay, b,
                        [[position_particle[0,0]*1000+400,position_particle[1,0]/2*1000+400],[position_particle[0,0]*1000+390,position_particle[1,0]/2*1000+390] ,
                        [position_particle[0,0]*1000+400,position_particle[1,0]/2*1000+410]])

    size = (position_particle[0,0]*1000+400-(std_x*2000)/2, position_particle[1,0]/2*1000+400-(2000*std_y)/2, std_x*2000, 2000*std_y)
    pygame.draw.ellipse(gameDisplay, red, size,1)
    points.append([position_particle[0,0]*1000+400,position_particle[1,0]/2*1000+400])
    points_gt.append([position_new_true[0,0]*1000+400,position_new_true[1,0]*1000+400])
    points_measure.append([position_measure[0,0]*1000+400,(position_measure[1,0]/2)*1000+400])
    pygame.draw.lines(gameDisplay,b,False,points,5) #b: mean position
    pygame.draw.lines(gameDisplay,g,False,points_gt,5) #g: ground truth
    pygame.draw.lines(gameDisplay,r,False,points_measure,5) #r: measurement
 

    pygame.draw.rect(gameDisplay,yellow,(position_particle[0,0]*1000+400,(position_particle[1,0]/2)*1000+400,10,10))
    pygame.draw.rect(gameDisplay,r,(position_measure[0,0]*1000+400,(position_measure[1,0]/2)*1000+400,10,10))
    pygame.draw.rect(gameDisplay,g,(position_new_true[0,0]*1000+400,(position_new_true[1,0])*1000+400,10,10))
    measured_positions_x.append(position_measure[0,0])
    measured_positions_y.append(position_measure[1,0])
    predicted_positions_x.append(position[0,0])
    predicted_positions_y.append(position[1,0])
    cov_hight.append(cov[0,0])
    cov_width.append(cov[1,1])
    pygame.display.update()
    clock.tick(6)
    t+=1

pygame.quit()
quit()
