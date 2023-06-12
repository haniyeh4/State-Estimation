from cProfile import label
from cmath import atan, exp, sqrt
import math
from math import pi
from hashlib import new
import random
from re import U
import pygame
import numpy as np
from pygame.locals import *
import matplotlib.pyplot as plt 
import math

pygame.init()

display_width = 1000
display_height = 1000

gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('Robot Trajectory')

white = (255,255,255)

clock = pygame.time.Clock()
crashed = False
points=[]
points.append([0,0])
points_measure=[]
points_measure.append([0,0])
points_gt=[]
points_gt.append([0,0])
measured_positions_x=[]
measured_positions_y=[]
predicted_positions_x=[]
predicted_positions_y=[]
cov_width=[]
cov_hight=[]
position=np.array([[0],[0]])
position_measure=position
position_new_true=position
cov=np.array([[0,0],[0,0]])
point_prev=[[0],[0]]
position_particle=np.array([[0],[0]])

#def Gaussian( mu, sigma, x):
#        return exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))
    
def measurement_prob(x,y):
    dist = (1.0/(2*np.pi*0.005*0.075))*np.exp(-(((x - position_measure[0][0])**2/(2*0.005*0.005))+((y - position_measure[1][0])**2/(2*0.075*0.075))))


    dist = dist + 1e-9
    return dist

def compute_cov(x,y):
    x=np.array(x)
    y=np.array(y)
    return np.mean(x),np.std(x),np.mean(y),np.std(y)

def particle_generation():
    global position
    global  position_measure
    N_particles=100
    particles=[]
    particles_x=np.random.normal(position[0][0],0.005,N_particles)
    particles_y=np.random.normal(position[1][0],0.075,N_particles)
    for i in range(0,N_particles):
        particles.append(np.array([[particles_x[i]],[particles_y[i]]]))  
    return particles

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
                  
        new_particles.append(np.array([[particles[index][0][0]],[particles[index][1][0]]]))
        
    return new_particles


def compute_weight(particles):

        global position_measure

        dist=np.zeros((1,len(particles)))

        new_x = 0
        new_y = 0

        for i in range(0,len(particles)):
            dist[0,i]=measurement_prob(particles[i][0][0],particles[i][1][0])


        dist=dist/np.sum(dist)
        
        for i in range(0, len(particles)):	
            new_x = new_x + dist[0,i] * particles[i][0][0]
            new_y = new_y + dist[0,i] * particles[i][1][0]

        pose=np.array([[new_x],[new_y]])
 
        return pose,dist



def estimate_pose(position,particles):
    x=[]
    y=[]
    global position_new_true
    
    A=np.array([[1,0],[0,1]])
    r=0.1
    delta_t=1/8
    B=np.array([[r/2*delta_t,r/2*delta_t],[r/2*delta_t,r/2*delta_t]])
    U=np.array([[1],[1]])
    
    position_new = np.matmul(A,position) + np.matmul(B,U) + np.array([[np.random.normal(0,0.1)],[np.random.normal(0,0.15)]])*delta_t
    position_new_true = np.matmul(A,position_new_true)+np.matmul(B,U)#ME
    
 
    for i in range(0,len(particles)):

            temp = np.matmul(A,particles[i]) + np.matmul(B,U)+ np.array([[np.random.normal(0,0.1)],[np.random.normal(0,0.15)]])*delta_t
            particles[i]=temp
            x.append(temp[0][0])
            y.append(temp[1][0])
    return particles,position_new,x,y

x_change = 0


def update():
    global cov 
    delta_t=1/8
    F=np.array([[1,0],[0,1]])
    temp=np.matmul(F,cov)
    Q=np.array([[0.1,0],[0,0.15]])* delta_t
    p_new=np.matmul(temp,F.transpose())+Q
    cov=p_new
    
#The measurement data is being computed 
def measurement():
    global position_measure
    global position
    C=np.array([[1,0],[0,2]])
    
    Z=np.matmul(C,position_new_true) + np.array([[np.random.normal(0,0.05)],[np.random.normal(0,0.075)]])
    
    position_measure=Z


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

    if first_flag==0:
        particles=particle_generation()
        first_flag=1
    particles,position,x,y=estimate_pose(position,particles)
    mean_x,std_x,mean_y,std_y=compute_cov(x,y)
    
    if(t%8==0):
        measurement()
        position_particle,dist=compute_weight(particles)
        particles=resample(particles, dist)
        
    gameDisplay.fill(white)
    b=(0,0,255)
    r=(255,0,0)
    g=(0,255,0)
    yellow=(255, 255, 0)
    
    for p in particles:        
        pygame.draw.rect(gameDisplay,r,(p[0][0]*1000+50,(p[1][0]/2)*1000+50,2,2))#particles are red

    red = (180, 50, 50)
    size = (mean_x*1000+50-(std_x*2000)/2, mean_y/2*1000+50-(2000*std_y)/2, std_x*2000, 2000*std_y)
    pygame.draw.ellipse(gameDisplay, red, size,1)  

    pygame.draw.polygon(gameDisplay, b,
                        [[position_particle[0,0]*1000+50,position_particle[1,0]/2*1000+50],[position_particle[0,0]*1000+40,position_particle[1,0]/2*1000+35] ,
                        [position_particle[0,0]*1000+40,position_particle[1,0]/2*1000+65]])#blue is the position
  
    points.append([position_particle[0,0]*1000+50,position_particle[1,0]/2*1000+50])
    points_gt.append([position_new_true[0,0]*1000+50,position_new_true[1,0]*1000+50])
    points_measure.append([position_measure[0,0]*1000+50,(position_measure[1,0]/2)*1000+50])
    pygame.draw.lines(gameDisplay,b,False,points,5) #b: mean position
    pygame.draw.lines(gameDisplay,g,False,points_gt,5) #g: ground truth
    pygame.draw.lines(gameDisplay,r,False,points_measure,5) #r: measurement



   
    pygame.draw.rect(gameDisplay,yellow,(position_particle[0,0]*1000+50,(position_particle[1,0]/2)*1000+50,10,10))

    measured_positions_x.append(position_measure[0,0])
    measured_positions_y.append(position_measure[1,0])
    predicted_positions_x.append(position_particle[0,0])
    predicted_positions_y.append(position_particle[1,0])
    cov_hight.append(cov[0,0])
    cov_width.append(cov[1,1])

    pygame.display.update()
    clock.tick(6) 
        
    t+=1
    
pygame.quit()
quit()
