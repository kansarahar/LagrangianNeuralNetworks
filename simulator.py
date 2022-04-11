from typing_extensions import _AnnotatedAlias
from numpy.core.numeric import Infinity
from numpy.lib.npyio import BagObj
import pygame
import sys
import numpy as np
from pygame.constants import DOUBLEBUF, K_LEFT, KEYDOWN

from physics import analytical_double_pendulum_fn, RK4_step

################ Pygame Canvas Setup ################

px = 800

pygame.init()
screen = pygame.display.set_mode((px, px))
pygame.display.set_caption('Double Pendulum')

clock = pygame.time.Clock()

bg_surface = pygame.Surface((4*px//5, 4*px//5))
bg_surface.fill('White')

########### Double Pendulum Calculations ############

class Double_Pendulum:
    def __init__(self, m1, m2, l1, l2, t1, t2, g=9.8):
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g
        self.state = np.array([t1, t2, 0, 0])

    def get_cartesian_coords(self):
        t1, t2, w1, w2 = self.state
        x1, y1 = self.l1*np.sin(t1), self.l1*np.cos(t1)
        x2, y2 = x1 + self.l2*np.sin(t2) , y1 + self.l2*np.cos(t2)
        return np.array([x1, y1, x2, y2])

    def get_potential_energy(self):
        x1, y1, x2, y2 = self.get_cartesian_coords()
        return -self.g * (self.m1 * y1 + self.m2 * y2)

    def get_kinetic_energy(self):
        t1, t2, w1, w2 = self.state 
        v1, v2 = self.l1 * w1, self.l2 * w2
        return 0.5 * (self.m1 * v1**2 + self.m2 * (v1**2 + v2**2 + 2*v1*v2*np.cos(t1 - t2)))

    def get_total_energy(self):
        return self.get_kinetic_energy() + self.get_potential_energy()

    def get_derivs(self, state, t=0):
        return analytical_double_pendulum_fn(state, t, self.m1, self.m2, self.l1, self.l2, self.g)

    def step_analytical(self, dt=0.001):
        self.state[0] %= (2*np.pi)
        self.state[1] %= (2*np.pi)
        self.state = RK4_step(self.get_derivs, self.state, 0, dt)[0]


################## Main Event Loop ##################

double_pendulum = Double_Pendulum(1, 1, 1, 1, np.pi/4, -np.pi*0.376547452)

while True:
    for event in pygame.event.get():
        
        # exit the game
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)
        
        # handle keyboard input
        #if event.type == pygame.KEYDOWN:
        #    if (event.key == pygame.K_d or event.key == pygame.K_RIGHT):
        #        double_pendulum.step_analytical(0.01)
        #        print(double_pendulum.state)
        #    if (event.key == pygame.K_a or event.key == pygame.K_LEFT):
        #        double_pendulum.step_analytical(-0.01)
   
    x1, y1, x2, y2 = 0.1 * px * double_pendulum.get_cartesian_coords()
    print('Kinetic Energy:', double_pendulum.get_kinetic_energy())
    print('Potential Energy:', double_pendulum.get_potential_energy())
    print('Total Energy:', double_pendulum.get_total_energy())
    double_pendulum.step_analytical()
    
    mass1_pos = (bg_surface.get_width()//2 + x1, y1)
    mass2_pos = (bg_surface.get_width()//2 + x2, y2)

    screen.blit(bg_surface, ((px - bg_surface.get_width())//2, (px - bg_surface.get_height())//2)) # block image transfer (one surface imposed on another surface)
    bg_surface.fill('White')
    pygame.draw.line(bg_surface, 'Black', (bg_surface.get_width()//2, 0), mass1_pos)
    pygame.draw.line(bg_surface, 'Black', mass1_pos, mass2_pos)
    pygame.draw.circle(bg_surface, 'Red', mass1_pos, 10)
    pygame.draw.circle(bg_surface, 'Blue', mass2_pos, 10)

    pygame.display.update()
    clock.tick(600) # prevents the while loop from running faster than 60Hz