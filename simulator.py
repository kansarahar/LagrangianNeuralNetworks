import pygame
import sys
import numpy as np

import physics

# ---------------------------------------------------------- 
# Pygame Canvas Setup
# ----------------------------------------------------------

px = 800

pygame.init()
screen = pygame.display.set_mode((px, px))
pygame.display.set_caption('Double Pendulum')

clock = pygame.time.Clock()

bg_surface = pygame.Surface((4*px//5, 4*px//5))
bg_surface.fill('White')

# ----------------------------------------------------------
# Main Event Loop
# ----------------------------------------------------------

double_pendulum = physics.Double_Pendulum(1, 1, 2, 2, np.pi*0.26, np.pi*0.82)

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
    clock.tick(120) # prevents the while loop from running faster than 60Hz
