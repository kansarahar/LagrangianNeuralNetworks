import os
import sys
import argparse
import pygame

import numpy as np
import torch

import physics
from lnn import LNN

# ----------------------------------------------------------
#   Args
# ----------------------------------------------------------

parser = argparse.ArgumentParser(
    description='A tool used to train and save an LNN model for a pendulum experiment',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--experiment', dest='experiment', type=str, choices=['double_pendulum'], default='double_pendulum', help='the type of experiment that the LNN is learning')
parser.add_argument('--model-name', dest='model_name', type=str, default='model.pt', help='the name of the LNN model to be saved or loaded')
parser.add_argument('--lnn', dest='lnn', action='store_true', help='use the trained model instead of the analytical solution')

args = parser.parse_args()

# ----------------------------------------------------------
#   File Paths
# ----------------------------------------------------------

dir_name = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(dir_name, 'models', 'double_pendulum_models')
model_path = os.path.join(model_dir, args.model_name)

# ---------------------------------------------------------- 
#   Pygame Canvas Setup
# ----------------------------------------------------------

px = 800

pygame.init()
screen = pygame.display.set_mode((px, px))
pygame.display.set_caption('Double Pendulum')

clock = pygame.time.Clock()

bg_surface = pygame.Surface((4*px//5, 4*px//5))
bg_surface.fill('White')

# ---------------------------------------------------------- 
#   Font and Text Setup
# ----------------------------------------------------------

pygame.font.init()
default_font = pygame.font.Font(pygame.font.get_default_font(), 16)

# ----------------------------------------------------------
#   Main Event Loop
# ----------------------------------------------------------

model = LNN(2)
model.load_state_dict(torch.load(model_path))
model.eval()
double_pendulum = physics.Double_Pendulum(np.pi*0.26, np.pi*0.82, 0, 0, 1, 1, 1, 1, 9.8, model)

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
   
    x1, y1, x2, y2 = 0.1 * px * double_pendulum.get_cartesian_coords().detach().numpy()
    double_pendulum.step_lnn() if args.lnn else double_pendulum.step_lagrangian()
    
    vertical_offset = 20
    mass1_pos = (bg_surface.get_width()//2 + x1, y1 + vertical_offset)
    mass2_pos = (bg_surface.get_width()//2 + x2, y2 + vertical_offset)

    screen.blit(bg_surface, ((px - bg_surface.get_width())//2, (px - bg_surface.get_height())//2)) # block image transfer (one surface imposed on another surface)
    bg_surface.fill('White')
    pygame.draw.line(bg_surface, 'Black', (bg_surface.get_width()//2, vertical_offset), mass1_pos)
    pygame.draw.line(bg_surface, 'Black', mass1_pos, mass2_pos)
    pygame.draw.circle(bg_surface, 'Red', mass1_pos, 10)
    pygame.draw.circle(bg_surface, 'Blue', mass2_pos, 10)

    potential_energy_text = default_font.render('Potential Energy: %s J' % round(double_pendulum.get_potential_energy().item(), 3), True, (0, 0, 0))
    kinetic_energy_text = default_font.render('Kinetic Energy: %s J' % round(double_pendulum.get_kinetic_energy().item(), 3), True, (0, 0, 0))
    total_energy_text = default_font.render('Total Energy: %s J' % round(double_pendulum.get_total_energy().item(), 3), True, (0, 0, 0))
    bg_surface.blit(potential_energy_text, (10, (bg_surface.get_height()//2 + 100)))
    bg_surface.blit(kinetic_energy_text, (10, (bg_surface.get_height()//2 + 125)))
    bg_surface.blit(total_energy_text, (10, (bg_surface.get_height()//2 + 150)))

    pygame.display.update()
    clock.tick(120) # prevents the while loop from running faster than 60Hz