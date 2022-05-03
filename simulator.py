import os
import sys
import argparse
import pygame

import numpy as np
import torch

from physics import Double_Pendulum, Spring_Pendulum, Cart_Pendulum
from lnn import LNN

# ----------------------------------------------------------
#   Args
# ----------------------------------------------------------

parser = argparse.ArgumentParser(
    description='A tool used to train and save an LNN model for a pendulum experiment',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--experiment', dest='experiment', type=str, choices=['double_pendulum', 'spring_pendulum', 'cart_pendulum'], default='double_pendulum', help='the type of experiment that the LNN is learning')
parser.add_argument('--model-name', dest='model_name', type=str, default='pretrained_lnn.pt', help='the name of the LNN model to be saved or loaded')
parser.add_argument('--lnn', dest='lnn', action='store_true', help='use the trained model instead of the analytical solution')

args = parser.parse_args()

# ----------------------------------------------------------
#   File Paths
# ----------------------------------------------------------

dir_name = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(dir_name, 'models', '%s_models' % args.experiment, 'LNN')
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

double_pendulum = Double_Pendulum(np.pi*0.26, np.pi*0.82)
spring_pendulum = Spring_Pendulum(np.pi*0.36, 1.2)
cart_pendulum = Cart_Pendulum(2, np.pi*0.26)

model = LNN()
if args.lnn:
    model.load_state_dict(torch.load(model_path))
    model.eval()
    double_pendulum.lnn = model
    spring_pendulum.lnn = model
    cart_pendulum.lnn = model

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

    mid_width, mid_height = bg_surface.get_width()//2, bg_surface.get_height()//2
    if (args.experiment == 'double_pendulum'):
   
        x1, y1, x2, y2 = 0.1 * px * double_pendulum.get_cartesian_coords().detach().numpy()
        double_pendulum.step_lnn() if args.lnn else double_pendulum.step_analytical()
        
        vertical_offset = 100
        mass1_pos = (bg_surface.get_width()//2 + x1, y1 + vertical_offset)
        mass2_pos = (bg_surface.get_width()//2 + x2, y2 + vertical_offset)

        screen.blit(bg_surface, ((px - bg_surface.get_width())//2, (px - bg_surface.get_height())//2)) # block image transfer (one surface imposed on another surface)
        bg_surface.fill('White')
        pygame.draw.line(bg_surface, 'Black', (mid_width, vertical_offset), mass1_pos)
        pygame.draw.line(bg_surface, 'Black', mass1_pos, mass2_pos)
        pygame.draw.circle(bg_surface, 'Black', (mid_width, vertical_offset), 4)
        pygame.draw.circle(bg_surface, 'Red', mass1_pos, 10)
        pygame.draw.circle(bg_surface, 'Blue', mass2_pos, 10)

        potential_energy_text = default_font.render('Potential Energy: %s J' % round(double_pendulum.get_potential_energy().item(), 3), True, (0, 0, 0))
        kinetic_energy_text = default_font.render('Kinetic Energy: %s J' % round(double_pendulum.get_kinetic_energy().item(), 3), True, (0, 0, 0))
        total_energy_text = default_font.render('Total Energy: %s J' % round(double_pendulum.get_total_energy().item(), 3), True, (0, 0, 0))
        bg_surface.blit(potential_energy_text, (10, (mid_height + 100)))
        bg_surface.blit(kinetic_energy_text, (10, (mid_height + 125)))
        bg_surface.blit(total_energy_text, (10, (mid_height + 150)))

        pygame.display.update()
        clock.tick(120) # prevents the while loop from running faster than 60Hz
    
    if (args.experiment == 'spring_pendulum'):
        spring_pendulum.lnn = model
   
        x, y = 0.1 * px * spring_pendulum.get_cartesian_coords().detach().numpy()
        spring_pendulum.step_lnn() if args.lnn else spring_pendulum.step_analytical()
        
        vertical_offset = 100
        mass_pos = (bg_surface.get_width()//2 + x, y + vertical_offset)

        screen.blit(bg_surface, ((px - bg_surface.get_width())//2, (px - bg_surface.get_height())//2)) # block image transfer (one surface imposed on another surface)
        bg_surface.fill('White')
        pygame.draw.line(bg_surface, 'Black', (mid_width, vertical_offset), mass_pos)
        pygame.draw.circle(bg_surface, 'Black', (mid_width, vertical_offset), 4)
        pygame.draw.circle(bg_surface, 'Red', mass_pos, 10)

        potential_energy_text = default_font.render('Potential Energy: %s J' % round(spring_pendulum.get_potential_energy().item(), 3), True, (0, 0, 0))
        kinetic_energy_text = default_font.render('Kinetic Energy: %s J' % round(spring_pendulum.get_kinetic_energy().item(), 3), True, (0, 0, 0))
        total_energy_text = default_font.render('Total Energy: %s J' % round(spring_pendulum.get_total_energy().item(), 3), True, (0, 0, 0))
        bg_surface.blit(potential_energy_text, (10, (mid_height + 100)))
        bg_surface.blit(kinetic_energy_text, (10, (mid_height + 125)))
        bg_surface.blit(total_energy_text, (10, (mid_height + 150)))

        pygame.display.update()
        clock.tick(120) # prevents the while loop from running faster than 60Hz

    if (args.experiment == 'cart_pendulum'):
        spring_pendulum.lnn = model
   
        xc, yc, xp, yp = 0.1 * px * cart_pendulum.get_cartesian_coords().detach().numpy()
        cart_pendulum.step_lnn() if args.lnn else cart_pendulum.step_analytical()
        
        vertical_offset = 100
        cart_pos = (bg_surface.get_width()//2 + xc, vertical_offset - 10)
        mass_pos = (bg_surface.get_width()//2 + xp, yp + vertical_offset)

        screen.blit(bg_surface, ((px - bg_surface.get_width())//2, (px - bg_surface.get_height())//2)) # block image transfer (one surface imposed on another surface)
        bg_surface.fill('White')
        cart = pygame.Rect(cart_pos[0]-15, cart_pos[1]-10, 30, 20)
        pygame.draw.line(bg_surface, 'Black', cart_pos, mass_pos)
        pygame.draw.line(bg_surface, 'Black', (mid_width - 200, vertical_offset - 10), (cart_pos[0], vertical_offset-10), 1)
        pygame.draw.line(bg_surface, 'Black', (mid_width - 100, vertical_offset), (mid_width + 100, vertical_offset), 2)
        pygame.draw.circle(bg_surface, 'Black', (mid_width - 200, vertical_offset - 10), 4)
        pygame.draw.circle(bg_surface, 'Red', mass_pos, 10)
        pygame.draw.rect(bg_surface, 'Blue', cart)

        potential_energy_text = default_font.render('Potential Energy: %s J' % round(cart_pendulum.get_potential_energy().item(), 3), True, (0, 0, 0))
        kinetic_energy_text = default_font.render('Kinetic Energy: %s J' % round(cart_pendulum.get_kinetic_energy().item(), 3), True, (0, 0, 0))
        total_energy_text = default_font.render('Total Energy: %s J' % round(cart_pendulum.get_total_energy().item(), 3), True, (0, 0, 0))
        bg_surface.blit(potential_energy_text, (10, (bg_surface.get_height()//2 + 100)))
        bg_surface.blit(kinetic_energy_text, (10, (bg_surface.get_height()//2 + 125)))
        bg_surface.blit(total_energy_text, (10, (bg_surface.get_height()//2 + 150)))

        pygame.display.update()
        clock.tick(120) # prevents the while loop from running faster than 60Hz