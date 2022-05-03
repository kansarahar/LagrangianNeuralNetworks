import os
import sys
import argparse
from tqdm import tqdm

import numpy as np
import torch
from lnn import LNN, ClassicalNN
from physics import Double_Pendulum, Spring_Pendulum, Cart_Pendulum

# ----------------------------------------------------------
#   Args
# ----------------------------------------------------------

parser = argparse.ArgumentParser(
    description='A tool used for creating double pendulum experimental data',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument('--experiment', dest='experiment', type=str, choices=['double_pendulum', 'spring_pendulum', 'cart_pendulum'], default='double_pendulum', help='the type of experiment that the LNN is learning')
parser.add_argument('--classical', dest='classical', action='store_true', help='train a linear model instead of an LNN model')
parser.add_argument('--model-name', dest='model_name', type=str, default='model.pt', help='the name of the LNN model to be saved or loaded')
parser.add_argument('--overwrite', dest='overwrite', action='store_true', help='indicates that you want to entirely rewrite a model')
parser.add_argument('--lr', dest='lr', type=float, default=0.003, help='learning rate')
parser.add_argument('--epochs', dest='epochs', type=int, default=1, help='the number of passes of the entire training dataset')
parser.add_argument('--num-states', dest='num_states', type=int, default=100000, help='the total number of random states to generate')
parser.add_argument('--save-freq', dest='save_freq', type=int, default=10000, help='how often to save the model while training')

args = parser.parse_args()

# ----------------------------------------------------------
#   File Paths
# ----------------------------------------------------------

dir_name = os.path.dirname(os.path.abspath(__file__))
model_type = 'Classical' if args.classical else 'LNN'
model_dir = os.path.join(dir_name, 'models', '%s_models' % args.experiment, model_type)
model_path = os.path.join(model_dir, args.model_name)
os.makedirs(model_dir, exist_ok=True)

# ----------------------------------------------------------
#   Data Creation
# ----------------------------------------------------------

state_data = np.zeros((args.num_states, 4))
deriv_data = np.zeros((args.num_states, 2))

print('Generating data...')

if (args.experiment == 'double_pendulum'):
    m1, m2, l1, l2, g = 1, 1, 1, 1, 9.8
    max_energy = g * (m1 * l1 + m2 * (l1 + l2)) # the maximum possible energy for a double pendulum system starting at rest
    
    double_pendulum = Double_Pendulum(0, 0, 0, 0, m1, m2, l1, l2, g)
    random_angles = np.random.rand(args.num_states, 2) * 2 * np.pi
    
    for count, angles in enumerate(tqdm(random_angles)):
        t1, t2 = angles
        potential_energy = -g * (m1 * l1 * np.cos(t1) + m2 * (l1 * np.cos(t1) + l2 * np.cos(t2)))
        free_energy = max_energy - potential_energy
        max_angular_velocity_m1 = np.sqrt(2 * free_energy / m1 / l1**2)
        max_angular_velocity_m2 = np.sqrt(2 * free_energy / m2 / l2**2)
        w1 = max_angular_velocity_m1 * (2 * np.random.rand() - 1)
        w2 = max_angular_velocity_m2 * (2 * np.random.rand() - 1)
        state = np.array([t1, t2, w1, w2])
        w1, w2, g1, g2 = double_pendulum.get_derivs_analytical(state)
        state_data[count] = state
        deriv_data[count] = g1, g2

if (args.experiment == 'spring_pendulum'):
    m, k, r0, g = 1, 10, 1, 9.8
    max_energy = 0.5 * k * r0**2 + m * g * 2 * r0 # assume that the spring always starts off < 2 x equilibrium length

    spring_pendulum = Spring_Pendulum(0, 0, 0, 0, m, k, r0, g)
    random_rs = np.random.rand(args.num_states) * 2 * r0
    random_thetas = np.random.rand(args.num_states) * 2 * np.pi

    for idx in tqdm(range(args.num_states)):
        r, theta = random_rs[idx], random_thetas[idx]
        potential_energy = 0.5 * k * (r - r0)**2 - m * g * r * np.cos(theta)
        free_energy = max_energy - potential_energy
        max_radial_velocity = np.sqrt(2 * free_energy / m)
        max_angular_velocity = np.sqrt(2 * free_energy / m) / r
        r_t = max_radial_velocity * (2 * np.random.rand() - 1)
        theta_t = max_angular_velocity * (2 * np.random.rand() - 1)
        state = np.array([r, theta, r_t, theta_t])
        r_t, theta_t, r_tt, theta_tt = spring_pendulum.get_derivs_analytical(state)
        state_data[idx] = state
        deriv_data[idx] = r_tt, theta_tt

if (args.experiment == 'cart_pendulum'):
    mc, mp, k, d0, l, g= 1, 1, 10, 1, 1, 9.8
    max_energy = 0.5 * k * d0**2 + mp * g * l # assume that the spring always starts off < 2 x equilibrium length

    cart_pendulum = Cart_Pendulum(0, 0, 0, 0, mc, mp, k, d0, l, g)
    random_ds = np.random.rand(args.num_states) * 2 * d0
    random_thetas = np.random.rand(args.num_states) * 2 * np.pi

    for idx in tqdm(range(args.num_states)):
        d, theta = random_ds[idx], random_thetas[idx]
        potential_energy = 0.5 * k * (d - d0)**2 - mp * g * l * np.cos(theta)
        free_energy = max_energy - potential_energy
        max_cart_velocity = np.sqrt(2 * free_energy / (mc + mp))
        max_angular_velocity = np.sqrt(2 * free_energy / mp) / l
        d_t = max_cart_velocity * (2 * np.random.rand() - 1)
        theta_t = max_angular_velocity * (2 * np.random.rand() - 1)
        state = np.array([d, theta, d_t, theta_t])
        d_t, theta_t, d_tt, theta_tt = cart_pendulum.get_derivs_analytical(state)
        state_data[idx] = state
        deriv_data[idx] = d_tt, theta_tt

print('Data generated')

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.state_data = torch.tensor(state_data).float()
        self.deriv_data = torch.tensor(deriv_data).float()
    def __len__(self):
        return len(self.state_data)
    def __getitem__(self, index):
        return self.state_data[index], self.deriv_data[index]

training_set = Dataset()
training_generator = torch.utils.data.DataLoader(training_set, shuffle=True)

# ----------------------------------------------------------
#   Train Model
# ----------------------------------------------------------

model = ClassicalNN() if args.classical else LNN()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_function = torch.nn.L1Loss()

print('Training %s model for %s experiment' % (model_type, args.experiment))
try:
    if (not args.overwrite):
        print('Loading saved model', args.model_name, '...')
        model.load_state_dict(torch.load(model_path))
        print('Model successfully loaded')
except:
    print('Model %s does not exist - creating new model' % args.model_name)

for epoch in range(args.epochs):
    epoch_loss = 0
    
    for count, (state, deriv) in enumerate(tqdm(training_generator)):
        state, deriv = state.flatten(), deriv.flatten()
        optimizer.zero_grad()
        loss = loss_function(model(state), deriv)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if count % args.save_freq == 0:
            print('\ncount=%s, saving model...' % count)
            torch.save(model.state_dict(), model_path)
            print('model saved')
            print('accumulated loss for current epoch:', epoch_loss)

    print('Epoch', epoch+1, 'complete! Total loss for this epoch:', epoch_loss)
    print('Saving model as', args.model_name, '...')
    torch.save(model.state_dict(), model_path)
    print('Model saved!')
print('Training complete')
