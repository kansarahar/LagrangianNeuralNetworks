import os
import sys
import argparse
from tqdm import tqdm

import numpy as np
import torch
from torch import tensor
from lnn import LNN
from physics import Double_Pendulum

# ----------------------------------------------------------
#   Args
# ----------------------------------------------------------

parser = argparse.ArgumentParser(
    description='A tool used for creating double pendulum experimental data',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument('--model-name', dest='model_name', type=str, default='model.pt', help='the name of the LNN model to be saved or loaded')
parser.add_argument('--overwrite', dest='overwrite', action='store_true', help='indicates that you want to entirely rewrite a model')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epochs', dest='epochs', type=int, default=1, help='the number of passes of the entire training dataset')
parser.add_argument('--num-states', dest='num_states', type=int, default=50000, help='the total number of random states to generate')

parser.add_argument('--m1', dest='m1', type=float, default=1, help='the value of the first mass')
parser.add_argument('--m2', dest='m2', type=float, default=1, help='the value of the second mass')
parser.add_argument('--l1', dest='l1', type=float, default=1, help='the length of the first pendulum')
parser.add_argument('--l2', dest='l2', type=float, default=1, help='the length of the second pendulum')
parser.add_argument('--g', dest='g', type=float, default=9.8, help='the acceleration due to gravity')

args = parser.parse_args()

# ----------------------------------------------------------
#   File Paths
# ----------------------------------------------------------

dir_name = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(dir_name, 'models', 'double_pendulum_models')
model_path = os.path.join(model_dir, args.model_name)
os.makedirs(model_dir, exist_ok=True)

# ----------------------------------------------------------
#   Data Creation
# ----------------------------------------------------------

# the maximum possible energy for a double pendulum system starting at rest
max_energy = 2 * args.g * (args.m1 * args.l1 + args.m2 * (args.l1 + args.l2))
max_theta = 2 * np.pi
max_theta_dot_1 = np.sqrt(2 * max_energy / args.m1) / args.l1
max_theta_dot_2 = np.sqrt(2 * max_energy / args.m2) / args.l2

print('Generating data...')
double_pendulum = Double_Pendulum(0, 0)
state_data = torch.rand(args.num_states, 4)
deriv_data = torch.zeros((args.num_states, 2))

for count, state in enumerate(tqdm(state_data)):
    state[0] *= max_theta
    state[1] *= max_theta
    state[2] *= max_theta_dot_1
    state[3] *= max_theta_dot_2

    w1, w2, g1, g2 = double_pendulum.get_derivs_analytical(state)
    deriv_data[count] = (torch.stack([g1,g2]))

print('Data generated')

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.state_data = state_data
        self.deriv_data = deriv_data
    def __len__(self):
        return len(self.state_data)
    def __getitem__(self, index):
        return self.state_data[index], self.deriv_data[index]

training_set = Dataset()
training_generator = torch.utils.data.DataLoader(training_set, shuffle=True)

# ----------------------------------------------------------
#   Train Model
# ----------------------------------------------------------

device = torch.device('cpu')
model = LNN(2)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_function = torch.nn.L1Loss()

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
        #with torch.autograd.detect_anomaly():
        state, deriv = state.flatten().to(device), deriv.flatten().to(device)
        optimizer.zero_grad()
        loss = loss_function(model(state), deriv)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if count % 5000 == 0:
            print('\ncount=%s, saving model...' % count)
            torch.save(model.state_dict(), model_path)
            print('model saved')

    print('Epoch', epoch+1, 'complete! Total loss for this epoch:', epoch_loss)
    print('Saving model as', args.model_name, '...')
    torch.save(model.state_dict(), model_path)
    print('Model saved!')
print('Training complete')
