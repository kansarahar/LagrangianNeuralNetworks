import os
import sys
import argparse

import numpy as np
import torch
from tqdm import tqdm

from lnn import LNN


# ----------------------------------------------------------
#   Args
# ----------------------------------------------------------

parser = argparse.ArgumentParser(
    description='A tool used to train and save an LNN model for a pendulum experiment',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--experiment', dest='experiment', type=str, choices=['double_pendulum'], default='double_pendulum', help='the type of experiment that the LNN is learning')

parser.add_argument('--data-dir-name', dest='dir_name', type=str, default='generated_data', help='the name of the saved data directory')
parser.add_argument('--model-name', dest='model_name', type=str, default='model.pt', help='the name of the LNN model to be saved or loaded')
parser.add_argument('--overwrite', dest='overwrite', action='store_true', help='indicates that you want to entirely rewrite a model')

parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epochs', dest='epochs', type=int, default=1, help='the number of passes of the entire training dataset')

parser.add_argument('--num-workers', dest='num_workers', type=int, default=0, help='how many subprocesses to use for data loading - 0 means that the data will be loaded in the main process')
parser.add_argument('--no-cuda', dest='no_cuda', action='store_true', help='indicates that you do not want to use cuda, even if it is available')

args = parser.parse_args()

# ----------------------------------------------------------
#   File Paths
# ----------------------------------------------------------

dir_name = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_name, args.experiment, 'data', args.dir_name)
data_file_path = os.path.join(data_path, 'data.npy')
params_path = os.path.join(data_path, 'params.npy')
model_path = os.path.join(dir_name, 'models', args.model_name)
os.makedirs(model_path, exist_ok=True)

# ----------------------------------------------------------
#   CUDA
# ----------------------------------------------------------

use_cuda = torch.cuda.is_available() and not args.no_cuda
device = torch.device('cuda:0' if use_cuda else 'cpu')

# ----------------------------------------------------------
#   Dataset Specification
# ----------------------------------------------------------

# https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

dataloader_args = {
    'batch_size': 1,
    'shuffle': True,
    'num_workers': args.num_workers
}

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = np.load(data_file_path)
        self.states = torch.tensor(self.data[:,:4]).float()
        self.accelerations = torch.tensor(self.data[:,4:]).float()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.states[index], self.accelerations[index]

training_set = Dataset()
training_generator = torch.utils.data.DataLoader(training_set, **dataloader_args)

# ----------------------------------------------------------
#   Training
# ----------------------------------------------------------

model = LNN(2)
model.to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
loss_function = torch.nn.L1Loss()

for epoch in range(args.epochs):
    epoch_loss = 0
    
    for state, accel in tqdm(training_generator):
        state, accel = state.flatten().to(device), accel.flatten().to(device)
        optimizer.zero_grad()
        loss = loss_function(model(state), accel)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print('Epoch', epoch+1, 'complete! Total loss for this epoch:', epoch_loss)
    print('Saving model as', args.model_name, '...')
    torch.save(model.state_dict(), model_path)
    print('Model saved!')