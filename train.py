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

parser.add_argument('--model-name', dest='model_name', type=str, default='model.pt', help='the name of the LNN model to be saved or loaded')
parser.add_argument('--overwrite', dest='overwrite', action='store_true', help='indicates that you want to entirely rewrite a model')

parser.add_argument('--lr', dest='lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--epochs', dest='epochs', type=int, default=1, help='the number of passes of the entire training dataset')

parser.add_argument('--num-workers', dest='num_workers', type=int, default=0, help='how many subprocesses to use for data loading - 0 means that the data will be loaded in the main process')
parser.add_argument('--no-cuda', dest='no_cuda', action='store_true', help='indicates that you do not want to use cuda, even if it is available')

args = parser.parse_args()

# ----------------------------------------------------------
#   File Paths
# ----------------------------------------------------------

dir_name = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(dir_name, 'data', '%s_data' % args.experiment)
data_path = os.path.join(data_dir, 'data.npy')
params_path = os.path.join(data_dir, 'params.npy')
model_dir = os.path.join(dir_name, 'models', '%s_model' % args.experiment)
model_path = os.path.join(model_dir, args.model_name)
os.makedirs(model_dir, exist_ok=True)

# ----------------------------------------------------------
#   CUDA
# ----------------------------------------------------------

use_cuda = torch.cuda.is_available() and not args.no_cuda
device = torch.device('cuda:0' if use_cuda else 'cpu')
print('CUDA is available') if torch.cuda.is_available() else print('CUDA is unavailable')
print('Using CUDA') if use_cuda else print('Not using CUDA')

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
        self.data = np.load(data_path)
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
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
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
    

    for count, (state, accel) in enumerate(tqdm(training_generator)):
        state, accel = state.flatten().to(device), accel.flatten().to(device)
        optimizer.zero_grad()
        loss = loss_function(model(state), accel)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if count % 1000 == 0:
            print('\ncount=%s, saving model...' % count)
            torch.save(model.state_dict(), model_path)
            print('model saved')

    print('Epoch', epoch+1, 'complete! Total loss for this epoch:', epoch_loss)
    print('Saving model as', args.model_name, '...')
    torch.save(model.state_dict(), model_path)
    print('Model saved!')
print('Training complete')