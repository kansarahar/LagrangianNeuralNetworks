import os
import sys
import argparse
from tqdm import tqdm

import numpy as np
from physics import Double_Pendulum

# ----------------------------------------------------------
#   Args
# ----------------------------------------------------------

parser = argparse.ArgumentParser(
    description='A tool used for creating double pendulum experimental data',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--experiment', dest='experiment', type=str, choices=['double_pendulum'], default='double_pendulum', help='the type of experiment that the LNN is learning')

parser.add_argument('--num-experiments', dest='num_experiments', type=int, default=100, help='the number of random double pendulum experimental starting conditions to be used')
parser.add_argument('--step-size', dest='step_size', type=float, default=0.01, help='the time step size (in seconds); WARNING - values > 0.01 may cause instability')
parser.add_argument('--duration', dest='duration', type=float, default=100, help='the duration of each experiment (in seconds)')

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

# directory where data will be stored
data_path = os.path.join(dir_name, 'data', '%s_data' % args.experiment)
data_file_path = os.path.join(data_path, 'data.npy')
params_path = os.path.join(data_path, 'params.npy')
os.makedirs(data_path, exist_ok=True)

# ----------------------------------------------------------
#   Data Creation
# ----------------------------------------------------------

params = (args.m1, args.m2, args.l1, args.l2, args.g)

# generate experiments with random starting conditions for t1, t2
num_exp = args.num_experiments
starting_conditions = np.random.rand(num_exp, 2) * np.pi * 2
experiments = [Double_Pendulum(starting_conditions[i][0], starting_conditions[i][1], 0, 0, *params) for i in range(num_exp)]

# for each experiment, step through and save state data
print('generating double pendulum data...')
total_steps = int(args.duration / args.step_size)
data = np.zeros((total_steps * num_exp, 6))
for step in tqdm(range(total_steps)):
    for (idx, experiment) in enumerate(experiments):
        w1, w2, a1, a2 = experiment.get_current_derivs_analytical()
        data[num_exp * step + idx] = np.concatenate((experiment.state, (a1, a2)))
        experiment.step_analytical()

print('saving data...')
np.save(data_file_path, data)
np.save(params_path, np.array(params))
print('data saved to', data_file_path)
print('model params saved to', params_path)
print('data shape:', data.shape)
print('model params: (m1, m2, l1, l2, g) =', params)
