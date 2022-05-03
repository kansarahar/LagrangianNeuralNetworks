import torch
import torch.nn as nn
from torch.autograd.functional import jacobian, hessian


# ----------------------------------------------------------
#   Model
# ----------------------------------------------------------

class LNN(nn.Module):
    '''
    A neural network composed of a lagrangian encoder and a derivative layer.
    To create an instance of lnn, pass in the number of parameters q.
    Inputs to the LNN should be of the form x=(q0, q1, q0_t, q1_t),
    
    :hidden_dimensions (default 128): The number of dimensions of the hidden layers of the encoding network
    '''
    def __init__(self, hidden_dimensions = 128):
        super().__init__()

        self.lagrangian = nn.Sequential(
            nn.Linear(4, hidden_dimensions),
            nn.Softplus(),
            nn.Linear(hidden_dimensions, hidden_dimensions),
            nn.Softplus(),
            nn.Linear(hidden_dimensions, hidden_dimensions),
            nn.Softplus(),
            nn.Linear(hidden_dimensions, hidden_dimensions),
            nn.Softplus(),
            nn.Linear(hidden_dimensions, 1)
        )

    def forward(self, x):
        # x = (q0, q1, q0_t, q1_t)
        H = hessian(self.lagrangian, x, True)
        J = jacobian(self.lagrangian, x, True)

        A = J[0, :2]
        B = H[2:, 2:]
        C = H[2:, :2]

        return torch.linalg.pinv(B) @ (A - C @ x[2:])

class ClassicalNN(nn.Module):
    '''
    A basic linear network that will be used as a baseline to compare against the performace of the LNN.
    Inputs to this network should be of the form x=(q0, q1, q0_t, q1_t)
    
    :hidden_dimensions (default 128): The number of dimensions of the hidden layers of the encoding network
    '''
    def __init__(self, hidden_dimensions = 128):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(4, hidden_dimensions),
            nn.Softplus(),
            nn.Linear(hidden_dimensions, hidden_dimensions),
            nn.Softplus(),
            nn.Linear(hidden_dimensions, hidden_dimensions),
            nn.Softplus(),
            nn.Linear(hidden_dimensions, hidden_dimensions),
            nn.Softplus(),
            nn.Linear(hidden_dimensions, 2)
        )

    def forward(self, x):
        # x = (q0, q1, q0_t, q1_t)
        return self.layers(x)