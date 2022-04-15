import torch
import torch.nn as nn
from torch.autograd.functional import jacobian, hessian


# ----------------------------------------------------------
#   Model
# ----------------------------------------------------------

class Lagrangian(nn.Module):
    def __init__(self, num_params, hidden_dimensions = 128):
        super().__init__()
        self.output = nn.Sequential(
            nn.Linear(2*num_params, hidden_dimensions),
            nn.Softplus(),
            nn.Linear(hidden_dimensions, hidden_dimensions),
            nn.Softplus(),
            nn.Linear(hidden_dimensions, 1)
        )

    def forward(self, x):
        # x = (q0, q1, ..., q0_t, q1_t, ...)
        return self.output(x)

class LNN(nn.Module):
    '''
    A neural network composed of a lagrangian encoder and a derivative layer.
    To create an instance of lnn, pass in the number of parameters q.
    Inputs to the LNN should be of the form x=(q0, q1, ..., q0_t, q1_t, ...),
        where len(x) = 2 * num_params
    
    :param num_params: The number of coordinates q
    :hidden_dimensions (default 128): The number of dimensions of the hidden layers of the encoding network
    '''
    def __init__(self, num_params, hidden_dimensions = 128):
        super().__init__()
        self.num_params = num_params
        self.lagrangian = Lagrangian(num_params, hidden_dimensions)

    def forward(self, x):
        H = hessian(self.lagrangian, x, True, True)
        D = jacobian(self.lagrangian, x, True, True)
        dq0 = D[0, :self.num_params]
        dq1 = H[self.num_params:, self.num_params:]
        dq2 = H[:self.num_params, self.num_params:]
        return torch.linalg.inv(dq1) @ (dq0 - dq2 @ x[self.num_params:])