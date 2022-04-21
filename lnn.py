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
    Inputs to the LNN should be of the form x=(q0, q1, ..., q0_t, q1_t, ...),
        where len(x) = 2 * num_params
    
    :param num_params: The number of coordinates q
    :hidden_dimensions (default 128): The number of dimensions of the hidden layers of the encoding network
    '''
    def __init__(self, num_params, hidden_dimensions = 128):
        super().__init__()
        self.num_params = num_params

        self.lagrangian = nn.Sequential(
            nn.Linear(2*num_params, hidden_dimensions),
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
        # x = (q0, q1, ..., q0_t, q1_t, ...)
        H = hessian(self.lagrangian, x, True)
        J = jacobian(self.lagrangian, x, True)

        A = J[0, :self.num_params]
        B = H[self.num_params:, self.num_params:]
        C = H[self.num_params:, :self.num_params]

        return torch.linalg.pinv(B) @ (A - C @ x[self.num_params:])
