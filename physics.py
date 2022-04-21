import sys
import math
import numpy as np
import torch
from torch import tensor
from torch.autograd.functional import jacobian, hessian

# ---------------------------------------------------------- 
#   Simple Runge Kutta Method Implementation
# ----------------------------------------------------------

def RK4_step(func, y0, t0, dt=0.01):
    '''
    Given a function for calculating the derivative of some known scalar or vector function y(t),
    as well as an initial value for y, calculate the next value of y
    
    :param func: the function f(y, t) that returns the derivative of y
    :param y0: (vector or scalar) initial value of y at time t0
    :param t0: (scalar) time t0
    :param h: step size
    :return: (y1, t1)
    '''

    k1 = func(y0, t0)
    k2 = func(y0 + dt*k1/2, t0 + dt/2)
    k3 = func(y0 + dt*k2/2, t0 + dt/2)
    k4 = func(y0 + dt*k3, t0 + dt)

    dy = dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6 
    
    return dy

# ----------------------------------------------------------
#   Double Pendulum
# ----------------------------------------------------------

class Double_Pendulum:
    def __init__(self, t1, t2, w1=0, w2=0, m1=1, m2=1, l1=1, l2=1, g=9.8, lnn=None):
        '''
        :param t1: the initial angle of the first mass w.r.t the y-axis (radians)
        :param t2: the initial angle of the second mass w.r.t the y-axis (radians)
        :param w1: the initial time rate of change of t1 (radians/s)
        :param w2: the initial time rate of change of t2 (radians/s)
        :param m1: the mass of the first pendulum (kg)
        :param m1: the mass of the second pendulum (kg)
        :param l1: the length of the first pendulum wire (m)
        :param l2: the length of the second pendulum wire (m)
        :param g: the acceleration due to gravity (m/s^2)
        :param lnn: an optional lnn model that can be attached to this experiment
        '''
        
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g
        self.state = tensor([t1, t2, w1, w2]).float()
        self.lnn = lnn

    def get_cartesian_coords(self, state=None):
        state = state if state is not None else self.state
        t1, t2, w1, w2 = state
        x1, y1 = self.l1 * torch.sin(t1), self.l1 * torch.cos(t1)
        x2, y2 = x1 + self.l2 * torch.sin(t2), y1 + self.l2 * torch.cos(t2)
        return torch.stack([x1, y1, x2, y2])

    def get_potential_energy(self, state=None):
        state = state if state is not None else self.state
        x1, y1, x2, y2 = self.get_cartesian_coords(state)
        return -self.g * (self.m1 * y1 + self.m2 * y2)

    def get_kinetic_energy(self, state=None):
        state = state if state is not None else self.state
        t1, t2, w1, w2 = state
        v1, v2 = self.l1 * w1, self.l2 * w2
        return 0.5 * (self.m1 * v1**2 + self.m2 * (v1**2 + v2**2 + 2*v1*v2*torch.cos(t1 - t2)))

    def get_total_energy(self, state=None):
        state = state if state is not None else self.state
        return self.get_kinetic_energy(state) + self.get_potential_energy(state)

    def get_lagrangian(self, state=None):
        state = state if state is not None else self.state
        return self.get_kinetic_energy(state) - self.get_potential_energy(state)

    def get_derivs_analytical(self, state=None, t=0):
        '''
        double pendulum dynamics from https://github.com/MilesCranmer/lagrangian_nns/blob/master/experiment_dblpend/physics.py

        :param state: the angles and angular velocities of the two masses
        :param t: dummy variable for runge-kutta calculation
        :returns: the angular velocities and accelerations of the two masses
        '''
        state = state if state is not None else self.state

        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g
        t1, t2, w1, w2 = state.tolist() # angles and angular velocities of the two masses

        a1 = (l2 / l1) * (m2 / (m1 + m2)) * math.cos(t1 - t2)
        a2 = (l1 / l2) * math.cos(t1 - t2)
        f1 = -(l2 / l1) * (m2 / (m1 + m2)) * (w2**2) * math.sin(t1 - t2) - (g / l1) * math.sin(t1)
        f2 = (l1 / l2) * (w1**2) * math.sin(t1 - t2) - (g / l2) * math.sin(t2)
        g1 = (f1 - a1 * f2) / (1 - a1 * a2)
        g2 = (f2 - a2 * f1) / (1 - a1 * a2)

        # return derivative of state
        return tensor([w1, w2, g1, g2]).float()

    def get_derivs_lagrangian(self, state=None, t=0):
        state = state if state is not None else self.state
 
        H = hessian(self.get_lagrangian, state)
        J = jacobian(self.get_lagrangian, state)

        A = J[:2]
        B = H[2:, 2:]
        C = H[2:, :2]

        q_tt = torch.linalg.pinv(B) @ (A - C @ state[2:])
        return torch.cat((state[2:], q_tt))

    def get_derivs_lnn(self, state=None, t=0):
        state = state if state is not None else self.state
        if (not self.lnn):
            print('No LNN is attached to this experiment')
            sys.exit(1)
        lnn_result = self.lnn(state)
        return torch.cat((state[2:], lnn_result))

    def step_analytical(self, dt=0.01):
        step = RK4_step(self.get_derivs_analytical, self.state, 0, dt)
        self.state += step
        self.state[0] %= (2*np.pi) # angles should be [0, 2pi)
        self.state[1] %= (2*np.pi)
        return step

    def step_lagrangian(self, dt=0.01):
        step = RK4_step(self.get_derivs_lagrangian, self.state, 0, dt)
        self.state += step
        self.state[0] %= (2*np.pi) # angles should be [0, 2pi)
        self.state[1] %= (2*np.pi)
        return step

    def step_lnn(self, dt=0.01):
        step = RK4_step(self.get_derivs_lnn, self.state, 0, dt)
        self.state += step
        self.state[0] %= (2*np.pi) # angles should be [0, 2pi)
        self.state[1] %= (2*np.pi)
        return step

class Spring_Pendulum:
    def __init__(self, theta, r, theta_t=0, r_t=0, m=1, k=10, r0=1, g=9.8, lnn=None):
        '''
        :param theta: the initial angle of the mass w.r.t the y-axis (radians)
        :param r: the initial length of the spring (m)
        :param theta_t: the initial time rate of change of theta (radians/s)
        :param r_t: the initial time rate of change of r (m/s)
        :param m: the mass of the pendulum (kg)
        :param k: the spring constant (N/m)
        :param r0: the equilibrium length of the spring (m)
        :param g: the acceleration due to gravity (m/s^2)
        :param lnn: an optional lnn model that can be attached to this experiment
        '''
        
        self.m = m
        self.r0 = r0
        self.k = k
        self.g = g
        self.state = tensor([theta, r, theta_t, r_t]).float()
        self.lnn = lnn

    def get_cartesian_coords(self, state=None):
        state = state if state is not None else self.state
        theta, r, theta_t, r_t = state
        x, y = r * torch.sin(theta), r * torch.cos(theta)
        return torch.stack([x, y])

    def get_potential_energy(self, state=None):
        state = state if state is not None else self.state
        theta, r, theta_t, r_t = state
        return -self.m * self.g * r * torch.cos(theta) + 0.5 * self.k * (r - self.r0)**2

    def get_kinetic_energy(self, state=None):
        state = state if state is not None else self.state
        theta, r, theta_t, r_t = state
        return 0.5 * self.m * (r_t**2 + (r * theta_t)**2)

    def get_total_energy(self, state=None):
        state = state if state is not None else self.state
        return self.get_kinetic_energy(state) + self.get_potential_energy(state)

    def get_lagrangian(self, state=None):
        state = state if state is not None else self.state
        return self.get_kinetic_energy(state) - self.get_potential_energy(state)

    def get_derivs_analytical(self, state=None, t=0):
        '''
        :param state: the angles and angular velocities of the two masses
        :param t: dummy variable for runge-kutta calculation
        :returns: the angular velocities and accelerations of the two masses
        '''
        state = state if state is not None else self.state

        m, k, r0, g = self.m, self.k, self.r0, self.g
        theta, r, theta_t, r_t = state.tolist() # angles and angular velocities of the two masses

        r = r if r is not 0 else 1e-3
        theta_tt = -(g * np.sin(theta) + 2 * r_t * theta_t) / r
        r_tt = r * theta_t**2 + g * np.cos(theta) - k * (r - r0) / m

        # return derivative of state
        return tensor([theta_t, r_t, theta_tt, r_tt]).float()

    def get_derivs_lagrangian(self, state=None, t=0):
        state = state if state is not None else self.state
 
        H = hessian(self.get_lagrangian, state)
        J = jacobian(self.get_lagrangian, state)

        A = J[:2]
        B = H[2:, 2:]
        C = H[2:, :2]

        q_tt = torch.linalg.pinv(B) @ (A - C @ state[2:])
        return torch.cat((state[2:], q_tt))

    def get_derivs_lnn(self, state=None, t=0):
        state = state if state is not None else self.state
        if (not self.lnn):
            print('No LNN is attached to this experiment')
            sys.exit(1)
        lnn_result = self.lnn(state)
        return torch.cat((state[2:], lnn_result))

    def step_analytical(self, dt=0.01):
        step = RK4_step(self.get_derivs_analytical, self.state, 0, dt)
        self.state += step
        self.state[0] %= (2*np.pi) # angles should be [0, 2pi)
        return step

    def step_lagrangian(self, dt=0.01):
        step = RK4_step(self.get_derivs_lagrangian, self.state, 0, dt)
        self.state += step
        self.state[0] %= (2*np.pi) # angles should be [0, 2pi)
        return step

    def step_lnn(self, dt=0.01):
        step = RK4_step(self.get_derivs_lnn, self.state, 0, dt)
        self.state += step
        self.state[0] %= (2*np.pi) # angles should be [0, 2pi)
        return step

class Cart_Pendulum:
    def __init__(self, d, theta, d_t=0, theta_t=0, mc=1, mp=1, k=10, d0=1, l=1, g=9.8, lnn=None):
        '''
        :param d: the initial length of the spring (m)
        :param theta: the initial angle of the mass w.r.t the y-axis (radians)
        :param d_t: the initial time rate of change of d (m/s)
        :param theta_t: the initial time rate of change of theta (radians/s)
        :param mc: the mass of the cart (kg)
        :param mp: the mass of the pendulum (kg)
        :param k: the spring constant (m)
        :param d0: the equilibrium length of the spring (m)
        :param l: the length of the pendulum wire (m)
        :param g: the acceleration due to gravity (m/s^2)
        :param lnn: an optional lnn model that can be attached to this experiment
        '''

        self.mc = mc
        self.mp = mp
        self.d0 = d0
        self.k = k
        self.l = l
        self.g = g
        self.state = tensor([d, theta, d_t, theta_t]).float()
        self.lnn = lnn

    def get_cartesian_coords(self, state=None):
        state = state if state is not None else self.state
        d, theta, d_t, theta_t = state
        xc, yc, xp, yp = d - self.d0, tensor(0), d - self.d0 + self.l * torch.sin(theta), self.l * torch.cos(theta)
        return torch.stack([xc, yc, xp, yp])

    def get_potential_energy(self, state=None):
        state = state if state is not None else self.state
        d, theta, d_t, theta_t = state
        return -self.mp * self.g * torch.cos(theta) + 0.5 * self.k * (d - self.d0)**2

    def get_kinetic_energy(self, state=None):
        state = state if state is not None else self.state
        d, theta, d_t, theta_t = state
        return 0.5 * (self.mc * d_t**2 + self.mp * (self.l * theta_t)**2)

    def get_total_energy(self, state=None):
        state = state if state is not None else self.state
        return self.get_kinetic_energy(state) + self.get_potential_energy(state)

    def get_lagrangian(self, state=None):
        state = state if state is not None else self.state
        return self.get_kinetic_energy(state) - self.get_potential_energy(state)

    def get_derivs_analytical(self, state=None, t=0):
        '''
        :param state: the angles and angular velocities of the two masses
        :param t: dummy variable for runge-kutta calculation
        :returns: the angular velocities and accelerations of the two masses
        '''
        state = state if state is not None else self.state

        d, theta, d_t, theta_t = state.tolist()

        d_tt = -self.k * (d - self.d0) / self.mc
        theta_tt = -self.g * np.sin(theta) / self.l**2
        # return derivative of state
        return tensor([d_t, theta_t, d_tt, theta_tt]).float()

    def get_derivs_lagrangian(self, state=None, t=0):
        state = state if state is not None else self.state
 
        H = hessian(self.get_lagrangian, state)
        J = jacobian(self.get_lagrangian, state)

        A = J[:2]
        B = H[2:, 2:]
        C = H[2:, :2]

        q_tt = torch.linalg.pinv(B) @ (A - C @ state[2:])
        return torch.cat((state[2:], q_tt))

    def get_derivs_lnn(self, state=None, t=0):
        state = state if state is not None else self.state
        if (not self.lnn):
            print('No LNN is attached to this experiment')
            sys.exit(1)
        lnn_result = self.lnn(state)
        return torch.cat((state[2:], lnn_result))

    def step_analytical(self, dt=0.01):
        step = RK4_step(self.get_derivs_analytical, self.state, 0, dt)
        self.state += step
        self.state[0] %= (2*np.pi) # angles should be [0, 2pi)
        return step

    def step_lagrangian(self, dt=0.01):
        step = RK4_step(self.get_derivs_lagrangian, self.state, 0, dt)
        self.state += step
        self.state[0] %= (2*np.pi) # angles should be [0, 2pi)
        return step

    def step_lnn(self, dt=0.01):
        step = RK4_step(self.get_derivs_lnn, self.state, 0, dt)
        self.state += step
        self.state[0] %= (2*np.pi) # angles should be [0, 2pi)
        return step
