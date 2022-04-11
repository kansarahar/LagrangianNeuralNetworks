import numpy as np



# ---------------------------------------------------------- 
# Simple Runge Kutta Method Implementation
# ----------------------------------------------------------

def RK4_step(func, y0, t0, h=0.01):
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
    k2 = func(y0 + h*k1/2, t0 + h/2)
    k3 = func(y0 + h*k2/2, t0 + h/2)
    k4 = func(y0 + h*k3, t0 + h)

    y1 = y0 + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6 
    t1 = t0 + h
    
    return (y1, t1)

# ----------------------------------------------------------
# Double Pendulum
# ----------------------------------------------------------

def analytical_double_pendulum_fn(state, t=0, m1=1, m2=1, l1=1, l2=2, g=9.8):
    t1, t2, w1, w2 = state # angles and angular velocities of the two masses
    a1 = (l2 / l1) * (m2 / (m1 + m2)) * np.cos(t1 - t2)
    a2 = (l1 / l2) * np.cos(t1 - t2)
    f1 = -(l2 / l1) * (m2 / (m1 + m2)) * (w2**2) * np.sin(t1 - t2) - (g / l1) * np.sin(t1)
    f2 = (l1 / l2) * (w1**2) * np.sin(t1 - t2) - (g / l2) * np.sin(t2)
    g1 = (f1 - a1 * f2) / (1 - a1 * a2)
    g2 = (f2 - a2 * f1) / (1 - a1 * a2)

    # return derivative of state
    return np.array([w1, w2, g1, g2])

class Double_Pendulum:
    def __init__(self, m1, m2, l1, l2, t1, t2, g=9.8):
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g
        self.state = np.array([t1, t2, 0, 0])

    def get_cartesian_coords(self):
        t1, t2, w1, w2 = self.state
        x1, y1 = self.l1*np.sin(t1), self.l1*np.cos(t1)
        x2, y2 = x1 + self.l2*np.sin(t2) , y1 + self.l2*np.cos(t2)
        return np.array([x1, y1, x2, y2])

    def get_potential_energy(self):
        x1, y1, x2, y2 = self.get_cartesian_coords()
        return -self.g * (self.m1 * y1 + self.m2 * y2)

    def get_kinetic_energy(self):
        t1, t2, w1, w2 = self.state
        v1, v2 = self.l1 * w1, self.l2 * w2
        return 0.5 * (self.m1 * v1**2 + self.m2 * (v1**2 + v2**2 + 2*v1*v2*np.cos(t1 - t2)))

    def get_total_energy(self):
        return self.get_kinetic_energy() + self.get_potential_energy()

    def get_derivs(self, state, t=0):
        return analytical_double_pendulum_fn(state, t, self.m1, self.m2, self.l1, self.l2, self.g)

    def get_derivs(self, state, t=0):
        '''
        double pendulum dynamics from https://github.com/MilesCranmer/lagrangian_nns/blob/master/experiment_dblpend/physics.py

        :param state: the two angles and angular velocities of the pendulums
        :param t: dummy variable for runge-kutta calculation
        '''
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g
        t1, t2, w1, w2 = state # angles and angular velocities of the two masses

        a1 = (l2 / l1) * (m2 / (m1 + m2)) * np.cos(t1 - t2)
        a2 = (l1 / l2) * np.cos(t1 - t2)
        f1 = -(l2 / l1) * (m2 / (m1 + m2)) * (w2**2) * np.sin(t1 - t2) - (g / l1) * np.sin(t1)
        f2 = (l1 / l2) * (w1**2) * np.sin(t1 - t2) - (g / l2) * np.sin(t2)
        g1 = (f1 - a1 * f2) / (1 - a1 * a2)
        g2 = (f2 - a2 * f1) / (1 - a1 * a2)

        # return derivative of state
        return np.array([w1, w2, g1, g2])

    def step_analytical(self, dt=0.01):
        self.state[0] %= (2*np.pi)
        self.state[1] %= (2*np.pi)
        self.state = RK4_step(self.get_derivs, self.state, 0, dt)[0]

