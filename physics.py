import numpy as np

# double pendulum dynamics from https://github.com/MilesCranmer/lagrangian_nns/blob/master/experiment_dblpend/physics.py
def analytical_double_pendulum_fn(state, t=0, m1=1, m2=1, l1=1, l2=2, g=9.8):
    t1, t2, w1, w2 = state # angles and angular velocities of the two masses
    a1 = (l2 / l1) * (m2 / (m1 + m2)) * np.cos(t1 - t2)
    a2 = (l1 / l2) * np.cos(t1 - t2)
    f1 = -(l2 / l1) * (m2 / (m1 + m2)) * (w2**2) * np.sin(t1 - t2) - (g / l1) * np.sin(t1)
    f2 = (l1 / l2) * (w1**2) * np.sin(t1 - t2) - (g / l2) * np.sin(t2)
    g1 = (f1 - a1 * f2) / (1 - a1 * a2)
    g2 = (f2 - a2 * f1) / (1 - a1 * a2)
    return np.array([w1, w2, g1, g2]) # returns d(state)/dt

 
def RK4_step(func, y0, t0, h=0.01):
    """
    Given a function for calculating the derivative of some known scalar or vector function y(t),
    as well as an initial value for y, calculate the next value of y
    
    :param func: the function f(y, t) that returns the derivative of y
    :param y0: (vector or scalar) initial value of y at time t0
    :param t0: (scalar) time t0
    :param h: step size
    :return: (y1, t1)
    """

    k1 = func(y0, t0)
    k2 = func(y0 + h*k1/2, t0 + h/2)
    k3 = func(y0 + h*k2/2, t0 + h/2)
    k4 = func(y0 + h*k3, t0 + h)

    y1 = y0 + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6 
    t1 = t0 + h
    
    return (y1, t1)

