import numpy as np

# double pendulum dynamics from https://github.com/MilesCranmer/lagrangian_nns/blob/master/experiment_dblpend/physics.py
def analytical_double_pendulum_fn(state, m1=1, m2=1, l1=1, l2=2, g=9.8):
  t1, t2, w1, w2 = state # angles and angular velocities of the two masses
  a1 = (l2 / l1) * (m2 / (m1 + m2)) * np.cos(t1 - t2)
  a2 = (l1 / l2) * np.cos(t1 - t2)
  f1 = -(l2 / l1) * (m2 / (m1 + m2)) * (w2**2) * np.sin(t1 - t2) - (g / l1) * np.sin(t1)
  f2 = (l1 / l2) * (w1**2) * np.sin(t1 - t2) - (g / l2) * np.sin(t2)
  g1 = (f1 - a1 * f2) / (1 - a1 * a2)
  g2 = (f2 - a2 * f1) / (1 - a1 * a2)
  return np.array([w1, w2, g1, g2]) # returns d(state)/dt

 
