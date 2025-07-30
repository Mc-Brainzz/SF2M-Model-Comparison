
import numpy as np
from scipy.integrate import solve_ivp

def van_der_pol_ode(t, y, mu=1.0): return [y[1], mu * (1 - y[0]**2) * y[1] - y[0]]

def biased_van_der_pol_ode(t, y, mu=1.5, k=1.0, f=0.3):
    x, v = y
    return [v, mu * (1 - x**2) * v - k*x + f*np.cos(t)]

def euler_solver(func, y0, t_span, dt=0.05):
    t = np.arange(t_span[0], t_span[1], dt); ys = np.zeros((len(t), len(y0))); ys[0] = y0
    for i in range(len(t) - 1):
        if np.any(np.isnan(ys[i])) or np.any(np.isinf(ys[i])):
            ys[i:] = np.nan; return t, ys
        ys[i+1] = ys[i] + dt * np.array(func(t[i], ys[i]))
    return t, ys

def generate_data(n_samples=2000, t_eval_time=5.0):
    initial_conditions = (np.random.rand(n_samples, 2) - 0.5) * 6
    clean_data = np.array([solve_ivp(van_der_pol_ode, [0, t_eval_time], y0, dense_output=True).sol(t_eval_time) for y0 in initial_conditions])
    biased_data = []
    for y0 in initial_conditions:
        _, ys = euler_solver(biased_van_der_pol_ode, y0, [0, t_eval_time])
        biased_data.append(ys[-1])
    biased_data = np.array(biased_data)
    biased_data = biased_data[~np.isnan(biased_data).any(axis=1)]
    num_valid = biased_data.shape[0]
    return biased_data, clean_data[:num_valid]
