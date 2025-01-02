from typing import Literal

import numpy as np
import spacetrace

def cr3bp_dynamics(state, t, mu):

    x, y, z = state[:3]
    xdot, ydot, zdot = state[3:]

    state_derivative = np.zeros_like(state)
    state_derivative[:3] = state[3:]

    minor_dist = np.sqrt(np.sum(np.square([x - 1 + mu, y, z])))
    major_dist = np.sqrt(np.sum(np.square([x + mu, y, z])))
    state_derivative[3] = (
        2 * ydot + x
        - (1 - mu) * (x + mu) / major_dist**3
        - mu * (x - 1 + mu) / minor_dist**3
    )
    state_derivative[4] = -2 * xdot + y - (1 - mu) * y / major_dist**3 - mu * y / minor_dist**3
    state_derivative[5] = - (1 - mu) / major_dist**3 * z - mu / minor_dist**3 * z

    return state_derivative


def rk4_step(f, x, t, dt, *args):
    k1 = f(x, t, *args)
    k2 = f(x + k1*dt/2, t + dt/2, *args)
    k3 = f(x + k2*dt/2, t + dt/2, *args)
    k4 = f(x + k3*dt, t + dt, *args)
    return x + (k1 + 2*k2 + 2*k3 + k4)*dt/6



def circular_restriced_three_body_problem(initial_state, mu):
    time_step = 0.01
    epochs = np.arange(0, np.pi*2, time_step)
    states = np.zeros((len(epochs), 6))
    state = initial_state.copy()
    for i, t in enumerate(epochs):
        state = rk4_step(cr3bp_dynamics, state, t, time_step, mu)
        states[i] = state

    return states, epochs


def generate_3bp_data(mode: Literal['normalized', 'inertial']):
    L1 = 0.8490659859092115

    m_1 = 5.974e24  # kg
    m_2 = 7.348e22  # kg
    mu = m_2 / (m_1 + m_2)

    initial_state = np.array([L1, 0, .01, 0, 0, 0])

    states, epochs = circular_restriced_three_body_problem(initial_state, mu)
    transformed_states = np.zeros_like(states)
    if mode == 'inertial':
        transformed_states[:, 0] = np.cos(epochs) * states[:, 0] - np.sin(epochs) * states[:, 1]
        transformed_states[:, 1] = np.sin(epochs) * states[:, 0] + np.cos(epochs) * states[:, 1]
        transformed_states[:, 2] = states[:, 2]
        transformed_states[:, 3] = np.cos(epochs) * states[:, 3] - np.sin(epochs) * states[:, 4]
        transformed_states[:, 4] = np.sin(epochs) * states[:, 3] + np.cos(epochs) * states[:, 4]
        transformed_states[:, 5] = states[:, 5]
        return transformed_states, epochs
    return states, epochs