from typing import Literal
import numpy as np

'''
    Generates data for example in '3_body_problem.py'.
'''

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


def get_sydonic_to_inertial_reference_frame(epochs):
    X, Y, Z = np.eye(3)
    transforms = np.zeros((len(epochs), 3, 3))
    transforms[:,:,0] =  np.outer(np.cos(epochs), X) + np.outer(np.sin(epochs), Y)
    transforms[:,:,1] = -np.outer(np.sin(epochs), X) + np.outer(np.cos(epochs), Y)
    transforms[:,:,2] =  Z
    return transforms


def generate_3bp_data(mode: Literal['sydonic', 'inertial']):
    L1 = 0.8490659859092115

    m_1 = 5.974e24  # kg
    m_2 = 7.348e22  # kg
    mu = m_2 / (m_1 + m_2)

    initial_state = np.array([L1, 0, .01, 0, 0, 0])

    states, epochs = circular_restriced_three_body_problem(initial_state, mu)
    if mode == 'inertial':
        transformed_states = np.zeros_like(states)
        transforms = get_sydonic_to_inertial_reference_frame(epochs)
        for i in range(len(epochs)):
            transformed_states[i,:3] = transforms[i] @ states[i,:3]
            transformed_states[i,3:] = transforms[i] @ states[i,3:]
        return transformed_states, epochs
    return states, epochs