import numpy as np
import spacetrace
from _3bp_source import generate_3bp_data

'''
    This example illusrates a usecase drawing multiple trajectories in the same space.
    A path in the Circular Restricted Three Body Problem (CR3BP) is plotted in both 
    inertial and normalized coordinates.
'''

# Generate both sets of data
states_inertial, epochs = generate_3bp_data('inertial')
states_normalized, _ = generate_3bp_data('normalized')

# Create the scene. Scale factor needs to be set to 1 as we are using normalized
# coordinates instead of meters
scene = spacetrace.Scene(scale_factor=1)
# Add both trajectories. Distinctive naming is important
scene.add_trajectory(epochs, states_inertial[:,:3], name='Orbit-Inertial', color='red')
scene.add_trajectory(epochs, states_normalized[:,:3], name='Orbit', color='green')

# Earth is at 0, 0
scene.add_static_body(0, 0, 0, radius=6.7/384, name='Earth', color='blue')

scene.add_static_body(0.8491, 0, 0, radius=0.03, name='L1', color='white', shape='cross')
scene.add_static_body(1.1678, 0, 0, radius=0.03, name='L2', color='white', shape='cross')

# Trajectory of the moon in inertial coordinates in the CR3BP
moon_path = np.array([np.cos(epochs), np.sin(epochs), np.zeros_like(epochs)]).T

scene.add_moving_body(epochs, moon_path, radius=1.6/384, name='Moon-Inertial', color='white')
scene.add_trajectory(epochs, moon_path, name='Moon-Inertial-Trajectory', color='white')
scene.add_static_body(1, 0, 0, radius=1.6/384, name='Moon', color='white')

spacetrace.show_scene(scene, focus='Orbit')
