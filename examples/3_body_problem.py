import numpy as np
import spacetrace
from _3bp_source import generate_3bp_data

# Normalized coordinates lead lets us know the scale in advance
states_inertial, epochs = generate_3bp_data('inertial')
states_normalized, _ = generate_3bp_data('normalized')

scene = spacetrace.Scene(scale_factor=1)
scene.add_trajectory(epochs, states_inertial[:,:3], name='Orbit-Inertial', color='red')
scene.add_trajectory(epochs, states_normalized[:,:3], name='Orbit', color='green')
scene.add_static_body(0, 0, 0, radius=6.7/384, name='Earth', color='blue')

moon_path = np.array([np.cos(epochs), np.sin(epochs), np.zeros_like(epochs)]).T
scene.add_moving_body(epochs, moon_path, radius=1.6/384, name='Moon-Inertial', color='white')
scene.add_trajectory(epochs, moon_path, name='Moon-Inertial-Trajectory', color='white')
scene.add_static_body(1, 0, 0, radius=1.6/384, name='Moon', color='white')

spacetrace.show_scene(scene, focus='Orbit')
