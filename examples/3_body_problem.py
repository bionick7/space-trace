import numpy as np
import spacetrace
from _3bp_source import generate_3bp_data, get_sydonic_to_inertial_reference_frame

'''
    This example illusrates a usecase drawing multiple trajectories in the same space.
    A path in the Circular Restricted Three Body Problem (CR3BP) is plotted in both 
    inertial and normalized coordinates.
'''

# Generate both sets of data
states_inertial, epochs = generate_3bp_data('inertial')
states_sydonic, _ = generate_3bp_data('sydonic')
transforms = get_sydonic_to_inertial_reference_frame(epochs)

angular_momentum = np.cross(states_inertial[:,:3], states_inertial[:,3:])

# Create the scene. Scale factor needs to be set to 1 as we are using normalized
# coordinates instead of meters
scene = spacetrace.Scene(scale_factor=1)
# Add both trajectories. Distinctive naming is important
scene.add(spacetrace.Trajectory(epochs, states_inertial[:,:3], name='Orbit-Inertial', color='red'))
scene.add(spacetrace.Trajectory(epochs, states_sydonic[:,:3], name='Orbit', color='green'))
# Angular momentum in inertial frame, to demonstrate Vector
scene.add(spacetrace.VectorShape(epochs, states_sydonic[:,:3], angular_momentum, name='Angular Momentum', color='grey'))

# Earth is at 0, 0
scene.add(spacetrace.Body.fixed(0, 0, 0, radius=6.7/384, name='Earth', color='blue'))

scene.add(spacetrace.Body.fixed(0.8491, 0, 0, radius=0.03, name='L1', color='white', shape='cross'))
scene.add(spacetrace.Body.fixed(1.1678, 0, 0, radius=0.03, name='L2', color='white', shape='cross'))

# Trajectory of the moon in inertial coordinates in the CR3BP
moon_path = np.array([np.cos(epochs), np.sin(epochs), np.zeros_like(epochs)]).T
scene.add(spacetrace.Body(epochs, moon_path, radius=1.6/384, name='Moon-Inertial', color='white'))
scene.add(spacetrace.Trajectory(epochs, moon_path, name='Moon-Inertial-Trajectory', color='white'))

# Fixed moon in sydonic coordinates
scene.add(spacetrace.Body.fixed(1, 0, 0, radius=1.6/384, name='Moon', color='white'))

# Sydonic to inertial transform
scene.add(spacetrace.TransformShape(epochs, np.zeros((len(epochs), 3)), transforms*.4, "Sydonic frame"))

spacetrace.show_scene(scene, focus='Orbit')
