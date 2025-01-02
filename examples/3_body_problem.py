import space_trace
from _3bp_source import generate_3bp_data

# Normalized coordinates lead lets us know the scale in advance
states, epochs = generate_3bp_data()

scene = space_trace.Scene(scale_factor=1)
scene.add_trajectory(epochs, states, name='Orbit')
scene.add_static_body(0, 0, 0, radius=6.7/384, name='Earth', color=(0,0.5,1))
scene.add_static_body(1, 0, 0, radius=1.6/384, name='Moon', color=(0.5,0.5,0.5))

# Detailed visualization control to set focus
space_trace.show_scene(scene, focus='Orbit')