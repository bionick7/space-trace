import numpy as np
import spacetrace

# Generate a simple circular orbit and epochs
N = 30_000
epochs = np.linspace(0, 3600*1.5, N)
thetas = np.linspace(0, 2*np.pi, N)
rr = np.array([np.cos(thetas), np.sin(thetas), np.zeros_like(thetas)]).T * 7e6

# Create the scene
scene = spacetrace.Scene()

# Add the generated trajectory
scene.add_trajectory(epochs, rr)

# Create a static, blue body at the center, representing Earth
scene.add_static_body(0, 0, 0, radius=6.7e6, name='Earth', color=(0,0.5,1))

# Show scene
spacetrace.show_scene(scene)
