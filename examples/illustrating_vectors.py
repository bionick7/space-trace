import numpy as np
from _3bp_source import rk4_step

import spacetrace

# Generate orbit, forces, etc
def f(y, t):
    mu_earth = 3.98e14

    dydt = np.zeros_like(y)
    dydt[:3] = y[3:]
    dydt[3:] = - mu_earth / np.linalg.norm(y[:3])**3 * y[:3]
    return dydt

epochs = np.linspace(0, 3600*4, 1000)
yy = np.zeros((len(epochs), 6))
dydt = np.zeros((len(epochs), 6))
rsw = np.zeros((len(epochs), 3, 3))

for i in range(len(epochs)):
    if i == 0:
        yy[0,0] = 7e6
        yy[0,4] = 8500
    else:
        yy[i] = rk4_step(f, yy[i-1], epochs[i-1], epochs[i] - epochs[i-1])
    dydt[i] = f(yy[i], epochs[i])
    rsw[i,:,0] = yy[i,:3] / np.linalg.norm(yy[i,:3])
    rsw[i,:,2] = np.eye(3)[2]
    rsw[i,:,1] = np.cross(rsw[i,:,2], rsw[i,:,0])

# Show output
scene = spacetrace.Scene()
scene.add(spacetrace.Trajectory(epochs, yy))
# Draw velocity and acceleration vectors over time (need to be scaled to new time)
scene.add(spacetrace.VectorShape(epochs, yy[:,:3], dydt[:,:3] * 1000, name="Velocities"))
scene.add(spacetrace.VectorShape(epochs, yy[:,:3], dydt[:,3:] * 1000**2, name="Accelerations", color="red"))
# Draw RSW coordinate system
scene.add(spacetrace.TransformShape(epochs, yy[:,:3], rsw / scene.scale_factor, name="RSW", axis_colors=('red', 'green', 'blue')))
# Add earth
scene.add(spacetrace.Body.fixed(0, 0, 0, radius=6731e3, name="Earth", color="blue"))
spacetrace.show_scene(scene)