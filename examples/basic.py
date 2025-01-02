import numpy as np
import spacetrace

def circular_orbit():
    N = 30_000
    tt = np.linspace(0, 3600*1.5, N)
    thetas = np.linspace(0, 2*np.pi, N)
    rr = np.array([np.cos(thetas), np.sin(thetas), np.zeros_like(thetas)]).T * 7e6

    scene = spacetrace.Scene()
    scene.add_trajectory(tt, rr)
    scene.add_static_body(0, 0, 0, radius=6.7e6, name='Earth', color=(0,0.5,1))
    spacetrace.show_scene(scene)

if __name__ == '__main__':
    circular_orbit()