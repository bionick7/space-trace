# Space Trace
Spacetrace is a simple, lightweight, low-abstraction tool for visualizing astrodynamic trajectories.
Spacetrace is agnostic of the physics, coordinate systems and the tools to generate the data. It's sole purpose is to
plot trajectories and inspect them visually.

This tool should be used similarly to matplotlib's 3d plotting tool, but optimized for interactivity and inspection, 
as well as drawing large trajectories at a smooth framerate.

![Screenshot 1](https://images/img1.png)
![Screenshot 2](https://images/img2.png)

## Installation
Spacetrace is a standard python package available on pypi:
```bash
pip install spacetrace
```
Spacetrace only depends on numpy as data interface and pyray for drawing.

## Usage

The most basic usecase is as follows:
```py
    import spacetrace

    scene = spacetrace.Scene()
    scene.add_trajectory(epochs, states)
    scene.add_static_body(0, 0, 0, radius=6.7e6, name='Earth', color=(0,0.5,1))
    spacetrace.show_scene(scene)
```
where `states` is a numpy array of size N x 3 or N x 6 and  epochs is a numpy array of size N, 
with the corresponding times values. This program will draw the trajectory and add a blue sphere,
representing Earth for reference.

For more details see the documentation within the source files (`spacetrace/*`) or the examples (`examples/*`)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
