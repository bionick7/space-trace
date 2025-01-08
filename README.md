# Space Trace
Spacetrace is a simple, lightweight, low-abstraction tool for visualizing astrodynamic trajectories.
Spacetrace is agnostic of the physics, coordinate systems and the tools to generate the data. It's sole purpose is to
plot trajectories and inspect them visually.

This tool should be used similarly to matplotlib's 3d plotting tool, but optimized for interactivity and inspection, 
as well as drawing large trajectories at a smooth framerate.

<!--![Screenshot 1](/images/img1.png) -->
![Screenshot 2](/images/img2.png)

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
    scene.add_static_body(0, 0, 0, radius=6.7e6, name='Earth', color='blue')
    spacetrace.show_scene(scene)
```
where `states` is a numpy array of size N x 3 or N x 6 and  epochs is a numpy array of size N, 
with the corresponding times values. This program will draw the trajectory and add a blue sphere,
representing Earth for reference.

For more details see the documentation within the source files (`spacetrace/*`) or the examples (`examples/*`)

## GUI

- Middle mouse button to pan camera
- RMB click to offset camera vertically
- shift + RMB to offset camera horizontally
- Drag slider at the bottom to readjust time
- Left click on entity label (top left) to hide/show
- Press F while hovering over entity label to focus

## Changelog
### 0.1.1
- Update readme.md

### 0.1.2
- Update how color palette works
- Older python version should now work (as matched is no longer required)
- Fixed Color type-hinting input
- Updated raylib requirement to only support version 5.5 due to recent breaking api change
- Clamped camera pitch to -pi/2 ... pi/2, preventing rendering glitch

### 0.2.0
- Fixed bug of add_trajectory changing the output
- Added transforms and vectors
- BREAKING: User is now responsible for object initialization, i.e.
  `add_xxx(...)` notation is replaced with more flexible `add(Xxx(...))`

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
