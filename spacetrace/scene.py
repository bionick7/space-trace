from typing import Literal, Callable
from math import ceil

import numpy as np
import pyray as rl
import raylib as rl_raw

ffi = rl.ffi

DEFAULT_WINDOWN_WIDTH = 800
DEFAULT_WINDOW_HEIGHT = 600
DEFAULT_FRAME_NAME = 'Default Frame'


# COLOR HANDLING
# ==============

def hex_to_color(hex: int) -> tuple[float, float, float]:
    return (
        ((hex >> 16) & 0xFF) / 255,
        ((hex >> 8) & 0xFF) / 255, 
        (hex & 0xFF) / 255
    )
__palette = {}
__palette['bg'] = hex_to_color(0x12141c)
__palette['blue'] = hex_to_color(0x454e7e)
__palette['green'] = hex_to_color(0x4Fc76C)
__palette['red'] = hex_to_color(0xFF5155)
__palette['white'] = hex_to_color(0xfaf7d5)
__palette['gray'] = hex_to_color(0x735e4c)
__palette['main'] = __palette['white']
__palette['accent'] = __palette['blue']
__palette['grey'] = __palette['gray']

_ColorIDLiteral = Literal['bg', 'blue', 'green', 'red', 'white', 'main', 'accent', 'gray', 'grey']
_ColorType = tuple[float, float, float] | _ColorIDLiteral

def default_palette(name: _ColorIDLiteral) -> tuple[float, float, float]:
    '''
    Default color palette for spacetrace.
    Simple function that returns the corresponding RGB values for a given color name.
    Returns aggressive magenta as error color.

    Pallette is a modification of https://lospec.com/palette-list/offshore
    '''

    return __palette.get(name, (1, 0, 1))


class Color():
    '''
    Simple class to handle colors.
    '''
    def __init__(self, c: _ColorType, 
                 palette: Callable[[_ColorIDLiteral], tuple[float, float, float]]=default_palette):
        if isinstance(c, tuple):
            self.rgb = c
        self.rgb = palette(c)

    def as_rl_color(self) -> rl.Color:
        r, g, b = self.rgb
        return rl.Color(int(r*255), int(g*255), int(b*255), 255)
    
    def as_array(self) -> rl.Color:
        return np.array([*self.rgb, 1], np.float32)


#     SCENE
# ==============


class SceneEntity():
    '''
    Base class for all entities in the scene
    Has a name, color, visibility flag as well as a trajectory through time.
    '''

    def __init__(self, name: str, color: _ColorType='main'):
        '''
        Initializes the entity with a name and a color
        name: str
            Identifieer used in the UI. Should be unique
        color: tuple[float, float, float] or str
            Color of the entity. Can be a tuple of RGB values or a identifies a 
            color in the color palette.
            Default colors are 'main', 'accent', 'bg', 'blue', 'green', 'red', 'white'
        '''
        self.name = name
        self.color = Color(color)
        self.positions = np.zeros((1,3))
        self.epochs = np.zeros(1)
        self.is_visible = True
    
    def set_trajectory(self, time: np.ndarray, positions: np.ndarray):
        self.epochs = time
        self.positions = positions

    def get_position(self, time: float):
        if len(self.epochs) == 1:
            return self.positions[0]
        idx = np.searchsorted(self.epochs, time)
        if idx == 0:
            return self.positions[0]
        if idx == len(self.epochs):
            return self.positions[-1]
        t0, t1 = self.epochs[idx-1], self.epochs[idx]
        p0, p1 = self.positions[idx-1], self.positions[idx]
        alpha = (time - t0) / (t1 - t0)
        return p0 + alpha * (p1 - p0)


class ReferenceFrame(SceneEntity):
    ''' 
    The main reference frame in the scene. Currently the scene only has one
    automatically created reference frame.
    '''
    def __init__(self, name: str, color: _ColorType='main'):
        super().__init__(name, color)
        self.positions = np.zeros((1,3))
        self.epochs = np.zeros(1)
        self.transforms = np.eye(3)[np.newaxis,:,:]


class Trajectory(SceneEntity):
    '''
    A trajectory is a sequence of positions in space over time.
    Internally, a trajectory can be multiple draw calls. 
    This is mostly to access the metadata and to support get_position
    '''
    def __init__(self, epochs: np.ndarray, positions: np.ndarray, 
                 name: str, color: _ColorType='main'):
        super().__init__(name, color)
        self.positions = positions
        self.epochs = epochs


class Body(SceneEntity):
    '''
    A body is a static or moving object in the scene.
    Represented by a colored sphere of a certain radius.
    Mostly represents a celestial body.
    '''
    def __init__(self, name: str, radius: float, color: _ColorType='main'):
        super().__init__(name, color)
        self.radius = radius


class Scene():
    '''
    All the data that is needed to render a scene in spacetrace
    The scene is created and populated by the user.
    Entities can be Trajectories, Bodies or the main Reference Frame.
    '''

    def __init__(self, scale_factor: float=1e-7):
        '''
        Initializes the scene with a scale factor. The scale factor is used to convert
        provided positions into rendering units. A scale factor of 10^-7 is provided,
        assuming that positions are in meters and the trajectories are on the scale of
        earth orbits.

        Adjust scale_factor, such that the largest dimensions is on the order of magnitude 1-10
        '''
        self.scale_factor = scale_factor
        self.trajectories = []
        self.bodies = []

        self.trajectory_patches = []
        self.time_bounds = [np.inf, -np.inf]
        self.reference_frame = ReferenceFrame(DEFAULT_FRAME_NAME)

    def add_trajectory(self, epochs: np.ndarray, states: np.ndarray, name:str="SpaceCraft", 
                       color: _ColorType='white') -> None:
        '''
        Adds a trajectory to the scene. The trajectory is a sequence of states in space over time.
        epochs: np.ndarray (N,)
            Time values for each state
        states: np.ndarray (N, 3) or (N, 6)
            Position or Positions and velocity states for each time step
            velocities are used to inform the direction of the curve for better rendering
            if velocities are not provided, they are calculated from the positions
        name: str
            Identifier used in the UI. Should be unique
        color: tuple[float, float, float] or str
            Color of the trajectory. Can be a tuple of RGB values or a identifies a 
            color in the color palette.
            Default colors are 'main', 'accent', 'bg', 'blue', 'green', 'red', 'white'
        '''
        if len(epochs) != len(states):
            raise ValueError("Epochs and states should have the same length")
        total_length = len(states)
        parts = int(ceil(total_length / 2**14))

        if states.shape[1] == 3:
            states[:,:3] = states[:,(0,2,1)] * np.array([1,1,-1])[np.newaxis,:] * self.scale_factor
        elif states.shape[1] == 6:
            states[:,:3] = states[:,(0,2,1)] * np.array([1,1,-1])[np.newaxis,:] * self.scale_factor
            states[:,3:] = states[:,(3,5,4)] * np.array([1,1,-1])[np.newaxis,:] * self.scale_factor
        else:
            raise ValueError("States should have 3 or 6 columns")

        for i in range(parts):
            start = max(0, i * 2**14 - 1)  # Link up t0.0.0 the previous one
            end = min((i+1) * 2**14, total_length)
            self._add_trajectory_path(
                epochs[start:end], states[start:end], len(self.trajectories))
            
        trajectory = Trajectory(epochs, states[:,:3], name, color)
        self.trajectories.append(trajectory)
        if self.time_bounds[0] > epochs[0]:
            self.time_bounds[0] = epochs[0]
        if self.time_bounds[1] < epochs[-1]:
            self.time_bounds[1] = epochs[-1]

    def add_static_body(self, x: float, y: float, z: float, radius: float=6e6, 
                        color: _ColorType=(1,1,1), name: str="Central Body") -> None:
        ''' 
        Adds a static body (without trajectory) to the scene. Usefull for central bodies
        in a body-centric reference frame.
        x: float
        y: float
        z: float
            Position of the body in space, ususally 0, 0, 0 for central bodies
        radius: float
            Radius of the body, in the same units as positions are provided
        color: tuple[float, float, float] or str
            Color of the body. Can be a tuple of RGB values or a identifies a 
            color in the color palette.
            Default colors are 'main', 'accent', 'bg', 'blue', 'green', 'red', 'white'
        '''
        body = Body(name, radius * self.scale_factor, color)
        body.set_trajectory(np.zeros(1), np.array([[x, z, -y]]) * self.scale_factor)
        self.bodies.append(body)

    def add_moving_body(self, epochs: np.ndarray, r: np.ndarray, radius: float=6e6, 
                        color: _ColorType=(1,1,1), name: str="Central Body") -> None:
        '''
        Similar to add_static_body, but instear of a single position, a trajectory is provided
        '''
        body = Body(name, radius * self.scale_factor, color)
        render_space_positions = r[:,(0,2,1)] * np.array([1,1,-1])[np.newaxis,:] * self.scale_factor
        body.set_trajectory(epochs, render_space_positions)
        self.bodies.append(body)
        if self.time_bounds[0] > epochs[0]:
            self.time_bounds[0] = epochs[0]
        if self.time_bounds[1] < epochs[-1]:
            self.time_bounds[1] = epochs[-1]

    def _add_trajectory_path(self, epochs: np.ndarray, states: np.ndarray, trajectory_index: int):
        '''
        Helper function for add_trajectory. Handle a lot of the low-level rendering setup
        '''
        if not rl.is_window_ready():
            _init_raylib_window()

        if states.shape[1] == 3:
            positions = states
            deltas = np.diff(positions, append=positions[-1:], axis=0)
        elif states.shape[1] == 6:
            positions = states[:,:3]
            deltas = states[:,3:]
        else:
            raise ValueError("States should have 3 or 6 columns")

        directions = deltas / np.linalg.norm(deltas, axis=1)[:,np.newaxis]
        directions[np.isnan(directions)] = 0
        if len(directions) > 1:
            directions[-1] = directions[-2]

        double_stiched_positions = np.repeat(positions, 2, axis=0)
        double_stiched_dirs = np.repeat(directions, 2, axis=0)
        double_stiched_time = np.repeat(epochs, 2, axis=0)

        vao = rl.rl_load_vertex_array()
        rl.rl_enable_vertex_array(vao)

        _create_vb_attribute(double_stiched_positions, 0)
        _create_vb_attribute(double_stiched_time[:,np.newaxis], 1)
        _create_vb_attribute(double_stiched_dirs, 2)

        """ 
        0 - 1
        | / |
        2 - 3 
        """

        triangle_buffer = np.zeros((len(states) - 1) * 6, np.uint16)
        enum = np.arange(0, (len(states) - 1)*2, 2)
        for offs, idx in enumerate([0,1,2,1,3,2]):
            triangle_buffer[offs::6] = enum + idx

        with ffi.from_buffer(triangle_buffer) as c_array:
            vbo = rl_raw.rlLoadVertexBufferElement(c_array, triangle_buffer.size*2, False)
        rl.rl_enable_vertex_buffer_element(vbo)

        rl.rl_disable_vertex_array()
        self.trajectory_patches.append((vao, len(triangle_buffer), trajectory_index))

    @property
    def entities(self):
        ''' Generates all entities in the scene '''
        yield self.reference_frame
        for trajectory in self.trajectories:
            yield trajectory
        for body in self.bodies:
            yield body


def _create_vb_attribute(array: np.ndarray, index: int):
    ''' Helper function to create vertex buffer attributes '''

    # Needs to be hardcode, since python raylib does not expose this to my knowledge
    GL_FLOAT = 0x1406

    assert array.ndim == 2
    array_32 = array.astype(np.float32)
    with ffi.from_buffer(array_32) as c_array:
        vbo = rl_raw.rlLoadVertexBuffer(c_array, array_32.size * 4, False)
    rl_raw.rlSetVertexAttribute(index, array.shape[1], GL_FLOAT, False, 0, 0)
    rl_raw.rlEnableVertexAttribute(index)
    return vbo


def _init_raylib_window():
    # Initiialize raylib graphics window
    rl.set_config_flags(rl.ConfigFlags.FLAG_MSAA_4X_HINT | rl.ConfigFlags.FLAG_WINDOW_RESIZABLE)
    rl.init_window(DEFAULT_WINDOWN_WIDTH, DEFAULT_WINDOW_HEIGHT, "Space Trace")
    rl.set_target_fps(60)
