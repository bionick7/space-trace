from typing import Literal, Callable, Optional, Any
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


def _transform_vectors_to_draw_space(inp: np.ndarray) -> np.ndarray:
    return inp[:,(0,2,1)] * np.array([1,1,-1])[np.newaxis,:]


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
    
    def set_trajectory(self, epochs: np.ndarray, positions: np.ndarray):
        self.epochs = epochs
        self.positions = positions

    def _get_index(self, time: float) -> tuple[int, float]:
        idx = np.searchsorted(self.epochs, time)
        if idx == 0:
            return 1, 0.0
        if idx == len(self.epochs):
            return len(self.epochs) - 1, 1.0
        t0, t1 = self.epochs[idx-1], self.epochs[idx]
        alpha = (time - t0) / (t1 - t0)
        return idx, alpha

    def _get_property(self, a: np.ndarray, time: float) -> Any:
        if len(self.epochs) == 1:
            return a[0]
        idx, alpha = self._get_index(time)
        yl, yr = a[idx-1], a[idx]
        return yl + alpha * (yr - yl)


    def get_position(self, time: float):
        return self._get_property(self.positions, time)


class Transform(SceneEntity):
    ''' 
    The main reference frame in the scene.
    By default, the identiy transform is always in the scene
    '''
    def __init__(self, epochs: np.ndarray, origins: np.ndarray, bases: np.ndarray, 
                 name: str='Transform', color: _ColorType='main', draw_space: float=False,
                 axis_colors: Optional[tuple[_ColorType, _ColorType, _ColorType]]=None):
        '''
        TODO
        '''
        super().__init__(name, color)
        N = len(epochs)
        if epochs.ndim != 1:
            raise ValueError("epochs must be 1-dimensional")
        if origins.shape != (N, 3):
            raise ValueError("Shape mismatch: origins shape must be 3 x N, where N is the size of epochs")
        if bases.shape != (N, 3, 3):
            raise ValueError("Shape mismatch: bases must be 3 x N, where N is the size of epochs")
        self.epochs = epochs
        self.positions = _transform_vectors_to_draw_space(origins)
        self.bases = np.zeros_like(bases)
        for i in range(3):
            self.bases[:,:,i] = _transform_vectors_to_draw_space(bases[:,:,i])

        self.draw_space = draw_space
        self.axis_colors = axis_colors

    def get_basis(self, time: float):
        return self._get_property(self.bases, time)
    
    def fixed(origin: np.ndarray, M: np.ndarray, **kwargs):
        ''' 
        Adds a transform (without trajectory) to the scene. Usefull to display
        rotations and 
        origin (3,):
            Origin of the transform
        basis (3, 3): 
            3x3 matrix, indicatinf scale and rotation of the transform
        color: tuple[float, float, float] or str
            Color of the body. Can be a tuple of RGB values or a identifies a 
            color in the color palette.
            Default colors are 'main', 'accent', 'bg', 'blue', 'green', 'red', 'white'
        '''
        return Transform(np.zeros(1), origin[np.newaxis,:], M[np.newaxis,:,:], **kwargs)

    def get_x_color(self) -> Color:
        if self.axis_colors is None:
            return self.color
        return Color(self.axis_colors[0])

    def get_y_color(self) -> Color:
        if self.axis_colors is None:
            return self.color
        return Color(self.axis_colors[1])

    def get_z_color(self) -> Color:
        if self.axis_colors is None:
            return self.color
        return Color(self.axis_colors[2])



class Vector(SceneEntity):
    def __init__(self, epochs: np.ndarray, origins: np.ndarray, vectors: np.ndarray, 
                 name: str, color: _ColorType='main', draw_space: float=False):
        '''
        TODO
        '''
        super().__init__(name, color)
        N = len(epochs)        
        if len(origins) == len(vectors) == N:
            self.positions = _transform_vectors_to_draw_space(origins)
            self.vectors = _transform_vectors_to_draw_space(vectors)
        elif origins.shape == (N, 3) and vectors.shape == (3,):
            self.positions = _transform_vectors_to_draw_space(origins)
            self.vectors = _transform_vectors_to_draw_space(np.hstack(self.positions[np.newaxis,:], N))
        elif vectors.shape == (N, 3) and origins.shape == (3,):
            self.positions = _transform_vectors_to_draw_space(np.hstack(self.positions[np.newaxis,:], N))
            self.vectors = _transform_vectors_to_draw_space(vectors)
        else:
            raise ValueError("Shape mismatch")
        self.epochs = epochs
        self.draw_space = draw_space

    def get_vector(self, time: float):
        return self._get_property(self.vectors, time)

    @staticmethod
    def fixed(x: float, y: float, z: float, vx: float, vy: float, vz: float, **kwargs):
        ''' 
        Adds a static vector (without trajectory) to the scene.
        x: float
        y: float
        z: float
            Origin of the vector in space
        vx: float
        vy: float
        vz: float
            Direction and magnitude of the vector in space
        color: tuple[float, float, float] or str
            Color of the body. Can be a tuple of RGB values or a identifies a 
            color in the color palette.
            Default colors are 'main', 'accent', 'bg', 'blue', 'green', 'red', 'white'
        '''
        return Vector(np.zeros(1), np.array([[x, y, z]]), np.array([[vx, vy, vz]]), **kwargs)


class Trajectory(SceneEntity):
    '''
    A trajectory is a sequence of positions in space over time.
    Internally, a trajectory can be multiple draw calls. 
    This is mostly to access the metadata and to support get_position
    '''
    def __init__(self, epochs: np.ndarray, states: np.ndarray, 
                 name: str, color: _ColorType='main'):
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
        super().__init__(name, color)

        if len(epochs) != len(states):
            raise ValueError("Epochs and states should have the same length")

        if states.shape[1] == 3:
            self.positions = _transform_vectors_to_draw_space(states)
            self.velocities = None
        elif states.shape[1] == 6:
            self.positions = _transform_vectors_to_draw_space(states)
            self.velocities = _transform_vectors_to_draw_space(states[:,3:])
        else:
            raise ValueError("States should have 3 or 6 columns")
            
        self.epochs = epochs

    @property
    def patches(self):
        total_length = len(self.positions)
        parts = int(ceil(total_length / 2**14))
        for i in range(parts):
            start = max(0, i * 2**14 - 1)  # Link up t0.0.0 the previous one
            end = min((i+1) * 2**14, total_length)
            yield self.epochs[start:end], self.positions[start:end], self.velocities


class Body(SceneEntity):
    '''
    A body is a static or moving object in the scene.
    Represented by a colored sphere of a certain radius.
    Mostly represents a celestial body.
    '''
    def __init__(self, epochs: np.ndarray, states: np.ndarray, name: str, radius: float, 
                 color: _ColorType='main', shape: Literal['sphere', 'cross'] = 'sphere'):
        ''' 
        Adds a static body (without trajectory) to the scene. Usefull for central bodies
        in a body-centric reference frame.
        epochs: np.ndarray (N,)
            Time values for each state
        states: np.ndarray (N, 3) or (N, 6)
            Positions or positions and velocities states for each time step
            velocities are ignored
        radius: float
            Radius of the body, in the same units as positions are provided
        color: tuple[float, float, float] or str
            Color of the body. Can be a tuple of RGB values or a identifies a 
            color in the color palette.
            Default colors are 'main', 'accent', 'bg', 'blue', 'green', 'red', 'white'
        shape: shape that will be rendered
            can be 'sphere' for planetary bodies or 'cross' for points of interest without dimension
        '''
        super().__init__(name, color)
        self.radius = radius
        self.shape = shape
        self.positions = states[:,(0, 2, 1)] * np.array([1,1,-1])[np.newaxis,:]
        self.epochs = epochs

    @staticmethod
    def fixed(x: float, y: float, z: float, **kwargs):
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
        shape: shape that will be rendered
            can be 'sphere' for planetary bodies or 'cross' for points of interest without dimension
        '''
        return Body(np.zeros(1), np.array([[x, y, z]]), **kwargs)


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
        self.vectors = []
        self.transforms = []

        self.trajectory_patches = []
        self.time_bounds = [np.inf, -np.inf]
        self.lookup = {}
        origin_frame = Transform.fixed(np.zeros(3), np.eye(3) * 100, name=DEFAULT_FRAME_NAME, 
                                       draw_space=True, axis_colors=('red', 'green', 'blue'))
        self.transforms.append(origin_frame)

    def get_entity(self, entity_name: str) -> SceneEntity:
        if entity_name in self.lookup:
            return self.lookup[entity_name]
        else:
            raise ValueError(f"No such entity: '{entity_name}'")

    def add(self, entity: SceneEntity) -> None:
        '''
        Adds scene entity to the scene. Scene entities can be Trajectory, Body, Vector or Transform
        '''
        entity_name_suffix_index = 0
        entity_original_name = entity.name
        while entity.name in [e.name for e in self.entities]:
            entity.name = entity_original_name + str(entity_name_suffix_index)
            entity_name_suffix_index += 1
        
        if isinstance(entity, Trajectory):
            for patch in entity.patches:
                self._add_trajectory_patch(*patch, len(self.trajectories))
            self.trajectories.append(entity)
        elif isinstance(entity, Body):
            self.bodies.append(entity)
        elif isinstance(entity, Vector):
            self.vectors.append(entity)
        elif isinstance(entity, Transform):
            self.transforms.append(entity)
        else:
            raise TypeError(f"Type not supported: '{type(entity)}'")
        
        if self.time_bounds[0] > entity.epochs[0]:
            self.time_bounds[0] = entity.epochs[0]
        if self.time_bounds[1] < entity.epochs[-1]:
            self.time_bounds[1] = entity.epochs[-1]

    def _add_trajectory_patch(self, epochs: np.ndarray, positions: np.ndarray, 
                              deltas: Optional[np.ndarray], trajectory_index: int):
        '''
        Helper function for add_trajectory. Handle a lot of the low-level rendering setup
        '''
        if not rl.is_window_ready():
            _init_raylib_window()

        if deltas is None:
            deltas = np.diff(positions, append=positions[-1:], axis=0)

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

        triangle_buffer = np.zeros((len(positions) - 1) * 6, np.uint16)
        enum = np.arange(0, (len(positions) - 1)*2, 2)
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
        for trajectory in self.trajectories:
            yield trajectory
        for body in self.bodies:
            yield body
        for transform in self.transforms:
            yield transform
        for vector in self.vectors:
            yield vector


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
