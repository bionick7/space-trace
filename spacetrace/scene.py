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

def __hex_to_color(hex: int) -> tuple[float, float, float]:
    return (
        ((hex >> 16) & 0xFF) / 255,
        ((hex >> 8) & 0xFF) / 255, 
        (hex & 0xFF) / 255
    )

# Preload in hidden global scope
__palette = {}
__palette['bg'] = __hex_to_color(0x12141c)
__palette['blue'] = __hex_to_color(0x454e7e)
__palette['green'] = __hex_to_color(0x4Fc76C)
__palette['red'] = __hex_to_color(0xFF5155)
__palette['white'] = __hex_to_color(0xfaf7d5)
__palette['gray'] = __hex_to_color(0x735e4c)
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
        else:
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
    
    def on_setup(self, scene, draw_app):
        ''' Gets called when the window is initialized '''
        pass


class TransformShape(SceneEntity):
    ''' 
    Reference frame, represented by 3 orthogonal arrows
    By default, the identiy transform is always in the scene
    '''
    def __init__(self, epochs: np.ndarray, origins: np.ndarray, bases: np.ndarray, 
                 name: str="Transform", color: _ColorType='main', draw_space: bool=False,
                 axis_colors: Optional[tuple[_ColorType, _ColorType, _ColorType]]=None):
        '''
        Initializes a transform entity
        epochs: np.ndarray (N,)
            Time values for each state
        origins: np.ndarray (N, 3)
            Origin for each epoch, where the transform is drawn from
        bases: np.ndarray (N, 3, 3)
            Array of 3x3 matrices, designating orientation and scale for each epoch
        name: str
            Identifier used in the UI. Should be unique
        color: tuple[float, float, float] or str
            Color of the trajectory. Can be a tuple of RGB values or a identifies a 
            color in the color palette.
            Default colors are 'main', 'accent', 'bg', 'blue', 'green', 'red', 'white'
        draw_space: bool
            If true, the coordinates are specified in 'draw space' and therefore not 
            affected by the scene's scale factor
        axis_colors: Optional[tuple[_ColorType, _ColorType, _ColorType]]
            axis colors
        '''
        super().__init__(name, color)
        N = len(epochs)
        
        if origins.shape == (N, 3):
            self.positions = _transform_vectors_to_draw_space(origins)
        elif origins.shape == (3,):
            self.positions = _transform_vectors_to_draw_space(np.repeat(origins[np.newaxis,:], N, axis=0))
        else:
            raise ValueError("origins must be of shape (N, 3) or (3,), where N is the epoch length")
        
        self.bases = np.zeros((N, 3, 3))
        if bases.shape == (N, 3, 3):
            for i in range(3):
                self.bases[:,:,i] = _transform_vectors_to_draw_space(bases[:,:,i])
        elif bases.shape == (3, 3):
            for i in range(3):
                self.bases[:,:,i] = _transform_vectors_to_draw_space(bases[:,i])
        else:
            raise ValueError("bases must be of shape (N, 3, 3) or (3, 3), where N is the epoch length")
        
        self.epochs = epochs
        self.draw_space = draw_space
        self.axis_colors = axis_colors

    def get_basis(self, time: float):
        return self._get_property(self.bases, time)
    
    @classmethod
    def fixed(cls, origin: np.ndarray, M: np.ndarray, *args, **kwargs):
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
        return cls(np.zeros(1), origin[np.newaxis,:], M[np.newaxis,:,:], *args, **kwargs)

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



class VectorShape(SceneEntity):
    def __init__(self, epochs: np.ndarray, origins: np.ndarray, vectors: np.ndarray, 
                 name: str="Vector", color: _ColorType='main', draw_space: float=False):
        '''
        Initializes a transform entity
        epochs: np.ndarray (N,)
            Time values for each state
        origins: np.ndarray (N, 3)
            Origin for each epoch, where the transform is drawn from
        vectors: np.ndarray (N, 3)
            vector (direction and magnitude) for each epoch
        name: str
            Identifier used in the UI. Should be unique
        color: tuple[float, float, float] or str
            Color of the trajectory. Can be a tuple of RGB values or a identifies a 
            color in the color palette.
            Default colors are 'main', 'accent', 'bg', 'blue', 'green', 'red', 'white'
        draw_space: bool
            If true, the coordinates are specified in 'draw space' and therefore not 
            affected by the scene's scale factor
        axis_colors: Optional[tuple[_ColorType, _ColorType, _ColorType]]
            axis colors
        '''
        super().__init__(name, color)
        N = len(epochs)        
        
        if origins.shape == (N, 3):
            self.positions = _transform_vectors_to_draw_space(origins)
        elif origins.shape == (3,):
            self.positions = _transform_vectors_to_draw_space(np.repeat(origins[np.newaxis,:], N, axis=0))
        else:
            raise ValueError("origins must be of shape (N, 3) or (3,), where N is the epoch length")
        
        if vectors.shape == (N, 3):
            self.vectors = _transform_vectors_to_draw_space(vectors)
        elif vectors.shape == (3,):
            self.vectors = _transform_vectors_to_draw_space(np.repeat(vectors[np.newaxis,:], N, axis=0))
        else:
            raise ValueError("directions must be of shape (N, 3) or (3,), where N is the epoch length")
        
        self.epochs = epochs
        self.draw_space = draw_space

    def get_vector(self, time: float):
        return self._get_property(self.vectors, time)

    @classmethod
    def fixed(cls, x: float, y: float, z: float, vx: float, vy: float, vz: float, *args, **kwargs):
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
        return VectorShape(cls, np.zeros(1), np.array([[x, y, z]]), np.array([[vx, vy, vz]]), *args, **kwargs)


class Trajectory(SceneEntity):
    '''
    A trajectory is a sequence of positions in space over time.
    Internally, a trajectory can be multiple draw calls. 
    This is mostly to access the metadata and to support get_position
    '''
    def __init__(self, epochs: np.ndarray, states: np.ndarray, 
                 name: str='Trajectory', color: _ColorType='main'):
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
        self._scene_index = -1

    @property
    def patches(self):
        total_length = len(self.positions)
        parts = int(ceil(total_length / 2**14))
        for i in range(parts):
            start = max(0, i * 2**14 - 1)  # Link up t0.0.0 the previous one
            end = min((i+1) * 2**14, total_length)
            yield self.epochs[start:end], self.positions[start:end], self.velocities

    def on_setup(self, scene, draw_app):
        for patch in self.patches:
            scene._add_trajectory_patch(*patch, self._scene_index)


class Body(SceneEntity):
    '''
    A body is a static or moving object in the scene.
    Represented by a colored sphere of a certain radius.
    Mostly represents a celestial body.
    '''
    def __init__(self, epochs: np.ndarray, states: np.ndarray, radius: float, name: str="Body", 
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

    @classmethod
    def fixed(cls, x: float, y: float, z: float, *args, **kwargs):
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
        return cls(np.zeros(1), np.array([[x, y, z]]), *args, **kwargs)


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
        origin_frame = TransformShape.fixed(np.zeros(3), np.eye(3) * 100, name=DEFAULT_FRAME_NAME, 
                                       draw_space=True, axis_colors=('red', 'green', 'blue'))
        self.transforms.append(origin_frame)

    def get_entity(self, entity_name: str) -> SceneEntity:
        if entity_name in self.lookup:
            return self.lookup[entity_name]
        else:
            raise ValueError(f"No such entity: '{entity_name}'")

    def add(self, entity: SceneEntity) -> None:
        '''
        Adds scene entity to the scene. Scene entities can be Trajectory, Body, VectorShape or TransformShape
        '''
        entity_name_suffix_index = 2
        entity_original_name = entity.name
        while entity.name in [e.name for e in self.entities]:
            entity.name = entity_original_name + " " + str(entity_name_suffix_index)
            entity_name_suffix_index += 1
        
        if isinstance(entity, Trajectory):
            entity._scene_index = len(self.trajectories)
            self.trajectories.append(entity)
        elif isinstance(entity, Body):
            self.bodies.append(entity)
        elif isinstance(entity, VectorShape):
            self.vectors.append(entity)
        elif isinstance(entity, TransformShape):
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
            raise Exception("Window not initialized, no graphics API exists")

        if deltas is None:
            deltas = np.diff(positions, append=positions[-1:], axis=0)

        directions = deltas / np.linalg.norm(deltas, axis=1)[:,np.newaxis]
        directions[np.isnan(directions)] = 0
        if len(directions) > 1:
            directions[-1] = directions[-2]

        double_stiched_positions = np.repeat(positions, 2, axis=0) * self.scale_factor
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

    def on_setup(self, draw_app):
        for entity in self.entities:
            entity.on_setup(self, draw_app)

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
