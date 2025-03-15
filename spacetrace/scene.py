from .utils import *
from typing import (
    Literal, Callable, Optional, Any
)
import pickle
from math import ceil

import numpy as np
import pyray as rl
import raylib as rl_raw

ffi = rl.ffi


#     SCENE
# ==============


class SceneEntity():
    '''
    Base class for all entities in the scene
    Has a name, color, visibility flag as well as a trajectory through time.
    '''

    def __init__(self, name: str, color: ColorType='main'):
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
        self.color_id = color
        # Purple: color not in palette
        # Yellow: setup not called
        self.color = hex_to_color(0xFFFF00)
        self.positions = np.zeros((1,3))
        self.epochs = np.zeros(1)
        self._is_visible = True
        self._uuid = -1  # set by Scene on addition
    
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
        self.color = scene.palette(self.color_id)

    @property
    def uuid(self) -> np.uint64:
        return self._uuid
    
    @property
    def is_visible(self) -> bool:
        return self._is_visible
    
    @is_visible.setter
    def is_visible(self, value: bool):
        self._is_visible = value


class Group(SceneEntity):
    '''
    Entity to structure scene Hierarchy
    '''
    def __init__(self, name: str, *members: SceneEntity):
        super().__init__(name, 'main')
        self._members = []
        self._hierarchy = {}
        self.folded = False
        for member in members:
            self.add(member)
        
    def add(self, entitiy: SceneEntity):
        self._members.append(entitiy)
        self._hierarchy[entitiy.name] = entitiy

    def remove(self, entity_name: str):
        if entity_name not in self._hierarchy:
            raise KeyError(f"No such entity: {entity_name}")
        self._members.remove(self._hierarchy[entity_name])
        del self._hierarchy[entity_name]

    def get_position(self, time: float):
        if len(self._members) == 0:
            return np.zeros(3)
        return self._members[0].get_position(time)
        
    @property
    def members(self):
        return self._members
    
    @property
    def is_visible(self) -> bool:
        return self._is_visible
    
    @is_visible.setter
    def is_visible(self, value: bool):
        self._is_visible = value
        for member in self.members:
            member.is_visible = value


class TransformShape(SceneEntity):
    ''' 
    Reference frame, represented by 3 orthogonal arrows
    By default, the identiy transform is always in the scene
    '''
    def __init__(self, epochs: np.ndarray, origins: np.ndarray, bases: np.ndarray, 
                 name: str="Transform", color: ColorType='main', draw_space: bool=False,
                 axis_colors: Optional[tuple[ColorType, ColorType, ColorType]]=['red', 'green', 'blue']):
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
            self.positions = transform_vectors_to_draw_space(origins)
        elif origins.shape == (3,):
            self.positions = transform_vectors_to_draw_space(np.repeat(origins[np.newaxis,:], N, axis=0))
        else:
            raise ValueError("origins must be of shape (N, 3) or (3,), where N is the epoch length")
        
        self.bases = np.zeros((N, 3, 3))
        if bases.shape == (N, 3, 3):
            for i in range(3):
                self.bases[:,:,i] = transform_vectors_to_draw_space(bases[:,:,i])
        elif bases.shape == (3, 3):
            for i in range(3):
                self.bases[:,:,i] = transform_vectors_to_draw_space(bases[:,i])
        else:
            raise ValueError("bases must be of shape (N, 3, 3) or (3, 3), where N is the epoch length")
        
        self.epochs = epochs
        self.draw_space = draw_space
        self.axis_colors_ids = axis_colors

    def get_basis(self, time: float):
        return self._get_property(self.bases, time)
    
    def on_setup(self, scene, draw_app):
        super().on_setup(scene, draw_app)
        if self.axis_colors_ids is None:
            self.axis_colors = None
        else:
            self.axis_colors = [
                scene.palette(axis_color) for axis_color in self.axis_colors_ids
            ]
    
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

    def get_x_color(self) -> ColorType:
        if self.axis_colors is None:
            return self.color
        return self.axis_colors[0]

    def get_y_color(self) -> ColorType:
        if self.axis_colors is None:
            return self.color
        return self.axis_colors[1]

    def get_z_color(self) -> ColorType:
        if self.axis_colors is None:
            return self.color
        return self.axis_colors[2]


class VectorShape(SceneEntity):
    def __init__(self, epochs: np.ndarray, origins: np.ndarray, vectors: np.ndarray, 
                 name: str="Vector", color: ColorType='main', draw_space: float=False):
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
            self.positions = transform_vectors_to_draw_space(origins)
        elif origins.shape == (3,):
            self.positions = transform_vectors_to_draw_space(np.repeat(origins[np.newaxis,:], N, axis=0))
        else:
            raise ValueError("origins must be of shape (N, 3) or (3,), where N is the epoch length")
        
        if vectors.shape == (N, 3):
            self.vectors = transform_vectors_to_draw_space(vectors)
        elif vectors.shape == (3,):
            self.vectors = transform_vectors_to_draw_space(np.repeat(vectors[np.newaxis,:], N, axis=0))
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
        return cls(np.zeros(1), np.array([[x, y, z]]), np.array([[vx, vy, vz]]), *args, **kwargs)


class Trajectory(SceneEntity):
    '''
    A trajectory is a sequence of positions in space over time.
    Internally, a trajectory can be multiple draw calls. 
    This is mostly to access the metadata and to support get_position
    '''
    def __init__(self, epochs: np.ndarray, states: np.ndarray, 
                 name: str='Trajectory', color: ColorType='main'):
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
            self.positions = transform_vectors_to_draw_space(states)
            self.velocities = None
        elif states.shape[1] == 6:
            self.positions = transform_vectors_to_draw_space(states)
            self.velocities = transform_vectors_to_draw_space(states[:,3:])
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
        super().on_setup(scene, draw_app)
        for patch in self.patches:
            scene._add_trajectory_patch(*patch, self._scene_index)


class Body(SceneEntity):
    '''
    A body is a static or moving object in the scene.
    Represented by a colored sphere of a certain radius.
    Mostly represents a celestial body.
    '''
    def __init__(self, epochs: np.ndarray, states: np.ndarray, radius: float=0, name: str="Body", 
                 color: ColorType='main', shape: Literal['sphere', 'cross'] = 'sphere',
                 albedo_map_path: Optional[str]=None, specular_map_path: Optional[str]=None,
                 normal_map_path: Optional[str]=None):
        ''' 
        Adds a body to the scene.
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
        albedo_map_path: Optional[str]
            Path to an color or albedo map for the body
        specular_map_path: Optional[str]
            Path to a specularity map for the body
        normal_map_path: Optional[str]
            Path to a normal map for the body
        '''

        super().__init__(name, color)
        self.radius = radius
        self.shape = shape
        self.positions = states[:,(0, 2, 1)] * np.array([1,1,-1])[np.newaxis,:]
        self.epochs = epochs

        self.albedo_map_path = albedo_map_path
        self.specular_map_path = specular_map_path
        self.normal_map_path = normal_map_path

        # Predefine in case 'on_setup' is not called
        self.maps_enabled = int(0)
        self.albedo_map = None
        self.specular_map = None
        self.normal_map = None

    def on_setup(self, scene, draw_app):
        self.maps_enabled = int(0)
        
        find_path = os.path.join(draw_app.resource_path, "planet_textures")
        if isinstance(self.albedo_map_path, str):
            self.albedo_map = rl.load_texture(get_local_or_global_path(find_path, self.albedo_map_path))
            if self.albedo_map is not None:
                self.maps_enabled |= 0x01
        if isinstance(self.specular_map_path, str):
            self.specular_map = rl.load_texture(get_local_or_global_path(find_path, self.specular_map_path))
            if self.specular_map is not None:
                self.maps_enabled |= 0x02
        if isinstance(self.normal_map_path, str):
            self.normal_map = rl.load_texture(get_local_or_global_path(find_path, self.normal_map_path))
            if self.normal_map is not None:
                self.maps_enabled |= 0x04

        super().on_setup(scene, draw_app)

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


def _find_all_in_lookup(lookup: dict[str, SceneEntity]):
    for k, v in lookup.items():
        yield v
        if isinstance(v, Group):
            for entity in _find_all_in_lookup(v._hierarchy):
                yield entity

class Scene():
    '''
    All the data that is needed to render a scene in spacetrace
    The scene is created and populated by the user.
    Entities can be Trajectories, Bodies or the main Reference Frame.
    '''

    def __init__(self, scale_factor: float=1e-7, palette: Callable[[str], ColorType]=Themes.default_palette):
        '''
        Initializes the scene with a scale factor. The scale factor is used to convert
        provided positions into rendering units. A scale factor of 10^-7 is provided,
        assuming that positions are in meters and the trajectories are on the scale of
        earth orbits.

        default_palette: Callable[[str], ColorType]
            Represents the color palette of the scene
            Function that returns the RGB values for a given color name
        '''
        self.scale_factor = scale_factor
        self.trajectories = []
        self.bodies = []
        self.vectors = []
        self.transforms = []
        self.groups = []

        self.palette = palette

        self.trajectory_patches = []
        self.time_bounds = [np.inf, -np.inf]
        self.hierarchy = {}
        self.lookup = {}

        origin_frame = TransformShape.fixed(np.zeros(3), np.eye(3) * 100, name="Origin", 
                                       draw_space=True, axis_colors=('x-axis', 'y-axis', 'z-axis'))
        self.add(origin_frame)

    def get_entity(self, entity_path: str) -> SceneEntity:
        path_elements = entity_path.split("/")
        cursor = self.hierarchy
        for path_element in path_elements[:-1]:
            if path_element in cursor and isinstance(cursor[path_element], Group):
                cursor = cursor[path_element]._hierarchy
            else:
                raise ValueError(f"Invalid path: '{entity_path}'")
        if path_elements[-1] in cursor:
            return cursor[path_elements[-1]]
        raise ValueError(f"Invalid path: '{entity_path}'")


    def add(self, *entities: SceneEntity) -> None:
        '''
        Adds scene entity to the scene. Scene entities can be Trajectory, Body, VectorShape or TransformShape
        '''
        for entity in entities:
            entity_name_suffix_index = 2
            entity_original_name = entity.name
            while entity.name in [e.name for e in self.entities]:
                entity.name = entity_original_name + " " + str(entity_name_suffix_index)
                entity_name_suffix_index += 1

            """
            path_elements = entity.name.split("/")
            cursor = self.lookup
            for path_element in path_elements[:-1]:
                if path_element not in cursor:
                    cursor[path_element] = Group([], path_element)
                cursor = cursor[path_element]
            cursor.add(entity)
            """
            self.hierarchy[entity.name] = entity

            if isinstance(entity, Group):
                attached_entities = list(_find_all_in_lookup(entity._hierarchy)) + [entity]
            else:
                attached_entities = [entity]

            for attached in attached_entities:
                uuid = np.random.randint(np.array([2**64]), size=1, dtype=np.uint64)[0]
                self.lookup[uuid] = attached
                attached._uuid = uuid

                if isinstance(attached, Trajectory):
                    attached._scene_index = len(self.trajectories)
                    self.trajectories.append(attached)
                elif isinstance(attached, Body):
                    self.bodies.append(attached)
                elif isinstance(attached, VectorShape):
                    self.vectors.append(attached)
                elif isinstance(attached, TransformShape):
                    self.transforms.append(attached)
                elif isinstance(attached, Group):
                    self.groups.append(attached)
                else:
                    raise TypeError(f"Type not supported: '{type(attached)}'")
                
                if self.time_bounds[0] > attached.epochs[0]:
                    self.time_bounds[0] = attached.epochs[0]
                if self.time_bounds[1] < attached.epochs[-1]:
                    self.time_bounds[1] = attached.epochs[-1]

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

        """  0 - 1
             | / |
             2 - 3  """

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

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
        return loaded

    @property
    def entities(self):
        ''' Generates all non-group entities in the scene '''
        return _find_all_in_lookup(self.hierarchy)

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
