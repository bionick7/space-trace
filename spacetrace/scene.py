from typing import Sequence, Literal, Callable
from math import ceil

import numpy as np
import pyray as rl
import raylib as rl_raw

ffi = rl.ffi
NULL = ffi.NULL

DEFAULT_WINDOWN_WIDTH = 800
DEFAULT_WINDOW_HEIGHT = 600

DEFAULT_FRAME_NAME = 'Default Frame'


def hex_to_color(hex: int) -> tuple[float, float, float]:
    return (
        ((hex >> 16) & 0xFF) / 255,
        ((hex >> 8) & 0xFF) / 255, 
        (hex & 0xFF) / 255
    )

_ColorIDLiteral = Literal['bg', 'blue', 'green', 'red', 'white', 'main', 'accent']
_ColorType = tuple[float, float, float] | _ColorIDLiteral

def default_palette(name: _ColorIDLiteral) -> tuple[float, float, float]:
    match name:
        case 'bg':
            return hex_to_color(0x12141c)
        case 'blue' | 'accent':
            return hex_to_color(0x454e7e)
        case 'green':
            return hex_to_color(0x4Fc76C)
        case 'red':
            return hex_to_color(0xFF5155)
        case 'white' | 'main':
            return hex_to_color(0xfaf7d5)
        case 'gray' | 'grey': 
            return hex_to_color(0x735e4c)
    return hex_to_color(0xFF00FF)

class Color():
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
    

class SceneEntity():
    def __init__(self, name: str, color: _ColorType='main'):
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
    def __init__(self, name: str, color: _ColorType='main'):
        super().__init__(name, color)
        self.positions = np.zeros((1,3))
        self.epochs = np.zeros(1)
        self.transforms = np.eye(3)[np.newaxis,:,:]


class Trajectory(SceneEntity):
    def __init__(self, epochs: np.ndarray, positions: np.ndarray, 
                 name: str, color: _ColorType='main'):
        super().__init__(name, color)
        self.positions = positions
        self.epochs = epochs


class Body(SceneEntity):
    def __init__(self, name: str, radius: float, color: _ColorType='main'):
        super().__init__(name, color)
        self.radius = radius


class Scene():
    def __init__(self, scale_factor: float=1e-7):
        self.scale_factor = scale_factor
        self.trajectories = []
        self.bodies = []

        self.trajectory_patches = []
        self.time_bounds = [np.inf, -np.inf]
        self.reference_frame = ReferenceFrame(DEFAULT_FRAME_NAME)

    def add_trajectory(self, epochs: np.ndarray, states: np.ndarray, name:str="SpaceCraft", 
                       color: _ColorType=(1,1,1)):
        assert len(epochs) == len(states)
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
            start = i * 2**14
            end = min((i+1) * 2**14, total_length)
            self.add_trajectory_path(
                epochs[start:end], states[start:end], len(self.trajectories))
            
        trajectory = Trajectory(epochs, states[:,:3], name, color)
        self.trajectories.append(trajectory)
        

    def add_static_body(self, x: float, y: float, z: float, radius: float=6e6, 
                        color: _ColorType=(1,1,1), name: str="Central Body"):
        body = Body(name, radius * self.scale_factor, color)
        body.set_trajectory(np.zeros(1), np.array([[x, z, -y]]) * self.scale_factor)
        self.bodies.append(body)


    def add_moving_body(self, t: np.ndarray, r: np.ndarray, radius: float=6e6, 
                        color: _ColorType=(1,1,1), name: str="Central Body"):
        body = Body(name, radius * self.scale_factor, color)
        render_space_positions = r[:,(0,2,1)] * np.array([1,1,-1])[np.newaxis,:] * self.scale_factor
        body.set_trajectory(t, render_space_positions)
        self.bodies.append(body)


    def add_trajectory_path(self, time: np.ndarray, states: np.ndarray, trajectory_index: int):
        if not rl.is_window_ready():
            init_raylib_window()

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
        directions[-1] = directions[-2]

        double_stiched_positions = np.repeat(positions, 2, axis=0)
        double_stiched_dirs = np.repeat(directions, 2, axis=0)
        double_stiched_time = np.repeat(time, 2, axis=0)

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
        if self.time_bounds[0] > time[0]:
            self.time_bounds[0] = time[0]
        if self.time_bounds[1] < time[-1]:
            self.time_bounds[1] = time[-1]

    @property
    def entities(self):
        yield self.reference_frame
        for trajectory in self.trajectories:
            yield trajectory
        for body in self.bodies:
            yield body


def _create_vb_attribute(array: np.ndarray, index: int):
    GL_FLOAT = 0x1406
    assert array.ndim == 2
    array_32 = array.astype(np.float32)
    with ffi.from_buffer(array_32) as c_array:
        vbo = rl_raw.rlLoadVertexBuffer(c_array, array_32.size * 4, False)
    rl_raw.rlSetVertexAttribute(index, array.shape[1], GL_FLOAT, False, 0, 0)
    rl_raw.rlEnableVertexAttribute(index)
    return vbo


def init_raylib_window():
    # Initiialize raylib graphics
    rl.set_config_flags(rl.ConfigFlags.FLAG_MSAA_4X_HINT | rl.ConfigFlags.FLAG_WINDOW_RESIZABLE)
    rl.init_window(DEFAULT_WINDOWN_WIDTH, DEFAULT_WINDOW_HEIGHT, "Space Trace")
    #rl.set_target_fps(60)
