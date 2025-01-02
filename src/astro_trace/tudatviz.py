from typing import Sequence

import numpy as np
import raylib as rl_raw
import pyray as rl
from math import ceil

ffi = rl.ffi
NULL = ffi.cast('void*', 0)

DEFAULT_WINDOWN_WIDTH = 800
DEFAULT_WINDOW_HEIGHT = 600


def _create_vb_attribute(array: np.ndarray, index: int):
    GL_FLOAT = 0x1406
    assert array.ndim == 2
    array_32 = array.astype(np.float32)
    with ffi.from_buffer(array_32) as c_array:
        vbo = rl_raw.rlLoadVertexBuffer(c_array, array_32.size * 4, False)
    rl_raw.rlSetVertexAttribute(index, array.shape[1], GL_FLOAT, False, 0, NULL)
    rl_raw.rlEnableVertexAttribute(index)
    return vbo


class SceneEntity():
    def __init__(self, name: str, color: tuple[float, float, float]=(1, 1, 1)):
        self.name = name
        self.color = color


class Trajectory(SceneEntity):
    def __init__(self, segment_indices: Sequence[int], name: str, color: tuple[float, float, float]=(1, 1, 1)):
        super().__init__(name, color)
        self.segmnets = segment_indices


class Body(SceneEntity):
    def __init__(self, name: str, radius: float, color: tuple[float, float, float]=(1, 1, 1)):
        super().__init__(name, color)
        self.radius = radius
        self.positions = np.zeros((1,3))
        self.times = np.zeros(1)

    def set_trajectory(self, time: np.ndarray, positions: np.ndarray):
        self.times = time
        self.positions = positions

    def get_position(self, time: float):
        if len(self.times) == 1:
            return self.positions[0]
        idx = np.searchsorted(self.times, time)
        if idx == 0:
            return self.positions[0]
        if idx == len(self.times):
            return self.positions[-1]
        t0, t1 = self.times[idx-1], self.times[idx]
        p0, p1 = self.positions[idx-1], self.positions[idx]
        alpha = (time - t0) / (t1 - t0)
        return p0 + alpha * (p1 - p0)


class Scene():
    scale_factor = 1e-7

    def __init__(self):
        self.trajectories = []
        self.bodies = []

        self.trajectory_patches = []
        self.time_bounds = [np.inf, -np.inf]

    def add_trajectory(self, time: np.ndarray, states: np.ndarray, name:str="SpaceCraft", 
                       color: tuple[float, float, float]=(1,1,1)):
        assert len(time) == len(states)
        total_length = len(states)
        parts = int(ceil(total_length / 2**14))

        patch_indices = range(
            len(self.trajectory_patches), 
            len(self.trajectory_patches) + parts, 
        )
        trajectory = Trajectory(patch_indices, name, color)
        self.trajectories.append(trajectory)

        for i in range(parts):
            start = i * 2**14
            end = min((i+1) * 2**14, total_length)
            self.add_trajectory_path(
                time[start:end], states[start:end], len(self.trajectories) - 1)
        


    def add_static_body(self, x: float, y: float, z: float, radius: float=6e6, 
                        color: tuple[float, float, float]=(1,1,1), name: str="Central Body"):
        body = Body(name, radius * self.scale_factor, color)
        body.set_trajectory(np.zeros(1), np.array([[x, z, -y]]) * self.scale_factor)
        self.bodies.append(body)


    def add_moving_body(self, t: np.ndarray, r: np.ndarray, radius: float=6e6, 
                        color: tuple[float, float, float]=(1,1,1), name: str="Central Body"):
        body = Body(name, radius * self.scale_factor, color)
        render_space_positions = r[:,(0,2,1)] * np.array([1,1,-1])[np.newaxis,:] * self.scale_factor
        body.set_trajectory(t, render_space_positions)
        self.bodies.append(body)


    def add_trajectory_path(self, time: np.ndarray, states: np.ndarray, trajectory_index: int):
        if not rl.is_window_ready():
            init_raylib_window()

        # Preallocate double positions for triangle strip
        if states.shape[1] == 3:
            positions = states[:,(0,2,1)] * np.array([1,1,-1])[np.newaxis,:]
            velocities = np.diff(positions, append=positions[-1:], axis=0)
        elif states.shape[1] == 6:
            positions = states[:,(0,2,1)] * np.array([1,1,-1])[np.newaxis,:]
            velocities = states[:,(3,5,4)] * np.array([1,1,-1])[np.newaxis,:]

        directions = velocities / np.linalg.norm(velocities, axis=1)[:,np.newaxis]
        directions[np.isnan(directions)] = 0
        directions[-1] = directions[-2]
        double_stiched_positions = np.repeat(positions, 2, axis=0) * self.scale_factor
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

    def show(self):
        with show_interactable(self) as app:
            while app.is_running():
                app.step()

def init_raylib_window():
    # Initiialize raylib graphics
    rl.set_config_flags(rl.ConfigFlags.FLAG_MSAA_4X_HINT | rl.ConfigFlags.FLAG_WINDOW_RESIZABLE)
    rl.init_window(DEFAULT_WINDOWN_WIDTH, DEFAULT_WINDOW_HEIGHT, "Tudat Viz")
    #rl.set_target_fps(60)


class DrawApplication():
    def __init__(self, scene: Scene):
        self.scene = scene
        self.camera_zoom = 5.0
        cam_start_pos = rl.Vector3(0, np.sqrt(2)/2* self.camera_zoom, np.sqrt(2)/2* self.camera_zoom)
        self.camera = rl.Camera3D(cam_start_pos, rl.vector3_zero(), 
                                  rl.Vector3(0, 1, 0), 45, rl.CAMERA_PERSPECTIVE)
        self.traj_shader = None
        self.time_bounds = [0, 0]
        self.current_time = 0

        self.focus = None  # focuses on the center


    def update_camera(self):
        ''' Updates rotation. Maintains target while updating yaw and pitch trhough mouse input. '''
        if self.focus is not None:
            for body in self.scene.bodies:
                if body.name == self.focus:
                    pos = body.get_position(self.current_time)
                    self.camera.target = rl.Vector3(pos[0], pos[1], pos[2])

        camera_pos = rl.vector3_subtract(self.camera.position, self.camera.target)
        self.camera_zoom = rl.vector3_length(camera_pos)
        camera_dir = rl.vector3_scale(camera_pos, 1 / self.camera_zoom)
        pitch = np.arcsin(camera_dir.y)
        yaw = np.arctan2(camera_dir.x, camera_dir.z)

        if rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_MIDDLE):
            mouse_delta = rl.get_mouse_delta()
            pitch += mouse_delta.y * 0.005
            yaw   -= mouse_delta.x * 0.005
        self.camera_zoom *= (1 - rl.get_mouse_wheel_move() * 0.05)
        
        pitch = np.clip(pitch, -np.pi/2, np.pi/2)
        camera_dir.x = np.sin(yaw) * np.cos(pitch)
        camera_dir.y = np.sin(pitch)
        camera_dir.z = np.cos(yaw) * np.cos(pitch)

        self.camera.position = rl.vector3_add(self.camera.target, rl.vector3_scale(camera_dir, self.camera_zoom))

    def setup(self):
        ''' Setup after all the drawing data has been specified'''
        if not rl.is_window_ready():
            init_raylib_window()

        if np.isfinite(self.scene.time_bounds[0]):
            self.time_bounds = self.scene.time_bounds
        else:
            self.time_bounds = [0, 0]
        self.current_time = self.time_bounds[1]

        # Load trajectory shader
        self.traj_shader = rl.load_shader("trajectory_shader.vs", "trajectory_shader.fs")
        self.traj_locs_window_size = rl.get_shader_location(self.traj_shader, "window_size")
        self.traj_locs_mvp = rl.get_shader_location(self.traj_shader, "mvp")
        self.traj_locs_color = rl.get_shader_location(self.traj_shader, "color")
        self.traj_locs_time = rl.get_shader_location(self.traj_shader, "current_t")


    def destroy(self):
        rl.unload_shader(self.traj_shader)
        rl.close_window()


    def set_focus(self, body_name: str):
        self.focus = body_name


    def _draw_trajectories(self):
        rl.begin_shader_mode(self.traj_shader)
        #rl.draw_cube(rl.vector3_zero(), 1, 1, 1, rl.BLUE)

        mat_projection = rl.rl_get_matrix_projection()
        mat_model_view = rl.rl_get_matrix_modelview()
        mat_model_view_projection = rl.matrix_multiply(mat_model_view, mat_projection)

        screen_size = rl.Vector2(rl.get_screen_width(), rl.get_screen_height())
        rl.set_shader_value(self.traj_shader, self.traj_locs_window_size, screen_size, 
                            rl.ShaderUniformDataType.SHADER_UNIFORM_VEC2)
        time_ptr = ffi.from_buffer(np.array([self.current_time], np.float32))
        rl.set_shader_value(self.traj_shader, self.traj_locs_time, time_ptr, 
                            rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)
        rl.set_shader_value_matrix(self.traj_shader, self.traj_locs_mvp, mat_model_view_projection)


        for traj in self.scene.trajectory_patches:
            vao, elems, traj_index = traj

            color = self.scene.trajectories[traj_index].color
            color_c = ffi.from_buffer(np.array([*color,1], np.float32))
            rl_raw.SetShaderValue(self.traj_shader, self.traj_locs_color, color_c, 
                                  rl.ShaderUniformDataType.SHADER_UNIFORM_VEC4)

            rl.rl_enable_vertex_array(vao)
            rl_raw.rlDrawVertexArrayElements(0, elems, NULL)
            rl.rl_disable_vertex_array()

        rl.end_shader_mode()

    def _draw_bodies(self):
        for body in self.scene.bodies:
            r = body.get_position(self.current_time)
            pos_3d = rl.Vector3(r[0], r[1], r[2])
            pos_2d = rl.Vector3(r[0], 0, r[2])
            color = rl.Color(*[int(c*255) for c in body.color], 255)

            rl.draw_line_3d(pos_2d, pos_3d, rl.color_alpha(color, 0.5))
            rl.draw_sphere_ex(pos_3d, body.radius, 32, 64, color)
            

    def _draw_time_bar(self):
        if self.time_bounds[1] <= self.time_bounds[0]:
            return
        
        TIMEBAR_HIGHT = 20
        rl.draw_rectangle(0, rl.get_screen_height() - TIMEBAR_HIGHT, rl.get_screen_width(), TIMEBAR_HIGHT, rl.GRAY)
        t = (self.current_time - self.time_bounds[0]) / (self.time_bounds[1] - self.time_bounds[0])
        rl.draw_rectangle(0, rl.get_screen_height() - TIMEBAR_HIGHT, int(t * rl.get_screen_width()), TIMEBAR_HIGHT, rl.YELLOW)

        if rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT):
            mouse_pos = rl.get_mouse_position()
            if mouse_pos.y > rl.get_screen_height() - TIMEBAR_HIGHT:
                self.current_time = self.time_bounds[0] + (mouse_pos.x / rl.get_screen_width()) * (self.time_bounds[1] - self.time_bounds[0])


    def _draw_grid(self):
        extend = self.camera_zoom * 100
        rl.draw_line_3d(rl.Vector3(-extend, 0, 0), rl.Vector3(extend, 0, 0), rl.color_tint(rl.RED,   rl.Color(120,120,120,255)))
        rl.draw_line_3d(rl.Vector3(0, -extend, 0), rl.Vector3(0, extend, 0), rl.color_tint(rl.GREEN, rl.Color(120,120,120,255)))
        rl.draw_line_3d(rl.Vector3(0, 0, -extend), rl.Vector3(0, 0, extend), rl.color_tint(rl.BLUE,  rl.Color(120,120,120,255)))
        for r in [1, 2, 5, 10, 20, 50]:
            rl.draw_circle_3d(rl.Vector3(0,0,0), r, rl.Vector3(1,0,0), 90, rl.GRAY)


    def step(self):
        self.update_camera()

        rl.begin_drawing()
        
        rl.clear_background(rl_raw.BLACK)

        rl.begin_mode_3d(self.camera)
        self._draw_grid()
        self._draw_trajectories()
        self._draw_bodies()
        rl.end_mode_3d()

        self._draw_time_bar()
        rl.draw_fps(10, 10)

        rl.end_drawing()

    def is_running(self):
        return not rl.window_should_close()
    
class show_interactable():

    def __init__(self, scene):
        self.app = DrawApplication(scene)

    def __enter__(self):
        self.app.setup()
        return self.app

    def __exit__(self, *args):
        self.app.destroy()


if __name__ == "__main__":
    N = 30_000
    tt = np.linspace(0, 3600*1.5, N)
    thetas = np.linspace(0, 2*np.pi, N)
    rr = np.array([np.cos(thetas), np.sin(thetas), np.zeros_like(thetas)]).T * 7e6

    init_raylib_window()
    scene = Scene()
    scene.add_trajectory(tt, rr)
    scene.add_static_body(0, 0, 0, radius=6.7e6, name='Earth', color=(0,0.5,1))
    scene.show()