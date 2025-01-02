from spacetrace._shaders import *
from spacetrace.scene import *

import numpy as np
import pyray as rl
import raylib as rl_raw



class DrawApplication():
    def __init__(self, scene: Scene, *args, **kwargs):
        self.scene = scene
        
        self.focus = kwargs.get("focus", DEFAULT_FRAME_NAME)
        self.camera_distance = kwargs.get("camera_distance", 5.0)
        self.pitch = kwargs.get("camera_pitch", 0.5)
        self.yaw = kwargs.get("camera_yaw", 0)
        self.scene.reference_frame.is_visible = kwargs.get("show_axes", True)

        camera_dir = rl.Vector3()
        camera_dir.x = np.sin(self.yaw) * np.cos(self.pitch)
        camera_dir.y = np.sin(self.pitch)
        camera_dir.z = np.cos(self.yaw) * np.cos(self.pitch)

        self.camera = rl.Camera3D(camera_dir, rl.vector3_zero(), 
                                  rl.Vector3(0, 1, 0), kwargs.get("camera_fov", 45), 
                                  rl.CAMERA_PERSPECTIVE)

        self.traj_shader = None
        self.time_bounds = [0, 0]
        self.current_time = 0

        self.draw_entity_list = kwargs.get("draw_entity_list", True)

        self.camera_state = None
        self.last_scroll_event = 1e6
        self.time_setting = False


    def _get_mouse_pos_on_plane(self, mouse_pos, plane):
        mouse_ray = rl.get_screen_to_world_ray(mouse_pos, self.camera)
        mouse_pos_3d = mouse_ray.position
        mouse_dir = mouse_ray.direction
        res = rl.Vector2(
            mouse_pos_3d.x - mouse_dir.x * (mouse_pos_3d.y - plane) / (mouse_dir.y - plane),
            mouse_pos_3d.z - mouse_dir.z * (mouse_pos_3d.y - plane) / (mouse_dir.y - plane)
        )
        return res

    @property
    def is_scrolling(self):
        return self.last_scroll_event < 0.2

    def update_camera(self):
        ''' Updates rotation. Maintains target while updating yaw and pitch trhough mouse input. '''
        if self.focus != DEFAULT_FRAME_NAME:
            for entity in self.scene.bodies + self.scene.trajectories:
                if entity.name == self.focus:
                    pos = entity.get_position(self.current_time)
                    self.camera.target = rl.Vector3(pos[0], pos[1], pos[2])

        self.camera_state = None
        if rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_MIDDLE):
            self.camera_state = 'rotating'
        elif rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_RIGHT):
            if rl.is_key_down(rl.KEY_LEFT_SHIFT):
                self.camera_state = 'dragging_vertical'
            else:
                self.camera_state = 'dragging_horizontal'
        if rl.get_mouse_wheel_move() == 0:
            self.last_scroll_event += rl.get_frame_time()
        else:
            self.last_scroll_event = 0

        if self.camera_state == 'rotating':
            mouse_delta = rl.get_mouse_delta()
            self.pitch += mouse_delta.y * 0.005
            self.yaw   -= mouse_delta.x * 0.005

        elif self.camera_state == 'dragging_horizontal':
            self.set_focus(None)
            mouse_pos = rl.get_mouse_position()
            prev_mouse_pos = rl.vector2_subtract(mouse_pos, rl.get_mouse_delta())
            plane_pos = self._get_mouse_pos_on_plane(mouse_pos, self.camera.target.y)
            prev_plane_pos = self._get_mouse_pos_on_plane(prev_mouse_pos, self.camera.target.y)
            delta = rl.vector2_subtract(prev_plane_pos, plane_pos)
            delta_l = rl.vector2_length(delta)
            limit_l = 20 * rl.get_frame_time() * self.camera_distance
            if delta_l > limit_l:
                delta = rl.vector2_scale(delta, limit_l / delta_l)
            self.camera.target.x += delta.x
            self.camera.target.z += delta.y
        
        elif self.camera_state == 'dragging_vertical':
            self.set_focus(None)
            mouse_delta = rl.get_mouse_delta()
            self.camera.target.y += mouse_delta.y * self.camera_distance / rl.get_screen_height()

        self.camera_distance *= (1 - rl.get_mouse_wheel_move() * 0.05)
        
        self.pitch = np.clip(self.pitch, -np.pi/2, np.pi/2)
        camera_dir = rl.Vector3()
        camera_dir.x = np.sin(self.yaw) * np.cos(self.pitch) * self.camera_distance
        camera_dir.y = np.sin(self.pitch) * self.camera_distance
        camera_dir.z = np.cos(self.yaw) * np.cos(self.pitch) * self.camera_distance

        self.camera.position = rl.vector3_add(self.camera.target, camera_dir)

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
        self.traj_shader = rl.load_shader_from_memory(trajectory_shader_vs, trajectory_shader_fs)
        self.traj_locs_window_size = rl.get_shader_location(self.traj_shader, "window_size")
        self.traj_locs_mvp = rl.get_shader_location(self.traj_shader, "mvp")
        self.traj_locs_color = rl.get_shader_location(self.traj_shader, "color")
        self.traj_locs_time = rl.get_shader_location(self.traj_shader, "current_t")


    def destroy(self):
        rl.unload_shader(self.traj_shader)
        rl.close_window()


    def set_focus(self, body_name: str|None):
        if body_name is None:
            body_name = DEFAULT_FRAME_NAME
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
        time_ptr = ffi.new('float *', self.current_time)
        rl.set_shader_value(self.traj_shader, self.traj_locs_time, time_ptr, 
                            rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)
        rl.set_shader_value_matrix(self.traj_shader, self.traj_locs_mvp, mat_model_view_projection)


        for traj in self.scene.trajectory_patches:
            vao, elems, traj_index = traj
            if not self.scene.trajectories[traj_index].is_visible:
                continue

            color = self.scene.trajectories[traj_index].color.as_array()
            color_c = ffi.cast("Color *", ffi.from_buffer(color))
            rl.set_shader_value(self.traj_shader, self.traj_locs_color, color_c, 
                                rl.ShaderUniformDataType.SHADER_UNIFORM_VEC4)

            rl.rl_enable_vertex_array(vao)
            rl_raw.rlDrawVertexArrayElements(0, elems, NULL)
            rl.rl_disable_vertex_array()

        rl.end_shader_mode()

    def _draw_bodies(self):
        for body in self.scene.bodies:
            if not body.is_visible:
                continue
            r = body.get_position(self.current_time)
            pos_3d = rl.Vector3(r[0], r[1], r[2])
            pos_2d = rl.Vector3(r[0], 0, r[2])
            color = body.color.as_rl_color()

            rl.draw_line_3d(pos_2d, pos_3d, rl.color_alpha(color, 0.5))
            rl.draw_sphere_ex(pos_3d, body.radius, 32, 64, color)
            

    def _draw_time_bar(self):
        if self.time_bounds[1] <= self.time_bounds[0]:
            return
        
        TIMEBAR_HIGHT = 20
        #rl.draw_rectangle(0, rl.get_screen_height() - TIMEBAR_HIGHT, rl.get_screen_width(), TIMEBAR_HIGHT, rl.GRAY)
        t = (self.current_time - self.time_bounds[0]) / (self.time_bounds[1] - self.time_bounds[0])
        rl.draw_rectangle(0, rl.get_screen_height() - TIMEBAR_HIGHT, int(t * rl.get_screen_width()), TIMEBAR_HIGHT, 
                          Color('main').as_rl_color())

        mouse_pos = rl.get_mouse_position()
        slider_hover = mouse_pos.y > rl.get_screen_height() - TIMEBAR_HIGHT
        if rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT) and slider_hover:
            self.time_setting = True
        if rl.is_mouse_button_up(rl.MouseButton.MOUSE_BUTTON_LEFT):
            self.time_setting = False
        if self.time_setting:
            self.current_time = self.time_bounds[0] + (mouse_pos.x / rl.get_screen_width()) * (self.time_bounds[1] - self.time_bounds[0])

    def _draw_list_element(self, e: SceneEntity, i: int):
        TEXT_SIZE = 20
        x0 = 2
        y0 = 2 + TEXT_SIZE * i
        color = Color('main').as_rl_color() if e.is_visible else Color('grey').as_rl_color()
        text = ("> " if e.name == self.focus else "  ") + e.name
        rl.draw_text(text, x0, y0, TEXT_SIZE, color)
        width = rl.measure_text(text, TEXT_SIZE)
        hover = rl.check_collision_point_rec(rl.get_mouse_position(), rl.Rectangle(x0, y0, width, TEXT_SIZE))
        if rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT) and hover:
            e.is_visible = not e.is_visible
        if rl.is_key_pressed(rl.KeyboardKey.KEY_F) and hover:
            self.set_focus(e.name)

    def _draw_ui(self):
        if self.draw_entity_list:
            for i, e in enumerate(self.scene.entities):
                self._draw_list_element(e, i)


    def _draw_grid(self):
        if self.scene.reference_frame.is_visible:
            extend = self.camera_distance * 100
            red, green, blue = Color('red').as_rl_color(), Color('green').as_rl_color(), Color('blue').as_rl_color()
            rl.draw_line_3d(rl.vector3_zero(), rl.Vector3( extend, 0, 0), rl.color_tint(red,   rl.Color(120,120,120,255)))
            rl.draw_line_3d(rl.vector3_zero(), rl.Vector3(-extend, 0, 0), rl.color_tint(red,   rl.Color(60, 60, 60,255)))
            rl.draw_line_3d(rl.vector3_zero(), rl.Vector3(0, 0, -extend), rl.color_tint(green, rl.Color(120,120,120,255)))
            rl.draw_line_3d(rl.vector3_zero(), rl.Vector3(0, 0,  extend), rl.color_tint(green, rl.Color(60, 60, 60,255)))
            rl.draw_line_3d(rl.vector3_zero(), rl.Vector3(0,  extend, 0), rl.color_tint(blue,  rl.Color(120,120,120,255)))
            rl.draw_line_3d(rl.vector3_zero(), rl.Vector3(0, -extend, 0), rl.color_tint(blue,  rl.Color(60, 60, 60,255)))

            for r in [1, 2, 5, 10, 20, 50]:
                rl.draw_circle_3d(rl.Vector3(0,0,0), r, rl.Vector3(1,0,0), 90, rl.GRAY)

        if (self.camera_state is not None or self.is_scrolling) and self.focus == DEFAULT_FRAME_NAME:
            ground_pos = rl.Vector3(self.camera.target.x, 0, self.camera.target.z)
            rl.draw_circle_3d(ground_pos, 0.01 * self.camera_distance, rl.Vector3(1,0,0), 90, Color('main').as_rl_color())
            rl.draw_line_3d(ground_pos, self.camera.target, Color('main').as_rl_color())


    def step(self):
        self.update_camera()

        rl.begin_drawing()
        
        rl.clear_background(Color('bg').as_rl_color())

        rl.begin_mode_3d(self.camera)
        self._draw_grid()
        self._draw_trajectories()
        self._draw_bodies()
        rl.end_mode_3d()

        self._draw_time_bar()
        self._draw_ui()
        #rl.draw_fps(10, 10)

        rl.end_drawing()

    def is_running(self):
        return not rl.window_should_close()
    
class show_interactable():
    def __init__(self, scene, *args, **kwargs):
        self.app = DrawApplication(scene, *args, **kwargs)

    def __enter__(self):
        self.app.setup()
        return self.app

    def __exit__(self, *args):
        self.app.destroy()


def show_scene(scene, *args, **kwargs):
    with show_interactable(scene, *args, **kwargs) as app:
        while app.is_running():
            app.step()