from spacetrace._shaders import *
from spacetrace.scene import *

import numpy as np
import pyray as rl
import raylib as rl_raw


class DrawApplication():
    '''
    Main class for drawing the scene. Handles the core application loop.

    Parameters:
    -----------
    scene: spacetrace.scene.Scene
        Contains all the data and is populated by the user.
    focus: str
        Name of the camera focuses on. Defaults to `DEFAULT_FRAME_NAME`.
    camera_distance: float
    camera_pitch: float
    camera_yaw: float
        Spherical coordinates for the camera position relative to the camera target.
        pitch and yaw are in radians.
    camera: rl.Camera3D
        Camera object used by raylib.
    time_bounds: list[float]
        first and last epochs mentionned by any scene entity.
    current_time: float
        Current time, everything is beeing rendered relative to.
    draw_entity_list: bool
        Whether to draw the list of entities on the left side of the window.
    camera_state: str
        Current state of camera manipulation.
        Can be 'rotating', 'dragging_vertical' or 'dragging_horizontal'.
    arrowhead_scaling_cap: float
        Above this length (in draw space), arrow length will not affect arrowhead size
    arrowhead_height_ratio: float
        Ratio between arrowhead heigth and arrow length
    arrowhead_radius_ratio: float
        Ratio between arrowhead radius and arrow length
    '''

    def __init__(self, scene: Scene, *args, **kwargs):
        '''
        scene: the scene to be drawn (c.f. spacetrace.scene.Scene)
        keyword arguments:
            focus: str
            camera_distance: float
            camera_pitch: float
            camera_yaw: float
            draw_entity_list: bool
                c.f. DrawApplication parameters
            
            show_axes: bool
                Whether to show the default reference frame.
            camera_fov: float
                Field of view of the camera in degrees.
        '''
        self.scene = scene
        
        self.focus = kwargs.get("focus", DEFAULT_FRAME_NAME)
        self.camera_distance = kwargs.get("camera_distance", 5.0)
        self.pitch = kwargs.get("camera_pitch", 0.5)
        self.yaw = kwargs.get("camera_yaw", 0)
        self.draw_entity_list = kwargs.get("draw_entity_list", True)
        self.scene.transforms[0].is_visible = kwargs.get("show_axes", True)

        camera_dir = rl.Vector3()
        camera_dir.x = np.sin(self.yaw) * np.cos(self.pitch)
        camera_dir.y = np.sin(self.pitch)
        camera_dir.z = np.cos(self.yaw) * np.cos(self.pitch)

        self.camera = rl.Camera3D(camera_dir, rl.vector3_zero(), 
                                  rl.Vector3(0, 1, 0), kwargs.get("camera_fov", 45), 
                                  rl.CAMERA_PERSPECTIVE)

        self.time_bounds = [0, 0]
        self.current_time = 0

        self.arrowhead_scaling_cap = 1
        self.arrowhead_radius_ratio = 0.05
        self.arrowhead_height_ratio = 0.2

        # internal state keeping
        self._traj_shader = None
        self._camera_state = None
        self._last_scroll_event = 1e6
        self._time_setting = False

    def _get_mouse_pos_on_plane(self, mouse_pos: rl.Vector2, plane: float):
        ''' Projects the mouse position from screen space onto a horizontal plane '''
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
        return self._last_scroll_event < 0.2

    def update_camera(self):
        ''' Updates rotation. Maintains target while updating yaw and pitch trhough mouse input. '''

        # Follow focus if applicable
        for entity in self.scene.entities:
            if entity.name == self.focus and self.focus != DEFAULT_FRAME_NAME:
                pos = entity.get_position(self.current_time) * self.scene.scale_factor
                self.camera.target = rl.Vector3(pos[0], pos[1], pos[2])

        # Get Camera state
        self._camera_state = None
        if rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_MIDDLE):
            self._camera_state = 'rotating'
        elif rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_RIGHT):
            if rl.is_key_down(rl.KEY_LEFT_SHIFT):
                self._camera_state = 'dragging_vertical'
            else:
                self._camera_state = 'dragging_horizontal'
        if rl.get_mouse_wheel_move() == 0:
            self._last_scroll_event += rl.get_frame_time()
        else:
            self._last_scroll_event = 0

        # Update camera position
        if self._camera_state == 'rotating':
            mouse_delta = rl.get_mouse_delta()
            self.pitch += mouse_delta.y * 0.005
            self.yaw   -= mouse_delta.x * 0.005
            self.pitch = min(max(self.pitch, -np.pi/2 + 1e-4), np.pi/2 - 1e-4)

        elif self._camera_state == 'dragging_horizontal':
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
        
        elif self._camera_state == 'dragging_vertical':
            self.set_focus(None)
            mouse_delta = rl.get_mouse_delta()
            self.camera.target.y += mouse_delta.y * self.camera_distance / rl.get_screen_height()

        # Handle scrolling
        self.camera_distance *= (1 - rl.get_mouse_wheel_move() * 0.05)
        
        # Transform back to cartesian coordinates
        self.pitch = np.clip(self.pitch, -np.pi/2, np.pi/2)
        camera_dir = rl.Vector3()
        camera_dir.x = np.sin(self.yaw) * np.cos(self.pitch) * self.camera_distance
        camera_dir.y = np.sin(self.pitch) * self.camera_distance
        camera_dir.z = np.cos(self.yaw) * np.cos(self.pitch) * self.camera_distance

        self.camera.position = rl.vector3_add(self.camera.target, camera_dir)

    def setup(self):
        ''' Setup application after the scene has been fully populated. '''
        if not rl.is_window_ready():
            _init_raylib_window()

        if np.isfinite(self.scene.time_bounds[0]):
            self.time_bounds = self.scene.time_bounds
        else:
            self.time_bounds = [0, 0]
        self.current_time = self.time_bounds[1]

        # Load trajectory shader
        self._traj_shader = rl.load_shader_from_memory(trajectory_shader_vs, trajectory_shader_fs)
        self.traj_locs_window_size = rl.get_shader_location(self._traj_shader, "window_size")
        self.traj_locs_mvp = rl.get_shader_location(self._traj_shader, "mvp")
        self.traj_locs_color = rl.get_shader_location(self._traj_shader, "color")
        self.traj_locs_time = rl.get_shader_location(self._traj_shader, "current_t")

    def destroy(self):
        ''' Clean up after the application is done to prevent memory leaks. '''
        rl.unload_shader(self._traj_shader)
        rl.close_window()

    def set_focus(self, body_name: str|None):
        ''' 
            Sets the camera focus to a specific body.
            The camera will follow the body through time.
            If focus is 'None' or 'DEFAULT_FRAME_NAME', 
            the camera will not follow anything.
        '''
        if body_name is None:
            body_name = DEFAULT_FRAME_NAME
        self.focus = body_name

    def _draw_trajectories(self):
        '''
            Draws all the trajectories through some lower-level graphics shenanigans.
        '''
        
        NULL = ffi.NULL
        
        rl.begin_shader_mode(self._traj_shader)
        #rl.draw_cube(rl.vector3_zero(), 1, 1, 1, rl.BLUE)

        mat_projection = rl.rl_get_matrix_projection()
        mat_model_view = rl.rl_get_matrix_modelview()
        mat_model_view_projection = rl.matrix_multiply(mat_model_view, mat_projection)

        screen_size = rl.Vector2(rl.get_screen_width(), rl.get_screen_height())
        rl.set_shader_value(self._traj_shader, self.traj_locs_window_size, screen_size, 
                            rl.ShaderUniformDataType.SHADER_UNIFORM_VEC2)
        time_ptr = ffi.new('float *', self.current_time)
        rl.set_shader_value(self._traj_shader, self.traj_locs_time, time_ptr, 
                            rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)
        rl.set_shader_value_matrix(self._traj_shader, self.traj_locs_mvp, mat_model_view_projection)


        for traj in self.scene.trajectory_patches:
            vao, elems, traj_index = traj
            if not self.scene.trajectories[traj_index].is_visible:
                continue

            color = self.scene.trajectories[traj_index].color.as_array()
            color_c = ffi.cast("Color *", ffi.from_buffer(color))
            rl.set_shader_value(self._traj_shader, self.traj_locs_color, color_c, 
                                rl.ShaderUniformDataType.SHADER_UNIFORM_VEC4)

            rl.rl_enable_vertex_array(vao)
            rl_raw.rlDrawVertexArrayElements(0, elems, NULL)
            rl.rl_disable_vertex_array()

        rl.end_shader_mode()

    def _draw_vector(self, origin: rl.Vector3, v: rl.Vector3, color: rl.Color):
        rl.draw_line_3d(origin, rl.vector3_add(origin, v), color)
        
        # Establish local coordinate system
        z = np.array([v.x, v.y, v.z]) / rl.vector3_length(v)
        x_start = np.array([1,0,0]) if v.x < 0.5 else np.array([0,1,0])
        x = (x_start - z * np.dot(x_start, z))
        x /= np.linalg.norm(x)
        y = np.cross(z, x)

        arrow_head_radius = self.arrowhead_radius_ratio * min(rl.vector3_length(v), self.arrowhead_scaling_cap)
        arrow_head_height = self.arrowhead_height_ratio * min(rl.vector3_length(v), self.arrowhead_scaling_cap)

        angles = np.linspace(0, 2*np.pi, 17)
        vectors = (np.outer(np.cos(angles), x) + np.outer(np.sin(angles), y)) * arrow_head_radius - z[np.newaxis,:] * arrow_head_height
        tip = rl.vector3_add(origin, v)
        root = rl.vector3_subtract(tip, rl.Vector3(*z* arrow_head_height))
        for i in range(16):
            v1 = rl.Vector3(*vectors[i])
            v2 = rl.Vector3(*vectors[i + 1])
            rl.draw_triangle_3d(tip, rl.vector3_add(tip, v1), rl.vector3_add(tip, v2), color)
            rl.draw_triangle_3d(root, rl.vector3_add(tip, v2), rl.vector3_add(tip, v1), color)
            #rl.draw_line_3d(tip, rl.vector3_add(tip, v1), color)

    def _draw_gizmos(self):
        for vector in self.scene.vectors:
            if not vector.is_visible:
                continue
            if vector.draw_space:
                o = vector.get_position(self.current_time)
                v = vector.get_vector(self.current_time)
            else:
                o = vector.get_position(self.current_time) * self.scene.scale_factor
                v = vector.get_vector(self.current_time) * self.scene.scale_factor
            self._draw_vector(rl.Vector3(*o), rl.Vector3(*v), vector.color.as_rl_color())
        
        for transform in self.scene.transforms:
            if not transform.is_visible:
                continue
            if transform.draw_space:
                o = transform.get_position(self.current_time)
                basis = transform.get_basis(self.current_time)
            else:
                o = transform.get_position(self.current_time) * self.scene.scale_factor
                basis = transform.get_basis(self.current_time) * self.scene.scale_factor
            self._draw_vector(rl.Vector3(*o), rl.Vector3(*basis[:,0]), transform.get_x_color().as_rl_color())
            self._draw_vector(rl.Vector3(*o), rl.Vector3(*basis[:,1]), transform.get_y_color().as_rl_color())
            self._draw_vector(rl.Vector3(*o), rl.Vector3(*basis[:,2]), transform.get_z_color().as_rl_color())

    def _draw_axis_cross(self, point: rl.Vector3, extend: float, rl_color: rl.Color):
        rl.draw_line_3d(rl.vector3_add(point, rl.Vector3(-extend,0,0)), rl.vector3_add(point, rl.Vector3(extend,0,0)), rl_color)
        rl.draw_line_3d(rl.vector3_add(point, rl.Vector3(0,-extend,0)), rl.vector3_add(point, rl.Vector3(0,extend,0)), rl_color)
        rl.draw_line_3d(rl.vector3_add(point, rl.Vector3(0,0,-extend)), rl.vector3_add(point, rl.Vector3(0,0,extend)), rl_color)

    def _draw_bodies(self):
        '''
            Draws all the bodies in the scene.
        '''
        for body in self.scene.bodies:
            if not body.is_visible:
                continue
            r = body.get_position(self.current_time) * self.scene.scale_factor
            pos_3d = rl.Vector3(r[0], r[1], r[2])
            pos_2d = rl.Vector3(r[0], 0, r[2])
            color = body.color.as_rl_color()

            if r[1] != 0:
                rl.draw_line_3d(pos_2d, pos_3d, rl.color_alpha(color, 0.5))
            if body.shape == 'cross':
                self._draw_axis_cross(pos_3d, body.radius, color)
            else:
                rl.draw_sphere_ex(pos_3d, body.radius, 32, 64, color)
            
    def _draw_time_bar(self):
        '''
            Draws the time slider at the bottom of the screen.
        '''
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
            self._time_setting = True
        if rl.is_mouse_button_up(rl.MouseButton.MOUSE_BUTTON_LEFT):
            self._time_setting = False
        if self._time_setting:
            self.current_time = self.time_bounds[0] + (mouse_pos.x / rl.get_screen_width()) * (self.time_bounds[1] - self.time_bounds[0])

    def _draw_list_element(self, e: SceneEntity, i: int):
        '''
            Draws a single element in the list of entities.
        '''
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
        '''
            Draws the rest of the UI, most notably the list of entities.
        '''
        if self.draw_entity_list:
            for i, e in enumerate(self.scene.entities):
                self._draw_list_element(e, i)

    def _draw_grid(self):
        '''
            Draws the primary reference frame and the camera target if no focus is set.
        '''
        if self.scene.transforms[0].is_visible:
            for r in [1, 2, 5, 10, 20, 50]:
                rl.draw_circle_3d(rl.Vector3(0,0,0), r, rl.Vector3(1,0,0), 90, rl.GRAY)

        if (self._camera_state is not None or self.is_scrolling) and self.focus == DEFAULT_FRAME_NAME:
            ground_pos = rl.Vector3(self.camera.target.x, 0, self.camera.target.z)
            rl.draw_circle_3d(ground_pos, 0.01 * self.camera_distance, rl.Vector3(1,0,0), 90, Color('main').as_rl_color())
            rl.draw_line_3d(ground_pos, self.camera.target, Color('main').as_rl_color())

    def step(self):
        '''
            A single step in the main application loop.
        '''
        self.update_camera()

        rl.begin_drawing()
        
        rl.clear_background(Color('bg').as_rl_color())

        rl.begin_mode_3d(self.camera)
        self._draw_grid()
        self._draw_trajectories()
        self._draw_bodies()
        self._draw_gizmos()
        rl.end_mode_3d()

        self._draw_time_bar()
        self._draw_ui()
        #rl.draw_fps(10, 10)

        rl.end_drawing()

    def is_running(self):
        ''' 
            Whether or not the application wants to continue running. 
            Is false when the window is closed.
        '''
        return not rl.window_should_close()
    
class show_interactable():
    '''
        Context manager to wrap around the scene. This can be called by the user to operate 
        the main application loop and gain more control over the rendering or input.
    '''
    def __init__(self, scene, *args, **kwargs):
        self.app = DrawApplication(scene, *args, **kwargs)

    def __enter__(self):
        self.app.setup()
        return self.app

    def __exit__(self, *args):
        self.app.destroy()


def show_scene(scene, *args, **kwargs):
    '''
        Fire and forget function to show a scene. This will block the main thread until the window is closed.
        for args and kwargs, c.f. DrawApplication initializ
    '''
    with show_interactable(scene, *args, **kwargs) as app:
        while app.is_running():
            app.step()