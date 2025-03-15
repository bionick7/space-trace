from ._shaders import *
from .utils import *
from .scene import *

import numpy as np
import pyray as rl
import raylib as rl_raw
import os.path


def _init_raylib_window():
    # Initiialize raylib graphics window
    rl.set_trace_log_level(rl.TraceLogLevel.LOG_WARNING)
    rl.set_config_flags(rl.ConfigFlags.FLAG_MSAA_4X_HINT 
                      | rl.ConfigFlags.FLAG_WINDOW_RESIZABLE
                      | rl.ConfigFlags.FLAG_VSYNC_HINT)
    rl.init_window(DEFAULT_WINDOWN_WIDTH, DEFAULT_WINDOW_HEIGHT, "Space Trace")
    rl.set_target_fps(rl.get_monitor_refresh_rate(0))

class DrawApplication():
    '''
    Main class for drawing the scene. Handles the core application loop.

    Parameters:
    -----------
    scene: spacetrace.scene.Scene
        Contains all the data and is populated by the user.
    focus: uint-64
        UUID of the camera focuses on. Defaults to `DEFAULT_FRAME_UUID`.
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

    text_size = 24
    timebar_height = 20

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
        self.main_font = None
        
        if 'focus' in kwargs:
            self.focus = self.scene.get_entity(kwargs['focus'])
        else: 
            self.focus = DEFAULT_FRAME_UUID
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
        self._planet_shader = None

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
            if entity.uuid == self.focus and self.focus != DEFAULT_FRAME_UUID:
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

        if np.isfinite(self.scene.time_bounds[0]):
            self.time_bounds = self.scene.time_bounds
        else:
            self.time_bounds = [0, 0]
        self.current_time = self.time_bounds[1]

        print("Initializing Window ...")
        if not rl.is_window_ready():
            _init_raylib_window()

        print("Loading default assets ...")
        self.resource_path = os.path.join(os.path.dirname(__file__), 'resources')
        main_font_path = os.path.join(self.resource_path, 'SpaceMono-Regular.ttf')
        self.main_font = rl.load_font_ex(main_font_path, self.text_size, ffi.NULL, 0)
        self.default_texture = rl.load_texture(os.path.join(self.resource_path, 'default_texture.png'))
        self.sphere_mesh = rl.gen_mesh_sphere(1, 32, 64)
        #self.sphere_mesh = rl.load_model(os.path.join(self.resource_path, 'sphere.obj')).meshes[0]
        
        print("Loading shaders ...")
        self._traj_shader = rl.load_shader_from_memory(trajectory_shader_vs, trajectory_shader_fs)
        self._traj_locs_window_size = rl.get_shader_location(self._traj_shader, "window_size")
        self._traj_locs_mvp = rl.get_shader_location(self._traj_shader, "mvp")
        self._traj_locs_color = rl.get_shader_location(self._traj_shader, "color")
        self._traj_locs_time = rl.get_shader_location(self._traj_shader, "current_t")

        self._planet_shader = rl.load_shader_from_memory(planet_shader_vs, planet_shader_fs)
        self._planet_locs_maps_enabled = rl.get_shader_location(self._planet_shader, "maps_enabled")
        self._planet_locs_mvp = rl.get_shader_location(self._planet_shader, "mvp")
        self._planet_locs_color = rl.get_shader_location(self._planet_shader, "color")
        self._planet_locs_albedo_map = rl.get_shader_location(self._planet_shader, "albedo_map")
        self._planet_locs_normal_map = rl.get_shader_location(self._planet_shader, "normal_map")
        self._planet_locs_specular_map = rl.get_shader_location(self._planet_shader, "specular_map")

        print("Setting up scene objects ...")
        self.scene.on_setup(self)

    def destroy(self):
        ''' Clean up after the application is done to prevent memory leaks. '''
        rl.unload_shader(self._traj_shader)
        rl.close_window()

    def set_focus(self, body_uuid: np.uint64|None):
        ''' 
            Sets the camera focus to a specific body.
            The camera will follow the body through time.
            If focus is 'None' or 'DEFAULT_FRAME_NAME', 
            the camera will not follow anything.
        '''
        if body_uuid is None:
            body_uuid = DEFAULT_FRAME_UUID
        self.focus = body_uuid

    def _draw_trajectories(self):
        '''
            Draws all the trajectories through some lower-level graphics shenanigans.
        '''
                
        rl.begin_shader_mode(self._traj_shader)

        mat_projection = rl.rl_get_matrix_projection()
        mat_model_view = rl.rl_get_matrix_modelview()
        mat_model_view_projection = rl.matrix_multiply(mat_model_view, mat_projection)

        screen_size = rl.Vector2(rl.get_screen_width(), rl.get_screen_height())
        rl.set_shader_value(self._traj_shader, self._traj_locs_window_size, screen_size, 
                            rl.ShaderUniformDataType.SHADER_UNIFORM_VEC2)
        time_ptr = ffi.new('float *', self.current_time)
        rl.set_shader_value(self._traj_shader, self._traj_locs_time, time_ptr, 
                            rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)
        rl.set_shader_value_matrix(self._traj_shader, self._traj_locs_mvp, mat_model_view_projection)


        for traj in self.scene.trajectory_patches:
            vao, elems, traj_index = traj
            if not self.scene.trajectories[traj_index].is_visible:
                continue

            color = self.scene.trajectories[traj_index].color.as_array()
            color_c = ffi.cast("Color *", ffi.from_buffer(color))
            rl.set_shader_value(self._traj_shader, self._traj_locs_color, color_c, 
                                rl.ShaderUniformDataType.SHADER_UNIFORM_VEC4)

            rl.rl_enable_vertex_array(vao)
            rl_raw.rlDrawVertexArrayElements(0, elems, ffi.NULL)
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

        # Calculate triangle vertex coordinates
        angles = np.linspace(0, 2*np.pi, 17)
        vectors = (np.outer(np.cos(angles), x) + np.outer(np.sin(angles), y)) * arrow_head_radius - z[np.newaxis,:] * arrow_head_height
        tip = rl.vector3_add(origin, v)
        root = rl.vector3_subtract(tip, rl.Vector3(*z* arrow_head_height))

        # Draw triangles
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

    def _draw_shaded_planet(self, body: Body, pos_3d: rl.Vector3, r: float, color: rl.Color):
        rl.begin_shader_mode(self._planet_shader)
        """
        rl.set_shader_value(self._planet_shader, self._planet_locs_maps_enabled, 
                            ffi.new('int *', body.maps_enabled),
                            rl.ShaderUniformDataType.SHADER_UNIFORM_INT)
        if body.albedo_map is None:
            rl.set_shader_value_texture(self._planet_shader, self._planet_locs_albedo_map, self.default_texture)
        else:
            rl.set_shader_value_texture(self._planet_shader, self._planet_locs_albedo_map, body.albedo_map)

        if body.specular_map is None:
            rl.set_shader_value_texture(self._planet_shader, self._planet_locs_specular_map, self.default_texture)
        else:
            rl.set_shader_value_texture(self._planet_shader, self._planet_locs_specular_map, body.specular_map)

        if body.normal_map is None:
            rl.set_shader_value_texture(self._planet_shader, self._planet_locs_normal_map, self.default_texture)
        else:
            rl.set_shader_value_texture(self._planet_shader, self._planet_locs_normal_map, body.normal_map)"
        """

        material = rl.load_material_default()
        material.shader = self._planet_shader
        if body.albedo_map is not None:
            material.maps[rl.MaterialMapIndex.MATERIAL_MAP_ALBEDO].texture = body.albedo_map
        if body.specular_map is not None:
            material.maps[rl.MaterialMapIndex.MATERIAL_MAP_ROUGHNESS].texture = body.specular_map
        if body.normal_map is not None:
            material.maps[rl.MaterialMapIndex.MATERIAL_MAP_NORMAL].texture = body.specular_map
            
        transform_matrix = rl.matrix_multiply(
            rl.matrix_multiply(
                rl.matrix_rotate_x(-rl.DEG2RAD * 90),
                rl.matrix_scale(r, r, r),
            ),
            rl.matrix_translate(pos_3d.x, pos_3d.y, pos_3d.z),
        )
        rl.draw_mesh(self.sphere_mesh, material, transform_matrix)
        #rl.draw_sphere_ex(pos_3d, r, 32, 64, color)
        rl.end_shader_mode()

    def _draw_bodies(self):
        '''
            Draws all the bodies in the scene.
        '''
        for body in self.scene.bodies:
            if not body.is_visible:
                continue
            pos = body.get_position(self.current_time) * self.scene.scale_factor
            r = body.radius * self.scene.scale_factor
            pos_3d = rl.Vector3(pos[0], pos[1], pos[2])
            pos_2d = rl.Vector3(pos[0], 0, pos[2])
            color = body.color.as_rl_color()

            if pos[1] != 0:
                rl.draw_line_3d(pos_2d, pos_3d, rl.color_alpha(color, 0.5))
            if body.shape == 'cross':
                self._draw_axis_cross(pos_3d, r, color)
            elif self._planet_shader is None:
                rl.draw_sphere_ex(pos_3d, r, 32, 64, color)
            else:
                self._draw_shaded_planet(body, pos_3d, r, color)
            
    def _draw_time_bar(self):
        '''
            Draws the time slider at the bottom of the screen.
        '''
        if self.time_bounds[1] <= self.time_bounds[0]:
            return
        
        #rl.draw_rectangle(0, rl.get_screen_height() - TIMEBAR_HIGHT, rl.get_screen_width(), TIMEBAR_HIGHT, rl.GRAY)
        t = (self.current_time - self.time_bounds[0]) / (self.time_bounds[1] - self.time_bounds[0])
        rl.draw_rectangle(0, rl.get_screen_height() - self.timebar_height, int(t * rl.get_screen_width()), self.timebar_height, 
                          self.scene.palette('main').as_rl_color())

        mouse_pos = rl.get_mouse_position()
        slider_hover = mouse_pos.y > rl.get_screen_height() - self.timebar_height
        if rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT) and slider_hover:
            self._time_setting = True
        if rl.is_mouse_button_up(rl.MouseButton.MOUSE_BUTTON_LEFT):
            self._time_setting = False
        if self._time_setting:
            self.current_time = self.time_bounds[0] + (mouse_pos.x / rl.get_screen_width()) * (self.time_bounds[1] - self.time_bounds[0])

    def _draw_list_element(self, e: SceneEntity, index: int, indent: int):
        '''
            Draws a single element in the list of entities.
        '''
        x0 = 2 + 5 * indent
        y0 = 2 + self.text_size * index
        display_name = e.name.split("/")[-1]
        if isinstance(e, Group):
            text_foling = ("> " if e.folded else "v ")
        else:
            text_foling = ""
        
        text_focus = ("x " if e.uuid == self.focus else "  ")
        text_visible = ("o " if e.is_visible else "- ")

        texts = [text_foling, text_focus, text_visible, display_name]

        color = self.scene.palette('main').as_rl_color() if e.is_visible else self.scene.palette('grey').as_rl_color()

        if self.main_font is None:
            rl.draw_text("".join(texts), x0, y0, self.text_size, color)
            widths = [rl.measure_text(text, self.text_size) for text in texts]
        else:
            rl.draw_text_ex(self.main_font, "".join(texts), rl.Vector2(x0, y0), self.text_size, 1, color)
            widths = [rl.measure_text_ex(self.main_font, text, self.text_size, 1).x for text in texts]

        x_coords = [x0 + sum(widths[:i+1]) for i in range(len(widths))]

        # Check for mouse hovering
        hover_index = -1
        hover = rl.check_collision_point_rec(rl.get_mouse_position(), rl.Rectangle(x0, y0, x_coords[-1] - x0, self.text_size))
        if hover:
            x_cursor = rl.get_mouse_position().x
            for i, x_c in enumerate([0] + x_coords[:-1]):
                if x_c < x_cursor < x_coords[i]:
                    hover_index = i

        # Handle input
        if rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT) and hover:
            if hover_index == 0 and isinstance(e, Group):
                e.folded = not e.folded
            elif hover_index in (1, 3):
                self.set_focus(e.uuid)
            elif hover_index == 2:
                e.is_visible = not e.is_visible
        if rl.is_key_pressed(rl.KeyboardKey.KEY_F) and hover:
            self.set_focus(e.uuid)
        
        # Handle groups recursively
        if isinstance(e, Group):
            index += 1
            for child in e.members:
                index = self._draw_list_element(child, index, indent + 1)
            return index
        else:
            return index + 1

    def _draw_ui(self):
        '''
            Draws the rest of the UI, most notably the list of entities.
        '''
        index = 0
        for v in self.scene.hierarchy.values():
            index = self._draw_list_element(v, index, 0)

    def _draw_grid(self):
        '''
            Draws the primary reference frame and the camera target if no focus is set.
        '''
        if self.scene.transforms[0].is_visible:
            for r in [1, 2, 5, 10, 20, 50]:
                rl.draw_circle_3d(rl.Vector3(0,0,0), r, rl.Vector3(1,0,0), 90, rl.GRAY)

        if (self._camera_state is not None or self.is_scrolling) and self.focus == DEFAULT_FRAME_UUID:
            ground_pos = rl.Vector3(self.camera.target.x, 0, self.camera.target.z)
            rl.draw_circle_3d(ground_pos, 0.01 * self.camera_distance, rl.Vector3(1,0,0), 90, self.scene.palette('main').as_rl_color())
            rl.draw_line_3d(ground_pos, self.camera.target, self.scene.palette('main').as_rl_color())

    def step(self):
        '''
            A single step in the main application loop.
        '''
        self.update_camera()

        rl.begin_drawing()
        
        rl.clear_background(self.scene.palette('bg').as_rl_color())

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