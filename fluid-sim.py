import os
import slangpy as spy
import numpy as np

COLOR_FORMAT = spy.Format.rgba32_float

def get_asset_path(asset):
    return os.path.join(os.path.dirname(__file__), asset)

def is_down(input_state, key):
    return key in input_state and input_state[key]

class FluidSimulator:
    def __init__(self, device : spy.Device):
        self.device = device
        self.enabled = False
        self.step_once = False
        self.initialized = False
        self.vars = {}
        
        self.passes = {
            entry: self.device.create_compute_kernel(self.device.load_program("coflip.cs.slang", [entry]))
            for entry in [
                "init_particles",

                "build_particle_map",
                "particle_to_grid",

                "compute_divergence",
                "pressure_projection",
                "apply_pressure_gradient",

                "advect_particles"
            ]
        }
        self.passes["particle_map_compute_offsets"] = self.device.create_compute_kernel(self.device.load_program("ParticleMap.cs.slang", ["compute_offsets"]))
        self.passes["particle_map_sort"] = self.device.create_compute_kernel(self.device.load_program("ParticleMap.cs.slang", ["sort"]))

    def setup_ui(self, window):
        def toggle_callback():
            self.enabled = not self.enabled
            self.toggle_button.label = "Pause simulation" if self.enabled else "Resume simulation"
        self.toggle_button = spy.ui.Button(window, "Resume simulation", callback=toggle_callback)

        def step_callback():
            self.step_once = True
        spy.ui.Button(window, "Step", callback=step_callback)

        def reset_callback(arg=None):
            self.initialized = False
        spy.ui.Button(window, "Reset", callback=reset_callback)

        self.resolution                  = spy.ui.DragInt2(window,  "Resolution", value=spy.int2(512,512), min=1, callback=reset_callback)
        self.num_particles               = spy.ui.DragInt(window,   "Particles", 10000, callback=reset_callback)
        self.advection_iterations        = spy.ui.DragInt(window,   "Advection iterations", value=3, min=1)
        self.pressure_project_iterations = spy.ui.DragInt(window,   "Pressure projection iterations", value=50, min=1)
        self.dt                          = spy.ui.DragFloat(window, "dt", value=0.1, min=0)
        self.smoke                       = spy.ui.CheckBox(window,  "Smoke")
    
    def initialize(self, command_encoder:spy.CommandEncoder):
        u_grid = self.device.create_texture(
            format=spy.Format.r32_float,
            width=self.resolution.value.x+1,
            height=self.resolution.value.y,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label="u_grid",
        )
        v_grid = self.device.create_texture(
            format=spy.Format.r32_float,
            width=self.resolution.value.x,
            height=self.resolution.value.y+1,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label="v_grid",
        )
        density_grid, divergence, p_grid, p_grid_out = [ self.device.create_texture(
            format=spy.Format.r32_float,
            width=self.resolution.value.x,
            height=self.resolution.value.y,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label=label,
        ) for label in ["density_grid", "divergence", "p_grid", "p_grid_out"]]

        particle_pos_impulse = self.device.create_buffer(
            element_count=self.num_particles.value,
            struct_size=16,
            format=spy.Format.rgba32_float,
            usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access
        )
        particle_volume = self.device.create_buffer(
            element_count=self.num_particles.value,
            struct_size=4,
            format=spy.Format.r32_float,
            usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access
        )

        self.particle_map_vars = {
            "data": self.device.create_buffer(
                element_count=self.num_particles.value,
                struct_size=4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access
            ),
            "sorted_data": self.device.create_buffer(
                element_count=self.num_particles.value,
                struct_size=4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access
            ),
            "data_indices": self.device.create_buffer(
                element_count=self.num_particles.value,
                struct_size=8,
                format=spy.Format.rg32_uint,
                usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access
            ),
            "data_counters": self.device.create_buffer(
                element_count=2,
                struct_size=4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access
            ),
            "cell_data_offsets": self.device.create_buffer(
                element_count=(self.resolution.value.x + 4)*(self.resolution.value.y + 4),
                struct_size=4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access
            ),
            "cell_counters": self.device.create_buffer(
                element_count=(self.resolution.value.x + 4)*(self.resolution.value.y + 4),
                struct_size=4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access
            ),
            "grid_size": [self.resolution.value.x+4, self.resolution.value.y+4],
            "grid_offset": [2, 2],
            "capacity": self.num_particles.value
        }
        self.vars={
            "particle_pos_impulse": particle_pos_impulse,
            "particle_volume": particle_volume,
            "density_grid": density_grid,
            "u_grid":     u_grid,
            "v_grid":     v_grid,
            "divergence": divergence,
            "p_grid":     p_grid,
            "p_grid_out": p_grid_out,
            "resolution": self.resolution.value,
            "cell_size":  1 / self.resolution.value.x,
            "num_particles": self.num_particles.value,
            "particle_map": self.particle_map_vars,
            "dt": self.dt.value,
            "init_seed": 1234
        }

        self.passes["init_particles"].dispatch([4096, (self.num_particles.value+4095)//4096, 1], self.vars, command_encoder)

        self.initialized = True

        self.particle_to_grid(command_encoder)

    def get_density(self):    return self.vars["density_grid"]
    def get_velocity_x(self): return self.vars["u_grid"]
    def get_velocity_y(self): return self.vars["v_grid"]

    def particle_to_grid(self, command_encoder:spy.CommandEncoder):
        particle_dispatch_dim = [4096, (self.num_particles.value+4095)//4096, 1]

        command_encoder.clear_buffer(self.particle_map_vars["data_counters"])
        command_encoder.clear_buffer(self.particle_map_vars["cell_counters"])
        self.passes["build_particle_map"].dispatch(particle_dispatch_dim, self.vars, command_encoder)

        self.passes["particle_map_compute_offsets"].dispatch(self.particle_map_vars["grid_size"] + [1], {"particle_map": self.particle_map_vars}, command_encoder)
        self.passes["particle_map_sort"].dispatch(particle_dispatch_dim, {"particle_map": self.particle_map_vars}, command_encoder)

        self.passes["particle_to_grid"].dispatch([self.resolution.value.x+1, self.resolution.value.y+1, 1], self.vars, command_encoder)
    
    def step(self, command_encoder:spy.CommandEncoder, dt):
        if not self.initialized:
            self.initialize(command_encoder)

        if not self.enabled and not self.step_once:
            return
        
        self.step_once = False

        self.vars["dt"] = self.dt.value / self.advection_iterations.value

        resolution = spy.uint2(self.resolution.value.x, self.resolution.value.y)

        # if self.smoke.value:
        #     self.emit_smoke_pass.dispatch(
        #         [self.density.width, self.density.height, 1],
        #         {
        #             "density":    self.density,
        #             "velocity_x": self.velocity_x,
        #             "velocity_y": self.velocity_y,
        #             "dim": resolution,
        #             "dt": self.dt.value,
        #         },
        #         command_encoder
        #     )

        particle_dispatch_dim = [4096, (self.num_particles.value+4095)//4096, 1]
        grid_dispatch_dim = [resolution.x, resolution.y, 1]
        vel_grid_dispatch_dim = [resolution.x+1, resolution.y+1, 1]

        def pressure_projection():
            self.passes["compute_divergence"].dispatch(grid_dispatch_dim, self.vars, command_encoder)
            self.vars["p_grid"], self.vars["p_grid_out"] = self.vars["p_grid_out"], self.vars["p_grid"]
            for _ in range(self.pressure_project_iterations.value):
                self.passes["pressure_projection"].dispatch(vel_grid_dispatch_dim, self.vars, command_encoder)
                self.vars["p_grid"], self.vars["p_grid_out"] = self.vars["p_grid_out"], self.vars["p_grid"]
            self.passes["apply_pressure_gradient"].dispatch(vel_grid_dispatch_dim, self.vars, command_encoder)
                
        # Particle to grid transfer
        self.particle_to_grid(command_encoder)
        pressure_projection()
        
        # Fixed-point iteration for implicit trapezoidal
        for iter in range(self.advection_iterations.value):
            # Advect particles
            self.passes["advect_particles"].dispatch(particle_dispatch_dim, self.vars, command_encoder)
            
            # Update grid from particles
            self.particle_to_grid(command_encoder)
            pressure_projection()

class App:
    def __init__(self):
        super().__init__()
        self.device = spy.create_device(
            type=spy.DeviceType.vulkan,
            include_paths=[os.path.abspath(os.path.dirname(__file__)), os.path.abspath("src")],
        )
        self.window = spy.Window(width=1400, height=(1400*9)//16, title="App", resizable=True)
        self.surface = self.device.create_surface(self.window)
        self.surface.configure(
            width = self.window.width,
            height = self.window.height,
            vsync = True
        )

        self.device.register_shader_hot_reload_callback(self.on_shader_reload)
        self.window.on_resize         = self.on_resize
        self.window.on_keyboard_event = self.on_keyboard_event
        self.window.on_mouse_event    = self.on_mouse_event

        self.ui = spy.ui.Context(self.device)

        self.pause = False
        self.minimized = False
        self.input_state = {}
        self.render_texture = None
        self.fps_avg = 0

        self.simulator = FluidSimulator(self.device)

        self.render_shader = self.device.create_compute_kernel(self.device.load_program("render-fluid.cs.slang", ["main"]))

        self.linear_sampler = self.device.create_sampler()

        self.setup_ui()
        
        self.render_offset = spy.float2(0,0)
        self.render_scale  = 1

    def setup_ui(self):
        screen = self.ui.screen
        window = spy.ui.Window(screen, "Settings", size=spy.float2(500, 300))

        self.fps_text = spy.ui.Text(window, "FPS: 0")
        
        self.simulator.setup_ui(spy.ui.Group(window, label="Simulation"))

        self.render_mode = spy.ui.ComboBox(window, "Render", 0, items=[ "Density", "Velocity", "Divergence" ])

    def on_resize(self, width: int, height: int):
        self.device.wait()
        if width > 0 and height > 0:
            self.surface.configure(width=width, height=height)
            self.minimized = False
        else:
            self.minimized = True
            self.surface.unconfigure()
            self.render_texture = None

    def on_shader_reload(self, e:spy.ShaderHotReloadEvent):
        self.history_valid = False

    def on_keyboard_event(self, event: spy.KeyboardEvent):
        has_focus = not self.ui.handle_keyboard_event(event)
        if has_focus and event.is_key_press(): self.input_state[event.key] = True
        if event.is_key_release(): self.input_state[event.key] = False
        
    def on_mouse_event(self, event: spy.MouseEvent):
        has_focus = not self.ui.handle_mouse_event(event)
        if event.is_move(): self.input_state["mouse"] = event.pos
        if has_focus and event.is_button_down(): self.input_state[event.button] = True
        if has_focus and event.is_scroll(): self.input_state["scroll"] = event.scroll.y
        if event.is_button_up(): self.input_state[event.button] = False

    def main_loop(self):
        self.frame_timer = spy.Timer()
        while not self.window.should_close():
            self.input_state["scroll"] = 0
            self.window.process_events()

            if self.minimized:
                continue

            surface_texture = self.surface.acquire_next_image()
            if not surface_texture:
                continue

            dt = self.frame_timer.elapsed_s()
            self.frame_timer.reset()
            self.fps_avg = 0.95 * self.fps_avg + 0.05 * (1.0 / dt)

            self.ui.begin_frame(surface_texture.width, surface_texture.height)

            command_encoder = self.device.create_command_encoder()

            self.simulator.step(command_encoder, dt)
        
            if self.render_texture is None or self.render_texture.width != surface_texture.width or self.render_texture.height != surface_texture.height:
                self.render_texture = self.device.create_texture(
                    format=COLOR_FORMAT,
                    width=surface_texture.width,
                    height=surface_texture.height,
                    usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access | spy.TextureUsage.render_target,
                    label="render_texture",
                )

            screen_size = spy.uint2(surface_texture.width, surface_texture.height)
            fluid_dim = spy.int2(self.simulator.get_density().width, self.simulator.get_density().height)
            rect_pos = spy.int2(
                int(screen_size.x/2 - self.render_scale*fluid_dim.x/2 - self.render_offset.x),
                int(screen_size.y/2 - self.render_scale*fluid_dim.y/2 - self.render_offset.y)
            )
            rect_size = spy.int2(
                int(fluid_dim.x * self.render_scale),
                int(fluid_dim.y * self.render_scale),
            )

            self.render_shader.dispatch(
                thread_count=[self.render_texture.width, self.render_texture.height, 1],
                vars={
                    "render_target": self.render_texture,
                    "density":    self.simulator.get_density(),
                    "velocity_x": self.simulator.get_velocity_x(),
                    "velocity_y": self.simulator.get_velocity_y(),
                    "sampler": self.linear_sampler,
                    "screen_rect": spy.int4(rect_pos, rect_pos + rect_size),
                    "screen_size": screen_size,
                    "render_mode": self.render_mode.value
                },
                command_encoder=command_encoder
            )
            
            command_encoder.blit(surface_texture, self.render_texture)

            self.fps_text.text = f"FPS: {self.fps_avg:.2f}"

            if is_down(self.input_state, spy.MouseButton.left) and "mouse" in self.input_state:
                if self.drag_start is not None:
                    if is_down(self.input_state, spy.KeyCode.left_shift):
                        uv0 = (self.drag_start - self.render_offset) / spy.float2(rect_size.x, rect_size.y)
                        uv1 = (self.input_state["mouse"] - self.render_offset) / spy.float2(rect_size.x, rect_size.y)
                        # self.simulator.emit_drag(command_encoder, uv0, uv1, 30/fluid_dim.x, 1/fluid_dim.x)
                    else:
                        self.render_offset -= self.input_state["mouse"] - self.drag_start
                self.drag_start = self.input_state["mouse"]
            else:
                self.drag_start = None
            self.render_scale *= 1 + .05*self.input_state["scroll"]

            self.ui.end_frame(surface_texture, command_encoder)

            self.device.submit_command_buffer(command_encoder.finish())
            del surface_texture

            self.surface.present()
            
        self.device.wait()

app = App()
app.main_loop()
