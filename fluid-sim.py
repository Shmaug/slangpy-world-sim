import os
import slangpy as spy
from ParticleMap import ParticleMap

COLOR_FORMAT = spy.Format.rgba32_float

INIT_NONE = 0
INIT_VORTEX = 1
INIT_LEAPFROG = 2

ADVECT_GRID = 0
ADVECT_PARTICLE = 1
ADVECT_COFLIP_GRID = 2
ADVECT_COFLIP_PARTICLE = 3

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
        self.grid_vars = {}
        self.passes = {}

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

        self.init_mode  = spy.ui.ComboBox(window, "Init mode", 0, reset_callback, ["None", "Vortex", "Leapfrog"])
        self.emit_smoke = spy.ui.CheckBox(window, "Emit smoke")
        self.advect_mode = spy.ui.ComboBox(window, "Advect mode", 0, items=["Grid", "Particle", "CO-FLIP Grid", "CO-FLIP Particle"])

        self.resolution                  = spy.ui.DragInt2(window,  "Resolution", value=spy.int2(512,512), min=1, callback=reset_callback)
        self.advect_dt                   = spy.ui.DragFloat(window, "Advection dt", value=0.1, min=0)
        self.advection_iterations        = spy.ui.DragInt(window,   "Advection iterations", value=3, min=1)
        self.pressure_project_iterations = spy.ui.DragInt(window,   "Pressure projection iterations", value=50, min=1)
        self.avg_pressure = spy.ui.Text(window, "Avg pressure: 0")
    
    def swap_grids(self):
        for name in ["velocity_x", "velocity_y", "pressure"]:
            name_out = f"{name}_rw"
            self.grid_vars[name], self.grid_vars[name_out] = self.grid_vars[name_out], self.grid_vars[name]

    def dispatch_pass(self, shader, entry, dim, vars, command_encoder):
        id = f"{shader}:{entry}"
        if id not in self.passes:
            self.passes[id] = self.device.create_compute_kernel(self.device.load_program(shader, [entry]))
        self.passes[id].dispatch(dim, vars, command_encoder)

    # must be called after updating velocities (advection), but before swapping the velocity grids
    def pressure_project(self, command_encoder:spy.CommandEncoder):
        if self.pressure_project_iterations.value == 0:
            return
        command_encoder.clear_texture_float(self.pressure_correction_0)
        command_encoder.clear_texture_float(self.pressure_correction_1)
        vars = {
            "grid": self.grid_vars,
            "dt":   self.advect_dt.value / self.advection_iterations.value,
            "divergence":             self.divergence,
            "pressure_correction":    self.pressure_correction_0,
            "pressure_correction_rw": self.pressure_correction_1,
        }

        self.dispatch_pass(
            "fluid-pressure-project.cs.slang",
            "update_divergence",
            self.grid_dispatch_dim,
            vars,
            command_encoder)
        for iter in range(self.pressure_project_iterations.value):
            self.dispatch_pass(
                "fluid-pressure-project.cs.slang",
                "step",
                self.grid_dispatch_dim,
                vars,
                command_encoder)
            vars["pressure_correction"], vars["pressure_correction_rw"] = vars["pressure_correction_rw"], vars["pressure_correction"]
        self.dispatch_pass(
            "fluid-pressure-project.cs.slang",
            "apply",
            self.grid_dispatch_dim,
            vars,
            command_encoder)

    def initialize(self, command_encoder:spy.CommandEncoder):
        def create_grid(width, height, name, mip_count = 1):
            return self.device.create_texture(
                format=spy.Format.r32_float,
                width=width,
                height=height,
                mip_count=mip_count,
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                label=name)
        
        w,h = self.resolution.value

        self.grid_vars = {
            "velocity_x":     create_grid(w+1, h, "velocity_x_0"),
            "velocity_x_rw":  create_grid(w+1, h, "velocity_x_1"),
            "velocity_y":     create_grid(w, h+1, "velocity_y_0"),
            "velocity_y_rw":  create_grid(w, h+1, "velocity_y_1"),
            "pressure":       create_grid(w, h,   "pressure_0", spy.ALL_MIPS),
            "pressure_rw":    create_grid(w, h,   "pressure_1", spy.ALL_MIPS),
            "linear_sampler": self.device.create_sampler(),
            "resolution":     self.resolution.value,
        }
        self.divergence            = create_grid(w, h, "divergence")
        self.pressure_correction_0 = create_grid(w, h, "pressure_correction_0")
        self.pressure_correction_1 = create_grid(w, h, "pressure_correction_1")

        self.particle_map = ParticleMap(self.device, w*h, 4*w*h)

        self.avg_pressure_buf = self.device.create_buffer(
            element_count=1,
            struct_size=4,
            format=spy.Format.r32_float,
            memory_type=spy.MemoryType.read_back,
            usage=spy.BufferUsage.copy_destination
        )

        self.grid_dispatch_dim     = [self.resolution.value.x,   self.resolution.value.y,   1]
        self.mac_grid_dispatch_dim = [self.resolution.value.x+1, self.resolution.value.y+1, 1]

        for name in ["velocity_x", "velocity_y", "pressure"]:
            command_encoder.clear_texture_float(self.grid_vars[name])
            command_encoder.clear_texture_float(self.grid_vars[f"{name}_rw"])
    
        if self.init_mode.value == INIT_VORTEX:
            self.dispatch_pass(
                "fluid-init.cs.slang",
                "init_vortex",
                self.mac_grid_dispatch_dim,
                { "grid": self.grid_vars },
                command_encoder)
            self.pressure_project(command_encoder)
            self.swap_grids()
        elif self.init_mode.value == INIT_LEAPFROG:
            self.dispatch_pass(
                "fluid-init.cs.slang",
                "init_leapfrog_vortex",
                self.mac_grid_dispatch_dim,
                { "grid": self.grid_vars },
                command_encoder)
            self.pressure_project(command_encoder)
            self.swap_grids()

        self.initialized = True

    def step(self, command_encoder:spy.CommandEncoder, dt):
        if not self.initialized:
            self.initialize(command_encoder)

        if not self.enabled and not self.step_once:
            return
        
        self.step_once = False

        if self.emit_smoke.value:
            self.dispatch_pass(
                "fluid-init.cs.slang",
                "emit_smoke",
                self.mac_grid_dispatch_dim,
                { "grid": self.grid_vars },
                command_encoder)
            self.pressure_project(command_encoder)
            self.swap_grids()

        for _ in range(self.advection_iterations.value):
            vars = { "grid": self.grid_vars,
                    "particle_map": self.particle_map.vars,
                    "advect_dt": self.advect_dt.value / self.advection_iterations.value }
            if self.advect_mode.value == ADVECT_GRID:
                self.dispatch_pass(
                    "fluid-advection.cs.slang",
                    "advect_grid",
                    self.mac_grid_dispatch_dim,
                    vars,
                    command_encoder)
            elif self.advect_mode.value == ADVECT_PARTICLE:
                self.particle_map.clear(command_encoder)
                self.dispatch_pass(
                    "fluid-advection.cs.slang",
                    "advect_particle",
                    self.grid_dispatch_dim,
                    vars,
                    command_encoder)
                self.particle_map.sort(command_encoder)
                self.dispatch_pass(
                    "fluid-advection.cs.slang",
                    "particle_to_grid",
                    self.mac_grid_dispatch_dim,
                    vars,
                    command_encoder)
            elif self.advect_mode.value == ADVECT_COFLIP_GRID:
                self.dispatch_pass(
                    "coflip-sim.cs.slang",
                    "advect_coflip_grid",
                    self.mac_grid_dispatch_dim,
                    vars,
                    command_encoder)
            elif self.advect_mode.value == ADVECT_COFLIP_PARTICLE:
                self.particle_map.clear(command_encoder)
                self.dispatch_pass(
                    "coflip-sim.cs.slang",
                    "advect_coflip_particle",
                    self.grid_dispatch_dim,
                    vars,
                    command_encoder)
                self.particle_map.sort(command_encoder)
                self.dispatch_pass(
                    "coflip-sim.cs.slang",
                    "coflip_particle_to_grid",
                    self.mac_grid_dispatch_dim,
                    vars,
                    command_encoder)
            self.pressure_project(command_encoder)
            self.swap_grids()

        command_encoder.generate_mips(self.grid_vars["pressure"])
        command_encoder.copy_texture_to_buffer(self.avg_pressure_buf, 0, 4, 256, self.grid_vars["pressure"], 0, self.grid_vars["pressure"].mip_count-1, [0,0,0], [1,1,1])
        self.avg_pressure.text = f"Avg pressure: {self.avg_pressure_buf.to_numpy()[0]}"

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

        self.setup_ui()
        
        self.render_offset = spy.float2(0,0)
        self.render_scale  = 1

    def setup_ui(self):
        screen = self.ui.screen
        window = spy.ui.Window(screen, "Settings", size=spy.float2(500, 300))

        self.fps_text = spy.ui.Text(window, "FPS: 0")
        
        self.simulator.setup_ui(spy.ui.Group(window, label="Simulation"))

        self.render_mode = spy.ui.ComboBox(window, "Render", 0, items=[ "Pressure", "Velocity", "Divergence", "Pressure (bspline)", "Velocity (bspline)" ])

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
            fluid_dim = self.simulator.grid_vars["resolution"]
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
                    "grid":          self.simulator.grid_vars,
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
