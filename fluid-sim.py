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
        self.density = None
        self.init_pass   = self.device.create_compute_kernel(self.device.load_program("fluid-init.cs.slang", ["main"]))
        self.advect_pass = self.device.create_compute_kernel(self.device.load_program("fluid-sim.cs.slang", ["advect"]))
        self.pressure_solve_pass = self.device.create_compute_kernel(self.device.load_program("fluid-sim.cs.slang", ["pressure_solve"]))
        self.pressure_project_pass = self.device.create_compute_kernel(self.device.load_program("fluid-sim.cs.slang", ["pressure_project"]))
        self.linear_sampler = self.device.create_sampler(
            address_u=spy.TextureAddressingMode.clamp_to_border,
            address_v=spy.TextureAddressingMode.clamp_to_border,
            border_color=spy.float4(0,0,0,0)
        )

    def setup_ui(self, window):
        def toggle_callback():
            self.enabled = not self.enabled
            self.toggle_button.label = "Pause simulation" if self.enabled else "Resume simulation"
        self.toggle_button = spy.ui.Button(window, "Resume simulation", callback=toggle_callback)

        def step_callback():
            self.step_once = True
        spy.ui.Button(window, "Step", callback=step_callback)

        def reset_callback(arg=None):
            self.density = None
        spy.ui.Button(window, "Reset", callback=reset_callback)

        self.resolution = spy.ui.DragInt2(window, "Resolution", value=spy.int2(512,512), min=1, callback=reset_callback)
        self.K         = spy.ui.DragFloat(window, "K", value=0.2, min=0)
        self.dt        = spy.ui.DragFloat(window, "dt", value=0.1, min=0)
    
    def initialize(self, command_encoder:spy.CommandEncoder):
        self.density, self.density_out = [ self.device.create_texture(
            format=spy.Format.r32_float,
            width=self.resolution.value.x,
            height=self.resolution.value.y,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label=f"density_{i}",
        ) for i in range(2) ]
        
        self.velocity_x, self.velocity_x_out = [ self.device.create_texture(
            format=spy.Format.rg32_float,
            width=self.resolution.value.x+1,
            height=self.resolution.value.y,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label=f"velocity_x_{i}",
        ) for i in range(2)]
        self.velocity_y, self.velocity_y_out = [ self.device.create_texture(
            format=spy.Format.rg32_float,
            width=self.resolution.value.x,
            height=self.resolution.value.y+1,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label=f"velocity_y_{i}",
        ) for i in range(2)]


        self.init_pass.dispatch(
            thread_count=[self.density.width, self.density.height, 1],
            vars={
                "density":  self.density,
                "velocity_x": self.velocity_x,
                "velocity_y": self.velocity_y,
                "dim": spy.uint2(self.density.width, self.density.height),
            },
            command_encoder=command_encoder
        )

    def step(self, command_encoder, dt):
        if self.density is None:
            self.initialize(command_encoder)

        if not self.enabled and not self.step_once:
            return
        
        self.step_once = False

        for checker_iteration in range(2):
            self.advect_pass.dispatch(
                thread_count=[self.density.width//2, self.density.height, 1],
                vars={
                    "density":        self.density,
                    "velocity_x":     self.velocity_x,
                    "velocity_y":     self.velocity_y,
                    "density_out":    self.density_out,
                    "velocity_x_out": self.velocity_x_out,
                    "velocity_y_out": self.velocity_y_out,
                    "linear_sampler": self.linear_sampler,
                    "dim": spy.uint2(self.density.width, self.density.height),
                    "checker_iteration": checker_iteration,
                    "K": self.K.value,
                    "dt": self.dt.value,
                },
                command_encoder=command_encoder
            )

        self.pressure_solve_pass.dispatch(
            thread_count=[self.density.width+1, self.density.height+1, 1],
            vars={
                "density":        self.density,
                "velocity_x":     self.velocity_x,
                "velocity_y":     self.velocity_y,
                "density_out":    self.density_out,
                "velocity_x_out": self.velocity_x_out,
                "velocity_y_out": self.velocity_y_out,
                "linear_sampler": self.linear_sampler,
                "dim": spy.uint2(self.density.width, self.density.height),
                "K": self.K.value,
                "dt": self.dt.value,
            },
            command_encoder=command_encoder
        )
        
        self.velocity_x, self.velocity_x_out = self.velocity_x_out, self.velocity_x
        self.velocity_y, self.velocity_y_out = self.velocity_y_out, self.velocity_y

        for checker_iteration in range(2):
            self.pressure_project_pass.dispatch(
                thread_count=[self.density.width//2, self.density.height, 1],
                vars={
                    "density":        self.density,
                    "velocity_x":     self.velocity_x,
                    "velocity_y":     self.velocity_y,
                    "density_out":    self.density_out,
                    "velocity_x_out": self.velocity_x_out,
                    "velocity_y_out": self.velocity_y_out,
                    "linear_sampler": self.linear_sampler,
                    "dim": spy.uint2(self.density.width, self.density.height),
                    "checker_iteration": checker_iteration,
                    "K": self.K.value,
                    "dt": self.dt.value,
                },
                command_encoder=command_encoder
            )

        self.density,  self.density_out  = self.density_out,  self.density

    def get_density(self):
        return self.density

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

    def setup_ui(self):
        screen = self.ui.screen
        window = spy.ui.Window(screen, "Settings", size=spy.float2(500, 300))

        self.fps_text = spy.ui.Text(window, "FPS: 0")
        
        self.simulator.setup_ui(spy.ui.Group(window, label="Simulation"))

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

            fluid_dim = spy.uint2(self.simulator.get_density().width, self.simulator.get_density().height)
            p0 = spy.uint2(
                3*(surface_texture.width//4)  - fluid_dim.x//2,
                surface_texture.height//2 - fluid_dim.y//2
            )

            self.render_shader.dispatch(
                thread_count=[self.render_texture.width, self.render_texture.height, 1],
                vars={
                    "render_target": self.render_texture,
                    "image": self.simulator.get_density(),
                    "sampler": self.linear_sampler,
                    "screen_size": [self.render_texture.width, self.render_texture.height],
                    "screen_rect": spy.uint4(p0, p0 + fluid_dim),
                },
                command_encoder=command_encoder
            )
            
            command_encoder.blit(surface_texture, self.render_texture)

            self.fps_text.text = f"FPS: {self.fps_avg:.2f}"

            self.ui.end_frame(surface_texture, command_encoder)

            self.device.submit_command_buffer(command_encoder.finish())
            del surface_texture

            self.surface.present()
            
        self.device.wait()

app = App()
app.main_loop()
