import slangpy as spy
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from FluidSimulator import FluidSimulator

COLOR_FORMAT = spy.Format.rgba32_float

def is_down(input_state, key):
    return key in input_state and input_state[key]

class App:
    def __init__(self):
        super().__init__()
        self.device = spy.create_device(
            include_paths=[
                os.path.abspath(os.path.dirname(__file__)),
                os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
                os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)),
            ],
        )
        self.window = spy.Window(width=1400, height=(1400*9)//16, title="App", resizable=True)
        self.surface = self.device.create_surface(self.window)
        self.surface.configure(
            width  = self.window.width,
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

        self.render_shader = self.device.create_compute_kernel(self.device.load_program("render-fluid.cs.slang", ["render"]))

        self.setup_ui()
        
        self.render_offset = spy.float2(0,0)
        self.render_scale  = 1

    def setup_ui(self):
        screen = self.ui.screen
        window = spy.ui.Window(screen, "Settings", size=spy.float2(500, 300))

        self.fps_text = spy.ui.Text(window, "FPS: 0")
        
        self.simulator.setup_ui(spy.ui.Group(window, label="Simulation"))

        self.render_mode = spy.ui.ComboBox(window, "Render", 0, items=[ "Pressure", "Velocity", "Divergence", "Curl" ])

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

            screen_size = spy.uint2(surface_texture.width, surface_texture.height)
            fluid_dim = self.simulator.get_resolution()
            rect_pos = spy.int2(
                int(screen_size.x/2 - self.render_scale*fluid_dim.x/2 - self.render_offset.x),
                int(screen_size.y/2 - self.render_scale*fluid_dim.y/2 - self.render_offset.y)
            )
            rect_size = spy.int2(
                int(fluid_dim.x * self.render_scale),
                int(fluid_dim.y * self.render_scale),
            )
            
            if is_down(self.input_state, spy.MouseButton.left) and "mouse" in self.input_state:
                if self.drag_start is not None:
                    if is_down(self.input_state, spy.KeyCode.left_shift):
                        def screen_to_grid(p):
                            uv = (p - rect_pos) / spy.float2(rect_size.x, rect_size.y)
                            return uv * spy.float2(fluid_dim.x, fluid_dim.y)
                        self.simulator.emit_drag(screen_to_grid(self.drag_start), screen_to_grid(self.input_state["mouse"]))
                    else:
                        self.render_offset -= self.input_state["mouse"] - self.drag_start
                self.drag_start = self.input_state["mouse"]
            else:
                self.drag_start = None

            scroll = self.input_state["scroll"]
            if scroll != 0:
                zoom_factor = 1 + .05 * scroll
                if "mouse" in self.input_state:
                    mouse = self.input_state["mouse"]
                    cx, cy = screen_size.x / 2, screen_size.y / 2
                    self.render_offset = spy.float2(
                        self.render_offset.x + (zoom_factor - 1) * (mouse.x - cx + self.render_offset.x),
                        self.render_offset.y + (zoom_factor - 1) * (mouse.y - cy + self.render_offset.y)
                    )
                self.render_scale *= zoom_factor

            self.fps_text.text = f"FPS: {self.fps_avg:.2f}"


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

            self.ui.end_frame(surface_texture, command_encoder)

            self.device.submit_command_buffer(command_encoder.finish())
            del surface_texture

            self.surface.present()
            
        self.device.wait()

app = App()
app.main_loop()