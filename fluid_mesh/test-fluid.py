import slangpy as spy
import pyobjloader
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from MeshFluidSimulator import MeshFluidSimulator
from world.Camera import Camera, InputState

COLOR_FORMAT = spy.Format.rgba32_float
DEPTH_FORMAT = spy.Format.d32_float

def get_asset_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(__file__), path)

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
        self.input_state = InputState()
        self.render_texture = None
        self.fps_avg = 0

        model = pyobjloader.load_model(os.path.join(os.path.dirname(__file__), os.pardir, "icosphere.obj"))
        vertices = np.array(model.vertex_points, dtype=np.float32)
        indices  = np.array(model.point_indices, dtype=np.uint32)
        self.simulator = MeshFluidSimulator(self.device, vertices, indices)
        self.camera = Camera()

        self.render_pipeline = self.device.create_render_pipeline(
            program=self.device.load_program(get_asset_path("render-fluid.3d.slang"), entry_point_names=["vs", "fs"]),
            input_layout=None,
            targets=[spy.ColorTargetDesc({
                "format": COLOR_FORMAT, 
                "color": spy.AspectBlendDesc({"src_factor": spy.BlendFactor.one, "dst_factor": spy.BlendFactor.zero, "op": spy.BlendOp.add}),
                "alpha": spy.AspectBlendDesc({"src_factor": spy.BlendFactor.one, "dst_factor": spy.BlendFactor.zero, "op": spy.BlendOp.add}),
                "write_mask": spy.RenderTargetWriteMask.all, 
                "enable_blend": False
            })],
            depth_stencil=spy.DepthStencilDesc({
                "format": DEPTH_FORMAT, 
                "depth_test_enable": True, 
                "depth_write_enable": True, 
                "depth_func": spy.ComparisonFunc.less, 
                "stencil_enable": False
            }),
            rasterizer=spy.RasterizerDesc({
                "fill_mode": spy.FillMode.solid,
                "cull_mode": spy.CullMode.back,
                "front_face": spy.FrontFaceMode.counter_clockwise,
            })
        )
        
        self.setup_ui()
        
        self.render_offset = spy.float2(0,0)
        self.render_scale  = 1

    def setup_ui(self):
        screen = self.ui.screen
        window = spy.ui.Window(screen, "Settings", size=spy.float2(500, 300))

        self.fps_text = spy.ui.Text(window, "FPS: 0")
        
        self.simulator.setup_ui(spy.ui.Group(window, label="Simulation"))

        self.render_mode = spy.ui.ComboBox(window, "Render", 0, items=[ "Smoke", "Velocity", "Streamfunction", "Vorticity" ])

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
        self.input_state.on_keyboard_event(event, has_focus)

    def on_mouse_event(self, event: spy.MouseEvent):
        has_focus = not self.ui.handle_mouse_event(event)
        self.input_state.on_mouse_event(event, has_focus)

    def main_loop(self):
        self.frame_timer = spy.Timer()
        while not self.window.should_close():
            self.input_state.update()
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
            self.fps_text.text = f"FPS: {self.fps_avg:.2f}"

            self.camera.update(self.input_state, dt)            
            
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
                self.depth_texture = self.device.create_texture(
                    format=DEPTH_FORMAT,
                    width=surface_texture.width,
                    height=surface_texture.height,
                    usage=spy.TextureUsage.depth_stencil,
                    label="depth_texture",
                )
                self.render_texture_view = self.render_texture.create_view({
                    "format": COLOR_FORMAT,
                    "label": "render_texture_view"
                })
                self.depth_texture_view = self.depth_texture.create_view({
                    "format": DEPTH_FORMAT,
                    "label": "depth_texture_view"
                })

            with command_encoder.begin_render_pass(spy.RenderPassDesc({
                "color_attachments": [ spy.RenderPassColorAttachment({
                    "view": self.render_texture_view,
                    "load_op": spy.LoadOp.clear,
                    "store_op": spy.StoreOp.store,
                    "clear_value": spy.float4(0,0,0,0)
                }) ],
                "depth_stencil_attachment": spy.RenderPassDepthStencilAttachment({
                    "view": self.depth_texture_view,
                    "depth_load_op": spy.LoadOp.clear,
                    "depth_store_op": spy.StoreOp.store,
                    "depth_clear_value": 1
                })
            })) as pass_encoder:
                shader = pass_encoder.bind_pipeline(self.render_pipeline)
                pass_encoder.set_render_state({
                    "viewports": [spy.Viewport.from_size(surface_texture.width, surface_texture.height)],
                    "scissor_rects": [ spy.ScissorRect.from_size(surface_texture.width, surface_texture.height) ],
                })

                camera_to_world = self.camera.camera_to_world()
                view = spy.math.inverse(camera_to_world)
                projection = self.camera.projection(surface_texture.width / surface_texture.height)

                cursor = spy.ShaderCursor(shader)
                cursor["view_projection"] = spy.math.mul(projection, view)
                cursor["render_mode"] = self.render_mode.value
                cursor["mesh"] = self.simulator.mesh_vars

                pass_encoder.draw(spy.DrawArguments({"vertex_count": self.simulator.num_triangles * 3}))
            
            command_encoder.blit(surface_texture, self.render_texture)

            self.ui.end_frame(surface_texture, command_encoder)

            self.device.submit_command_buffer(command_encoder.finish())
            del surface_texture

            self.surface.present()
            
        self.device.wait()

app = App()
app.main_loop()