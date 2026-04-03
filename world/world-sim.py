import os
import slangpy as spy
import numpy as np
from Camera import Camera, InputState

COLOR_FORMAT = spy.Format.rgba32_float
DEPTH_FORMAT = spy.Format.d32_float

def get_asset_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(__file__), path)

class WorldRenderer:
    def __init__(self, device : spy.Device):
        self.device = device
        self.texture_loader = spy.TextureLoader(self.device)
            
        def load_single_channel(path, channel, usage):
            bitmap = spy.Bitmap(path)
            bitmap1 = spy.Bitmap(pixel_format=spy.Bitmap.PixelFormat.r, component_type=spy.Bitmap.ComponentType.uint16, width=bitmap.width, height=bitmap.height, channel_count=1)
            pixels  = np.array(bitmap, copy=False)
            pixels1 = np.array(bitmap1, copy=False)
            pixels1[:,:] = 65535.0 * (pixels[:,:,channel] / 255.0)
            return self.texture_loader.load_texture(bitmap=bitmap1, options=spy.TextureLoader.Options({ "load_as_normalized": True, "usage": usage, "allocate_mips": False, "load_as_srgb": False }))

        self.surface_albedo_texture = self.texture_loader.load_texture(get_asset_path("earth_albedo.png"), options=spy.TextureLoader.Options({ "allocate_mips": False, "load_as_srgb": True }))
        self.surface_height_texture = load_single_channel(get_asset_path("earth_height.png"), 0, spy.TextureUsage.shader_resource)
        self.cloud_texture  = load_single_channel(get_asset_path("earth_clouds.png"), 3, spy.TextureUsage.shader_resource)

        self.render_pipeline = self.device.create_render_pipeline(
            program=self.device.load_program(get_asset_path("render-sphere.3d.slang"), entry_point_names=["vs", "fs"]),
            input_layout=None,
            targets=[spy.ColorTargetDesc({
                "format": COLOR_FORMAT, 
                "color": spy.AspectBlendDesc({"src_factor": spy.BlendFactor.one, "dst_factor": spy.BlendFactor.src_alpha, "op": spy.BlendOp.add}), # transmittance blending
                "alpha": spy.AspectBlendDesc({"src_factor": spy.BlendFactor.one, "dst_factor": spy.BlendFactor.zero,      "op": spy.BlendOp.add}),
                "write_mask": spy.RenderTargetWriteMask.all, 
                "enable_blend": True
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
                "front_face": spy.FrontFaceMode.clockwise,
            })
        )
        
        self.texture_sampler = self.device.create_sampler(spy.SamplerDesc())
        self.sphere_resolution = 64
        self.frame_seed = 0

    def setup_ui(self, window):        
        self.planet_radius              = spy.ui.DragFloat   (window, "Planet radius (km)",           value=6371,                              min=0)
        self.terrain_height             = spy.ui.DragFloat   (window, "Terrain height (km)",          value=6.4,                               min=0)
        self.cloud_height               = spy.ui.DragFloat2  (window, "Cloud height range (km)",      value=spy.float2(2,10),                  min=0)
        self.surface_rotation           = spy.ui.SliderFloat (window, "Surface rotation",             value=0.0,                               min=0, max=1)
        self.cloud_rotation             = spy.ui.SliderFloat (window, "Cloud rotation",               value=0.0,                               min=0, max=1)
        self.cloud_density              = spy.ui.DragFloat   (window, "Cloud density",                value=50.0,                              min=0)
        
        self.atmosphere_height          = spy.ui.DragFloat   (window, "Atmosphere height (km)",       value=100,                               min=0)
        self.atmosphere_rayleigh_height = spy.ui.DragFloat   (window, "Rayleigh scatter height (km)", value=4,                                 min=0, speed=0.01)
        self.atmosphere_mie_height      = spy.ui.DragFloat   (window, "Mie scatter height (km)",      value=0.6,                               min=0, speed=0.01)
        self.atmosphere_rayleigh_color  = spy.ui.DragFloat3  (window, "Rayleigh scatter factor",      value=spy.float3(6.605, 12.344, 29.412), min=0, speed=0.01)
        self.atmosphere_mie_color       = spy.ui.DragFloat   (window, "Mie scatter factor",           value=3.996,                             min=0, speed=0.01)
        self.atmosphere_density         = spy.ui.DragFloat   (window, "Atmosphere density",           value=1,                                 min=0, speed=0.01)

        self.sun_color                  = spy.ui.SliderFloat3(window, "Sun color",                    value=spy.float3(1,1,1),                 min=0, max=1)
        self.sun_strength               = spy.ui.DragFloat   (window, "Sun strength",                 value=10,                                min=0, speed=0.1)
        self.sun_direction              = spy.ui.SliderFloat3(window, "Sun direction",                value=spy.float3(0,0,1),                 min=-1, max=1)

    def render(self,
               command_encoder : spy.CommandEncoder,
               camera : Camera,
               render_width,
               render_height,
               render_texture_view : spy.TextureView,
               depth_texture_view : spy.TextureView
               ):
        with command_encoder.begin_render_pass(spy.RenderPassDesc({
            "color_attachments": [ spy.RenderPassColorAttachment({
                "view": render_texture_view,
                "load_op": spy.LoadOp.clear,
                "store_op": spy.StoreOp.store,
                "clear_value": spy.float4(0,0,0,0)
            }) ],
            "depth_stencil_attachment": spy.RenderPassDepthStencilAttachment({
                "view": depth_texture_view,
                "depth_load_op": spy.LoadOp.clear,
                "depth_store_op": spy.StoreOp.store,
                "depth_clear_value": 1
            })
        })) as pass_encoder:
            shader = pass_encoder.bind_pipeline(self.render_pipeline)
            pass_encoder.set_render_state({
                "viewports": [spy.Viewport.from_size(render_width, render_height)],
                "scissor_rects": [ spy.ScissorRect.from_size(render_width, render_height) ],
            })

            camera_to_world = camera.camera_to_world()
            view = spy.math.inverse(camera_to_world)
            projection = camera.projection(render_width / render_height)

            cursor = spy.ShaderCursor(shader)
            cursor["planet_albedo"]              = self.surface_albedo_texture
            cursor["planet_height"]              = self.surface_height_texture
            cursor["planet_clouds"]              = self.cloud_texture
            cursor["sampler"]                    = self.texture_sampler
            cursor["view_projection"]            = spy.math.mul(projection, view)
            cursor["camera_position"]            = spy.math.transform_point(camera_to_world, spy.float3(0,0,0))
            cursor["sphere_resolution"]          = self.sphere_resolution
            cursor["sun_emission"]               = self.sun_color.value * self.sun_strength.value
            cursor["cloud_density"]              = self.cloud_density.value
            cursor["sun_direction"]              = spy.math.normalize(self.sun_direction.value)
            cursor["cloud_height_range"]         = self.cloud_height.value / self.planet_radius.value
            cursor["terrain_height"]             = self.terrain_height.value / self.planet_radius.value
            cursor["surface_rotation"]           = self.surface_rotation.value
            cursor["cloud_rotation"]             = self.cloud_rotation.value
            cursor["atmosphere_height"]          = self.atmosphere_height.value / self.planet_radius.value
            cursor["atmosphere_rayleigh_height"] = 1 / max(1e-9, self.atmosphere_rayleigh_height.value / self.planet_radius.value)
            cursor["atmosphere_mie_height"]      = 1 / max(1e-9, self.atmosphere_mie_height.value / self.planet_radius.value)
            cursor["atmosphere_rayleigh_color"]  = self.atmosphere_rayleigh_color.value
            cursor["atmosphere_mie_color"]       = self.atmosphere_mie_color.value
            cursor["atmosphere_density"]         = self.atmosphere_density.value
            cursor["frame_seed"]                 = self.frame_seed

            pass_encoder.draw(spy.DrawArguments({"vertex_count": self.sphere_resolution*self.sphere_resolution*6}))

            self.frame_seed += 1

class App:
    def __init__(self):
        super().__init__()
        self.device = spy.create_device(include_paths=[os.path.abspath("."), os.path.abspath("src")])
        self.window = spy.Window(width=1400, height=(1400*9)//16, title="App", resizable=True)
        self.surface = self.device.create_surface(self.window)
        self.surface.configure({
            "width":  self.window.width,
            "height": self.window.height,
            "vsync": True
        })

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

        self.renderer   = WorldRenderer(self.device)
        self.camera = Camera()

        self.tonemapper = self.device.create_compute_kernel(self.device.load_program(get_asset_path("tonemap.cs.slang"), ["tonemap"]))

        self.setup_ui()

    def setup_ui(self):
        screen = self.ui.screen
        window = spy.ui.Window(screen, "Settings", size=spy.float2(500, 300))

        self.fps_text = spy.ui.Text(window, "FPS: 0")
        
        def pause_callback():
            self.pause = not self.pause
            self.pause_button.label = "Resume rendering" if self.pause else "Pause rendering"
        self.pause_button = spy.ui.Button(window, "Pause rendering", callback=pause_callback)

        self.camera_pos_text = spy.ui.Text(window, "Camera: 0")
        self.exposure = spy.ui.SliderFloat(window, "Exposure", value=0.0, min=-12, max=12)
        
        self.renderer.setup_ui(spy.ui.Group(window, label="Renderer"))

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
        timer = spy.Timer()
        while not self.window.should_close():
            self.input_state.update()
            self.window.process_events()

            if self.minimized:
                continue

            surface_texture = self.surface.acquire_next_image()
            if not surface_texture:
                continue

            dt = timer.elapsed_s()
            timer.reset()
            self.fps_avg = 0.95 * self.fps_avg + 0.05 * (1.0 / dt)

            command_encoder = self.device.create_command_encoder()

            if not self.pause:
                self.camera.update(self.input_state, dt)

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

                self.renderer.render(command_encoder, self.camera, surface_texture.width, surface_texture.height, self.render_texture_view, self.depth_texture_view)

                self.tonemapper.dispatch(
                    thread_count=[self.render_texture.width, self.render_texture.height, 1],
                    vars={
                        "exposure": self.exposure.value,
                        "image": self.render_texture
                    },
                    command_encoder=command_encoder
                )
                
            if self.render_texture is not None:
                command_encoder.blit(surface_texture, self.render_texture)

            self.camera_pos_text.text = f"Camera: {self.camera.position.x:.3f} {self.camera.position.y:.3f} {self.camera.position.z:.3f}"
            self.fps_text.text = f"FPS: {self.fps_avg:.2f}"
            self.ui.begin_frame(surface_texture.width, surface_texture.height)
            self.ui.end_frame(surface_texture, command_encoder)            

            self.device.submit_command_buffer(command_encoder.finish())
            del surface_texture

            self.surface.present()

        self.device.wait()

app = App()
app.main_loop()
