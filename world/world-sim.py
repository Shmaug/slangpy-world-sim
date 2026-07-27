import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
import slangpy as spy
import numpy as np
from Camera import Camera, InputState
from fluid_cubemap.CubeFluidSimulator import CubeFluidSimulator

COLOR_FORMAT = spy.Format.rgba32_float
DEPTH_FORMAT = spy.Format.d32_float

def get_asset_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(__file__), path)

def intersect_sphere(origin:spy.float3, direction:spy.float3, radius:float) -> spy.float2|None:
    # precise version
    r2 = radius * radius
    DD = spy.math.dot(direction, direction)
    Df = spy.math.dot(origin, direction)
    DD2 = DD*DD
    l = DD * origin - direction*Df
    a2_r2 = DD2*r2
    ll = spy.math.dot(l,l)
    if a2_r2 < ll:
        return None
    det = a2_r2 - ll
    rcp_a = 1.0 / DD
    det = spy.math.sqrt(det * rcp_a)    
    return (-Df + spy.float2(-det, det)) * rcp_a

class App:
    def __init__(self):
        super().__init__()
        self.device = spy.create_device(include_paths=[os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), os.path.pardir)])
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

        # Render state

        self.pause = False
        self.minimized = False
        self.input_state = InputState()
        self.render_texture = None
        self.fps_avg = 0

        self.cloud_sim = CubeFluidSimulator(self.device)
        self.camera = Camera()
        self.tonemapper = self.device.create_compute_kernel(self.device.load_program(get_asset_path("tonemap.cs.slang"), ["tonemap"]))        

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

        # Textures
        
        self.texture_loader = spy.TextureLoader(self.device)
            
        self.surface_albedo_texture = self.texture_loader.load_texture(get_asset_path("earth_albedo.png"), options=spy.TextureLoader.Options({ "allocate_mips": False, "load_as_srgb": True }))

        heightmap_bitmap_rgba = spy.Bitmap(get_asset_path("earth_height.png"))
        heightmap_rgba = np.array(heightmap_bitmap_rgba, copy=False)

        heightmap_r = spy.Bitmap(pixel_format=spy.Bitmap.PixelFormat.r, component_type=spy.Bitmap.ComponentType.uint16, width=heightmap_bitmap_rgba.width, height=heightmap_bitmap_rgba.height, channel_count=1)
        dst_pixels = np.array(heightmap_r, copy=False)
        dst_pixels[:,:] = heightmap_rgba[:,:,0]
        self.surface_height_texture = self.texture_loader.load_texture(heightmap_r, options=spy.TextureLoader.Options({ "load_as_normalized": True, "usage": spy.TextureUsage.shader_resource|spy.TextureUsage.unordered_access, "allocate_mips": True, "load_as_srgb": False }))

        heightmap_r2 = spy.Bitmap(pixel_format=spy.Bitmap.PixelFormat.r, component_type=spy.Bitmap.ComponentType.uint16, width=heightmap_bitmap_rgba.width, height=heightmap_bitmap_rgba.height, channel_count=1)
        dst_pixels2 = np.array(heightmap_r2, copy=False)
        dst_pixels2[:,:] = (255.0 * ((heightmap_rgba[:,:,0]/255.0)**2)).astype(np.uint16)
        self.surface_height_texture_sqr = self.texture_loader.load_texture(heightmap_r2, options=spy.TextureLoader.Options({ "load_as_normalized": True, "usage": spy.TextureUsage.shader_resource|spy.TextureUsage.unordered_access, "allocate_mips": True, "load_as_srgb": False }))

        command_encoder = self.device.create_command_encoder()
        command_encoder.generate_mips(self.surface_height_texture)
        command_encoder.generate_mips(self.surface_height_texture_sqr)
        self.device.wait_for_submit(self.device.submit_command_buffer(command_encoder.finish()))

        self.cloud_sim.heightmap     = self.surface_height_texture
        self.cloud_sim.heightmap_sqr = self.surface_height_texture_sqr

        # UI

        self.ui = spy.ui.Context(self.device)

        widget = spy.ui.Window(self.ui.screen, "Settings", size=spy.float2(500, 300))
        self.fps_text = spy.ui.Text(widget, "FPS: 0")
        def pause_callback():
            self.pause = not self.pause
            self.pause_button.label = "Resume rendering" if self.pause else "Pause rendering"
        self.pause_button = spy.ui.Button(widget, "Pause rendering", callback=pause_callback)

        self.camera_pos_text = spy.ui.Text(widget, "Camera: 0")
        self.exposure = spy.ui.SliderFloat(widget, "Exposure", value=0.0, min=-12, max=12)
        
        self.texture_sampler = self.device.create_sampler(address_v=spy.TextureAddressingMode.mirror_repeat)
        self.sphere_resolution = 64
        self.frame_seed = 0

        def reset_cb():
            self.cloud_sim.reset = True
        def update_cb(value):
            self.cloud_sim.resolution = 1 << self.cloud_resolution.value
            self.cloud_sim.vertical_resolution = self.cloud_vertical_resolution.value
            self.cloud_sim.radius = 1.0
            self.cloud_sim.terrain_height = self.terrain_height.value / self.planet_radius.value
            self.cloud_sim.thickness = self.cloud_max_height.value / self.planet_radius.value
            self.cloud_sim.density_scale_height = 1 / max(1e-9, self.atmosphere_rayleigh_height.value / self.planet_radius.value)
            self.cloud_sim.solver_iterations = self.solver_iterations.value
            self.cloud_sim.solver_fine_iterations = self.solver_fine_iterations.value
            self.cloud_sim.multires_solve = self.solver_multires.value
            self.cloud_sim.preserve_smoke = True
            self.cloud_sim._create_buffers()

        self.planet_radius              = spy.ui.DragFloat   (widget, "Planet radius (km)",           value=6371,                              min=0, callback=update_cb)
        self.terrain_height             = spy.ui.DragFloat   (widget, "Terrain height (km)",          value=8.8,                               min=0, callback=update_cb)
        self.atmosphere_height          = spy.ui.DragFloat   (widget, "Atmosphere height (km)",       value=100,                               min=0, callback=update_cb)
        self.cloud_max_height           = spy.ui.DragFloat   (widget, "Cloud max height (km)",        value=10,                                min=0, callback=update_cb)
        self.atmosphere_rayleigh_height = spy.ui.DragFloat   (widget, "Rayleigh scatter height (km)", value=4,                                 min=0, speed=0.01, callback=update_cb)
        self.atmosphere_mie_height      = spy.ui.DragFloat   (widget, "Mie scatter height (km)",      value=0.6,                               min=0, speed=0.01)
        self.atmosphere_rayleigh_color  = spy.ui.DragFloat3  (widget, "Rayleigh scatter factor",      value=spy.float3(6.605, 12.344, 29.412), min=0, speed=0.01)
        self.atmosphere_mie_color       = spy.ui.DragFloat   (widget, "Mie scatter factor",           value=3.996,                             min=0, speed=0.01)
        self.atmosphere_density         = spy.ui.DragFloat   (widget, "Atmosphere density",           value=1,                                 min=0, speed=0.01)
        self.cloud_density              = spy.ui.DragFloat   (widget, "Cloud density",                value=50.0,                              min=0)

        self.sun_color                  = spy.ui.SliderFloat3(widget, "Sun color",                    value=spy.float3(1,1,1),                 min=0, max=1)
        self.sun_strength               = spy.ui.DragFloat   (widget, "Sun strength",                 value=10,                                min=0, speed=0.1)
        self.sun_direction              = spy.ui.SliderFloat3(widget, "Sun direction",                value=spy.float3(0,0,1),                 min=-1, max=1)

        group = spy.ui.Group(widget, "Cloud simulation")
        spy.ui.Button(group, "Reset sim", reset_cb)
        self.cloud_timestep = spy.ui.DragFloat(group, "Timestep", 1.0/60.0, speed=1/120.0)
        self.cloud_resolution = spy.ui.DragInt(group, "Resolution (log2)", 9, min=0, max=12, callback=update_cb)
        self.cloud_vertical_resolution = spy.ui.DragInt(group, "Vertical resolution", 4, min=1, max=16, callback=update_cb)
        self.solver_iterations = spy.ui.DragInt(group, "Solver iterations", 10, callback=update_cb)
        self.solver_multires = spy.ui.CheckBox(group, "Multiresolution Solver", value=True, callback=update_cb)
        self.solver_fine_iterations = spy.ui.DragInt(group, "Multiresolution sub-iterations", 4, callback=update_cb)
        update_cb(None)

        self.emit_drag_kernel = self.device.create_compute_kernel(self.device.load_program("spawn-clouds.cs.slang", ["emit_drag"]))
        self.emit_radius = spy.ui.DragFloat(widget, "Emit radius", value=2, speed=0.01)
        self.emit_speed = spy.ui.DragFloat(widget, "Emit speed", value=1, speed=0.01)
        self.emit_vertical_speed = spy.ui.DragFloat(widget, "Emit vertical speed", value=0.1, speed=0.01)
        self.drag:list[spy.float3] = []
        def emit_drag(smoke,velocity,command_encoder):
            if len(self.drag) > 1:
                self.emit_drag_kernel.dispatch(
                    self.cloud_sim.dispatch_dim(),
                    {
                        "smoke": smoke,
                        "velocity": velocity,
                        "start_dir": self.drag[-2],
                        "end_dir": self.drag[-1],
                        "radius": spy.math.radians(self.emit_radius.value),
                        "speed": self.emit_speed.value,
                        "vertical_speed": self.emit_vertical_speed.value,
                    },
                    command_encoder)
        self.cloud_sim.emitters.append(emit_drag)
        
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

    def render(self, camera_position:spy.float3, view_projection:spy.float4x4, command_encoder : spy.CommandEncoder):
        if self.render_texture is None:
            return
        
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
                "viewports": [spy.Viewport.from_size(self.render_texture.width, self.render_texture.height)],
                "scissor_rects": [ spy.ScissorRect.from_size(self.render_texture.width, self.render_texture.height) ],
            })

            cursor = spy.ShaderCursor(shader)
            cursor["planet_albedo"]              = self.surface_albedo_texture
            cursor["planet_height"]              = self.surface_height_texture
            cursor["sampler"]                    = self.texture_sampler
            cursor["planet_clouds"]              = self.cloud_sim.smoke_field()
            cursor["view_projection"]            = view_projection
            cursor["camera_position"]            = camera_position
            cursor["sphere_resolution"]          = self.sphere_resolution
            cursor["sun_emission"]               = self.sun_color.value * self.sun_strength.value
            cursor["cloud_density"]              = self.cloud_density.value
            cursor["sun_direction"]              = spy.math.normalize(self.sun_direction.value)
            cursor["terrain_height"]             = self.terrain_height.value / self.planet_radius.value
            cursor["atmosphere_height"]          = self.atmosphere_height.value / self.planet_radius.value
            cursor["atmosphere_rayleigh_height"] = 1 / max(1e-9, self.atmosphere_rayleigh_height.value / self.planet_radius.value)
            cursor["atmosphere_mie_height"]      = 1 / max(1e-9, self.atmosphere_mie_height.value / self.planet_radius.value)
            cursor["atmosphere_rayleigh_color"]  = self.atmosphere_rayleigh_color.value
            cursor["atmosphere_mie_color"]       = self.atmosphere_mie_color.value
            cursor["atmosphere_density"]         = self.atmosphere_density.value
            cursor["frame_seed"]                 = self.frame_seed

            pass_encoder.draw(spy.DrawArguments({"vertex_count": self.sphere_resolution*self.sphere_resolution*6}))

            self.frame_seed += 1

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


                camera_to_world = self.camera.camera_to_world()
                view = spy.math.inverse(camera_to_world)
                projection = self.camera.projection(self.render_texture.width / self.render_texture.height)
                view_projection = spy.math.mul(projection, view)
                inv_view_projection = spy.math.inverse(view_projection)

                mouse_pos = self.input_state.get("mouse")
                if mouse_pos is not None:
                    if self.input_state.get(spy.MouseButton.right):
                        clip_pos = 2 * (mouse_pos / spy.float2(self.render_texture.width, self.render_texture.height)) - 1
                        clip_pos.y = -clip_pos.y
                        ray_direction = spy.math.normalize(spy.math.mul(inv_view_projection, spy.float4(clip_pos.x, clip_pos.y, 1, 1)).xyz)
                        hit = intersect_sphere(self.camera.position, ray_direction, 1.0)
                        if hit is not None and hit.x > 0:
                            self.drag.append(spy.math.normalize(self.camera.position + ray_direction * hit.x))
                        else:
                            self.drag.clear()
                    else:
                        self.drag.clear()

                self.cloud_sim.step(command_encoder, self.cloud_timestep.value)

                self.render(spy.math.transform_point(camera_to_world, spy.float3(0,0,0)), view_projection, command_encoder)

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
