import slangpy as spy
import numpy as np
import os
from collections.abc import Callable

class CubeFluidSimulator:
    def __init__(self, device:spy.Device, widget:spy.ui.Widget|None = None):
        self.device = device

        def create_kernel(shader_file, entry):
            path = os.path.join(os.path.dirname(__file__), shader_file)
            program = self.device.load_program(path, [entry])
            return self.device.create_compute_kernel(program)
        
        self.advect_kernel = create_kernel("fluid-advection.cs.slang", "advect")
        self.generate_mip_kernel = create_kernel("fluid-mipgen.cs.slang", "generate_mip")
        self.conserve_smoke_kernel = create_kernel("fluid-conservation.cs.slang", "conserve_smoke")
        self.compute_divergence_kernel        = create_kernel("fluid-pressure-project.cs.slang", "compute_divergence")
        self.pressure_project_step_kernel     = create_kernel("fluid-pressure-project.cs.slang", "step")
        self.pressure_project_resample_kernel = create_kernel("fluid-pressure-project.cs.slang", "resample")
        self.pressure_project_apply_kernel    = create_kernel("fluid-pressure-project.cs.slang", "apply")

        self.reset = True
        self.step_once = False

        self.smoke_buf:list[spy.Buffer] = []
        self.velocity_buf:list[spy.Buffer] = []
        self.divergence_buf:spy.Buffer = None # type:ignore
        self.pressure_correction_buf:spy.Buffer = None # type:ignore
        self.smoke_readback:spy.Buffer = None # type:ignore
        self.divergence_readback:spy.Buffer = None # type:ignore

        self.heightmap:spy.Texture|None = None
        self.heightmap_sqr:spy.Texture|None = None
        self.linear_sampler = self.device.create_sampler(address_v=spy.TextureAddressingMode.mirror_repeat)
        self.terrain_height = 0.0

        self.resolution = 512
        self.vertical_resolution = 1
        self.channels = 1
        self.radius = 1.0
        self.thickness = 0.1
        self.density_scale_height = 0.05
        self.solver_iterations = 5
        self.solver_fine_iterations = 4
        self.multires_solve = True
        self.preserve_smoke = True
        self.emitters: list[Callable[[spy.CommandEncoder]]] = []

        self._create_buffers()

        if widget is not None:
            def reset_cb():
                self.reset = True
            def step_cb():
                self.step_once = True
            def update_cb(value):
                self.vertical_resolution = min(max(1,1+self.vertical_resolution_ui.value), 8)
                self.resolution = min(max(1,1<<self.resolution_ui.value), 1<<12)
                self.radius = self.radius_ui.value
                self.channels = self.channels_ui.value
                self.thickness = max(1e-4, self.thickness_ui.value)
                self.density_scale_height = self.atmosphere_scale_height_ui.value
                self.solver_iterations = self.solver_iterations_ui.value
                self.solver_fine_iterations = self.solver_fine_iterations_ui.value
                self.multires_solve = self.multires_solve_ui.value
                self.preserve_smoke = self.preserve_smoke_ui.value
            def create_cb(value):
                update_cb(value)
                self._create_buffers()
            self.paused = spy.ui.CheckBox(widget, "Pause")
            spy.ui.Button(widget, "Step", callback=step_cb)
            self.reset_button = spy.ui.Button(widget, "Reset", callback=reset_cb)
            self.channels_ui = spy.ui.DragInt(widget, "Channels", self.channels, min=1, max=4, callback=create_cb)
            self.resolution_ui = spy.ui.ComboBox(widget, "Resolution", int(spy.math.ceil(spy.math.log2(self.resolution))), items=[ str(1 << i) for i in range(13) ], callback=create_cb)
            self.vertical_resolution_ui = spy.ui.ComboBox(widget, "Vertical resolution", self.vertical_resolution-1, items=[ str(i) for i in range(1,8) ], callback=create_cb)
            self.radius_ui = spy.ui.DragFloat(widget, "Radius", self.radius, min=1e-4, speed = 0.01, callback=update_cb)
            self.thickness_ui = spy.ui.DragFloat(widget, "Thickness", self.thickness, min=1e-4, speed = 0.01, callback=update_cb)
            self.atmosphere_scale_height_ui = spy.ui.DragFloat(widget, "Atmosphere scale height", self.density_scale_height, min=1e-4, speed = 0.01, callback=update_cb)
            self.solver_iterations_ui = spy.ui.DragInt(widget, "Solver iterations", value=self.solver_iterations, callback=update_cb)
            self.solver_fine_iterations_ui = spy.ui.DragInt(widget, "Solver fine iterations", value=self.solver_fine_iterations, callback=update_cb)
            self.multires_solve_ui = spy.ui.CheckBox(widget, "Multiresolution solver", value=self.multires_solve, callback=update_cb)
            self.preserve_smoke_ui = spy.ui.CheckBox(widget, "Preserve smoke quantity", value=self.preserve_smoke, callback=update_cb)
            self.smoke_amount_ui = spy.ui.Text(widget, f"Smoke: {float(0):.3f}")
            self.divergence_ui = spy.ui.Text(widget, f"Divergence: {float(0):.3f}")
        else:
            self.smoke_amount_ui = None
            self.divergence_ui = None
            self.paused = None

    def _create_buffers(self):
        total_texels = sum(self.vertical_resolution * 6 * (self.resolution >> i) * (self.resolution >> i) for i in range(self.mip_count()))
        self.smoke_buf = [self.device.create_buffer(
            element_count = total_texels * self.channels,
            struct_size = 4,
            usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
        ) for _ in range(2) ]
        self.velocity_buf = [self.device.create_buffer(
            element_count = self.vertical_resolution * 6 * self.resolution * self.resolution,
            struct_size = 12,
            usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
        ) for _ in range(2) ]
        self.divergence_buf = self.device.create_buffer(
            element_count = total_texels,
            struct_size = 4,
            usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
        )
        self.pressure_correction_buf = self.device.create_buffer(
            element_count = total_texels,
            struct_size = 4,
            usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
        )

        self.smoke_readback = self.device.create_buffer(
            element_count = 6 * self.vertical_resolution * self.channels,
            struct_size = 4,
            format = spy.Format.r32_float,
            memory_type = spy.MemoryType.read_back,
            usage = spy.BufferUsage.copy_destination,
        )
        self.divergence_readback = self.device.create_buffer(
            element_count = 6 * self.vertical_resolution,
            struct_size = 4,
            format = spy.Format.r32_float,
            memory_type = spy.MemoryType.read_back,
            usage = spy.BufferUsage.copy_destination,
        )

        self.reset = True

    def mip_count(self):
        return 1 + int(spy.math.ceil(spy.math.log2(self.resolution)))

    def velocity_field(self, pingpong=0):
        return {
            "data": self.velocity_buf[pingpong],
            "radius": self.radius,
            "thickness": self.thickness,
            "scale_height": self.density_scale_height,
            "resolution": self.resolution,
            "vertical_resolution": self.vertical_resolution
        }
    
    def smoke_field(self, pingpong=0):
        return {
            "data": self.smoke_buf[pingpong],
            "radius": self.radius,
            "thickness": self.thickness,
            "scale_height": self.density_scale_height,
            "resolution": self.resolution,
            "vertical_resolution": self.vertical_resolution,
            "channels": self.channels,
        }

    def dispatch_dim(self, mip:int = 0):
        res = self.resolution >> mip
        return [res, res, 6 * self.vertical_resolution]

    def step(self, command_encoder:spy.CommandEncoder, dt):
        def swap(vars):
            vars[0], vars[1] = vars[1], vars[0]

        if self.reset:
            for t in self.smoke_buf + self.velocity_buf:
                command_encoder.clear_buffer(t)
            self.reset = False

        if self.paused is not None and self.paused.value and not self.step_once:
            return
        self.step_once = False

        def field_vars(buf):
            return self.smoke_field() | { "data": buf, "channels": 1 }

        def pressure_project_vars(dst_mip = 0, src_mip = 0):
            vars = {
                "velocity": self.velocity_field(0),
                "divergence": field_vars(self.divergence_buf),
                "pressure_correction": field_vars(self.pressure_correction_buf),
                "dst_mip_level": dst_mip,
                "src_mip_level": src_mip,
                "terrain_height": 0,
            }
            if self.heightmap is not None:
                vars |= {
                    "surface_height": self.heightmap,
                    "surface_height2": self.heightmap_sqr,
                    "linear_sampler": self.linear_sampler,
                    "terrain_height": self.terrain_height,
                }
            return vars
        
        def dispatch(kernel, vars, mip = 0):
            kernel.dispatch(self.dispatch_dim(mip), vars, command_encoder)

        def generate_mips(buf, channels=1):
            for mip in range(1, self.mip_count()):
                dispatch(self.generate_mip_kernel, {
                    "field": self.smoke_field() | { "data": buf, "channels": channels },
                    "dst_mip": mip,
                }, mip)

        # handle emitters
        if len(self.emitters) > 0:
            for emit in self.emitters:
                emit(command_encoder)
                swap(self.smoke_buf)
                swap(self.velocity_buf)

        generate_mips(self.smoke_buf[0], self.channels)

        if dt > 0:
            # advect velocity and smoke
            dispatch(self.advect_kernel, {
                "smoke": self.smoke_field(),
                "velocity": self.velocity_field(),
                "smoke_out": self.smoke_field(1),
                "velocity_out": self.velocity_field(1),
                "dt": dt,
            })

            if self.preserve_smoke:
                # ensure total amount of smoke stays the same after advection
                generate_mips(self.smoke_buf[1], self.channels)
                dispatch(self.conserve_smoke_kernel, {
                    "fluid_pre": self.smoke_field(),
                    "fluid_post": self.smoke_field(1),
                    "mip_count": self.mip_count(),
                })

            swap(self.smoke_buf)
            swap(self.velocity_buf)

        # solve divergence
        if self.solver_iterations > 0:
            dispatch(self.compute_divergence_kernel, pressure_project_vars())
            def solver_step(dst_mip: int = 0, src_mip: int = 0):
                for color in range(4):
                    dispatch(self.pressure_project_step_kernel, pressure_project_vars(dst_mip, src_mip) | {"color": color}, dst_mip)
            if self.multires_solve:
                generate_mips(self.divergence_buf)

                for _ in range(self.solver_iterations):
                    # pre-smooth on the fine level
                    for _ in range(self.solver_fine_iterations):
                        solver_step()

                    # fine -> coarse
                    for mip in range(1,self.mip_count()):
                        dispatch(self.pressure_project_resample_kernel, pressure_project_vars(mip, mip-1), mip)

                    # coarse -> fine
                    for mip in range(self.mip_count()-1, 2, -1):
                        # interpolate mip+1 -> mip
                        dispatch(self.pressure_project_resample_kernel, pressure_project_vars(mip, mip+1), mip)
                        # solve on mip
                        solver_step(mip)
            else:
                for _ in range(self.solver_iterations):
                    solver_step()

            dispatch(self.pressure_project_apply_kernel, pressure_project_vars())

        # show some stats
        
        generate_mips(self.smoke_buf[0])
        
        if self.divergence_ui is not None:
            dispatch(self.compute_divergence_kernel, pressure_project_vars())
            generate_mips(self.divergence_buf)
            command_encoder.copy_buffer(
                self.divergence_readback, 0,
                self.divergence_buf, self.divergence_buf.size - self.divergence_readback.size,
                self.divergence_readback.size
            )
            self.divergence_ui.text = f"Divergence: {self.divergence_readback.to_numpy().view(np.float32).mean():.2e}"
        if self.smoke_amount_ui is not None:
            command_encoder.copy_buffer(
                self.smoke_readback, 0,
                self.smoke_buf[0], self.smoke_buf[0].size - self.smoke_readback.size,
                self.smoke_readback.size
            )
            self.smoke_amount_ui.text = f"Smoke: {self.smoke_readback.to_numpy().view(np.float32).mean() * self.vertical_resolution * 6 * (self.resolution * self.resolution):.2e}"

