import slangpy as spy
import numpy as np
import os

class CubeFluidSimulator:
    def __init__(self, device:spy.Device, widget:spy.ui.Widget):
        self.device = device

        def create_kernel(shader_file, entry):
            path = os.path.join(os.path.dirname(__file__), shader_file)
            program = self.device.load_program(path, [entry])
            return self.device.create_compute_kernel(program)
        
        self.emit_kernel = create_kernel("fluid-init.cs.slang", "emit_plume")
        self.advect_kernel = create_kernel("fluid-advection.cs.slang", "advect")
        self.generate_mip_kernel = create_kernel("fluid-mipgen.cs.slang", "generate_mip")
        self.conserve_smoke_kernel = create_kernel("fluid-conservation.cs.slang", "conserve_smoke")
        self.compute_divergence_kernel       = create_kernel("fluid-pressure-project.cs.slang", "compute_divergence")
        self.pressure_project_step_kernel    = create_kernel("fluid-pressure-project.cs.slang", "step")
        self.pressure_project_f2c_kernel = create_kernel("fluid-pressure-project.cs.slang", "fine_to_coarse")
        self.pressure_project_c2f_kernel = create_kernel("fluid-pressure-project.cs.slang", "coarse_to_fine")
        self.pressure_project_apply_kernel   = create_kernel("fluid-pressure-project.cs.slang", "apply")

        self.reset = True
        self.step_once = False

        self.smoke_buf:list[spy.Buffer] = []
        self.velocity_buf:list[spy.Buffer] = []
        self.divergence_buf:spy.Buffer = None # type:ignore
        self.pressure_correction_buf:list[spy.Buffer] = []

        self.smoke_readback = self.device.create_buffer(
            element_count = 6,
            struct_size = 4,
            format = spy.Format.r32_float,
            memory_type = spy.MemoryType.read_back,
            usage = spy.BufferUsage.copy_destination,
        )
        self.divergence_readback = self.device.create_buffer(
            element_count = 6,
            struct_size = 4,
            format = spy.Format.r32_float,
            memory_type = spy.MemoryType.read_back,
            usage = spy.BufferUsage.copy_destination,
        )

        self.resolution = 512
        def create_buffers():
            total_texels = sum(6 * (self.resolution >> i) * (self.resolution >> i) for i in range(self.mip_count()))
            self.smoke_buf = [self.device.create_buffer(
                element_count = total_texels,
                struct_size = 4,
                usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
            ) for _ in range(2) ]
            self.velocity_buf = [self.device.create_buffer(
                element_count = 6 * self.resolution * self.resolution,
                struct_size = 8,
                usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
            ) for _ in range(2) ]
            self.divergence_buf = self.device.create_buffer(
                element_count = total_texels,
                struct_size = 4,
                usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
            )
            self.pressure_correction_buf = [ self.device.create_buffer(
                element_count = total_texels,
                struct_size = 4,
                usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
            ) for _ in range(2) ]

            self.reset = True

        create_buffers()

        def reset_cb():
            self.reset = True
        def step_cb():
            self.step_once = True
        def res_cb(value):
            self.resolution = min(max(1,1<<self.resolution_ui.value), 1<<12)
            create_buffers()
        self.paused = spy.ui.CheckBox(widget, "Pause")
        spy.ui.Button(widget, "Step", callback=step_cb)
        self.reset_button = spy.ui.Button(widget, "Reset", callback=reset_cb)
        self.resolution_ui = spy.ui.ComboBox(widget, "Resolution", int(spy.math.ceil(spy.math.log2(self.resolution))), items=[ str(1 << i) for i in range(13) ], callback=res_cb)
        self.solver_iterations = spy.ui.DragInt(widget, "Solver iterations", value=10)
        self.solver_fine_iterations = spy.ui.DragInt(widget, "Solver fine iterations", value=4)
        self.multires_solve = spy.ui.CheckBox(widget, "Multiresolution solver", value=True)
        self.preserve_smoke = spy.ui.CheckBox(widget, "Preserve smoke quantity", value=True)
        self.dt = spy.ui.DragFloat(widget, "Timestep", 0.01)
        self.emit_plume = spy.ui.CheckBox(widget, "Emit plume")
        self.smoke_amount_ui = spy.ui.Text(widget, f"Smoke: {float(0):.3f}")
        self.divergence_ui = spy.ui.Text(widget, f"Divergence: {float(0):.3f}")

    def mip_count(self):
        return 1 + int(spy.math.ceil(spy.math.log2(self.resolution)))

    def shader_vars(self):
        return {
            "_smoke": self.smoke_buf[0],
            "_smoke_rw": self.smoke_buf[1],
            "_velocity": self.velocity_buf[0],
            "_velocity_rw": self.velocity_buf[1],
            "resolution": self.resolution,
        }

    def step(self, command_encoder:spy.CommandEncoder, dt):
        def swap(vars):
            vars[0], vars[1] = vars[1], vars[0]

        if self.reset:
            for t in self.smoke_buf + self.velocity_buf + self.pressure_correction_buf:
                command_encoder.clear_buffer(t)
            self.reset = False

        if self.paused.value and not self.step_once:
            return
        self.step_once = False

        def pressure_project_vars(mip = 0):
            return {
                "fluid": self.shader_vars(),
                "divergence": self.divergence_buf,
                "divergence_rw": self.divergence_buf,
                "pressure_correction": self.pressure_correction_buf[0],
                "pressure_correction_rw": self.pressure_correction_buf[1],
                "mip_level": mip,
            }
        
        def dispatch(kernel, vars, mip = 0):
            res = self.resolution >> mip
            kernel.dispatch([res, res, 6], vars, command_encoder)

        def generate_mips(buf):
            for mip in range(1, self.mip_count()):
                dispatch(self.generate_mip_kernel, {
                    "data": buf,
                    "resolution": self.resolution,
                    "dst_mip": mip,
                }, mip)


        # advect velocity and smoke
        dispatch(self.advect_kernel, {
            "fluid": self.shader_vars(),
            "dt": self.dt.value,
        })

        if self.preserve_smoke.value:
            # ensure total amount of smoke stays the same after advection
            generate_mips(self.smoke_buf[0])
            generate_mips(self.smoke_buf[1])
            dispatch(self.conserve_smoke_kernel, {
                "fluid": self.shader_vars(),
                "mip_count": self.mip_count(),
            })

        # handle emitters
        if self.emit_plume.value:
            dispatch(self.emit_kernel, {
                "fluid":        self.shader_vars(),
                "target_pos":   spy.float3(0,0,1),
                "target_angle": np.radians(1),
                "target_dir":   spy.float3(0,.1,0),
            })

        # solve divergence
        if self.solver_iterations.value > 0:            
            dispatch(self.compute_divergence_kernel, pressure_project_vars())
            if self.multires_solve.value:
                generate_mips(self.divergence_buf)

                for _ in range(self.solver_iterations.value):
                    # pre-smooth on the fine level
                    for _ in range(self.solver_fine_iterations.value):
                        dispatch(self.pressure_project_step_kernel, pressure_project_vars())
                        swap(self.pressure_correction_buf)

                    # fine -> coarse
                    for mip in range(self.mip_count()-1):
                        dispatch(self.pressure_project_f2c_kernel, pressure_project_vars(mip+1))

                    # coarse -> fine
                    for mip in range(self.mip_count()-1, 2, -1):
                        # interpolate mip+1 -> mip
                        dispatch(self.pressure_project_c2f_kernel, pressure_project_vars(mip), mip=mip)
                        swap(self.pressure_correction_buf)
                        # solve on mip
                        dispatch(self.pressure_project_step_kernel, pressure_project_vars(mip), mip=mip)
                        swap(self.pressure_correction_buf)
            else:
                for _ in range(self.solver_iterations.value):
                    dispatch(self.pressure_project_step_kernel, pressure_project_vars())
                    swap(self.pressure_correction_buf)

            dispatch(self.pressure_project_apply_kernel, pressure_project_vars())

        swap(self.smoke_buf)
        swap(self.velocity_buf)

        # show some stats
        
        dispatch(self.compute_divergence_kernel, pressure_project_vars())
        generate_mips(self.divergence_buf)
        command_encoder.copy_buffer(
            self.divergence_readback, 0,
            self.divergence_buf, 4 * sum(6 * (self.resolution >> i) * (self.resolution >> i) for i in range(self.mip_count()-1)),
            6 * 4
        )
        self.divergence_ui.text = f"Divergence: {self.divergence_readback.to_numpy().view(np.float32).mean() * (6 * self.resolution * self.resolution):.3f}"

        generate_mips(self.smoke_buf[0])
        command_encoder.copy_buffer(
            self.smoke_readback, 0,
            self.smoke_buf[0], 4 * sum(6 * (self.resolution >> i) * (self.resolution >> i) for i in range(self.mip_count()-1)),
            6 * 4
        )
        self.smoke_amount_ui.text = f"Smoke: {self.smoke_readback.to_numpy().view(np.float32).mean():.3f}"

