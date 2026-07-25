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
        self.conserve_smoke_kernel = create_kernel("fluid-conservation.cs.slang", "conserve_smoke")
        self.compute_divergence_kernel       = create_kernel("fluid-pressure-project.cs.slang", "compute_divergence")
        self.pressure_project_step_kernel    = create_kernel("fluid-pressure-project.cs.slang", "step")
        self.pressure_project_f2c_kernel = create_kernel("fluid-pressure-project.cs.slang", "fine_to_coarse")
        self.pressure_project_c2f_kernel = create_kernel("fluid-pressure-project.cs.slang", "coarse_to_fine")
        self.pressure_project_apply_kernel   = create_kernel("fluid-pressure-project.cs.slang", "apply")

        self.reset = True
        self.step_once = False

        self.smoke_tex:list[spy.Texture] = []
        self.velocity_tex:list[spy.Texture] = []
        self.divergence_tex:spy.Texture = None # type:ignore
        self.pressure_correction_tex:list[spy.Texture] = []

        self.resolution = 512
        def create_textures():
            self.smoke_tex = [self.device.create_texture(
                type = spy.TextureType.texture_2d,
                format = spy.Format.r32_float,
                width = 6 * self.resolution,
                height = self.resolution,
                mip_count = int(spy.math.ceil(spy.math.log2(self.resolution))) + 1,
                usage = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access
            ) for _ in range(2) ]
            self.velocity_tex = [self.device.create_texture(
                type = spy.TextureType.texture_2d,
                format = spy.Format.rg32_float,
                width = 6 * self.resolution,
                height = self.resolution,
                usage = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access
            ) for _ in range(2) ]
            self.divergence_tex = self.device.create_texture(
                type = spy.TextureType.texture_2d,
                format = spy.Format.rg32_float,
                width = 6 * self.resolution,
                height = self.resolution,
                mip_count = int(spy.math.ceil(spy.math.log2(self.resolution))) + 1,
                usage = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access
            )
            self.pressure_correction_tex = [ self.device.create_texture(
                type = spy.TextureType.texture_2d,
                format = spy.Format.r32_float,
                width = 6 * self.resolution,
                height = self.resolution,
                mip_count = int(spy.math.ceil(spy.math.log2(self.resolution))) + 1,
                usage = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access
            ) for _ in range(2) ]

            self.reset = True

        create_textures()

        def reset_cb():
            self.reset = True
        def step_cb():
            self.step_once = True
        def res_cb(value):
            self.resolution = min(max(1,1<<self.resolution_ui.value), 1<<12)
            create_textures()
        self.paused = spy.ui.CheckBox(widget, "Pause")
        spy.ui.Button(widget, "Step", callback=step_cb)
        self.reset_button = spy.ui.Button(widget, "Reset", callback=reset_cb)
        self.resolution_ui = spy.ui.ComboBox(widget, "Resolution", int(spy.math.ceil(spy.math.log2(self.resolution))), items=[ str(1 << i) for i in range(13) ], callback=res_cb)
        self.solver_iterations = spy.ui.DragInt(widget, "Solver iterations", value=5)
        self.solver_fine_iterations = spy.ui.DragInt(widget, "Solver fine iterations", value=4)
        self.multires_solve = spy.ui.CheckBox(widget, "Multiresolution solver", value=True)
        self.dt = spy.ui.DragFloat(widget, "Timestep", 0.01)
        self.emit_plume = spy.ui.CheckBox(widget, "Emit plume")

    def shader_vars(self):
        return {
            "_smoke": self.smoke_tex[0],
            "_smoke_rw": self.smoke_tex[1],
            "_velocity": self.velocity_tex[0],
            "_velocity_rw": self.velocity_tex[1],
            "resolution": self.resolution,
        }

    def step(self, command_encoder:spy.CommandEncoder, dt):
        def swap(vars):
            vars[0], vars[1] = vars[1], vars[0]

        if self.reset:
            for t in self.smoke_tex + self.velocity_tex + self.pressure_correction_tex:
                command_encoder.clear_texture_float(t)
            self.reset = False

        if self.paused.value and not self.step_once:
            return
        self.step_once = False

        def dispatch(kernel, vars, mip = 0):
            res = self.resolution >> mip
            kernel.dispatch([res, res, 6], vars, command_encoder)

        # advect velocity and smoke
        dispatch(self.advect_kernel, {
            "fluid": self.shader_vars(),
            "dt": self.dt.value,
        })

        # compute total amount of smoke via mip maps
        command_encoder.generate_mips(self.smoke_tex[0])
        command_encoder.generate_mips(self.smoke_tex[1])
        # ensure total amount of smoke stays the same after advection
        dispatch(self.conserve_smoke_kernel, {
            "fluid":      self.shader_vars(),
            "sum_pre":    self.smoke_tex[0],
            "sum_post":   self.smoke_tex[1],
            "mip_count":  self.smoke_tex[0].mip_count
        })

        if self.solver_iterations.value > 0:
            def pressure_project_vars(mip = 0):
                return {
                    "fluid": self.shader_vars(),
                    "divergence": self.divergence_tex,
                    "divergence_rw": self.divergence_tex,
                    "pressure_correction": self.pressure_correction_tex[0],
                    "pressure_correction_rw": self.pressure_correction_tex[1].create_view(mip=mip, mip_count=1),
                    "mip_level": mip,
                }
            
            dispatch(self.compute_divergence_kernel, pressure_project_vars())

            if self.multires_solve.value:
                command_encoder.generate_mips(self.divergence_tex)

                for _ in range(self.solver_iterations.value):
                    # pre-smooth on the fine level
                    for _ in range(self.solver_fine_iterations.value):
                        dispatch(self.pressure_project_step_kernel, pressure_project_vars())
                        swap(self.pressure_correction_tex)

                    # fine -> coarse
                    for mip in range(self.pressure_correction_tex[0].mip_count-1):
                        dispatch(self.pressure_project_f2c_kernel, pressure_project_vars(mip+1))

                    # coarse -> fine
                    for mip in range(self.pressure_correction_tex[0].mip_count-1, 2, -1):
                        # interpolate mip+1 -> mip
                        dispatch(self.pressure_project_c2f_kernel, pressure_project_vars(mip), mip=mip)
                        swap(self.pressure_correction_tex)
                        # solve on mip
                        dispatch(self.pressure_project_step_kernel, pressure_project_vars(mip), mip=mip)
                        swap(self.pressure_correction_tex)
            else:
                for _ in range(self.solver_iterations.value):
                    dispatch(self.pressure_project_step_kernel, pressure_project_vars())
                    swap(self.pressure_correction_tex)

            dispatch(self.pressure_project_apply_kernel, pressure_project_vars())

        if self.emit_plume.value:
            dispatch(self.emit_kernel, {
                "fluid":        self.shader_vars(),
                "target_pos":   spy.float3(0,0,1),
                "target_angle": np.radians(10),
                "target_dir":   spy.float3(0,1,0),
            })
        
        swap(self.smoke_tex)
        swap(self.velocity_tex)
