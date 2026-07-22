import slangpy as spy
import numpy as np
import os

class CubeFluidSimulator:
    def __init__(self, device:spy.Device, widget:spy.ui.Widget|None = None):
        self.device = device

        def create_kernel(shader_file, entry):
            path = os.path.join(os.path.dirname(__file__), shader_file)
            program = self.device.load_program(path, [entry])
            return self.device.create_compute_kernel(program)
        
        self.emit_kernel = create_kernel("fluid-init.cs.slang", "emit_plume")
        self.advect_kernel = create_kernel("fluid-advection.cs.slang", "advect")
        self.conserve_smoke_kernel = create_kernel("fluid-conservation.cs.slang", "conserve_smoke")
        self.compute_divergence_kernel     = create_kernel("fluid-pressure-project.cs.slang", "compute_divergence")
        self.pressure_project_kernel       = create_kernel("fluid-pressure-project.cs.slang", "step")
        self.pressure_project_apply_kernel = create_kernel("fluid-pressure-project.cs.slang", "apply")

        self.reset = True
        self.step_once = False

        self.pre_sum_texture:spy.Texture = None # type:ignore
        self.post_sum_texture:spy.Texture = None # type:ignore

        self.resolution = 1024
        self.fluid_vars = {}
        self.pressure_project_vars = {}
        def create_textures():
            smoke_tex = [self.device.create_texture(
                type = spy.TextureType.texture_2d_array,
                format = spy.Format.r32_float,
                width = self.resolution,
                height = self.resolution,
                array_length = 6,
                usage = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access
            ) for _ in range(2) ]
            velocity_tex = [self.device.create_texture(
                type = spy.TextureType.texture_2d_array,
                format = spy.Format.rg32_float,
                width = self.resolution,
                height = self.resolution,
                array_length = 6,
                usage = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access
            ) for _ in range(2) ]
            self.fluid_vars = {
                "smoke": smoke_tex[0],
                "smoke_rw": smoke_tex[1],
                "velocity": velocity_tex[0],
                "velocity_rw": velocity_tex[1],
                "resolution": self.resolution,
            }
            self.pressure_project_vars = {
                "divergence": self.device.create_texture(
                    type = spy.TextureType.texture_2d_array,
                    format = spy.Format.rg32_float,
                    width = self.resolution,
                    height = self.resolution,
                    array_length = 6,
                    usage = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access
                ),
                "pressure_correction": self.device.create_texture(
                    type = spy.TextureType.texture_2d_array,
                    format = spy.Format.r32_float,
                    width = self.resolution,
                    height = self.resolution,
                    array_length = 6,
                    usage = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access
                ),
                "pressure_correction_rw": self.device.create_texture(
                    type = spy.TextureType.texture_2d_array,
                    format = spy.Format.r32_float,
                    width = self.resolution,
                    height = self.resolution,
                    array_length = 6,
                    usage = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access
                ),
            }
            self.reset = True

            self.pre_sum_texture, self.post_sum_texture = [ self.device.create_texture(
                type = spy.TextureType.texture_2d,
                format = spy.Format.r32_float,
                width = self.resolution * 6,
                height = self.resolution,
                mip_count = int(spy.math.ceil(spy.math.log2(self.resolution))) + 1,
                usage = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access
            ) for _ in range(2) ]

        create_textures()

        if widget is not None:
            def reset_cb():
                self.reset = True
            def step_cb():
                self.step_once = True
            def res_cb(value):
                self.resolution = min(max(1,self.resolution_ui.value), 4096)
                create_textures()
            self.paused = spy.ui.CheckBox(widget, "Pause")
            spy.ui.Button(widget, "Step", callback=step_cb)
            self.reset_button = spy.ui.Button(widget, "Reset", callback=reset_cb)
            self.resolution_ui = spy.ui.DragInt(widget, "Resolution", value=1024, min=1, max=4096, callback=res_cb)
            self.solver_iterations = spy.ui.DragInt(widget, "Solver iterations", value=10)
            self.dt = spy.ui.DragFloat(widget, "Timestep", 0.01)
            self.emit_plume = spy.ui.CheckBox(widget, "Emit plume")

    def step(self, command_encoder:spy.CommandEncoder, dt):
        def swap(vars, name):
            vars[name], vars[f"{name}_rw"] = vars[f"{name}_rw"], vars[name]

        if self.reset:
            for n in ["velocity", "smoke"]:
                command_encoder.clear_texture_float(self.fluid_vars[n])
                command_encoder.clear_texture_float(self.fluid_vars[f"{n}_rw"])
            self.reset = False

        if self.paused.value and not self.step_once:
            return
        self.step_once = False

        # advect velocity and smoke
        self.advect_kernel.dispatch(
            [ self.resolution, self.resolution, 6 ],
            vars={
                "fluid": self.fluid_vars,
                "dt": self.dt.value,
            },
            command_encoder=command_encoder
        )

        # copy smoke to temporary 2d texture
        for i in range(6):
            command_encoder.copy_texture(
                self.pre_sum_texture,
                spy.SubresourceRange({ "layer": 0, "layer_count": 1, "mip": 0, "mip_count": 1 }),
                spy.uint3(self.resolution * i, 0, 0),
                self.fluid_vars["smoke"],
                spy.SubresourceRange({"layer": i, "layer_count": 1, "mip": 0, "mip_count": 1}),
                spy.uint3(0,0,0),
                spy.uint3(self.resolution, self.resolution, 1)
            )
            command_encoder.copy_texture(
                self.post_sum_texture,
                spy.SubresourceRange({ "layer": 0, "layer_count": 1, "mip": 0, "mip_count": 1 }),
                spy.uint3(self.resolution * i, 0, 0),
                self.fluid_vars["smoke_rw"],
                spy.SubresourceRange({"layer": i, "layer_count": 1, "mip": 0, "mip_count": 1}),
                spy.uint3(0,0,0),
                spy.uint3(self.resolution, self.resolution, 1)
            )
        # compute total amount of smoke via mip maps
        command_encoder.generate_mips(self.pre_sum_texture)
        command_encoder.generate_mips(self.post_sum_texture)
        # ensure total amount of smoke stays the same after advection
        self.conserve_smoke_kernel.dispatch(
            [ self.resolution, self.resolution, 6 ],
            vars={
                "fluid":      self.fluid_vars,
                "sum_pre":    self.pre_sum_texture,
                "sum_post":   self.post_sum_texture,
                "mip_count":  self.pre_sum_texture.mip_count
            },
            command_encoder=command_encoder
        )

        if self.solver_iterations.value > 0:
            self.compute_divergence_kernel.dispatch(
                [ self.resolution, self.resolution, 6 ],
                vars=self.pressure_project_vars | { "fluid": self.fluid_vars },
                command_encoder=command_encoder
            )
            for _ in range(self.solver_iterations.value):
                self.pressure_project_kernel.dispatch(
                    [ self.resolution, self.resolution, 6 ],
                    vars=self.pressure_project_vars | { "fluid": self.fluid_vars },
                    command_encoder=command_encoder
                )
                swap(self.pressure_project_vars, "pressure_correction")
            self.pressure_project_apply_kernel.dispatch(
                [ self.resolution, self.resolution, 6 ],
                vars=self.pressure_project_vars | { "fluid": self.fluid_vars },
                command_encoder=command_encoder
            )

        if self.emit_plume.value:
            self.emit_kernel.dispatch(
                [ self.resolution, self.resolution, 6 ],
                vars={
                    "fluid":        self.fluid_vars,
                    "target_pos":   spy.float3(0,0,1),
                    "target_angle": np.radians(10),
                    "target_dir":   spy.float3(0,1,0),
                },
                command_encoder=command_encoder
            )
        
        swap(self.fluid_vars, "smoke")
        swap(self.fluid_vars, "velocity")
