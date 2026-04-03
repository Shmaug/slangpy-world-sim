import os
import slangpy as spy
from ParticleMap import ParticleMap

INIT_NONE = 0
INIT_VORTEX = 1
INIT_LEAPFROG = 2

ADVECT_GRID = 0
ADVECT_PARTICLE = 1

def get_asset_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(__file__), path)

class FluidSimulator:
    def __init__(self, device : spy.Device):
        self.device = device
        self.step_once = False
        self.initialized = False
        self.grid_vars = {}
        self.passes = {}
        self.drag = None

    def setup_ui(self, window):
        self.enabled = spy.ui.CheckBox(window, "Enabled", True)

        def step_callback():
            self.step_once = True
        spy.ui.Button(window, "Step", callback=step_callback)

        def reset_callback(arg=None):
            self.initialized = False
        spy.ui.Button(window, "Reset", callback=reset_callback)

        self.init_mode  = spy.ui.ComboBox(window, "Init mode", 0, reset_callback, ["None", "Vortex", "Leapfrog"])
        self.emit_smoke = spy.ui.CheckBox(window, "Emit smoke")
        self.advect_mode = spy.ui.ComboBox(window, "Advect mode", 0, items=["Grid", "Particle"])

        self.resolution                  = spy.ui.DragInt2(window,  "Resolution", value=spy.int2(512,512), min=1, callback=reset_callback)
        self.advect_dt                   = spy.ui.DragFloat(window, "Advection dt", value=0.1, min=0)
        self.advection_iterations        = spy.ui.DragInt(window,   "Advection iterations", value=1, min=1)
        self.pressure_project_iterations = spy.ui.DragInt(window,   "Pressure projection iterations", value=50, min=1)
        self.avg_pressure = spy.ui.Text(window, "Avg pressure: 0")
        self.use_pressure_correction = spy.ui.CheckBox(window, "Pressure correction")
        self.vorticity_confinement_amount = spy.ui.DragFloat(window, "Vorticity confinement")
    
    def get_resolution(self):
        return self.resolution.value

    def swap_grids(self):
        for name in ["velocity_x", "velocity_y", "pressure"]:
            name_out = f"{name}_rw"
            self.grid_vars[name], self.grid_vars[name_out] = self.grid_vars[name_out], self.grid_vars[name]

    def dispatch_pass(self, shader, entry, dim, vars, command_encoder):
        id = f"{shader}:{entry}"
        if id not in self.passes:
            self.passes[id] = self.device.create_compute_kernel(self.device.load_program(get_asset_path(shader), [entry]))
        self.passes[id].dispatch(dim, vars, command_encoder)

    def initialize_grids(self, command_encoder:spy.CommandEncoder):
        def create_grid(width, height, name, mip_count = 1):
            return self.device.create_texture(
                format=spy.Format.r32_float,
                width=width,
                height=height,
                mip_count=mip_count,
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                label=name)
        
        w,h = self.get_resolution()

        self.grid_vars = {
            "velocity_x":     create_grid(w+1, h, "velocity_x_0"),
            "velocity_x_rw":  create_grid(w+1, h, "velocity_x_1"),
            "velocity_y":     create_grid(w, h+1, "velocity_y_0"),
            "velocity_y_rw":  create_grid(w, h+1, "velocity_y_1"),
            "pressure":       create_grid(w, h,   "pressure_0", spy.ALL_MIPS),
            "pressure_rw":    create_grid(w, h,   "pressure_1", spy.ALL_MIPS),
            "linear_sampler": self.device.create_sampler(),
            "resolution":     self.get_resolution(),
        }
        self.divergence            = create_grid(w, h, "divergence")
        self.pressure_correction_0 = create_grid(w, h, "pressure_correction_0")
        self.pressure_correction_1 = create_grid(w, h, "pressure_correction_1")

        self.curl = create_grid(w, h, "curl")

        self.particle_map = ParticleMap(self.device, w*h, 4*w*h)

        self.avg_pressure_gpu = self.device.create_buffer(
            element_count=1,
            struct_size=4,
            format=spy.Format.r32_float,
            usage=spy.BufferUsage.copy_destination|spy.BufferUsage.shader_resource
        )
        self.avg_pressure_cpu = self.device.create_buffer(
            element_count=1,
            struct_size=4,
            format=spy.Format.r32_float,
            memory_type=spy.MemoryType.read_back,
            usage=spy.BufferUsage.copy_destination
        )

        self.grid_dispatch_dim     = [w,   h,   1]
        self.mac_grid_dispatch_dim = [w+1, h+1, 1]

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

    def emit_drag(self, drag_start, drag_end):
        self.drag = (drag_start, drag_end)

    def advect(self, command_encoder:spy.CommandEncoder):
        vars = {
            "grid": self.grid_vars,
            "particle_map": self.particle_map.vars,
            "advect_dt": self.advect_dt.value / self.advection_iterations.value
        }
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

    def add_vorticity_confinement(self, command_encoder:spy.CommandEncoder):
        vars = {
            "grid": self.grid_vars,
            "curl": self.curl,
            "confinement_scale": self.vorticity_confinement_amount.value
        }

        self.dispatch_pass(
            "fluid-vorticity-confinement.cs.slang",
            "compute_curl",
            self.grid_dispatch_dim,
            vars,
            command_encoder)
        
        self.dispatch_pass(
            "fluid-vorticity-confinement.cs.slang",
            "add_confinement_force",
            self.mac_grid_dispatch_dim,
            vars,
            command_encoder)

    def pressure_project(self, command_encoder:spy.CommandEncoder):
        if self.pressure_project_iterations.value <= 0:
            return
        command_encoder.clear_texture_float(self.pressure_correction_0)
        command_encoder.clear_texture_float(self.pressure_correction_1)
        vars = {
            "grid": self.grid_vars,
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
        for _ in range(self.pressure_project_iterations.value):
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

    def step(self, command_encoder:spy.CommandEncoder, dt):
        if not self.initialized:
            self.initialize_grids(command_encoder)

        if not self.enabled.value and not self.step_once:
            return
        
        self.step_once = False

        for _ in range(self.advection_iterations.value):
            if self.use_pressure_correction.value:
                command_encoder.generate_mips(self.grid_vars["pressure"])

            self.advect(command_encoder)
            
            if self.vorticity_confinement_amount.value > 0:
                self.add_vorticity_confinement(command_encoder)

            if self.emit_smoke.value:
                self.dispatch_pass(
                    "fluid-init.cs.slang",
                    "emit_smoke",
                    self.mac_grid_dispatch_dim,
                    { "grid": self.grid_vars },
                    command_encoder)
            if self.drag is not None:
                drag_start, drag_end = self.drag
                self.drag = None
                self.dispatch_pass(
                    "fluid-init.cs.slang",
                    "emit_drag",
                    self.mac_grid_dispatch_dim,
                    { "grid": self.grid_vars, "drag_start": drag_start, "drag_end": drag_end },
                    command_encoder)
                
            self.pressure_project(command_encoder)
            
            if self.use_pressure_correction.value:
                # force avg pressure to match before the step
                command_encoder.generate_mips(self.grid_vars["pressure_rw"])
                command_encoder.copy_texture_to_buffer(self.avg_pressure_gpu, 0, 4, 256, self.grid_vars["pressure_rw"], 0, self.grid_vars["pressure_rw"].mip_count-1, [0,0,0], [1,1,1])
                self.dispatch_pass(
                    "pressure-correction.cs.slang",
                    "pressure_correction",
                    self.grid_dispatch_dim,
                    {"grid":self.grid_vars, "current_pressure": self.avg_pressure_gpu },
                    command_encoder)
            
            self.swap_grids()

        command_encoder.generate_mips(self.grid_vars["pressure"])
        command_encoder.copy_texture_to_buffer(self.avg_pressure_cpu, 0, 4, 256, self.grid_vars["pressure"], 0, self.grid_vars["pressure"].mip_count-1, [0,0,0], [1,1,1])
        self.avg_pressure.text = f"Avg pressure: {self.avg_pressure_cpu.to_numpy()[0]}"
