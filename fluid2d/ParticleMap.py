import slangpy as spy

class ParticleMap:
    def __init__(self, device : spy.Device, num_particles, num_cells):
        self.num_particles = num_particles
        self.num_cells = num_cells
        self.vars = {
            "data":          device.create_buffer(element_count=num_particles, struct_size=16, usage=spy.BufferUsage.unordered_access|spy.BufferUsage.shader_resource),
            "sorted_data":   device.create_buffer(element_count=num_particles, struct_size=16, usage=spy.BufferUsage.unordered_access|spy.BufferUsage.shader_resource),
            "data_indices":  device.create_buffer(element_count=num_particles, struct_size=8,  usage=spy.BufferUsage.unordered_access|spy.BufferUsage.shader_resource),
            "cell_counters": device.create_buffer(element_count=num_cells,     struct_size=4,  usage=spy.BufferUsage.unordered_access|spy.BufferUsage.shader_resource),
            "cell_offsets":  device.create_buffer(element_count=num_cells,     struct_size=4,  usage=spy.BufferUsage.unordered_access|spy.BufferUsage.shader_resource),
            "counters":      device.create_buffer(element_count=4,             struct_size=4,  usage=spy.BufferUsage.unordered_access|spy.BufferUsage.shader_resource),
            "max_cells":     num_cells,
            "max_data":      num_particles,
        }

        self.compute_offsets_pass = device.create_compute_kernel(device.load_program("ParticleMap.cs.slang", ["compute_offsets"]))
        self.sort_pass = device.create_compute_kernel(device.load_program("ParticleMap.cs.slang", ["sort"]))

    def clear(self, command_encoder:spy.CommandEncoder):
        command_encoder.clear_buffer(self.vars["cell_counters"])
        command_encoder.clear_buffer(self.vars["counters"])

    def sort(self, command_encoder:spy.CommandEncoder):
        self.compute_offsets_pass.dispatch([4096, (self.num_cells + 4095) // 4096, 1], {"particle_map":self.vars}, command_encoder)
        self.sort_pass.dispatch([4096, (self.num_particles + 4095) // 4096, 1], {"particle_map":self.vars}, command_encoder)
