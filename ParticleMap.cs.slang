import ParticleMap;

uniform ParticleMap particle_map;

[shader("compute")]
[numthreads(32, 1, 1)]
void compute_offsets(uint3 threadId: SV_DispatchThreadID)
{
    uint2 cell = threadId.xy;
    if (any(cell >= particle_map.grid_size))
        return;
    uint i;
    InterlockedAdd(particle_map.data_counters[1], particle_map.cell_counters[cell.y * particle_map.grid_size.x + cell.x], i);
    particle_map.cell_data_offsets[cell.y * particle_map.grid_size.x + cell.x] = i;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void sort(uint3 threadId: SV_DispatchThreadID)
{
    const uint index = threadId.x + threadId.y * 4096;
    if (index >= min(particle_map.capacity, particle_map.data_counters[0]))
        return;
    const uint2 p = particle_map.data_indices[index];
    const uint cell = p.x;;
    const uint index_in_cell = p.y;
    particle_map.sorted_data[particle_map.cell_data_offsets[p.x] + index_in_cell] = particle_map.data[index];
}
