import ParticleMap;

uniform ParticleMap particle_map;

[shader("compute")]
[numthreads(32, 1, 1)]
void compute_offsets(uint3 threadId: SV_DispatchThreadID)
{
    const uint cell = threadId.x + threadId.y * 4096;
    if (cell >= particle_map.max_cells)
        return;

    uint i;
    InterlockedAdd(particle_map.counters[1], particle_map.cell_counters[cell], i);
    particle_map.cell_offsets[cell] = i;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void sort(uint3 threadId: SV_DispatchThreadID)
{
    const uint index = threadId.x + threadId.y * 4096;
    if (index >= min(particle_map.max_data, particle_map.counters[0]))
        return;
    
    const uint2 p = particle_map.data_indices[index];
    const uint cell = p.x;
    const uint index_in_cell = p.y;
    const uint dst_index = particle_map.cell_offsets[cell] + index_in_cell;

    particle_map.sorted_data_v[dst_index] = particle_map.data_v[index];
    particle_map.sorted_data_d[dst_index] = particle_map.data_d[index];
}
