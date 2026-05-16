import Math;
import FluidMesh;

uniform FluidMesh mesh;
uniform float dt;
uniform uint level;

float interpolate<let N : int>(StructuredBuffer<float> data, vector<uint,N> face, vector<float,N> bary)
{
    vector<float, N> v_data;

    [ForceUnroll]
    for (uint i = 0; i < N; i++)
        v_data[i] = data[face[i]];

    float v = 0;
    [ForceUnroll]
    for (uint i = 0; i < N; i++)
        v += bary[i] * v_data[i];

    return v;
}

// compute (L @ psi) / A
[shader("compute")]
[numthreads(32, 1, 1)]
void streamfunction_to_vorticity(uint3 thread: SV_DispatchThreadID)
{
    uint vertex_index = thread.y * 4096 + thread.x;
    if (vertex_index >= mesh.num_vertices)
        return;

    uint offset = mesh.vertex_offsets[mesh.levels-1];
    vertex_index += offset;

    float lap  = 0;
    float diag = 0;
    for (uint i = 0; i < mesh.num_adjacencies; i++)
    {
        uint neighbor_index = mesh.adjacencies[vertex_index * mesh.num_adjacencies + i];
        if (neighbor_index == 0xFFFFFFFF)
            break;

        float w = mesh.edge_weights[vertex_index * mesh.num_adjacencies + i];
        lap  += w * mesh.psi[offset + neighbor_index];
        diag += w;
    }
    mesh.vorticity_rw[vertex_index] = (diag * mesh.psi[vertex_index] - lap) / mesh.vertex_weights[vertex_index];
}

[shader("compute")]
[numthreads(32, 1, 1)]
void advect(uint3 thread: SV_DispatchThreadID)
{
    uint vertex_index = thread.y * 4096 + thread.x;
    if (vertex_index >= mesh.num_vertices)
        return;

    uint offset = mesh.vertex_offsets[mesh.levels-1];

    float3 vertex = mesh.load_vertex(vertex_index);
    float3 velocity = mesh.compute_velocity_at_vertex(mesh.levels-1, vertex_index, vertex);

    // advect backwards + project to sphere
    float3 pos = normalize(vertex - velocity * dt);

    float vorticity = 0;
    float smoke = 0;

    uint3 face;
    float3 bary;
    if (mesh.find_face(pos, face, bary))
    {
        vorticity = interpolate(mesh.vorticity, offset + face, bary);
        smoke     = interpolate(mesh.smoke, face, bary);
    }
    else
    {
        vorticity = mesh.vorticity[offset + vertex_index];
        smoke     = mesh.smoke[vertex_index];
    }
    
    mesh.vorticity_rw[offset + vertex_index] = vorticity;
    mesh.smoke_rw[vertex_index] = smoke;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void vorticity_to_streamfunction(uint3 thread: SV_DispatchThreadID)
{
    uint vertex_index = thread.y * 4096 + thread.x;
    if (vertex_index >= mesh.vertex_counts[level])
        return;

    uint offset = mesh.vertex_offsets[level];
    vertex_index += offset;

    // pin vertex 0 to fix gauge
    if (all(thread == 0))
    {
        mesh.psi_rw[vertex_index] = 0;
        return;
    }

    float sum = 0;
    float diag = 0;
    for (uint i = 0; i < mesh.num_adjacencies; i++)
    {
        uint neighbor_index = mesh.adjacencies[vertex_index * mesh.num_adjacencies + i];
        if (neighbor_index == 0xFFFFFFFF)
            break;

        float w = mesh.edge_weights[vertex_index * mesh.num_adjacencies + i];
        sum  += w * mesh.psi[offset + neighbor_index];
        diag += w;
    }
    float new_psi = (mesh.vertex_weights[vertex_index] * mesh.vorticity[vertex_index] + sum) / diag;
    mesh.psi_rw[vertex_index] = new_psi;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void streamfunction_fine_to_coarse(uint3 thread: SV_DispatchThreadID)
{
    uint coarse_vertex_index = thread.y * 4096 + thread.x;
    if (coarse_vertex_index >= mesh.vertex_counts[level])
        return;

    uint coarse_offset = mesh.vertex_offsets[level];
    uint fine_offset = mesh.vertex_offsets[level+1];
    
    // sum psi over neighbors in fine mesh
    float neighbor_psi = 0;
    uint count = 0;
    for (uint i = 0; i < mesh.num_adjacencies; i++)
    {
        uint neighbor_index = mesh.adjacencies[(fine_offset + coarse_vertex_index) * mesh.num_adjacencies + i];
        if (neighbor_index == 0xFFFFFFFF)
            break;

        neighbor_psi += mesh.psi_rw[fine_offset + neighbor_index];
        count++;
    }
    // average between neighors and current value
    mesh.psi_rw[coarse_offset + coarse_vertex_index] = (mesh.psi_rw[fine_offset + coarse_vertex_index] + neighbor_psi);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void streamfunction_coarse_to_fine(uint3 thread: SV_DispatchThreadID)
{
    uint vertex_index = thread.y * 4096 + thread.x;
    if (vertex_index >= mesh.vertex_counts[level])
        return;

    uint coarse_offset = mesh.vertex_offsets[level-1];
    uint fine_offset = mesh.vertex_offsets[level];

    uint coarse_count = mesh.vertex_counts[level-1];
    
    if (vertex_index < coarse_count)
    {
        // This vertex existed at the coarse level - transfer its value
        mesh.psi_rw[fine_offset + vertex_index] = mesh.psi[coarse_offset + vertex_index];
    }
    else
    {
        // each fine vertex lies on the midpoint of a coarse edge
        uint2 coarse_edge = mesh.parent_edges[vertex_index];
        mesh.psi_rw[fine_offset + vertex_index] = (mesh.psi[coarse_offset + coarse_edge.x] + mesh.psi[coarse_offset + coarse_edge.y]) * 0.5;
    }
}
