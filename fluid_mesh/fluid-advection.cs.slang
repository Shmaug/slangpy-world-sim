import Math;
import FluidMesh;

uniform FluidMesh mesh;
uniform float dt;

// compute (L @ psi) / A
[shader("compute")]
[numthreads(32, 1, 1)]
void streamfunction_to_vorticity(uint3 thread: SV_DispatchThreadID)
{
    uint vertex_index = thread.y * 4096 + thread.x;
    if (vertex_index >= mesh.num_vertices)
        return;
    
    float lap  = 0;
    float diag = 0;
    for (uint i = 0; i < mesh.num_adjacencies; i++)
    {
        uint neighbor_index = mesh.adjacencies[vertex_index * mesh.num_adjacencies + i];
        if (neighbor_index >= mesh.num_vertices)
            break;

        float w = mesh.edge_weights[vertex_index * mesh.num_adjacencies + i];
        lap  += w * mesh.psi[neighbor_index];
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
    
    float3 vertex = mesh.load_vertex(vertex_index);
    float3 velocity = compute_velocity(mesh, vertex_index, vertex);

    // advect backwards + project to sphere
    float3 pos = normalize(vertex - velocity * dt);

    float2 bary;
    uint3 tri = mesh.get_triangle(pos, bary);

    float w0 = mesh.vorticity[tri.x];
    float w1 = mesh.vorticity[tri.y];
    float w2 = mesh.vorticity[tri.z];
    
    float smoke0 = mesh.smoke[tri.x];
    float smoke1 = mesh.smoke[tri.y];
    float smoke2 = mesh.smoke[tri.z];

    float w = w0 + (w1 - w0) * bary.x + (w2 - w0) * bary.y;
    float smoke = smoke0 + (smoke1 - smoke0) * bary.x + (smoke2 - smoke0) * bary.y;
    
    mesh.smoke_rw[vertex_index] = smoke;
    mesh.vorticity_rw[vertex_index] = w;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void vorticity_to_streamfunction(uint3 thread: SV_DispatchThreadID)
{
    uint vertex_index = thread.y * 4096 + thread.x;
    if (vertex_index >= mesh.num_vertices)
        return;

    // pin vertex 0 to fix gauge
    if (vertex_index == 0)
    {
        mesh.psi_rw[vertex_index] = 0;
        return;
    }

    float sum = 0;
    float diag = 0;
    for (uint i = 0; i < mesh.num_adjacencies; i++)
    {
        uint neighbor_index = mesh.adjacencies[vertex_index * mesh.num_adjacencies + i];
        if (neighbor_index >= mesh.num_vertices)
            break;

        float w = mesh.edge_weights[vertex_index * mesh.num_adjacencies + i];
        sum  += w * mesh.psi[neighbor_index];
        diag += w;
    }
    
    mesh.psi_rw[vertex_index] = (mesh.vertex_weights[vertex_index] * mesh.vorticity[vertex_index] + sum) / diag;
}
