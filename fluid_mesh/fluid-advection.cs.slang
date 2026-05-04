import Math;
import FluidMesh;

uniform FluidMesh mesh;
uniform float dt;
uniform float omega_sor;
uniform uint2 color_range;

float2 compute_barycentrics(float3 pos, float3 v0, float3 v1, float3 v2)
{
    float3 v0v1 = v1 - v0;
    float3 v0v2 = v2 - v0;
    float3 v0p = pos - v0;
    
    float dot00 = dot(v0v1, v0v1);
    float dot01 = dot(v0v1, v0v2);
    float dot02 = dot(v0v1, v0p);
    float dot11 = dot(v0v2, v0v2);
    float dot12 = dot(v0v2, v0p);
    
    float inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01);
    float u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    float v = (dot00 * dot12 - dot01 * dot02) * inv_denom;
    
    return float2(u, v);
}

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

float interpolate(StructuredBuffer<float> data, uint3 tri, float2 bary)
{
    float v0 = data[tri.x];
    float v1 = data[tri.y];
    float v2 = data[tri.z];
    return v0 * (1 - bary.x - bary.y) + v1 * bary.x + v2 * bary.y;
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

    float vorticity = 0;
    float smoke = 0;

    bool found = false;
    uint3 tri;
    float2 bary;
    for (uint i = 0; i < mesh.num_triangles; i++)
    {
        tri = mesh.load_indices(i);

        float3 v0 = mesh.load_vertex(tri.x);
        float3 v1 = mesh.load_vertex(tri.y);
        float3 v2 = mesh.load_vertex(tri.z);
        
        if (dot((v0 + v1 + v2) / 3.0, pos) < 0)
            continue;
        
        float3 n = normalize(cross(v1 - v0, v2 - v0));
        float3 p = pos - n * dot(pos - v0, n); // project to plane

        bary = compute_barycentrics(pos, v0, v1, v2);

        if (bary.x >= 0 && bary.y >= 0 && bary.x + bary.y <= 1)
        {
            found = true;
            break;
        }
    }
    if (found)
    {
        vorticity = interpolate(mesh.vorticity, tri, bary);
        smoke = interpolate(mesh.smoke, tri, bary);
    }
    else
    {
        vorticity = mesh.vorticity[vertex_index];
        smoke = mesh.smoke[vertex_index];
    }
    
    mesh.smoke_rw[vertex_index] = smoke;
    mesh.vorticity_rw[vertex_index] = vorticity;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void vorticity_to_streamfunction_jacobi(uint3 thread: SV_DispatchThreadID)
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

[shader("compute")]
[numthreads(32, 1, 1)]
void vorticity_to_streamfunction_gauss_seidel(uint3 thread: SV_DispatchThreadID)
{
    uint thread_index = color_range.x + thread.y * 4096 + thread.x;
    if (thread_index >= color_range.y)
        return;

    uint vertex_index = mesh.colored_vertices[thread_index];

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
        sum  += w * mesh.psi_rw[neighbor_index];
        diag += w;
    }
    float new_psi = (mesh.vertex_weights[vertex_index] * mesh.vorticity[vertex_index] + sum) / diag;

    float psi = mesh.psi_rw[vertex_index];
    mesh.psi_rw[vertex_index] = psi + omega_sor * (new_psi - psi);
}
