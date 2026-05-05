import Math;
import FluidMesh;

uniform FluidMesh mesh;
uniform float dt;
uniform float omega_sor;
uniform uint2 color_range;

// stores vertices in columns
matrix<float,N,3> load_face_vertices<let N : int>(FluidMesh mesh, vector<uint,N> indices)
{
    matrix<float,N,3> f;
    [ForceUnroll]
    for (uint i = 0; i < N; i++)
        f[i] = mesh.load_vertex(indices[i]);
    return f;
}

float3 face_centroid<let N : int>(matrix<float,N,3> vertices)
{
    float3 v = 0;
    [ForceUnroll]
    for (uint i = 0; i < N; i++)
        v += vertices[i];
    return v / 3.0;
}

float3 triangle_barycentrics(float3 pos, float3x3 vertices)
{
    // project pos to plane
    float3 n = normalize(cross(vertices[1] - vertices[0], vertices[2] - vertices[0]));
    float3 p = pos - n * dot(pos - vertices[0], n);

    float3 v0v1 = vertices[1] - vertices[0];
    float3 v0v2 = vertices[2] - vertices[0];
    float3 v0p = p - vertices[0];
    
    float dot00 = dot(v0v1, v0v1);
    float dot01 = dot(v0v1, v0v2);
    float dot02 = dot(v0v1, v0p);
    float dot11 = dot(v0v2, v0v2);
    float dot12 = dot(v0v2, v0p);
    
    float inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01);
    float u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    float v = (dot00 * dot12 - dot01 * dot02) * inv_denom;
    
    return float3(1 - u - v, u, v);
}

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
    float3 velocity = compute_velocity_at_vertex(mesh, vertex_index, vertex);

    // advect backwards + project to sphere
    float3 pos = normalize(vertex - velocity * dt);

    float vorticity = 0;
    float smoke = 0;

    bool found = false;
    uint3 face;
    float3 bary;
    for (uint i = 0; i < mesh.num_faces; i++)
    {
        face = mesh.load_indices(i);
        let vertices = load_face_vertices(mesh, face);
        
        if (dot(face_centroid(vertices), pos) < 0)
            continue;
        
        bary = triangle_barycentrics(pos, vertices);

        if (all(bary >= 0) && dot(bary, 1) <= 1)
        {
            found = true;
            break;
        }
    }
    
    if (found)
    {
        vorticity = interpolate(mesh.vorticity, face, bary);
        smoke     = interpolate(mesh.smoke, face, bary);
    }
    else
    {
        vorticity = mesh.vorticity[vertex_index];
        smoke     = mesh.smoke[vertex_index];
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
    float new_psi = (mesh.vertex_weights[vertex_index] * mesh.vorticity[vertex_index] + sum) / diag;
    mesh.psi_rw[vertex_index] = new_psi;
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
    mesh.psi_rw[vertex_index] = lerp(psi, new_psi, omega_sor);
}
