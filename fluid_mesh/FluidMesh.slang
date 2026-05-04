import Math;

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

struct FluidMesh
{
    ByteAddressBuffer vertices;
    ByteAddressBuffer indices; // 32 bit
    StructuredBuffer<uint>  adjacencies; // list of adjacent vertices for each vertex
    StructuredBuffer<float> vertex_weights; // barycentric area weight
    StructuredBuffer<float> edge_weights; // list of edge weights for each adjacency
    
    StructuredBuffer<float>   vorticity; // per-vertex vorticity
    RWStructuredBuffer<float> vorticity_rw;

    StructuredBuffer<float>   psi; // per-vertex streamfunction
    RWStructuredBuffer<float> psi_rw;
    
    StructuredBuffer<float>   smoke; // per-vertex
    RWStructuredBuffer<float> smoke_rw;

    uniform uint num_vertices;
    uniform uint num_triangles;
    uniform uint num_adjacencies;

    float3 load_vertex(uint index)
    {
        return vertices.Load<float3>(index * sizeof(float3));
    }

    // load vertex indices for a triangle
    uint3 load_indices(uint triangle_index)
    {
        return indices.Load3(triangle_index * sizeof(uint3));
    }

    uint3 get_triangle(float3 pos, out float2 bary)
    {
        for (uint i = 0; i < num_triangles; i++)
        {
            uint3 tri = load_indices(i);
            float3 v0 = load_vertex(tri.x);
            float3 v1 = load_vertex(tri.y);
            float3 v2 = load_vertex(tri.z);
            
            if (dot((v0 + v1 + v2) / 3.0, pos) < 0)
                continue;
            
            float3 n = normalize(cross(v1 - v0, v2 - v0));
            float3 p = pos - n * dot(pos - v0, n); // project to plane

            bary = compute_barycentrics(p, v0, v1, v2);

            if (bary.x >= 0 && bary.y >= 0 && bary.x + bary.y <= 1)
            {
                return tri;
            }
        }
        
        return -1;
    }
};

float3 compute_velocity(FluidMesh mesh, uint vertex_index, float3 vertex)
{
    float psi = mesh.psi[vertex_index];
    float3 normal = normalize(vertex);

    float3 velocity = 0;
    uint edge_n = 0;
    for (uint i = 0; i < mesh.num_adjacencies; i++)
    {
        uint neighbor_index = mesh.adjacencies[vertex_index * mesh.num_adjacencies + i];
        if (neighbor_index >= mesh.num_vertices)
            break;

        float3 vi = mesh.load_vertex(neighbor_index);
        float psi_i = mesh.psi[neighbor_index];

        float3 dv = vi - vertex;
        float len = length(dv);
        
        // dpsi/dn: psi gradient along the edge direction
        float dpsi_ds = (psi_i - psi) / len;
        
        // Velocity is perpendicular to both surface normal and edge gradient
        // u = n × (dpsi/ds * edge_dir) = (dpsi/ds) * (n × edge_dir)
        velocity += dpsi_ds * cross(normal, dv / len);
        edge_n++;
    }
    return edge_n > 0 ? velocity / edge_n : edge_n;
}