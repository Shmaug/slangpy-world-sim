import Math;

struct FluidMesh
{
    ByteAddressBuffer vertices;
    ByteAddressBuffer indices; // 32 bit
    StructuredBuffer<uint>  adjacencies; // list of adjacent vertices for each vertex
    StructuredBuffer<uint>  colored_vertices; // concatenated red and black vertices for gauss-seidel
    StructuredBuffer<float> vertex_weights; // barycentric area weight
    StructuredBuffer<float> edge_weights; // list of edge weights for each adjacency
    
    StructuredBuffer<float>   psi; // per-vertex streamfunction
    RWStructuredBuffer<float> psi_rw;
    
    StructuredBuffer<float>   vorticity; // per-vertex vorticity
    RWStructuredBuffer<float> vorticity_rw;

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
};

float3 compute_velocity_at_vertex(FluidMesh mesh, uint vertex_index, float3 vertex)
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