import Math;

bool in_spherical_triangle(float3 p, float3x3 vertices)
{
    // Signed volume of tetrahedra formed by origin + edge + P.
    float d0 = dot(cross(vertices[0], vertices[1]), p);
    float d1 = dot(cross(vertices[1], vertices[2]), p);
    float d2 = dot(cross(vertices[2], vertices[0]), p);

    // P is inside iff all three have the same sign (all +ve or all -ve).
    // The sign depends on the winding order of (A,B,C).
    return (d0 >= 0.0f && d1 >= 0.0f && d2 >= 0.0f)
        || (d0 <= 0.0f && d1 <= 0.0f && d2 <= 0.0f);
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

float3 triangle_centroid(float3x3 vertices)
{
    float3 v = 0;
    [ForceUnroll]
    for (uint i = 0; i < 3; i++)
        v += vertices[i];
    return v / 3.0;
}

struct FluidMesh
{
    ByteAddressBuffer       node_vertex_indices;
    StructuredBuffer<uint4> node_children;
    StructuredBuffer<uint>  node_payloads;

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

    static const uint num_adjacencies = 6; // icosphere max vertex degree

    uniform uint num_vertices;
    uniform uint num_faces;
    uniform uint levels;

    float3 load_vertex(uint index)
    {
        return vertices.Load<float3>(index * sizeof(float3));
    }

    // load vertex indices for a triangle
    uint3 load_indices(uint triangle_index)
    {
        return indices.Load3(triangle_index * sizeof(uint3));
    }

    // stores vertices in columns
    float3x3 load_face_vertices(uint3 indices)
    {
        float3x3 f;
        [ForceUnroll]
        for (uint i = 0; i < 3; i++)
            f[i] = load_vertex(indices[i]);
        return f;
    }

    bool find_face(float3 pos, out uint3 face, out float3 bary)
    {
        // acceleration structure search

        uint node_index = -1;
        face = -1;
        bary = 0;

        float3x3 vertices;

        // base icosphere has 20 triangles
        for (uint i = 0; i < 20; i++)
        {
            face = node_vertex_indices.Load3(i * sizeof(uint3));
            vertices = load_face_vertices(face);
            if (dot(triangle_centroid(vertices), pos) > 0)
            {
                if (in_spherical_triangle(pos, vertices))
                {
                    node_index = i;
                    break;
                }
            }
        }
        if (node_index == -1)
            return false; // failed to select a root node

        for (uint l = 1; l < levels; l++)
        {
            uint4 children = node_children[node_index];
            [ForceUnroll]
            for (uint j = 0; j < 4; j++)
            {
                face = node_vertex_indices.Load3(children[j] * sizeof(uint3));
                vertices = load_face_vertices(face);
                if (in_spherical_triangle(pos, vertices))
                {
                    node_index = children[j];
                    break;
                }
            }
            if (!any(children == node_index))
                return false; // failed to select a child node
        }

        bary = triangle_barycentrics(pos, vertices);
        return true;
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