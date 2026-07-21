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

// multiresolution icosphere fluid
struct FluidMesh
{
    ByteAddressBuffer vertices;
    ByteAddressBuffer faces;       // list of vertices for each face
    StructuredBuffer<uint2> edges; // pair of vertices for each edge

    StructuredBuffer<uint> face_children; // index of first child face for each face
    ByteAddressBuffer face_edges;         // list of edges for each face where face.edges[0] is opposite of face.vertices[0]
    ByteAddressBuffer face_adjacencies;  // list of adjacent faces for each face

    StructuredBuffer<uint> level_face_offsets; // offset in faces for each subdivision level
    StructuredBuffer<uint> level_face_counts; // number of faces in each subdivision level

    RWStructuredBuffer<float> velocity; // per-edge tangent velocity
    RWStructuredBuffer<float> velocity_rw; // per-edge tangent velocity

    StructuredBuffer<float>   smoke; // per-face smoke quantity
    RWStructuredBuffer<float> smoke_rw; // per-face smoke quantity

    uniform uint edge_count;
    uniform uint levels;

    float3 load_vertex(uint index)
    {
        return vertices.Load<float3>(index * sizeof(float3));
    }

    // load vertex indices for a face
    uint3 load_face(uint face_index)
    {
        return faces.Load3(face_index * sizeof(uint3));
    }

    // load edge indices for a face
    uint3 load_face_edges(uint face_index)
    {
        return face_edges.Load3(face_index * sizeof(uint3));
    }

    // load adjacent face indices for a face
    uint3 load_adjacent_faces(uint face_index)
    {
        return face_adjacencies.Load3(face_index * sizeof(uint3));
    }

    // stores vertices in columns
    float3x3 load_face_vertices(uint3 face)
    {
        float3x3 f;
        [ForceUnroll]
        for (uint i = 0; i < 3; i++)
            f[i] = load_vertex(face[i]);
        return f;
    }

    bool find_face(float3 pos, out uint face_index, out uint3 face, out float3x3 vertices, uint max_level = 0xFFFFFFFF)
    {
        // acceleration structure search

        face_index = -1;
        face = -1;

        // base icosphere has 20 triangles
        for (uint i = 0; i < 20; i++)
        {
            face = faces.Load3(i * sizeof(uint3));
            vertices = load_face_vertices(face);
            if (dot(triangle_centroid(vertices), pos) > 0)
            {
                if (in_spherical_triangle(pos, vertices))
                {
                    face_index = i;
                    break;
                }
            }
        }
        if (face_index == -1)
            return false; // failed to select a root node

        for (uint l = 1; l < levels; l++)
        {
            uint first_child = face_children[face_index];

            [ForceUnroll]
            for (uint j = 0; j < 4; j++)
            {
                face = faces.Load3((first_child + j) * sizeof(uint3));
                vertices = load_face_vertices(face);
                if (in_spherical_triangle(pos, vertices))
                {
                    face_index = first_child + j;
                    break;
                }
            }
            if (face_index < first_child || face_index >= first_child + 4)
                return false; // failed to select a child node
        }

        return true;
    }

    // Reconstructs the 3D velocity vector anywhere within a triangle
    float3 sample_velocity(float3 bary, uint3 tri, float3x3 tri_vertices, uint3 tri_edges)
    {
        float3 normal = 0.5 * cross(tri_vertices[1] - tri_vertices[0], tri_vertices[2] - tri_vertices[0]);
        float area = length(normal);
        normal /= area;

        float inv_2a = 1 / (2 * area);

        float3x3 grads;
        grads[0] = cross(normal, tri_vertices[2] - tri_vertices[1]) * inv_2a;
        grads[1] = cross(normal, tri_vertices[0] - tri_vertices[2]) * inv_2a;
        grads[2] = cross(normal, tri_vertices[1] - tri_vertices[0]) * inv_2a;

        float3 v = 0;

        [ForceUnroll]
        for (uint k = 0; k < 3; ++k)
        {
            uint a = (k + 1) % 3; // local edge k connects local verts a -> b
            uint b = (k + 2) % 3;

            uint edge_index = tri_edges[k];
            float c = velocity[edge_index];

            // Match this edge's globally-stored orientation (ev0 -> ev1) against
            // the triangle-local orientation (a -> b) to get the correct sign.
            uint2 edge = edges[edge_index];

            float sign = 1.0;
            if (tri[a] == edge[1] && tri[b] == edge[0])
                sign = -1.0;
            // else (tri[a] == edge[0] && tri[b] == edge[1]) -> sign stays +1

            // Whitney edge basis function: W_ab = lambda_a * grad_b - lambda_b * grad_a
            float3 W = bary[a] * grads[b] - bary[b] * grads[a];

            v += sign * c * W;
        }

        return v;
    }

    float sample_smoke(float3 pos, uint face_index, float3x3 face_vertices)
    {
        float3 normal = normalize(cross(face_vertices[1] - face_vertices[0], face_vertices[2] - face_vertices[0]));
        float3 tangent = normalize(face_vertices[1] - face_vertices[0]);
        float3 bitangent = cross(normal, tangent);
        float3 centroid = triangle_centroid(face_vertices);

        // project pos to triangle
        pos -= normal * dot(pos - centroid, normal);

        // TODO: compute per-vertex smoke values and use barycentric interpolation
    }

    void write_velocity<bool accumulate>(float3 velocity, uint edge_index, uint local_edge_index, uint face_index, uint adjacent_face, uint3 face, float3x3 vertices)
    {
        float3 face_normal = normalize(cross(vertices[1] - vertices[0], vertices[2] - vertices[0]));
        velocity -= dot(velocity, face_normal) * face_normal;
        
        float3 edge_vec = vertices[(local_edge_index + 2) % 3] - vertices[(local_edge_index + 1) % 3];
        if (face[(local_edge_index + 2) % 3] == edges[edge_index][0])
            edge_vec = -edge_vec;

        // store velocities for both face and sum them later
        uint k = face_index > adjacent_face;
        if (accumulate)
            velocity_rw[2 * edge_index + k] += dot(edge_vec, velocity);
        else
            velocity_rw[2 * edge_index + k] = dot(edge_vec, velocity);
    }
};
