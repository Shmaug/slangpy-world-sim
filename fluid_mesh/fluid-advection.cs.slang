import Math;
import FluidMesh;

uniform FluidMesh mesh;
uniform float dt;
uniform uint level;

[shader("compute")]
[numthreads(32, 1, 1)]
void advect(uint3 thread: SV_DispatchThreadID)
{
    uint face_index = thread.y * 4096 + thread.x;
    if (face_index >= mesh.level_face_counts[level])
        return;
    face_index += mesh.level_face_offsets[level];

    uint3 face = mesh.load_face(face_index);
    float3x3 vertices = mesh.load_face_vertices(face);
    uint3 edges = mesh.load_face_edges(face_index);

    // advect smoke
    {
        float3 velocity = mesh.sample_velocity(1 / 3.0, face, vertices, edges);
        
        // advect backwards + project to sphere
        float3 sampled_pos = normalize(triangle_centroid(vertices) - velocity * dt);

        float sampled_smoke;

        uint sampled_face_index;
        uint3 sampled_face;
        float3x3 sampled_face_vertices;
        if (mesh.find_face(sampled_pos, sampled_face_index, sampled_face, sampled_face_vertices))
        {
            sampled_smoke = mesh.sample_smoke(sampled_pos, sampled_face_index, sampled_face_vertices);
        }
        else
        {
            sampled_smoke = mesh.smoke[face_index];
        }

        mesh.smoke_rw[face_index] = sampled_smoke;
    }

    // advect edge velocities
    uint3 adjacent_faces = mesh.load_adjacent_faces(face_index);
    [ForceUnroll]
    for (uint i = 0; i < 3; i++)
    {
        uint edge_index = edges[i];
        uint adjacent_face = adjacent_faces[i];

        float3 pos = normalize(0.5 * (vertices[(i+1)%3] + vertices[(i+2)%3]));

        // compute barycentric coordinate for edge midpoint
        float3 midpoint_bary = 0.5;
        midpoint_bary[i] = 0; // edge i is opposite vertex i
        float3 velocity = mesh.sample_velocity(midpoint_bary, face, vertices, edges);

        // advect backwards + project to sphere
        float3 sampled_pos = normalize(pos - velocity * dt);

        float3 sampled_velocity;
        uint sampled_face_index;
        uint3 sampled_face;
        float3x3 sampled_face_vertices;
        if (mesh.find_face(sampled_pos, sampled_face_index, sampled_face, sampled_face_vertices))
        {
            float3 bary = triangle_barycentrics(sampled_pos, sampled_face_vertices);
            sampled_velocity = mesh.sample_velocity(bary, sampled_face, sampled_face_vertices, mesh.load_face_edges(sampled_face_index));
        }
        else
        {
            sampled_velocity = velocity;
        }

        mesh.write_velocity<false>(sampled_velocity, edge_index, i, face_index, adjacent_face, face, vertices);
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void swap_velocity(uint3 thread: SV_DispatchThreadID)
{
    uint edge_index = thread.y * 4096 + thread.x;
    if (edge_index >= mesh.edge_count)
        return;

    float v0 = mesh.velocity_rw[2 * edge_index + 0];
    float v1 = mesh.velocity_rw[2 * edge_index + 1];

    mesh.velocity[edge_index] = 0.5 * (v0 + v1);
}