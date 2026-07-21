import Math;
import FluidMesh;

uniform FluidMesh mesh;

uniform float3 target_pos;
uniform float  target_angle;
uniform float3 target_dir;

[shader("compute")]
[numthreads(32, 1, 1)]
void emit_plume(uint3 thread: SV_DispatchThreadID)
{
    uint face_index = thread.y * 4096 + thread.x;
    if (face_index >= mesh.level_face_counts[mesh.levels - 1])
        return;
    face_index += mesh.level_face_offsets[mesh.levels - 1];

    uint3 face = mesh.load_face(face_index);
    float3x3 vertices = mesh.load_face_vertices(face);

    // init smoke
    {
        float3 pos = normalize(triangle_centroid(vertices));

        // Distance from emission point
        float angle = acos(clamp(dot(target_pos, pos), -1.0, 1.0));
        
        // Create a plume at target_pos
        if (angle < target_angle * 2.0)
        {
            float normalized_dist = angle / (target_angle * 2.0);
            float gaussian = exp(-normalized_dist * normalized_dist * 4.0);
            
            mesh.smoke_rw[face_index] += gaussian;
        }
    }

    // Add velocity towards target_dir
    uint3 edges = mesh.load_face_edges(face_index);
    uint3 adjacent_faces = mesh.load_adjacent_faces(face_index);
    [ForceUnroll]
    for (uint i = 0; i < 3; i++)
    {
        uint edge_index = edges[i];
        uint adjacent_face = adjacent_faces[i];

        float3 pos = normalize(0.5 * (vertices[(i+1)%3] + vertices[(i+2)%3]));

        // Distance from emission point
        float angle = acos(clamp(dot(target_pos, pos), -1.0, 1.0));

        // Create a plume at target_pos
        if (angle < target_angle * 2.0)
        {
            float normalized_dist = angle / (target_angle * 2.0);
            float gaussian = exp(-normalized_dist * normalized_dist * 4.0);

            mesh.write_velocity<true>(target_dir * gaussian, edge_index, i, face_index, adjacent_face, face, vertices);
        }
    }
}