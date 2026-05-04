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
    uint vertex_index = thread.y * 4096 + thread.x;
    if (vertex_index >= mesh.num_vertices)
        return;
    
    float3 pos = mesh.load_vertex(vertex_index);
    float smoke = mesh.smoke_rw[vertex_index];
    float psi = mesh.psi_rw[vertex_index];

    float3 n = normalize(pos);

    // Distance from emission point
    float angle = acos(clamp(dot(target_pos, n), -1.0, 1.0));
    
    // Create a plume at target_pos towards target_dir
    if (angle < target_angle * 2.0)
    {
        float normalized_dist = angle / (target_angle * 2.0);
        float gaussian = exp(-normalized_dist * normalized_dist * 4.0);
        
        // Create tangent basis at emission point
        float3 t1 = normalize(cross(target_dir, target_pos));
        float psi_target = dot(pos - target_pos, t1) * gaussian;
        
        psi   = lerp(psi, psi_target, 0.9);
        smoke = lerp(smoke, gaussian, 0.9);
    }

    mesh.smoke_rw[vertex_index] = smoke;
    mesh.psi_rw[vertex_index] = psi;
}