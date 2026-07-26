import Math;
import SphericalField;

uniform SphericalScalarField smoke;
uniform SphericalVelocityField velocity;

uniform float3 target_pos;
uniform float  target_angle;
uniform float3 target_dir;

[shader("compute")]
[numthreads(8, 8, 1)]
void emit_plume(uint3 thread: SV_DispatchThreadID)
{
    if (any(thread.xy >= velocity.resolution) || thread.z >= 6)
        return;

    float3 pos = cube_uvw_to_dir(float3((thread.xy + 0.5) / velocity.resolution, thread.z));

    // Distance from emission point
    float angle = acos(clamp(dot(target_pos, pos), -1.0, 1.0));
    
    // Create a plume at target_pos
    if (angle < target_angle * 2.0)
    {
        float normalized_dist = angle / (target_angle * 2.0);
        float gaussian = exp(-normalized_dist * normalized_dist * 4.0);

        smoke.store(thread, smoke.load(thread) + gaussian);
        velocity.store(thread, velocity.load(thread) + gaussian * CubeMapTangentFrame(pos).to_local(target_dir));
    }
}