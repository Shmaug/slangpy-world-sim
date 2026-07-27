import Math;
import SphericalField;

uniform SphericalScalarField smoke;
uniform SphericalVelocityField velocity;

uniform float3 target_dir;
uniform float  target_angle;
uniform float3 target_vel;
uniform uint   target_channel;

[shader("compute")]
[numthreads(8, 8, 1)]
void emit_plume(uint3 thread: SV_DispatchThreadID)
{
    CubeCoord cell = CubeCoord(thread.xy, thread.z % 6, thread.z / 6, target_channel);
    if (!smoke.bounds_check(cell))
        return;

    // only inject at the ground layer -- the plume rises into higher
    // layers via radial advection instead of being stamped into all of them
    if (cell.layer != 0)
        return;

    float3 pos = cube_uvw_to_dir(float3((cell.texel + 0.5) / velocity.resolution, cell.face));

    // Distance from emission point
    float angle = acos(clamp(dot(target_dir, pos), -1.0, 1.0));

    // Create a plume at target_pos
    if (angle < target_angle * 2.0)
    {
        float normalized_dist = angle / (target_angle * 2.0);
        float gaussian = exp(-normalized_dist * normalized_dist * 4.0);

        float3 kick = CubeMapTangentFrame(pos).to_local3(target_vel);

        velocity.store(cell, velocity.load(cell) + gaussian * kick);
        
        smoke.store(cell, smoke.load<1>(cell) + gaussian);
    }
}