import Math;
import FluidCube;

uniform FluidCube fluid;

uniform float3 target_pos;
uniform float  target_angle;
uniform float3 target_dir;

[shader("compute")]
[numthreads(8, 8, 1)]
void emit_plume(uint3 thread: SV_DispatchThreadID)
{
    if (any(thread.xy >= fluid.resolution) || thread.z >= 6)
        return;

    // init smoke
    {
        float3 pos = cube_uvw_to_dir(float3((thread.xy + 0.5) / fluid.resolution, thread.z));

        // Distance from emission point
        float angle = acos(clamp(dot(target_pos, pos), -1.0, 1.0));
        
        // Create a plume at target_pos
        if (angle < target_angle * 2.0)
        {
            float normalized_dist = angle / (target_angle * 2.0);
            float gaussian = exp(-normalized_dist * normalized_dist * 4.0);
            
            fluid.smoke_rw[thread] += gaussian;
        }
    }

    // init velocity

    float2 velocity = fluid.velocity_rw[thread];

    {
        float3 pos = cube_uvw_to_dir(float3((thread.xy + float2(0, 0.5)) / fluid.resolution, thread.z));

        // Distance from emission point
        float angle = acos(clamp(dot(target_pos, pos), -1.0, 1.0));

        // Create a plume at target_pos
        if (angle < target_angle * 2.0)
        {
            float normalized_dist = angle / (target_angle * 2.0);
            float gaussian = exp(-normalized_dist * normalized_dist * 4.0);

            velocity.x += fluid.world_to_tangent(pos, target_dir).x * gaussian;
        }
    }

    {
        float3 pos = cube_uvw_to_dir(float3((thread.xy + float2(0.5, 0)) / fluid.resolution, thread.z));

        // Distance from emission point
        float angle = acos(clamp(dot(target_pos, pos), -1.0, 1.0));

        // Create a plume at target_pos
        if (angle < target_angle * 2.0)
        {
            float normalized_dist = angle / (target_angle * 2.0);
            float gaussian = exp(-normalized_dist * normalized_dist * 4.0);

            velocity.y += fluid.world_to_tangent(pos, target_dir).y * gaussian;
        }
    }

    fluid.velocity_rw[thread] = velocity;
}