import Math;
import SphericalField;

uniform SphericalScalarField smoke;
uniform SphericalVelocityField velocity;
uniform SphericalScalarField smoke_out;
uniform SphericalVelocityField velocity_out;

uniform float3 target_dir;
uniform float  target_angle;
uniform float3 target_vel;
uniform float  target_vertical_vel;
uniform float  target_temp;
uniform float  bouyancy;
uniform float  dt;

float map_temperature(float value)
{
    return value*1000 + 100;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void add_forces(uint3 thread: SV_DispatchThreadID)
{
    CubeCoord cell = CubeCoord(thread.xy, thread.z % 6, thread.z / 6, 0);
    if (!smoke.bounds_check(cell))
        return;

    // inject new smoke at the ground layer
    // layers via radial advection instead of being stamped into all of them
    float3 dir = cube_uvw_to_dir(float3((cell.texel + 0.5) / smoke.resolution, cell.face));
    float dr = smoke.thickness / smoke.vertical_resolution;
    float r = 1.0 + (cell.layer + 0.5) * dr;

    float3 vel = velocity.load(cell);
    float temperature = smoke.load<1>(cell);

    float ambient_temperature = smoke.sample<1>(dir * r, 0, 2);

    vel.z += dt * bouyancy * (temperature - ambient_temperature);
    // temperature = lerp(temperature, ambient_temperature.x, dt);

    if (cell.layer == 0 && target_angle > 0)
    {
        // Distance from emission point
        float angle = acos(clamp(dot(target_dir, dir), -1.0, 1.0));

        // Create a plume at target_pos
        if (angle < target_angle * 2.0)
        {
            float normalized_dist = angle / (target_angle * 2.0);
            float gaussian = exp(-normalized_dist * normalized_dist * 8.0);

            float3 local_target_vel = CubeMapTangentFrame(dir).to_local3(sphere_parallel_transport(target_dir, dir, target_vel));
            local_target_vel.z += target_vertical_vel;
            vel = lerp(vel, local_target_vel, gaussian);
            temperature = lerp(temperature, target_temp, gaussian);
        }
    }

    velocity_out.store(cell, vel);
    smoke_out.store(cell, temperature);
}