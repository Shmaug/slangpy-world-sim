import Math;
import FluidCube;

uniform FluidCube fluid;
uniform float dt;

float3 integrate_velocity(float3 pos)
{
    // https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    float3 k1 = fluid.sample_velocity(pos);
    float3 k2 = fluid.sample_velocity(normalize(pos + k1 * dt / 2.0));
    float3 k3 = fluid.sample_velocity(normalize(pos + k2 * dt / 2.0));
    float3 k4 = fluid.sample_velocity(normalize(pos + k3 * dt));
    
    float3 d = (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);

    return normalize(pos - d);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void advect(uint3 thread: SV_DispatchThreadID)
{
    if (any(thread.xy > fluid.resolution) || thread.z >= 6)
        return;

    float3 pos = cube_uvw_to_dir(float3((thread.xy + 0.5) / fluid.resolution, thread.z));
    float3 sampled_pos = integrate_velocity(pos);

    float sampled_smoke = fluid.sample_smoke(sampled_pos);
    float3 sampled_velocity = fluid.sample_velocity(sampled_pos);

    fluid.smoke_rw[thread] = sampled_smoke;
    fluid.velocity_rw[thread] = CubeMapTangentFrame(pos).to_local(sampled_velocity);
}
