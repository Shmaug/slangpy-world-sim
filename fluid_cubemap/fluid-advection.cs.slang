import Math;
import FluidCube;

uniform FluidCube fluid;
uniform float dt;

float3 integrate_velocity(float3 pos)
{
    // https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    // manifold RK4 (Munthe-Kaas style): every stage steps from `pos` along
    // the sphere's surface (sphere_exp_map), and every sampled velocity is
    // parallel-transported back into pos's tangent plane before it's used
    // as a step direction or folded into the weighted sum -- k2/k3/k4 are
    // sampled at three different points, so treating them as if they lived
    // in the same tangent space as k1 would reintroduce the same
    // flat-vector-space bug this is meant to fix.
    float3 k1 = fluid.sample_velocity(pos);

    float3 p2 = sphere_exp_map(pos, k1 * dt / 2.0);
    float3 k2 = sphere_parallel_transport(p2, pos, fluid.sample_velocity(p2));

    float3 p3 = sphere_exp_map(pos, k2 * dt / 2.0);
    float3 k3 = sphere_parallel_transport(p3, pos, fluid.sample_velocity(p3));

    float3 p4 = sphere_exp_map(pos, k3 * dt);
    float3 k4 = sphere_parallel_transport(p4, pos, fluid.sample_velocity(p4));

    float3 d = (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);

    return sphere_exp_map(pos, -d);
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

    // sampled_velocity is tangent to the sphere at sampled_pos, not pos --
    // rotate it into pos's tangent plane before storing it there
    float3 transported_velocity = sphere_parallel_transport(sampled_pos, pos, sampled_velocity);

    fluid.store_smoke(thread, sampled_smoke);
    fluid.store_velocity(thread, CubeMapTangentFrame(pos).to_local(transported_velocity));
}
