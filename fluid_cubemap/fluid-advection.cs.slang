import Math;
import SphericalField;

uniform SphericalScalarField   smoke;
uniform SphericalVelocityField velocity;

uniform SphericalScalarField   smoke_out;
uniform SphericalVelocityField velocity_out;

uniform float dt;

// advances unit vector `pos` along the great circle in the direction of
// tangent vector `v` (tangent to the sphere at `pos`) by angle |v|
[Differentiable]
float3 sphere_exp_map(float3 dir, float3 v)
{
    float theta = length(v);
    if (theta < 1e-8) return dir;
    return dir * cos(theta) + (v / theta) * sin(theta);
}

float3 integrate_velocity(float3 pos, uint vertical_layer)
{
    // https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    // manifold RK4 (Munthe-Kaas style): every stage steps from `pos` along
    // the sphere's surface (sphere_exp_map), and every sampled velocity is
    // parallel-transported back into pos's tangent plane before it's used
    // as a step direction or folded into the weighted sum -- k2/k3/k4 are
    // sampled at three different points, so treating them as if they lived
    // in the same tangent space as k1 would reintroduce the same
    // flat-vector-space bug this is meant to fix.
    float3 k1 = velocity.sample(pos, vertical_layer);

    float3 p2 = sphere_exp_map(pos, k1 * dt / 2.0);
    float3 k2 = sphere_parallel_transport(p2, pos, velocity.sample(p2, vertical_layer));

    float3 p3 = sphere_exp_map(pos, k2 * dt / 2.0);
    float3 k3 = sphere_parallel_transport(p3, pos, velocity.sample(p3, vertical_layer));

    float3 p4 = sphere_exp_map(pos, k3 * dt);
    float3 k4 = sphere_parallel_transport(p4, pos, velocity.sample(p4, vertical_layer));

    float3 d = (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);

    return sphere_exp_map(pos, -d);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void advect(uint3 thread: SV_DispatchThreadID)
{
    if (any(thread.xy > velocity.resolution) || thread.z >= 6 * velocity.vertical_resolution)
        return;

    float3 pos = cube_uvw_to_dir(float3((thread.xy + 0.5) / velocity.resolution, thread.z));
    float3 sampled_pos = integrate_velocity(pos, thread.z / 6);

    float sampled_smoke = smoke.sample(sampled_pos, 0, thread.z / 6);
    float3 sampled_velocity = velocity.sample(sampled_pos, thread.z / 6);

    // sampled_velocity is tangent to the sphere at sampled_pos, not pos --
    // rotate it into pos's tangent plane before storing it there
    float3 transported_velocity = sphere_parallel_transport(sampled_pos, pos, sampled_velocity);

    smoke_out.store(thread, sampled_smoke);
    velocity_out.store(thread, velocity.to_tangent(pos, transported_velocity));
}
