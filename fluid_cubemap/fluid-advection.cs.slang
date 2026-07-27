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

// converts a physical radius (in [1, 1+thickness]) to the continuous,
// cell-centered layer coordinate used by SphericalScalarField/
// SphericalVelocityField's fractional-layer `sample`
float radius_to_vertical_layer(float r)
{
    float dr = velocity.thickness / velocity.vertical_resolution;
    return (r - 1.0) / dr - 0.5;
}

void integrate_velocity(float3 pos, float r, out float3 departure_pos, out float departure_r)
{
    #if 1

    float3 vel = velocity.sample(pos, radius_to_vertical_layer(r));
    float radial = dot(vel, pos);
    float3 tang = vel - pos * radial;

    departure_pos = sphere_exp_map(pos, -dt*tang);
    departure_r = clamp(r - dt*radial, 1.0, 1.0 + velocity.thickness);

    #else

    // https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    // manifold RK4 (Munthe-Kaas style): every stage steps from `pos` along
    // the sphere's surface (sphere_exp_map) and along the radial axis
    // (plain scalar integration -- there's no curvature to worry about
    // there), and every sampled velocity is split into its tangential and
    // radial parts and the tangential part is parallel-transported back
    // into pos's tangent plane before it's folded into the weighted sum --
    // k2/k3/k4 are sampled at three different points, so treating them as
    // if they lived in the same tangent space as k1 would reintroduce the
    // same flat-vector-space bug this is meant to fix.
    float3 vel1 = velocity.sample(pos, radius_to_vertical_layer(r));
    float radial1 = dot(vel1, pos);
    float3 tang1 = vel1 - pos * radial1;

    float3 p2 = sphere_exp_map(pos, tang1 * dt / 2.0);
    float  r2 = r + radial1 * dt / 2.0;
    float3 vel2_raw = velocity.sample(p2, radius_to_vertical_layer(r2));
    float radial2 = dot(vel2_raw, p2);
    float3 tang2 = sphere_parallel_transport(p2, pos, vel2_raw - p2 * radial2);

    float3 p3 = sphere_exp_map(pos, tang2 * dt / 2.0);
    float  r3 = r + radial2 * dt / 2.0;
    float3 vel3_raw = velocity.sample(p3, radius_to_vertical_layer(r3));
    float radial3 = dot(vel3_raw, p3);
    float3 tang3 = sphere_parallel_transport(p3, pos, vel3_raw - p3 * radial3);

    float3 p4 = sphere_exp_map(pos, tang3 * dt);
    float  r4 = r + radial3 * dt;
    float3 vel4_raw = velocity.sample(p4, radius_to_vertical_layer(r4));
    float radial4 = dot(vel4_raw, p4);
    float3 tang4 = sphere_parallel_transport(p4, pos, vel4_raw - p4 * radial4);

    float3 d_tang = (dt / 6.0) * (tang1 + 2 * tang2 + 2 * tang3 + tang4);
    float d_radial = (dt / 6.0) * (radial1 + 2 * radial2 + 2 * radial3 + radial4);

    departure_pos = sphere_exp_map(pos, -d_tang);
    departure_r = clamp(r - d_radial, 1.0, 1.0 + velocity.thickness);
    
    #endif
}

[shader("compute")]
[numthreads(8, 8, 1)]
void advect(uint3 thread: SV_DispatchThreadID)
{
    CubeCoord cell = CubeCoord(thread.xy, thread.z % 6, thread.z / 6, 0);
    if (!velocity.bounds_check(cell))
        return;

    float dr = velocity.thickness / velocity.vertical_resolution;
    float r = 1.0 + dr * (cell.layer + 0.5);

    float3 pos = cube_uvw_to_dir(float3((cell.texel + 0.5) / velocity.resolution, cell.face));

    float3 sampled_pos;
    float sampled_r;
    integrate_velocity(pos, r, sampled_pos, sampled_r);
    float sampled_layer = radius_to_vertical_layer(sampled_r);

    float3 sampled_velocity = velocity.sample(sampled_pos, sampled_layer);

    // sampled_velocity is expressed at sampled_pos, not pos -- rotate it
    // into pos's tangent plane before storing it there (this correctly
    // carries the radial component too: transport is a rigid rotation
    // mapping sampled_pos to pos, so a component along sampled_pos maps to
    // the same-magnitude component along pos)
    float3 transported_velocity = sphere_parallel_transport(sampled_pos, pos, sampled_velocity);
    velocity_out.store(cell, velocity.to_local(pos, transported_velocity));

    for (cell.channel = 0; cell.channel < smoke.channels; cell.channel += 4)
    {
        uint c_left = smoke.channels - cell.channel;
        if (c_left >= 4)
        {
            smoke_out.store<4>(cell, smoke.sample<4>(sampled_pos, cell.channel, 0));
        }
        else if (c_left == 3)
        {
            smoke_out.store<3>(cell, smoke.sample<3>(sampled_pos, cell.channel, 0));
        }
        else if (c_left == 2)
        {
            smoke_out.store<2>(cell, smoke.sample<2>(sampled_pos, cell.channel, 0));
        }
        else if (c_left == 1)
        {
            smoke_out.store<1>(cell, smoke.sample<1>(sampled_pos, cell.channel, 0));
        }
    }
}
