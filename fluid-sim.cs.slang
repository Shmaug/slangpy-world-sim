import Math;
import RNG;
import ParticleMap;

struct Particle {
    float2 pos;
    float2 vel;
};

RWStructuredBuffer<Particle> particle_buffer;

RWTexture2D<float> u_grid;
RWTexture2D<float> v_grid;

RWTexture2D<float> density_grid;
RWTexture2D<float> divergence;
Texture2D<float>   p_grid;
RWTexture2D<float> p_grid_out;

uniform uint2 resolution;
uniform uint  num_particles;
uniform float dt;
uniform uint  init_seed;

uniform ParticleMap particle_map;

[Differentiable]
float2 interpolate_velocity(const float2 particle_pos) {
    float2 vel = 0;

    float2 pos = particle_pos * resolution;

    int2 p0 = max(0, int2(floor(pos - 2)));
    int2 p1 = min(resolution, int2(ceil(pos + 2)));
    
    [MaxIters(5)]
    for (int x = p0.x; x <= p1.x; x++)
    {
        [MaxIters(5)]
        for (int y = p0.y; y <= p1.y; y++)
        {
            if (y < resolution.y)
            {
                float wx = bspline(pos.x - x);
                float wy = bspline(pos.y - (y + .5));
                vel.x += u_grid[int2(x, y)] * wx * wy;
            }

            if (x < resolution.x)
            {
                float wx = bspline(pos.x - (x+.5));
                float wy = bspline(pos.y - y);
                vel.y += v_grid[int2(x, y)] * wx * wy;
            }
        }
    }

    return vel;
}

float2 interpolate_velocity_gradient(float2 pos, out float2x2 grad)
{
    var dv_dx = diffPair(pos);
    __bwd_diff(interpolate_velocity)(dv_dx, float2(1, 0));
    grad[0] = dv_dx.d;

    var dv_dy = diffPair(pos);
    __bwd_diff(interpolate_velocity)(dv_dy, float2(0, 1));
    grad[1] = dv_dy.d;

    return interpolate_velocity(pos);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void emit_vortex(uint3 thread: SV_DispatchThreadID)
{
    uint particle_index = thread.y * 4096 + thread.x;
    if (particle_index >= num_particles)
        return;

    static const float2 center0 = float2(resolution.x * .3, resolution.y * .5);
    static const float2 center1 = float2(resolution.x * .7, resolution.y * .5);
    static const float  radius = resolution.x*0.2;

    RNG rng = RNG(pcg(particle_index + pcg(init_seed)));
    bool left = rng.nextFloat() < 0.5;
    float theta = rng.nextFloat() * M_2PI;
    float2 s = float2(cos(theta), sin(theta)) * sqrt(rng.nextFloat());

    Particle p;
    p.pos = (select(left, center0, center1) + s * radius + 0.5) / resolution;
    p.vel = float2(-s.y, s.x) * 0.25;//select(left, -0.25, 0.25);
    particle_buffer[particle_index] = p;

    int2 cell = int2(floor(p.pos * resolution));
    particle_map.insert(cell.y * resolution.x + cell.x, particle_index);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void emit_smoke(uint3 thread: SV_DispatchThreadID)
{
    uint particle_index = thread.y * 4096 + thread.x;
    if (particle_index >= num_particles)
        return;

    static const uint4 box = uint4(resolution.x / 4, (resolution.y * 7) / 8, (resolution.x * 3) / 4, resolution.y);

    RNG rng = RNG(pcg(particle_index + pcg(init_seed)));
    Particle p;
    p.pos = lerp(box.xy, box.zw, float2(rng.nextFloat(), rng.nextFloat())) / resolution;
    p.vel = float2(0, -10);
    particle_buffer[particle_index];

    int2 cell = int2(floor(p.pos * resolution));
    particle_map.insert(cell.y * resolution.x + cell.x, particle_index);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void advect(uint3 thread : SV_DispatchThreadID)
{
    uint particle_index = thread.y*4096 + thread.x;
    if (particle_index >= num_particles)
        return;

    Particle particle = particle_buffer[particle_index];
    particle.vel = interpolate_velocity(particle.pos);

    #if 0
    {
        // https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods

        // Impulse evolution: du/dt = -(∇v)ᵀ u
        float2x2 grad;

        float2 k1 = interpolate_velocity_gradient(particle.pos, grad);
        float2 k1_vel = -mul(grad, particle.vel);

        float2 k2 = interpolate_velocity_gradient(particle.pos + k1 * dt / 2.0, grad);
        float2 k2_vel = -mul(grad, particle.vel + k1_vel * dt / 2.0);

        float2 k3 = interpolate_velocity_gradient(particle.pos + k2 * dt / 2.0, grad);
        float2 k3_vel = -mul(grad, particle.vel + k2_vel * dt / 2.0);

        float2 k4 = interpolate_velocity_gradient(particle.pos + k3 * dt, grad);
        float2 k4_vel = -mul(grad, particle.vel + k3_vel * dt);

        particle.pos += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
        particle.vel += (dt / 6.0) * (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel);
    }
    #else
    particle.pos += particle.vel * dt;
    #endif

    if (particle.pos.x <  0) { particle.pos.x =    -particle.pos.x; particle.vel.x *= -1; }
    if (particle.pos.x >= 1) { particle.pos.x = 2 - particle.pos.x; particle.vel.x *= -1; }
    if (particle.pos.y <  0) { particle.pos.y =    -particle.pos.y; particle.vel.y *= -1; }
    if (particle.pos.y >= 1) { particle.pos.y = 2 - particle.pos.y; particle.vel.y *= -1; }

    particle_buffer[particle_index] = particle;

    int2 cell = clamp(int2(floor(particle.pos * resolution)), 0, resolution - 1);

    particle_map.insert(cell.y * resolution.x + cell.x, particle_index);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void particle_to_grid(uint3 thread: SV_DispatchThreadID)
{
    const int2 cell = int2(thread.xy);
    if (any(cell >= resolution + 1))
        return;

    float density_sum = 0;
    float u_sum = 0;
    float u_weight = 0;
    float v_sum = 0;
    float v_weight = 0;

    int2 p0 = max(0, cell - 2);
    int2 p1 = min(resolution-1, cell+2);

    for (int x = p0.x; x <= p1.x; x++)
    {
        for (int y = p0.y; y <= p1.y; y++)
        {
            uint2 range = particle_map.get_range(y * resolution.x + x);
            for (int j = range.x; j < range.y; j++)
            {
                Particle p = particle_buffer[particle_map.get(j)];
                float2 pos = p.pos * resolution;

                float wx = bspline(pos.x - cell.x);
                float wy = bspline(pos.y - cell.y);
                float wcx = bspline(pos.x - (cell.x + .5));
                float wcy = bspline(pos.y - (cell.y + .5));

                density_sum += 4.0 * wcx * wcy;

                u_sum += p.vel.x * wx * wcy;
                u_weight += wx * wcy;

                v_sum += p.vel.y * wcx * wy;
                v_weight += wcx * wy;
            }
        }
    }

    if (all(cell < resolution)) {
        density_grid[cell] = density_sum;
    }
    if (cell.y < resolution.y) {
        u_grid[cell] = cell.x > 0 && cell.x < resolution.x ? safe_divide(u_sum, u_weight) : 0;
    }
    if (cell.x < resolution.x) {
        v_grid[cell] = cell.y > 0 && cell.y < resolution.y ? safe_divide(v_sum, v_weight) : 0;
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_divergence(uint3 thread: SV_DispatchThreadID)
{
    uint2 cell = thread.xy;
    if (any(cell >= resolution))
        return;

    float2 cell_size = 1;

    float d = (u_grid[cell + int2(1,0)] - u_grid[cell]) / cell_size.x +
              (v_grid[cell + int2(0,1)] - v_grid[cell]) / cell_size.y;
    divergence[cell] = d;
    p_grid_out[cell] = 0;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void pressure_projection(uint3 thread: SV_DispatchThreadID)
{
    uint2 cell = thread.xy;
    if (any(cell >= resolution))
        return;

    // Solve Poisson equation ∇²p = div with Gauss-Seidel

    float p_left  = (cell.x > 0)                ? p_grid[cell - int2(1,0)] : 0.0;
    float p_right = (cell.x < resolution.y - 1) ? p_grid[cell + int2(1,0)] : 0.0;
    float p_down  = (cell.y > 0)                ? p_grid[cell - int2(0,1)] : 0.0;
    float p_up    = (cell.y < resolution.y - 1) ? p_grid[cell + int2(0,1)] : 0.0;

    float2 cell_size = 1;

    p_grid_out[cell] = (p_left + p_right + p_down + p_up - cell_size.x * cell_size.y * divergence[cell]) * 0.25;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void apply_pressure_gradient(uint3 thread: SV_DispatchThreadID)
{
    uint2 cell = thread.xy;
    if (any(cell >= resolution+1))
        return;

    // Apply pressure gradient to make velocity divergence-free

    float u = 0;
    float v = 0;

    if (all(cell < resolution))
    {
        float p = p_grid[cell];
        if (cell.x > 0) {
            u = u_grid[cell] - (p - p_grid[cell - int2(1, 0)]);
        }
        if (cell.y > 0) {
            v = v_grid[cell] - (p - p_grid[cell - int2(0,1)]);
        }
    }
    if (cell.y < resolution.y)
        u_grid[cell] = u;
    if (cell.x < resolution.x)
        v_grid[cell] = v;
}
