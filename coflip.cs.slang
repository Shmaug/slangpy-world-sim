import Math;
import RNG;
import ParticleMap;

RWTexture2D<float> density_grid;
RWTexture2D<float> u_grid;
RWTexture2D<float> v_grid;

RWTexture2D<float> divergence;
Texture2D<float>   p_grid;
RWTexture2D<float> p_grid_out;

uniform uint2 resolution;
uniform float dt;
uniform uint  init_seed;

uniform ParticleMap particle_map;

[Differentiable]
float2 interpolate_velocity(const float2 pos) {
    float2 vel = 0;

    int2 p0 = int2(floor(pos));
    
    for (int di = -2; di <= 2; di++)
    {
        for (int dj = -2; dj <= 2; dj++)
        {
            int2 cell = p0 + int2(di, dj);

            if (cell.x >= 0 && cell.x < resolution.x + 1 && cell.y >= 0 && cell.y < resolution.y)
            {
                float wx = bspline(pos.x - cell.x);
                float wy = bspline(pos.y - (cell.y+.5));
                vel.x += u_grid[cell] * wx * wy;
            }

            if (cell.x >= 0 && cell.x < resolution.x && cell.y >= 0 && cell.y < resolution.y + 1)
            {
                float wx = bspline(pos.x - (cell.x+.5));
                float wy = bspline(pos.y - cell.y);
                vel.y += v_grid[cell] * wx * wy;
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

float2 integrate_velocity_rk4(const float2 pos, const float2 velocity, out float2 new_velocity)
{
    // https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods

    // Impulse evolution: du/dt = -(∇v)ᵀ u
    float2x2 grad;

    float2 k1 = interpolate_velocity_gradient(pos, grad);
    float2 k1_vel = -mul(grad, velocity);

    float2 k2 = interpolate_velocity_gradient(pos + k1 * dt / 2.0, grad);
    float2 k2_vel = -mul(grad, velocity + k1_vel * dt / 2.0);

    float2 k3 = interpolate_velocity_gradient(pos + k2 * dt / 2.0, grad);
    float2 k3_vel = -mul(grad, velocity + k2_vel * dt/ 2.0);

    float2 k4 = interpolate_velocity_gradient(pos + k3 * dt, grad);
    float2 k4_vel = -mul(grad, velocity + k3_vel * dt);

    new_velocity = velocity + (dt / 6.0) * (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel);

    return pos + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void init(uint3 thread: SV_DispatchThreadID)
{
    uint2  cell = thread.xy;

    #if 0

    static const float2 center = resolution * 0.5 + 0.5;
    static const float radius = max(resolution.x, resolution.y)*0.25;

    float2 px = (cell + float2(0, 0.5) - center);
    float2 py = (cell + float2(0.5, 0) - center);

    float2 vx = 10 * float2(-px.y, px.x) / max(resolution.x, resolution.y);
    float2 vy = 10 * float2(-py.y, py.x) / max(resolution.x, resolution.y);

    if (cell.y < resolution.y && cell.x < resolution.x+1)
        u_grid[cell] = cell.x > 0 && cell.x < resolution.x ? vx.x : 0;
    if (cell.x < resolution.x && cell.y < resolution.y+1)
        v_grid[cell] = cell.y > 0 && cell.y < resolution.y ? vy.y : 0;
    if (all(cell < resolution))
        density_grid[cell] = length(cell + 0.5 - center) < radius ? 1 : 0;

    #else
    
    uint4 box = uint4(resolution.x / 4, (resolution.y * 7) / 8, (resolution.x * 3) / 4, resolution.y);
    if (all(cell >= box.xy) && all(cell < box.zw))
    {
        v_grid[cell] = -10;
        density_grid[cell] = 1;
    }

    #endif
}

[shader("compute")]
[numthreads(8, 8, 1)]
void emit_smoke(uint3 thread: SV_DispatchThreadID)
{
    uint2 cell = thread.xy;
    uint4 box = uint4(resolution.x / 4, resolution.y - 5, (resolution.x * 3) / 4, resolution.y);
    if (all(cell >= box.xy) && all(cell < box.zw))
    {
        v_grid[cell] = -10;
        density_grid[cell] = 1;
    }
}

void splat_particle(float2 pos, float2 vel, float density)
{
    int2 p0 = (int2)floor(pos);

    float d_norm = 0;
    for (int x = p0.x - 2; x <= p0.x + 2; x++)
    {
        for (int y = p0.y - 2; y <= p0.y + 2; y++)
        {
            int2 cell = int2(x, y);

            float wcx = bspline(pos.x - (cell.x + 0.5));
            float wcy = bspline(pos.y - (cell.y + 0.5));

            if (all(cell >= 0) && any(cell < resolution + 1))
                d_norm += wcx * wcy;
        }
    }

    for (int x = p0.x - 2; x <= p0.x + 2; x++)
    {
        for (int y = p0.y - 2; y <= p0.y + 2; y++)
        {
            int2 cell = int2(x, y);

            float wcx = bspline(pos.x - (cell.x + 0.5));
            float wcy = bspline(pos.y - (cell.y + 0.5));

            float wu = bspline(pos.x - cell.x) * wcy;
            float wv = bspline(pos.y - cell.y) * wcx;

            if (all(cell >= 0) && any(cell < resolution+1))
            {
                particle_map.insert(cell.y * (resolution.x + 1) + cell.x, float4(vel, wu, wv), d_norm > 0 ? density * wcx * wcy / d_norm : 0);
            }
        }
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void advect(uint3 thread : SV_DispatchThreadID)
{
    uint2 cell = thread.xy;
    if (any(cell >= resolution))
        return;

    const float2 pos = cell + 0.5;
    const float2 vel = interpolate_velocity(pos);
    const float  density = density_grid[cell];

    float2 new_vel;
    float2 new_pos = integrate_velocity_rk4(pos, vel, new_vel);

    // Boundary handling (simple bounce-back)
    if (new_pos.x < 0)             { new_pos.x =                 -new_pos.x; new_vel.x *= -1; }
    if (new_pos.x >= resolution.x) { new_pos.x = 2*resolution.x - new_pos.x; new_vel.x *= -1; }
    if (new_pos.y < 0)             { new_pos.y =                 -new_pos.y; new_vel.y *= -1; }
    if (new_pos.y >= resolution.y) { new_pos.y = 2*resolution.y - new_pos.y; new_vel.y *= -1; }

    splat_particle(new_pos, new_vel, density);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void particle_to_grid(uint3 thread: SV_DispatchThreadID)
{
    const int2 cell = int2(thread.xy);
    if (any(cell >= resolution + 1))
        return;

    float density_sum = 0;

    // left edge of cell
    float u_sum = 0;
    float u_weight = 0;

    // top edge of cell
    float v_sum = 0;
    float v_weight = 0;

    float2 cp = (cell + 0.5) / 4 + 1;
    int2 c0 = select(fract(cp) < 0.5, int2(cp - 1), int2(cp));

    uint2 range = particle_map.get_range(cell.y * (resolution.x+1) + cell.x);
    for (int j = range.x; j < range.y; j++)
    {
        float4 v = particle_map.get_v(j);
        float  d = particle_map.get_d(j);

        float2 impulse = v.xy;
        float wu = v.z;
        float wv = v.w;

        density_sum += d;

        u_sum += impulse.x * wu;
        u_weight += wu;

        v_sum += impulse.y * wv;
        v_weight += wv;
    }

    if (all(cell < resolution)) {
        density_grid[cell] = density_sum;
    }
    if (cell.x < resolution.x+1 && cell.y < resolution.y) {
        u_grid[cell] = cell.x > 0 && cell.x < resolution.x && u_weight > 1e-6 ? u_sum / u_weight : 0;
    }
    if (cell.x < resolution.x && cell.y < resolution.y+1) {
        v_grid[cell] = cell.y > 0 && cell.y < resolution.y && v_weight > 1e-6 ? v_sum / v_weight : 0;
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
