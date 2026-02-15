import Math;
import RNG;
import ParticleMap;

RWStructuredBuffer<float4> particle_pos_impulse;
RWStructuredBuffer<float>  particle_volume;
RWTexture2D<float> density_grid;
RWTexture2D<float> u_grid;
RWTexture2D<float> v_grid;

RWTexture2D<float> divergence;
Texture2D<float>   p_grid;
RWTexture2D<float> p_grid_out;

uniform uint2 resolution;
uniform float cell_size;
uniform uint  num_particles;
uniform float dt;
uniform uint  init_seed;

uniform ParticleMap particle_map;

[Differentiable]
float2 interpolate_velocity(const float2 pos) {
    int2 p = (int2)floor(pos / cell_size);
    return float2(u_grid[p], v_grid[p]);
    /*
    float2 vel = 0;
    
    float2 fu = pos / cell_size - float2(0.5, 0);
    float2 fv = pos / cell_size - float2(0, 0.5);
    no_diff int2 fu0 = (int2)floor(fu);
    no_diff int2 fv0 = (int2)floor(fv);
    
    for (int di = -2; di <= 2; di++)
    {
        for (int dj = -2; dj <= 2; dj++)
        {
            int2 pu = fu0 + int2(di, dj);
            if (pu.x >= 0 && pu.x < resolution.x + 1 && pu.y >= 0 && pu.y < resolution.y)
            {
                float wx = bspline(fu.x - pu.x);
                float wy = bspline(fu.y - pu.y);
                vel.x += u_grid[pu] * wx * wy;
            }
            int2 pv = fv0 + int2(di, dj);
            if (pv.x >= 0 && pv.x < resolution.x && pv.y >= 0 && pv.y < resolution.y + 1)
            {
                float wx = bspline(fv.x - pv.x);
                float wy = bspline(fv.y - pv.y);
                vel.y += v_grid[pv] * wx * wy;
            }
        }
    }

    return vel;
    */
}

float2 interpolate_velocity_gradient(float2 pos, out float2x2 grad)
{
    var dv_dx = diffPair(pos);
    __bwd_diff(interpolate_velocity)(dv_dx, float2(1, 0));
    var dv_dy = diffPair(pos);
    __bwd_diff(interpolate_velocity)(dv_dy, float2(0, 1));

    grad[0] = dv_dx.d;
    grad[1] = dv_dy.d;
    return dv_dy.p;
}

float4 integrate_velocity(const float2 pos, const float2 impulse)
{
    #if 1

    float2x2 grad_k1;
    float2 vel = interpolate_velocity_gradient(pos, grad_k1);

    float2 new_pos = pos + vel * dt;
    float2 new_impulse = impulse;// - mul(grad_k1, impulse);*dt;

    #else

    // RK4
    
    float2x2 grad_k1;
    float2 k1_vel = interpolate_velocity_gradient(pos, grad_k1);
    float2 k1_imp = -mul(grad_k1, impulse);

    float2x2 grad_k2;
    float2 k2_vel = interpolate_velocity_gradient(pos + k1_vel * (dt * 0.5), grad_k2);
    float2 k2_imp = -mul(grad_k2, impulse + k1_imp * (dt * 0.5));

    float2x2 grad_k3;
    float2 k3_vel = interpolate_velocity_gradient(pos + k2_vel * (dt * 0.5), grad_k3);
    float2 k3_imp = -mul(grad_k3, impulse + k2_imp * (dt * 0.5));

    float2x2 grad_k4;
    float2 k4_vel = interpolate_velocity_gradient(pos + k3_vel * dt, grad_k4);
    float2 k4_imp = -mul(grad_k4, impulse + k3_imp * dt);

    float2 new_vel = pos + (k1_vel + k2_vel * 2.0 + k3_vel * 2.0 + k4_vel) * (dt / 6.0);
    float2 new_impulse = impulse + (k1_imp + k2_imp * 2.0 + k3_imp * 2.0 + k4_imp) * (dt / 6.0);

    #endif

    return float4(new_pos, new_impulse);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void init_particles(uint3 thread: SV_DispatchThreadID)
{
    uint particle_index = thread.y * 4096 + thread.x;
    if (particle_index >= num_particles)
        return;

    static const float2 center = 0.5;
    static const float  radius = 0.25;

    RNG rng = RNG(pcg(particle_index + pcg(init_seed)));
    float u = rng.nextFloat();

    float angle = 2.0 * M_PI * particle_index / float(num_particles);
    float2 pos = center + radius * sqrt(u) * float2(cos(angle), sin(angle));
    float2 vel = 0.5 * float2(-(pos.y - center.y), pos.x - center.x);

    particle_pos_impulse[particle_index] = float4(pos, vel);
    particle_volume[particle_index] = cell_size * cell_size * 0.5;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void build_particle_map(uint3 thread : SV_DispatchThreadID)
{
    uint particle_index = thread.y * 4096 + thread.x;
    if (particle_index >= num_particles)
        return;

    const float4 pos_impulse = particle_pos_impulse[particle_index];
    const float2 pos = pos_impulse.xy;
    particle_map.insert(int2(floor(pos / cell_size)), particle_index);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void particle_to_grid(uint3 thread: SV_DispatchThreadID)
{
    const int2 cell = int2(thread.xy);

    float u_sum = 0; // left edge of cell
    float u_weight = 0; // left edge of cell
    float v_sum = 0; // top edge of cell
    float v_weight = 0; // top edge of cell
    float density = 0;

    for (int x = cell.x - 2; x <= cell.x + 2; x++)
    {
        for (int y = cell.y - 2; y <= cell.y + 2; y++)
        {
            const uint2 range = particle_map.get_range(int2(x,y));
            for (int j = range.x; j < range.y; j++)
            {
                uint   particle_index = particle_map.get(j);
                float4 pos_impulse = particle_pos_impulse[particle_index];
                float  volume = particle_volume[particle_index];
                float2 pos     = pos_impulse.xy;
                float2 impulse = pos_impulse.zw;
                float2 f = pos / cell_size - cell;

                float wx  = bspline(f.x);
                float wy  = bspline(f.y);
                float wcx = bspline(f.x - 0.5);
                float wcy = bspline(f.y - 0.5);

                density += wcx * wcy * volume / (cell_size * cell_size);

                float wu = wx * wcy * volume;
                u_sum += impulse.x * wu;
                u_weight += wu;

                float wv = wcx * wy * volume;
                v_sum += impulse.y * wv;
                v_weight += wv;
            }
        }
    }
    
    if (all(cell < resolution)) {
        density_grid[cell] = density;
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

    float d = (u_grid[cell + int2(1,0)] - u_grid[cell]) / cell_size +
              (v_grid[cell + int2(0,1)] - v_grid[cell]) / cell_size;
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
    
    p_grid_out[cell] = (p_left + p_right + p_down + p_up - cell_size * cell_size * divergence[cell]) * 0.25;
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
            u = u_grid[cell] - (p - p_grid[cell - int2(1, 0)]) / cell_size;
        }
        if (cell.y > 0) {
            v = v_grid[cell] - (p - p_grid[cell - int2(0,1)]) / cell_size;
        }
    }
    if (cell.y < resolution.y)
        u_grid[cell] = u;
    if (cell.x < resolution.x)
        v_grid[cell] = v;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void advect_particles(uint3 thread : SV_DispatchThreadID)
{
    uint particle_index = thread.y * 4096 + thread.x;
    if (particle_index >= num_particles)
        return;

    const float4 pos_impulse = particle_pos_impulse[particle_index];

    float4 new_pos_impulse = integrate_velocity(pos_impulse.xy, pos_impulse.zw);

    // Boundary handling (simple bounce-back)
    if (new_pos_impulse.x < 0) { new_pos_impulse.x =    -new_pos_impulse.x; new_pos_impulse.z *= -1; }
    if (new_pos_impulse.x > 1) { new_pos_impulse.x = 2 - new_pos_impulse.x; new_pos_impulse.z *= -1; }
    if (new_pos_impulse.y < 0) { new_pos_impulse.y =    -new_pos_impulse.y; new_pos_impulse.w *= -1; }
    if (new_pos_impulse.y > 1) { new_pos_impulse.y = 2 - new_pos_impulse.y; new_pos_impulse.w *= -1; }

    particle_pos_impulse[particle_index] = new_pos_impulse;
}
