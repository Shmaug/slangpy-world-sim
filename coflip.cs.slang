import Math;
import RNG;

RWStructuredBuffer<float4> particle_pos_impulse;
RWStructuredBuffer<float>  particle_volume;
RWTexture2D<float> density;
RWTexture2D<float> u_grid;
RWTexture2D<float> v_grid;

RWTexture2D<float> u_weight;
RWTexture2D<float> v_weight;
RWTexture2D<float> u_sum;
RWTexture2D<float> v_sum;

RWTexture2D<float> divergence;
Texture2D<float>   p_grid;
RWTexture2D<float> p_grid_out;

uniform uint2 resolution;
uniform float cell_size;
uniform uint num_particles;
uniform float dt;
uniform uint init_seed;

// B-spline basis function (quadratic)
float bspline(float r) {
    r = abs(r);
    if (r < 0.5) return 0.75 - r * r;
    if (r < 1.5) return 0.5 * (1.5 - r) * (1.5 - r);
    return 0.0;
}

// B-spline derivative
float bspline_derivative(float r) {
    float ar = abs(r);
    float sgn = (r >= 0) ? 1.0 : -1.0;
    if (ar < 0.5) return -2.0 * r;
    if (ar < 1.5) return sgn * (ar - 1.5);
    return 0.0;
}

// Grid to particle interpolation (mimetic, divergence-free)
float2 interpolate_velocity(const float2 pos) {
    float2 vel = 0;
    
    // Interpolate u-component
    float fx = pos.x / cell_size - 0.5;
    float fy = pos.y / cell_size;
    int i0 = (int)floor(fx);
    int j0 = (int)floor(fy);
    
    for (int di = -1; di <= 2; di++) {
        for (int dj = -1; dj <= 2; dj++) {
            int i = i0 + di;
            int j = j0 + dj;
            if (i >= 0 && i < resolution.x + 1 && j >= 0 && j < resolution.y) {
                float wx = bspline(fx - i);
                float wy = bspline(fy - j);
                vel.x += u_grid[int2(i,j)] * wx * wy;
            }
        }
    }
    
    // Interpolate v-component
    fx = pos.x / cell_size;
    fy = pos.y / cell_size - 0.5;
    i0 = (int)floor(fx);
    j0 = (int)floor(fy);
    
    for (int di = -1; di <= 2; di++) {
        for (int dj = -1; dj <= 2; dj++) {
            int i = i0 + di;
            int j = j0 + dj;
            if (i >= 0 && i < resolution.x && j >= 0 && j < resolution.y + 1) {
                float wx = bspline(fx - i);
                float wy = bspline(fy - j);
                vel.y += v_grid[int2(i,j)] * wx * wy;
            }
        }
    }
    
    return vel;
}

// Compute velocity gradient for impulse advection
float2x2 compute_velocity_gradient(const float2 pos) {
    float2x2 grad = 0.0;
    
    // Gradient of u-component
    float fx = pos.x / cell_size - 0.5;
    float fy = pos.y / cell_size;
    int i0 = (int)floor(fx);
    int j0 = (int)floor(fy);
    
    for (int di = -1; di <= 2; di++) {
        for (int dj = -1; dj <= 2; dj++) {
            int i = i0 + di;
            int j = j0 + dj;
            if (i >= 0 && i < resolution.x + 1 && j >= 0 && j < resolution.y) {
                float u_val = u_grid[int2(i,j)];
                float wx = bspline(fx - i);
                float wy = bspline(fy - j);
                float dwx = bspline_derivative(fx - i) / cell_size;
                float dwy = bspline_derivative(fy - j) / cell_size;
                
                grad[0][0] += u_val * dwx * wy;   // du/cell_size
                grad[0][1] += u_val * wx * dwy;   // du/dy
            }
        }
    }
    
    // Gradient of v-component
    fx = pos.x / cell_size;
    fy = pos.y / cell_size - 0.5;
    i0 = (int)floor(fx);
    j0 = (int)floor(fy);
    
    for (int di = -1; di <= 2; di++) {
        for (int dj = -1; dj <= 2; dj++) {
            int i = i0 + di;
            int j = j0 + dj;
            if (i >= 0 && i < resolution.x && j >= 0 && j < resolution.y + 1) {
                float v_val = v_grid[int2(i,j)];
                float wx = bspline(fx - i);
                float wy = bspline(fy - j);
                float dwx = bspline_derivative(fx - i) / cell_size;
                float dwy = bspline_derivative(fy - j) / cell_size;
                
                grad[1][0] += v_val * dwx * wy;   // dv/cell_size
                grad[1][1] += v_val * wx * dwy;   // dv/dy
            }
        }
    }

    return grad;
}

// RK4 for impulse (covector transformation)
float2 advect_impulse(const float2 pos, const float2 imp) {
    float2x2 grad = compute_velocity_gradient(pos);
    // Impulse evolution: du/dt = -(∇v)ᵀ u
    return float2(-(grad[0][0] * imp.x + grad[1][0] * imp.y),
                  -(grad[0][1] * imp.x + grad[1][1] * imp.y));
};


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
    float2 pos = center + radius * u * float2(cos(angle), sin(angle));
    float2 vel = 0.5 * float2(-(pos.y - center.y), pos.x - center.x);

    particle_pos_impulse[particle_index] = float4(pos, vel);
    particle_volume[particle_index] = cell_size * cell_size * 0.5;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void particle_to_grid_clear(uint3 thread: SV_DispatchThreadID)
{
    uint2 i = thread.xy;
    if (all(i < resolution)) {
        density[i] = 0;
    }
    if (i.x < resolution.x + 1 && i.y < resolution.y) {
        u_sum[i] = 0;
        u_weight[i] = 0;
    }
    if (i.x < resolution.x && i.y < resolution.y + 1) {
        v_sum[i] = 0;
        v_weight[i] = 0;
    }
}

[shader("compute")]
[numthreads(32,1,1)]
void particle_to_grid(uint3 thread : SV_DispatchThreadID)
{
    uint particle_index = thread.y * 4096 + thread.x;
    if (particle_index >= num_particles)
        return;

    const float4 pos_impulse = particle_pos_impulse[particle_index];
    const float2 pos     = pos_impulse.xy;
    const float2 impulse = pos_impulse.zw;
    const float  volume = particle_volume[particle_index];
    
    // Transfer to u-grid
    float2 f = pos / cell_size - float2(0.5, 0);
    int2 p0 = int2(floor(f));

    for (int di = -1; di <= 1; di++) {
        for (int dj = -1; dj <= 1; dj++) {
            int2 p = p0 + int2(di,dj);
            if (all(p >= 0) && all(p < int2(resolution.x+1, resolution.y))) {
                float wx = bspline(f.x - p.x);
                float wy = bspline(f.y - p.y);
                float w = wx * wy * volume;
                u_sum[p] += impulse.x * w;
                u_weight[p] += w;
            }
        }
    }

    // Transfer to v-grid
    f = pos / cell_size - float2(0, 0.5);
    p0 = int2(floor(f));

    for (int di = -1; di <= 1; di++) {
        for (int dj = -1; dj <= 1; dj++) {
            int2 p = p0 + int2(di, dj);
            if (all(p >= 0) && all(p < int2(resolution.x, resolution.y+1))) {
                float wx = bspline(f.x - p.x);
                float wy = bspline(f.y - p.y);
                float w = wx * wy * volume;
                v_sum[p] += impulse.y * w;
                v_weight[p] += w;
            }
        }
    }

    // Transfer to density grid
    f = pos / cell_size;
    p0 = int2(floor(f));

    for (int di = -1; di <= 2; di++) {
        for (int dj = -1; dj <= 2; dj++) {
            int2 p = p0 + int2(di, dj);
            if (all(p >= 0) && all(p < resolution)) {
                float wx = bspline(f.x - (p.x + 0.5));
                float wy = bspline(f.y - (p.y + 0.5));
                float w = wx * wy * volume;
                density[p] += w / (cell_size * cell_size);
            }
        }
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void particle_to_grid_normalize(uint3 thread: SV_DispatchThreadID)
{
    uint2 i = thread.xy;
    if (i.x < resolution.x+1 && i.y < resolution.y) {
        float w = u_weight[i];
        if (w > 1e-6) u_grid[i] = u_sum[i] / w;
    }
    if (i.x < resolution.x && i.y < resolution.y+1) {
        float w = v_weight[i];
        if (w > 1e-6) v_grid[i] = v_sum[i] / w;
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
[numthreads(32,1,1)]
void advect_particles(uint3 thread : SV_DispatchThreadID)
{
    uint particle_index = thread.y * 4096 + thread.x;
    if (particle_index >= num_particles)
        return;

    const float4 pos_impulse = particle_pos_impulse[particle_index];
    const float2 pos = pos_impulse.xy;
    const float2 impulse = pos_impulse.zw;
    
    // RK4 for position
    float2 k1_pos = interpolate_velocity(pos);
    float2 k2_pos = interpolate_velocity(pos + k1_pos * (dt * 0.5));
    float2 k3_pos = interpolate_velocity(pos + k2_pos * (dt * 0.5));
    float2 k4_pos = interpolate_velocity(pos + k3_pos * dt);
    float2 new_pos = pos + (k1_pos + k2_pos * 2.0 + k3_pos * 2.0 + k4_pos) * (dt / 6.0);
    
    float2 k1_imp = advect_impulse(pos, impulse);
    float2 k2_imp = advect_impulse(pos + k1_pos * (dt * 0.5), impulse + k1_imp * (dt * 0.5));
    float2 k3_imp = advect_impulse(pos + k2_pos * (dt * 0.5), impulse + k2_imp * (dt * 0.5));
    float2 k4_imp = advect_impulse(pos + k3_pos * dt,         impulse + k3_imp * dt);
    float2 new_impulse = impulse + (k1_imp + k2_imp * 2.0 + k3_imp * 2.0 + k4_imp) * (dt / 6.0);
    
    // Boundary handling (simple bounce-back)
    if (new_pos.x < 0) { new_pos.x = 0; new_impulse.x *= -0.5; }
    if (new_pos.x > resolution.x * cell_size) { new_pos.x = resolution.x * cell_size; new_impulse.x *= -0.5; }
    if (new_pos.y < 0) { new_pos.y = 0; new_impulse.y *= -0.5; }
    if (new_pos.y > resolution.y * cell_size) { new_pos.y = resolution.y * cell_size; new_impulse.y *= -0.5; }

    particle_pos_impulse[particle_index] = float4(new_pos, new_impulse);
}
