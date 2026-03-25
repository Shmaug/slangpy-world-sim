import Math;
import FluidGrid;
import ParticleMap;

uniform FluidGrid grid;
uniform ParticleMap particle_map;
uniform float advect_dt;

float2 integrate_velocity_linear(float2 grid_pos, float dt)
{
    #if 1
    // https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    float2 k1 = grid.sample_velocity(grid_pos);
    float2 k2 = grid.sample_velocity(grid_pos + k1 * dt / 2.0);
    float2 k3 = grid.sample_velocity(grid_pos + k2 * dt / 2.0);
    float2 k4 = grid.sample_velocity(grid_pos + k3 * dt);
    grid_pos += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
    #else
    // basic Euler integration
    grid_pos += dt * grid.sample_velocity(grid_pos);
    #endif

    grid_pos = (grid_pos + grid.resolution) % grid.resolution;

    return grid_pos;
}

float2 integrate_velocity_bspline(float2 grid_pos, float dt)
{
    #if 1
    // https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    float2 k1 = grid.sample_velocity_bspline(grid_pos);
    float2 k2 = grid.sample_velocity_bspline(grid_pos + k1 * dt / 2.0);
    float2 k3 = grid.sample_velocity_bspline(grid_pos + k2 * dt / 2.0);
    float2 k4 = grid.sample_velocity_bspline(grid_pos + k3 * dt);
    grid_pos += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
    #else
    // basic Euler integration
    grid_pos += dt * grid.sample_velocity_bspline(grid_pos);
    #endif

    grid_pos = (grid_pos + grid.resolution) % grid.resolution;

    return grid_pos;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void advect_grid(uint3 thread : SV_DispatchThreadID)
{
    uint2 cell = thread.xy;

    float2 pos_x = integrate_velocity_linear(float2(cell.x,       cell.y + 0.5), -advect_dt);
    float2 pos_y = integrate_velocity_linear(float2(cell.x + 0.5, cell.y),       -advect_dt);
    float2 pos_c = integrate_velocity_linear(float2(cell.x + 0.5, cell.y + 0.5), -advect_dt);

    if (cell.x < grid.resolution.x + 1 && cell.y < grid.resolution.y)
    {
        grid.velocity_x_rw[cell] = grid.sample_velocity(pos_x).x;
    }

    if (cell.x < grid.resolution.x && cell.y < grid.resolution.y + 1)
    {
        grid.velocity_y_rw[cell] = grid.sample_velocity(pos_y).y;
    }

    if (cell.x < grid.resolution.x && cell.y < grid.resolution.y)
    {
        grid.pressure_rw[cell] = grid.sample_pressure(pos_c);
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void advect_particle(uint3 thread : SV_DispatchThreadID)
{
    uint2 cell = thread.xy;
    if (any(cell >= grid.resolution))
        return;

    float pressure = grid.sample_pressure_bspline(float2(cell) + 0.5);
    float2 vel = grid.sample_velocity_bspline(float2(cell) + 0.5);
    float2 pos = integrate_velocity_bspline(float2(cell) + 0.5, advect_dt);

    Particle p;
    p.pos_in_cell = frac(pos);
    p.velocity = vel;
    p.pressure = pressure;
    particle_map.insert(int(pos.y) * grid.resolution.x + int(pos.x), p);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void particle_to_grid(uint3 thread: SV_DispatchThreadID)
{
    int2 dst_cell = thread.xy;

    float pressure = 0;
    float w_pressure = 0;
    
    float vel_x = 0;
    float w_vel_x = 0;

    float vel_y = 0;
    float w_vel_y = 0;

    for (int y = -2; y <= 2; y++)
    {
        for (int x = -2; x <= 2; x++)
        {
            int2 cell = (dst_cell + int2(x,y) + grid.resolution) % grid.resolution;
            uint2 range = particle_map.get_range(cell.y * grid.resolution.x + cell.x);
            for (uint i = range.x; i < range.y; i++)
            {
                Particle p = particle_map.get(i);

                float2 pos = cell + p.pos_in_cell;

                float2 dx = pos - float2(dst_cell.x, dst_cell.y + 0.5);
                float wx = bspline(dx.x) * bspline(dx.y);
                vel_x += p.velocity.x * wx;
                w_vel_x += wx;

                float2 dy = pos - float2(dst_cell.x + 0.5, dst_cell.y);
                float wy = bspline(dy.x) * bspline(dy.y);
                vel_y += p.velocity.y * wy;
                w_vel_y += wy;

                float2 dc = pos - (dst_cell + 0.5);
                float wc = bspline(dc.x) * bspline(dc.y);
                pressure += p.pressure * wc;
                w_pressure += wc;
            }
        }
    }

    if (dst_cell.x < grid.resolution.x + 1 && dst_cell.y < grid.resolution.y)
        grid.velocity_x_rw[dst_cell] = w_vel_x > 0 ? vel_x / w_vel_x : 0;
    if (dst_cell.x < grid.resolution.x && dst_cell.y < grid.resolution.y + 1)
        grid.velocity_y_rw[dst_cell] = w_vel_y > 0 ? vel_y / w_vel_y : 0;
    if (dst_cell.x < grid.resolution.x && dst_cell.y < grid.resolution.y)
        grid.pressure_rw[dst_cell] = w_pressure > 0 ? pressure / w_pressure : 0;
}
