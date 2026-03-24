import Math;
import FluidGrid;
// import ParticleMap;

uniform FluidGrid grid;
// uniform ParticleMap particle_map;
uniform float dt;

float2 integrate_velocity(float2 grid_pos)
{
    #if 1
    // https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    float2 k1 = grid.sample_velocity(grid_pos);
    float2 k2 = grid.sample_velocity(grid_pos + k1 * dt / 2.0);
    float2 k3 = grid.sample_velocity(grid_pos + k2 * dt / 2.0);
    float2 k4 = grid.sample_velocity(grid_pos + k3 * dt);
    grid_pos -= (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
    #else
    // basic Euler integration
    grid_pos -= dt * grid.sample_velocity(grid_pos);
    #endif

    grid_pos = (grid_pos + grid.resolution) % grid.resolution;

    return grid_pos;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void advect(uint3 thread : SV_DispatchThreadID)
{
    uint2 cell = thread.xy;

    float2 pos_x = integrate_velocity(float2(cell.x,       cell.y + 0.5));
    float2 pos_y = integrate_velocity(float2(cell.x + 0.5, cell.y));
    float2 pos_c = integrate_velocity(float2(cell.x + 0.5, cell.y + 0.5));

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
