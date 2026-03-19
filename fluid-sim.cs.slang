import Math;
import FluidGrid;

uniform FluidGrid grid;
RWTexture2D<float> divergence;
Texture2D<float>   pressure_correction;
RWTexture2D<float> pressure_correction_rw;
uniform float dt;

[shader("compute")]
[numthreads(8, 8, 1)]
void init_vortex(uint3 thread: SV_DispatchThreadID)
{
    uint2 cell = thread.xy;
    if (any(cell >= grid.resolution+1))
        return;

    const float2 circles[2] = {
        float2(.5, .5) * grid.resolution,
        float2(.75, .5) * grid.resolution,
    };
    const float radius = 0.3 * min(grid.resolution.x, grid.resolution.y);

    float pressure = 0;
    float2 vel = 0;

    for (uint i = 0; i < 1; i++)
    {
        float2 s = (cell + 0.5) - circles[i];
        
        if (length(s) < radius)
        {
            pressure += 0.5;
        }

        vel += float2(-s.y, s.x) * 0.1;
    }

    if (all(cell < grid.resolution))
        grid.pressure_rw[cell] = pressure;
    if (cell.y < grid.resolution.y)
        grid.velocity_x_rw[cell] = vel.x;
    if (cell.x < grid.resolution.x)
        grid.velocity_y_rw[cell] = vel.y;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void emit_smoke(uint3 thread: SV_DispatchThreadID)
{
    uint2 cell = thread.xy;
    if (any(cell >= grid.resolution + 1))
        return;

    static const float4 box = float4(
        grid.resolution.x * 0.25, grid.resolution.y * 0.8,
        grid.resolution.x * 0.75, grid.resolution.y * 0.9);

    float pressure = grid.pressure[cell];
    float2 vel = float2(grid.velocity_x[cell], grid.velocity_y[cell]);

    if (all(cell >= box.xy) && all(cell < box.zw))
    {
        pressure = 1;
        vel = float2(0, -0.1) * grid.resolution;
    }

    if (all(cell < grid.resolution))
        grid.pressure_rw[cell] = pressure;
    if (cell.y < grid.resolution.y)
        grid.velocity_x_rw[cell] = vel.x;
    if (cell.x < grid.resolution.x)
        grid.velocity_y_rw[cell] = vel.y;
}

float2 integrate_velocity(float2 grid_pos)
{
    #if 0
    // https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    float2 k1 = grid.sample_velocity(grid_pos);
    float2 k2 = grid.sample_velocity(grid_pos + k1 * dt / 2.0);
    float2 k3 = grid.sample_velocity(grid_pos + k2 * dt / 2.0);
    float2 k4 = grid.sample_velocity(grid_pos + k3 * dt);
    grid_pos += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
    #else
    // basic Euler integration
    grid_pos += -dt * grid.sample_velocity(grid_pos);
    #endif

    grid_pos = grid_pos % grid.resolution;

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

[shader("compute")]
[numthreads(8, 8, 1)]
void update_divergence(uint3 thread : SV_DispatchThreadID)
{
    uint2 cell = thread.xy;
    if (any(cell >= grid.resolution))
        return;

    float vel_x0 = grid.velocity_x_rw[cell];
    float vel_x1 = grid.velocity_x_rw[cell + int2(1,0)];
    float vel_y0 = grid.velocity_y_rw[cell];
    float vel_y1 = grid.velocity_y_rw[cell + int2(0,1)];

    divergence[cell] = (vel_x1 - vel_x0) + (vel_y1 - vel_y0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void pressure_project_step(uint3 thread: SV_DispatchThreadID)
{
    uint2 cell = thread.xy;
    if (any(cell >= grid.resolution))
        return;

    float p = 0;
    p += pressure_correction[(cell - int2(1, 0) + grid.resolution) % grid.resolution];
    p += pressure_correction[(cell + int2(1, 0) + grid.resolution) % grid.resolution];
    p += pressure_correction[(cell - int2(0, 1) + grid.resolution) % grid.resolution];
    p += pressure_correction[(cell + int2(0, 1) + grid.resolution) % grid.resolution];

    p -= divergence[cell];

    pressure_correction_rw[cell] = p * 0.25;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void pressure_project_apply(uint3 thread: SV_DispatchThreadID)
{
    uint2 cell = thread.xy;
    if (any(cell >= grid.resolution))
        return;

    float p = pressure_correction[cell];
    float px = pressure_correction[(cell - int2(1, 0) + grid.resolution) % grid.resolution];
    float py = pressure_correction[(cell - int2(0, 1) + grid.resolution) % grid.resolution];

    grid.velocity_x_rw[cell] -= dt * (p - px);
    grid.velocity_y_rw[cell] -= dt * (p - py);
}