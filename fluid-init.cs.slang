import FluidGrid;

uniform FluidGrid grid;

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
