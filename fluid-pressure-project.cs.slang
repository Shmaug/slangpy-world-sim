import Math;
import FluidGrid;

uniform FluidGrid grid;
RWTexture2D<float> divergence;
Texture2D<float>   pressure_correction;
RWTexture2D<float> pressure_correction_rw;
uniform float dt;

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
void step(uint3 thread: SV_DispatchThreadID)
{
    uint2 cell = thread.xy;
    if (any(cell >= grid.resolution))
        return;

    float p = 0;
    p += pressure_correction[wrap_index(cell - int2(1, 0), grid.resolution)];
    p += pressure_correction[wrap_index(cell + int2(1, 0), grid.resolution)];
    p += pressure_correction[wrap_index(cell - int2(0, 1), grid.resolution)];
    p += pressure_correction[wrap_index(cell + int2(0, 1), grid.resolution)];

    p -= divergence[cell];

    pressure_correction_rw[cell] = p * 0.25;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void apply(uint3 thread: SV_DispatchThreadID)
{
    uint2 cell = thread.xy;
    if (any(cell >= grid.resolution))
        return;

    float p = pressure_correction[cell];
    float px = pressure_correction[wrap_index(cell - int2(1, 0), grid.resolution)];
    float py = pressure_correction[wrap_index(cell - int2(0, 1), grid.resolution)];

    grid.velocity_x_rw[cell] -= dt * (p - px);
    grid.velocity_y_rw[cell] -= dt * (p - py);
}