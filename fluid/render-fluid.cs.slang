import Math;
import FluidGrid;

uniform FluidGrid grid;

RWTexture2D<float4> render_target;

uniform int4  screen_rect;
uniform uint2 screen_size;
uniform uint  render_mode;

enum RenderMode
{
    Pressure = 0,
    Velocity = 1,
    Divergence = 2,
    Curl = 3,
}

float3 viridis(float t) {
    const float3 c0 = float3(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
    const float3 c1 = float3(0.1050930431085774, 1.404613529898575, 1.384590162594685);
    const float3 c2 = float3(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
    const float3 c3 = float3(-4.634230498983486, -5.799100973351585, -19.33244095627987);
    const float3 c4 = float3(6.228269936347081, 14.17993336680509, 56.69055260068105);
    const float3 c5 = float3(4.776384997670288, -13.74514537774601, -65.35303263337234);
    const float3 c6 = float3(-5.435455855934631, 4.645852612178535, 26.3124352495832);
    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
}

[shader("compute")]
[numthreads(8,8,1)]
void render(int2 pixel : SV_DispatchThreadID)
{
    if (any(pixel >= screen_size))
        return;

    float4 c = float4(0,0,0,1);

    float2 uv = (pixel + 0.5 - screen_rect.xy) / float2(screen_rect.zw - screen_rect.xy);
    float2 grid_pos = uv * grid.resolution;
    if (all(grid_pos >= 0) && all(grid_pos < grid.resolution))
    {
        if (render_mode == RenderMode::Pressure)
        {
            float p = grid.sample_pressure(grid_pos);
            c.rgb = saturate(float3(1 - exp(p), 1 - exp(-p), 0.1));
        }
        else if (render_mode == RenderMode::Velocity)
        {
            float2 vel = grid.sample_velocity(grid_pos);
            c.rgb = float3(100 * (vel / grid.resolution) * .5 + .5, 1);
        }
        else if (render_mode == RenderMode::Divergence)
        {
            int2 cell = int2(floor(grid_pos));
            float vel_x0 = grid.velocity_x[cell];
            float vel_x1 = grid.velocity_x[cell + uint2(1,0)];
            float vel_y0 = grid.velocity_y[cell];
            float vel_y1 = grid.velocity_y[cell + uint2(0,1)];

            float div = (vel_x1 - vel_x0) + (vel_y1 - vel_y0);
            c.rgb = saturate(float3(-div, div, 0) / max(grid.resolution.x, grid.resolution.y));
        }
        else if (render_mode == RenderMode::Curl)
        {
            int2 cell = int2(floor(grid_pos));
            float vx0 = grid.velocity_x_rw[wrap_index(int2(cell) + int2(0, -1), grid.resolution + uint2(1, 0))];
            float vx1 = grid.velocity_x_rw[cell];
            float vy0 = grid.velocity_y_rw[wrap_index(int2(cell) + int2(-1, 0), grid.resolution + uint2(0, 1))];
            float vy1 = grid.velocity_y_rw[cell];

            float curl = (vx1 - vx0) + (vy1 - vy0);
            c.rgb = saturate(float3(-curl, curl, 0) / max(grid.resolution.x, grid.resolution.y));
        }
    }

    render_target[pixel] = c;
}