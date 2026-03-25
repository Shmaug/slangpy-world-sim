import Math;

struct FluidGrid
{
    Texture2D<float>   velocity_x;
    Texture2D<float>   velocity_y;
    Texture2D<float>   pressure;

    RWTexture2D<float> velocity_x_rw;
    RWTexture2D<float> velocity_y_rw;
    RWTexture2D<float> pressure_rw;

    SamplerState linear_sampler;

    uniform uint2 resolution;

    float sample_pressure(float2 grid_pos)
    {
        return pressure.SampleLevel(linear_sampler, grid_pos / resolution, 0);
    }

    float2 sample_velocity(float2 grid_pos)
    {
        return float2(
            velocity_x.SampleLevel(linear_sampler, (grid_pos - float2(0, 0.5) + 0.5) / float2(resolution.x+1, resolution.y), 0),
            velocity_y.SampleLevel(linear_sampler, (grid_pos - float2(0.5, 0) + 0.5) / float2(resolution.x, resolution.y+1), 0),
        );
    }

    float sample_pressure_bspline(float2 grid_pos)
    {
        float p = 0;
        float wp = 0;

        for (int y = -2; y <= 2; y++)
        {
            for (int x = -2; x <= 2; x++)
            {
                int2 cell = (int2(grid_pos) + int2(x, y) + resolution) % resolution;

                float2 d = grid_pos - (cell + 0.5);

                float w = bspline(d.x) * bspline(d.y);
                p += w * pressure[cell];
                wp += w;
            }
        }

        return wp > 0 ? p / wp : 0;
    }

    float2 sample_velocity_bspline(float2 grid_pos)
    {
        float vx = 0;
        float wvx = 0;

        float vy = 0;
        float wvy = 0;

        for (int y = -2; y <= 2; y++)
        {
            for (int x = -2; x <= 2; x++)
            {
                int2 cell = (int2(grid_pos) + int2(x, y) + resolution) % resolution;

                float2 dx = grid_pos - float2(cell.x, cell.y + 0.5);
                float2 dy = grid_pos - float2(cell.x + 0.5, cell.y);

                float wx = bspline(dx.x) * bspline(dx.y);
                float wy = bspline(dy.x) * bspline(dy.y);

                vx += wx * velocity_x[cell];
                vy += wy * velocity_y[cell];

                wvx += wx;
                wvy += wy;
            }
        }

        return float2(wvx > 0 ? vx / wvx : 0, wvy > 0 ? vy / wvy : 0);
    }
}