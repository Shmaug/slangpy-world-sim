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
}