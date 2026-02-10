import Math;

Texture2D<float> density;
Texture2D<float> velocity_x;
Texture2D<float> velocity_y;
SamplerState linear_sampler;

RWTexture2D<float> density_out;
RWTexture2D<float> velocity_x_out;
RWTexture2D<float> velocity_y_out;

uniform uint2 dim;
uniform uint checker_iteration;
uniform float K;
uniform float dt;

float load_density(int2 coord)
{
    return all(coord >= 0) && all(coord < dim) ? density[coord] : 0;
}

float2 sample_velocity(float2 coord)
{
    int2 cell = clamp(int2(floor(coord)), 0, dim-1);

    float vx0 = velocity_x[cell];
    float vx1 = velocity_x[cell + int2(1, 0)];
    float vy0 = velocity_y[cell];
    float vy1 = velocity_y[cell + int2(0, 1)];

    float d  = density[cell];
    if (cell.x > 0)     vx0 -= d - density[cell + int2(-1,  0)];
    if (cell.x < dim.x) vx1 += d - density[cell + int2( 1,  0)];
    if (cell.y > 0)     vy0 -= d - density[cell + int2( 0, -1)];
    if (cell.x < dim.y) vy1 += d - density[cell + int2( 0,  1)];

    float2 w = frac(clamp(coord, 0, dim - 1));
    float vx = lerp(vx0, vx1, w.x);
    float vy = lerp(vy0, vy1, w.y);
    return float2(vx, vy);
}

[shader("compute")]
[numthreads(8,8,1)]
void advect(uint3 index : SV_DispatchThreadID)
{
    int2 cell = int2(2 * index.x + ((index.y + checker_iteration) & 1), index.y);
    if (any(cell >= dim))
        return;

    float2 velocity = sample_velocity(cell + 0.5);

    float2 p = (cell + 0.5) - dt * velocity;
    
    density_out[cell] = density.SampleLevel(linear_sampler, p / dim, 0);

    float2 advected_velocity = sample_velocity(p);

    float avx0 = advected_velocity.x * 0.5;
    float avx1 = advected_velocity.x * 0.5;
    float avy0 = advected_velocity.y * 0.5;
    float avy1 = advected_velocity.y * 0.5;
    if (checker_iteration != 0)
    {
        avx0 += velocity_x_out[cell];
        avx1 += velocity_x_out[cell + int2(1, 0)];
        avy0 += velocity_y_out[cell];
        avy1 += velocity_y_out[cell + int2(0, 1)];
    }

    velocity_x_out[cell]              = avx0;
    velocity_x_out[cell + int2(1, 0)] = avx1;
    velocity_y_out[cell]              = avy0;
    velocity_y_out[cell + int2(0, 1)] = avy1;
}
