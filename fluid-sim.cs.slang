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
uniform float K; // 1/pressure
uniform float dt;

float load_density(int2 coord)
{
    return all(coord >= 0) && all(coord < dim) ? density[coord] : 0;
}

float2 interpolate_velocity(float2 coord)
{
    int2 cell = int2(floor(clamp(coord, 0, dim-1)));

    float vx0 = velocity_x[cell];
    float vx1 = velocity_x[cell + int2(1,0)];
    float vy0 = velocity_y[cell];
    float vy1 = velocity_y[cell + int2(0, 1)];

    float2 w = frac(coord);

    return float2(lerp(vx0, vx1, w.x), lerp(vy0, vy1, w.y));
}

[shader("compute")]
[numthreads(8,8,1)]
void advect(uint3 index : SV_DispatchThreadID)
{
    int2 cell = int2(2 * index.x + ((index.y + checker_iteration) & 1), index.y);
    if (any(cell >= dim))
        return;

    float vx0 = velocity_x[cell];
    float vx1 = velocity_x[cell + int2(1, 0)];
    float vy0 = velocity_y[cell];
    float vy1 = velocity_y[cell + int2(0, 1)];
    float2 velocity = float2(vx1 + vx0, vy0 + vy1) * 0.5;

    float2 Was = (cell + 0.5) - dt * velocity;
    density_out[cell] = density.SampleLevel(linear_sampler, Was / dim, 0);

    float2 advected = interpolate_velocity(Was);

    float2 ax = advected.x * 0.5;
    float2 ay = advected.y * 0.5;

    if (checker_iteration > 0)
    {
        ax[0] += velocity_x_out[cell];
        ax[1] += velocity_x_out[cell + int2(1, 0)];
        ay[0] += velocity_y_out[cell];
        ay[1] += velocity_y_out[cell + int2(0, 1)];
    }

    velocity_x_out[cell]              = ax[0];
    velocity_x_out[cell + int2(1, 0)] = ax[1];
    velocity_y_out[cell]              = ay[0];
    velocity_y_out[cell + int2(0, 1)] = ay[1];
}

// Implements eqs 22 and 23
// https://cg.informatik.uni-freiburg.de/intern/seminar/gridFluids_fluid-EulerParticle.pdf
[shader("compute")]
[numthreads(8, 8, 1)]
void pressure_solve(uint3 index: SV_DispatchThreadID)
{
    int2 pos = index.xy;

    float p = load_density(pos);

    if (pos.y < dim.y)
    {
        float vx = 0;
        if (pos.x > 0 && pos.x <= dim.x)
        {
            float px = load_density(pos - int2(1, 0));
            vx = velocity_x_out[pos] - dt * K * (p - px);
        }
        velocity_x_out[pos] = vx;
    }

    if (pos.x < dim.x)
    {
        float vy = 0;
        if (pos.y > 0 && pos.y <= dim.x)
        {
            float py = load_density(pos + int2(0, 1));
            vy = velocity_y_out[pos] - dt * K * (p - py);
        }
        velocity_y_out[pos] = vy;
    }
}

// Remove divergence
[shader("compute")]
[numthreads(8, 8, 1)]
void pressure_project(uint3 index: SV_DispatchThreadID)
{
    int2 cell = int2(2 * index.x + ((index.y + checker_iteration) & 1), index.y);
    if (any(cell >= dim))
        return;

    float vx0 = velocity_x[cell];
    float vx1 = velocity_x[cell + int2(1, 0)];
    float vy0 = velocity_y[cell];
    float vy1 = velocity_y[cell + int2(0, 1)];

    float divergence = vx1 - vx0 + vy1 - vy0;

    velocity_x_out[cell] -= divergence * 0.25;
    velocity_x_out[cell + int2(1, 0)] += divergence * 0.25;
    velocity_y_out[cell] -= divergence * 0.25;
    velocity_y_out[cell + int2(0, 1)] += divergence * 0.25;
}