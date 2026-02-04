import Math;

// Implements "Simple and Fast Fluids": https://inria.hal.science/inria-00596050/document

Texture2D<float>  density;
Texture2D<float2> velocity;

RWTexture2D<float>  density_out;
RWTexture2D<float2> velocity_out;

uniform uint2 dim;
uniform float K; // 0.2
uniform float viscosity;
uniform float dt;

float2 sample_forces(int2 coord)
{
    return 0;
}
bool is_boundary(int2 coord)
{
    return false;
}

float load_density(int2 coord)
{
    if (coord.y < 0)
    {
        coord.y = 0;
        coord.x += dim.x / 2;
    }
    else if (coord.y >= dim.y)
    {
        coord.y = dim.y-1;
        coord.x += dim.x / 2;
    }
    coord.x = (coord.x + dim.x) % dim.x;
    return density[coord];
}

float2 load_velocity(int2 coord)
{
    bool flip = false;
    if (coord.y < 0)
    {
        coord.y = 0;
        coord.x += dim.x / 2;
        flip = true;
    }
    else if (coord.y >= dim.y)
    {
        coord.y = dim.y - 1;
        coord.x += dim.x / 2;
        flip = true;
    }
    coord.x = (coord.x + dim.x) % dim.x;

    float2 v = velocity[coord];
    if (flip) v.y = -v.y;
    return v;

}

float3 load_field(int2 coord)
{
    bool flip = false;
    if (coord.y < 0)
    {
        coord.y = 0;
        coord.x += dim.x / 2;
        flip = true;
    }
    else if (coord.y >= dim.y)
    {
        coord.y = dim.y - 1;
        coord.x += dim.x / 2;
        flip = true;
    }
    coord.x = (coord.x + dim.x) % dim.x;

    float3 f = float3(velocity[coord], density[coord]);
    if (flip) f.y = -f.y;
    return f;
}

float2 interpolate_velocity(float2 coord)
{
    int2 p00 = int2(floor(coord));
    p00 = select(frac(coord) < 0.5, p00 - 1, p00);
    int2 p10 = p00 + int2(1,0);
    int2 p01 = p00 + int2(0,1);
    int2 p11 = p00 + int2(1,1);

    float2 v00 = load_velocity(p00);
    float2 v10 = load_velocity(p10);
    float2 v01 = load_velocity(p01);
    float2 v11 = load_velocity(p11);

    float2 w = frac(coord - 0.5);

    return lerp(lerp(v00, v10, w.x), lerp(v01, v11, w.x), w.y);
}

[shader("compute")]
[numthreads(8,8,1)]
void advect(uint3 index : SV_DispatchThreadID)
{
    int2 pos = index.xy;
    if (any(pos >= dim))
        return;

    float CScale = 0.5;
    float S = K / dt;

    float3 FC = load_field(pos);
    float3 FR = load_field(pos + int2( 1, 0));
    float3 FL = load_field(pos + int2(-1, 0));
    float3 FT = load_field(pos + int2( 0, 1));
    float3 FD = load_field(pos + int2( 0,-1));

    float4x3 fieldMat = float4x3(FR, FL, FT, FD);

    // du/dx, du/dy
    float3 UdX = (fieldMat[0] - fieldMat[1]) * CScale;
    float3 UdY = (fieldMat[2] - fieldMat[3]) * CScale;

    float Udiv = UdX.x + UdY.y;
    float2 DdX = float2(UdX.z, UdY.z);

    // Solve for density
    FC.z -= dt * dot(float3(DdX, Udiv), FC);

    // stability enhancement
    // FC.z = clamp(FC.z, 0.5, 3.0);

    // Solve for velocity
    float2 PdX = S * DdX;
    float2 laplacian = mul(float4(1, 1, 1, 1), (float4x2)fieldMat) - 4 * FC.xy;
    float2 viscosityForce = viscosity * laplacian;

    // semi-lagrangian advection
    float2 Was = (pos + 0.5) - dt * FC.xy;
    FC.xy = interpolate_velocity(Was);

    FC.xy += dt * (viscosityForce - PdX + sample_forces(pos));

    // boundary condition
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            int2 dir = select(int2(i, j) == 0, int2(-1), int2(1));
            if (is_boundary(pos + dir))
            {
                // zero the velocity in the boundary direction
                FC.xy *= 1 - abs(dir);
            }
        }
    }

    velocity_out[index.xy] = FC.xy;
    density_out[index.xy] = FC.z;
}