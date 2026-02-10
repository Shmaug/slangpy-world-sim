RWTexture2D<float> density;
RWTexture2D<float> velocity_x;
RWTexture2D<float> velocity_y;

uniform uint2 dim;

[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 index : SV_DispatchThreadID)
{
    uint2 pos = index.xy;
    if (any(pos >= dim))
        return;

    const float r = max(dim.x, dim.y)/10.0;
    const float2 s0 = dim * 0.5 - float2(r * 1.5, r * 0.5);
    const float2 s1 = dim - s0;

    float2 d0 = (pos + 0.5) - (s0 + 0.5);
    float2 d1 = (pos + 0.5) - (s1 + 0.5);

    float d = 0;
    float2 v = 0;
    if (dot(d0, d0) < r*r)
    {
        v = float2(1, 0);// + 0.25 * float2(-d0.y, -d0.x) / r;
        d = 1;
    }
    if (dot(d1, d1) < r*r)
    {
        v = float2(-1, 0);// - 0.25 * float2(-d1.y, -d1.x) / r;
        d = 1;
    }

    density[pos] = d;
    velocity_x[pos] = v.x;
    velocity_y[pos] = v.y;
}