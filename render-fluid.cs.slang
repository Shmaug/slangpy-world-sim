RWTexture2D<float4> render_target;

Texture2D<float> density;
Texture2D<float> velocity_x;
Texture2D<float> velocity_y;
SamplerState sampler;

uniform uint4 screen_rect;
uniform uint2 screen_size;
uniform uint  render_mode;

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
void main(uint2 pixel : SV_DispatchThreadID)
{
    if (any(pixel >= screen_size))
        return;

    float4 c = float4(0,0,0,1);

    if (all(pixel >= screen_rect.xy) && all(pixel < screen_rect.zw))
    {
        float2 duv = 1 / float2(screen_rect.zw - screen_rect.xy);
        float2 uv = (pixel - screen_rect.xy + 0.5) * duv;
        if (render_mode == 0)
            c.rgb = viridis(saturate(density.SampleLevel(sampler, uv, 0)));
        else
        {
            float vx0 = velocity_x.SampleLevel(sampler, uv, 0);
            float vy0 = velocity_y.SampleLevel(sampler, uv, 0);
            float vx1 = velocity_x.SampleLevel(sampler, uv + float2(duv.x, 0), 0);
            float vy1 = velocity_y.SampleLevel(sampler, uv + float2(0, duv.y), 0);
            c.rgb = float3(.5+(vx0 + vx1) * .5, .5+(vy0 + vy1) * .5, 1);
        }
    }
    render_target[pixel] = c;
}