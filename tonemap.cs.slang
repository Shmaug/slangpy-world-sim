float3 aces_film(float3 x)
{
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

RWTexture2D<float4> image;
uniform float exposure;

[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 index : SV_DispatchThreadID)
{
    uint2 dim;
    image.GetDimensions(dim.x, dim.y);
    if (all(index.xy < dim))
    {
        float4 color = image[index.xy];
        color.rgb *= exp2(exposure);
        color.rgb = aces_film(color.rgb);
        color.rgb = pow(color.rgb, 1.0 / 2.2);
        image[index.xy] = color;
    }
}