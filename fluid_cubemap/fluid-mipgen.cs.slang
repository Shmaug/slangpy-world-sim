import SphericalField;

SphericalScalarField field;
uint dst_mip;

[shader("compute")]
[numthreads(8, 8, 1)]
void generate_mip(uint3 thread: SV_DispatchThreadID)
{
    uint res = field.resolution >> dst_mip;
    if (any(thread.xy >= res) || thread.z >= 6)
        return;

    float f00 = field.load(uint3(thread.xy*2 + uint2(0, 0), thread.z), dst_mip - 1);
    float f10 = field.load(uint3(thread.xy*2 + uint2(1, 0), thread.z), dst_mip - 1);
    float f01 = field.load(uint3(thread.xy*2 + uint2(0, 1), thread.z), dst_mip - 1);
    float f11 = field.load(uint3(thread.xy*2 + uint2(1, 1), thread.z), dst_mip - 1);

    field.store(thread, 0.25 * (f00 + f10 + f01 + f11), dst_mip);
}
