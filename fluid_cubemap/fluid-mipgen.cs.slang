import FluidCube;

RWStructuredBuffer<float> data;
uint resolution;
uint dst_mip;

[shader("compute")]
[numthreads(8, 8, 1)]
void generate_mip(uint3 thread: SV_DispatchThreadID)
{
    uint res = resolution >> dst_mip;
    if (any(thread.xy >= res) || thread.z >= 6)
        return;

    data[flatten_cube_texel(thread, resolution, dst_mip)] = 0.25 * (
        data[flatten_cube_texel(uint3(thread.xy*2 + uint2(0, 0), thread.z), resolution, dst_mip - 1)] +
        data[flatten_cube_texel(uint3(thread.xy*2 + uint2(1, 0), thread.z), resolution, dst_mip - 1)] +
        data[flatten_cube_texel(uint3(thread.xy*2 + uint2(0, 1), thread.z), resolution, dst_mip - 1)] +
        data[flatten_cube_texel(uint3(thread.xy*2 + uint2(1, 1), thread.z), resolution, dst_mip - 1)]);
}
