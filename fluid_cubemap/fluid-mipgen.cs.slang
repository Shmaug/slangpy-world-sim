import SphericalField;

SphericalScalarField field;
uint dst_mip;

void generate<int N>(uint3 thread, uint channel)
{
    vector<float, N> s = 0;
    s += field.load<N>(CubeCoord(thread.xy * 2 + uint2(0, 0), thread.z % 6, thread.z / 6, channel), dst_mip - 1);
    s += field.load<N>(CubeCoord(thread.xy * 2 + uint2(1, 0), thread.z % 6, thread.z / 6, channel), dst_mip - 1);
    s += field.load<N>(CubeCoord(thread.xy * 2 + uint2(0, 1), thread.z % 6, thread.z / 6, channel), dst_mip - 1);
    s += field.load<N>(CubeCoord(thread.xy * 2 + uint2(1, 1), thread.z % 6, thread.z / 6, channel), dst_mip - 1);
    field.store<N>(CubeCoord(thread.xy, thread.z % 6, thread.z / 6, channel), 0.25 * s, dst_mip);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void generate_mip(uint3 thread: SV_DispatchThreadID)
{
    uint res = field.resolution >> dst_mip;
    if (any(thread.xy >= res) || thread.z >= 6 * field.vertical_resolution)
        return;

    for (uint channel = 0; channel < field.channels; channel += 4)
    {
        uint c_left = field.channels - channel;
        if (c_left >= 4)
        {
            generate<4>(thread, channel);
        }
        else if (c_left == 3)
        {
            generate<3>(thread, channel);
        }
        else if (c_left == 2)
        {
            generate<2>(thread, channel);
        }
        else if (c_left == 1)
        {
            generate<1>(thread, channel);
        }        
    }
}
