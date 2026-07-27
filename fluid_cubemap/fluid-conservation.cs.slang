import SphericalField;

uniform SphericalScalarField fluid_pre;
uniform SphericalScalarField fluid_post;
uniform uint mip_count;

void conserve<int N>(CubeCoord cell)
{
    vector<float, N> pre = 0;
    vector<float, N> post = 0;
    for (uint face = 0; face < 6 * fluid_pre.vertical_resolution; face++)
    {
        pre  += fluid_pre.load<N>(CubeCoord(0, face, cell.layer, cell.channel), mip_count - 1);
        post += fluid_post.load<N>(CubeCoord(0, face, cell.layer, cell.channel), mip_count - 1);
    }
    fluid_post.store<N>(cell, fluid_post.load<N>(cell) * select(abs(post) > 0, pre / post, 0));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void conserve_smoke(uint3 thread: SV_DispatchThreadID)
{
    CubeCoord cell = CubeCoord(thread.xy, thread.z % 6, thread.z / 6, 0);
    if (!fluid_post.bounds_check(cell))
        return;

    for (cell.channel = 0; cell.channel < fluid_pre.channels; cell.channel += 4)
    {
        uint c_left = fluid_pre.channels - cell.channel;
        if (c_left >= 4)
        {
            conserve<4>(cell);
        }
        else if (c_left == 3)
        {
            conserve<3>(cell);
        }
        else if (c_left == 2)
        {
            conserve<2>(cell);
        }
        else if (c_left == 1)
        {
            conserve<1>(cell);
        }
    }
}