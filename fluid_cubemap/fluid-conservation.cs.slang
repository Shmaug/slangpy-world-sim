import FluidCube;

uniform FluidCube fluid;
uniform uint mip_count;

[shader("compute")]
[numthreads(8, 8, 1)]
void conserve_smoke(uint3 thread: SV_DispatchThreadID)
{
    if (any(thread.xy >= fluid.resolution) || thread.z >= 6)
        return;

    float pre = 0;
    float post = 0;
    for (uint i = 0; i < 6; i++)
    {
        pre += fluid._smoke[fluid.flatten_index(uint3(0,0,i), mip_count - 1)];
        post += fluid._smoke_rw[fluid.flatten_index(uint3(0,0,i), mip_count - 1)];
    }
    fluid._smoke_rw[fluid.flatten_index(thread)] *= (abs(post) > 0 ? pre / post : 0);
}