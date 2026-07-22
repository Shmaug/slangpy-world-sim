import FluidCube;

uniform FluidCube fluid;
Texture2D<float> sum_pre;
Texture2D<float> sum_post;
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
        pre += sum_pre.Load(uint3(i, 0, mip_count-1));
        post += sum_post.Load(uint3(i, 0, mip_count-1));
    }
    fluid.smoke_rw[thread] *= abs(post) > 0 ? pre / post : 0;
}