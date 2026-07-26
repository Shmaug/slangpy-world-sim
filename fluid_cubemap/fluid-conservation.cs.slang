import SphericalField;

uniform SphericalScalarField fluid_pre;
uniform SphericalScalarField fluid_post;
uniform uint mip_count;

[shader("compute")]
[numthreads(8, 8, 1)]
void conserve_smoke(uint3 thread: SV_DispatchThreadID)
{
    if (any(thread.xy >= fluid_post.resolution) || thread.z >= 6 * fluid_pre.vertical_resolution)
        return;

    // sum across the whole shell (all layers, not just this thread's own
    // layer) since radial advection now moves smoke between layers -- the
    // conserved quantity is the shell's total mass, not each layer's
    float pre = 0;
    float post = 0;
    for (uint face = 0; face < 6 * fluid_pre.vertical_resolution; face++)
    {
        pre += fluid_pre.load(uint3(0, 0, face), mip_count - 1);
        post += fluid_post.load(uint3(0, 0, face), mip_count - 1);
    }
    fluid_post.store(thread, fluid_post.load(thread) * (abs(post) > 0 ? pre / post : 0));
}